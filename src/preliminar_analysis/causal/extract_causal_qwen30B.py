from __future__ import annotations

import os

# ==============================================================================
# CONFIGURAZIONE GPU / MULTIPROCESSING
# ==============================================================================
# Default: prime 4 GPU fisiche. Per scegliere altre GPU senza modificare il file:
#   MAIA_GPU_IDS=1,2,4,6 python causal_qwen30b_4gpu_safe.py
GPU_IDS = [gpu.strip() for gpu in os.environ.get("MAIA_GPU_IDS", "0,1,2,3").split(",") if gpu.strip()]
if len(GPU_IDS) != 4:
    raise RuntimeError(
        f"MAIA_GPU_IDS deve contenere esattamente 4 GPU, ricevute: {GPU_IDS}"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(GPU_IDS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Non disabilitiamo P2P o InfiniBand a priori. In vLLM sono workaround
# diagnostici, non una configurazione da imporre sempre.
# In caso di errore NCCL il processo deve fallire invece di restare appeso.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "1")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "180")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1048576")
os.environ.setdefault("NCCL_DEBUG", "WARN")
# Se viene scelto TP, vLLM verifica realmente il P2P prima di usare custom all-reduce.
os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "0")

import argparse
import json
from pathlib import Path

# vLLM viene importato solo quando viene costruito l'Inferencer, quindi dopo
# il bootstrap del modulo e con tutte le variabili di ambiente già impostate.

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
EVENT_DIR = Path("data/preliminar_analysis/event/qwen-30B")
OUTPUT_DIR = Path("data/preliminar_analysis/causal/qwen-30B")

ALLOWED_RELATIONS = {"causes", "enables", "motivates", "prevents"}
ALLOWED_EVIDENCE = {"direct", "inferred", "uncertain"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json(text: str) -> dict:
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("La risposta non contiene un oggetto JSON.")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        from json_repair import repair_json
        return json.loads(repair_json(candidate))


def compact(payload: dict) -> dict:
    return {
        "id_video": payload.get("id_video"),
        "events": [
            {
                "event_id": event.get("event_id"),
                "description": event.get("description"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "participants": event.get("participants", []),
                "evidence_segments": event.get("evidence_segments", []),
                "evidence_type": event.get("evidence_type"),
                "confidence": event.get("confidence"),
            }
            for event in payload.get("events", [])
        ],
        "temporal_relations": payload.get("temporal_relations", []),
    }


def build_prompt(payload: dict) -> str:
    return f"""Analizza la rappresentazione consolidata degli eventi di un video e individua esclusivamente le relazioni causali supportate dagli eventi forniti.

Regole:
- usa esclusivamente gli eventi e le relazioni temporali presenti nell'input;
- non usare domande, caption, foil o altre informazioni esterne;
- la semplice successione temporale NON implica causalità;
- crea una relazione soltanto quando il contenuto degli eventi supporta una dipendenza causa-effetto, una condizione abilitante, una motivazione oppure una prevenzione;
- cause_event ed effect_event devono essere ID di eventi presenti nell'input;
- usa evidence_type="direct" quando la dipendenza è chiaramente supportata dall'interazione o dal cambiamento osservato;
- usa evidence_type="inferred" quando è necessaria un'inferenza contestuale o intenzionale;
- usa evidence_type="uncertain" solo quando esiste un indizio causale ma l'evidenza è debole;
- supporting_events contiene soltanto eventuali ulteriori eventi necessari a sostenere la relazione;
- non inventare eventi, oggetti, intenzioni o cause non ricavabili dalla rappresentazione fornita;
- se non esiste evidenza sufficiente, restituisci causal_relations come lista vuota;
- restituisci ESCLUSIVAMENTE un oggetto JSON valido.

Schema:
{{
  "causal_relations": [
    {{
      "cause_event": "E0001",
      "relation": "causes|enables|motivates|prevents",
      "effect_event": "E0002",
      "evidence_type": "direct|inferred|uncertain",
      "supporting_events": ["E0003"],
      "explanation": "breve spiegazione basata esclusivamente sugli eventi forniti",
      "confidence": 0.0
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


def normalize_result(result: dict, payload: dict) -> dict:
    valid_ids = {event.get("event_id") for event in payload.get("events", []) if event.get("event_id")}
    normalized = []
    seen = set()

    for relation in result.get("causal_relations", []):
        if not isinstance(relation, dict):
            continue
        cause = relation.get("cause_event")
        effect = relation.get("effect_event")
        rel = relation.get("relation")
        evidence = relation.get("evidence_type", "inferred")
        if cause not in valid_ids or effect not in valid_ids or cause == effect:
            continue
        if rel not in ALLOWED_RELATIONS:
            continue
        if evidence not in ALLOWED_EVIDENCE:
            evidence = "inferred"

        key = (cause, rel, effect)
        if key in seen:
            continue
        seen.add(key)

        supporting = [
            event_id for event_id in relation.get("supporting_events", [])
            if event_id in valid_ids and event_id not in {cause, effect}
        ]
        try:
            confidence = float(relation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        normalized.append({
            "cause_event": cause,
            "relation": rel,
            "effect_event": effect,
            "evidence_type": evidence,
            "supporting_events": list(dict.fromkeys(supporting)),
            "explanation": str(relation.get("explanation", "")).strip(),
            "confidence": max(0.0, min(1.0, confidence)),
        })

    return {"causal_relations": normalized}


class Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        gpu_utilization: float,
        parallel_mode: str,
    ) -> None:
        # Import locale: evita side effect CUDA/vLLM durante il re-import dei worker spawn.
        from vllm import LLM, SamplingParams

        if parallel_mode == "tp":
            tensor_parallel_size = 4
            pipeline_parallel_size = 1
        else:
            tensor_parallel_size = 1
            pipeline_parallel_size = 4

        print(
            f"Inizializzazione vLLM su 4 GPU: TP={tensor_parallel_size}, "
            f"PP={pipeline_parallel_size}...",
            flush=True,
        )
        print(f"GPU fisiche: {','.join(GPU_IDS)}", flush=True)

        # Default PP=4: più robusto su nodi PCIe senza NVLink e con meno
        # collettive all-reduce. TP=4 resta disponibile con --parallel-mode tp.
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            distributed_executor_backend="mp",
            distributed_timeout_seconds=180,
            cpu_distributed_timeout_seconds=180,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=32768,
            max_num_seqs=1,
            limit_mm_per_prompt={
                "image": 0,
                "audio": 0,
                "video": 0,
            },
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def __call__(self, prompt: str) -> str:
        outputs = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            sampling_params=self.sampling,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()


def infer_json(inferencer: Inferencer, prompt: str, attempts: int = 2) -> dict:
    raw = ""
    error = None
    for attempt in range(attempts):
        raw = inferencer(
            prompt if attempt == 0 else
            prompt + "\n\nLa risposta precedente non era JSON valido. Rispondi soltanto con l'oggetto JSON richiesto."
        )
        try:
            return parse_json(raw)
        except Exception as current_error:
            error = current_error
    raise ValueError(f"JSON non ottenuto dopo {attempts} tentativi. Ultima risposta: {raw[:500]!r}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal inference Qwen3-Omni-30B dagli eventi consolidati.")
    parser.add_argument("event_directory", nargs="?", type=Path, default=EVENT_DIR)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument(
        "--parallel-mode",
        choices=("pp", "tp"),
        default="pp",
        help=(
            "Parallelismo su 4 GPU. pp è il default più robusto su PCIe; "
            "tp usa Tensor Parallelism 4-way."
        ),
    )
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.event_directory.glob("*_events.json"))
    if args.limit_videos:
        files = files[:args.limit_videos]
    if not files:
        parser.error(f"Nessun *_events.json trovato in {args.event_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    inferencer = Inferencer(
        args.model,
        args.max_new_tokens,
        args.gpu_utilization,
        args.parallel_mode,
    )

    for index, path in enumerate(files, 1):
        payload = read_json(path)
        video_id = payload.get("id_video") or path.stem.removesuffix("_events")
        output_path = args.output_directory / f"{video_id}_causal.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] SKIP {video_id}", flush=True)
            continue

        mode_label = "PP=4, TP=1" if args.parallel_mode == "pp" else "PP=1, TP=4"
        print(
            f"[{index}/{len(files)}] {video_id} su GPU fisiche {','.join(GPU_IDS)} "
            f"({mode_label})",
            flush=True,
        )
        compact_payload = compact(payload)
        try:
            result = infer_json(inferencer, build_prompt(compact_payload))
            result = normalize_result(result, compact_payload)
            result.update({
                "id_video": video_id,
                "model": args.model,
                "source_event_file": str(path),
            })
        except Exception as error:
            print(f"ERRORE {video_id}: {type(error).__name__}: {error}", flush=True)
            result = {
                "id_video": video_id,
                "model": args.model,
                "source_event_file": str(path),
                "causal_relations": [],
                "error": f"{type(error).__name__}: {error}",
            }

        write_json(output_path, result)

    print(f"Creato output in: {args.output_directory}", flush=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()