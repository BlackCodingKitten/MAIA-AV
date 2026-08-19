from __future__ import annotations

import os

GPU_IDS = [
    gpu.strip()
    for gpu in os.environ.get("MAIA_GPU_IDS", "3,5,2,7").split(",")
    if gpu.strip()
]

if len(GPU_IDS) != 4:
    raise RuntimeError(
        f"MAIA_GPU_IDS deve contenere esattamente 4 GPU: {GPU_IDS}"
    )

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(GPU_IDS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_FLASHINFER_SAMPLING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["VLLM_CUSTOM_ALL_REDUCE"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["VLLM_RPC_TIMEOUT"] = "300"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import argparse
import json
import traceback

from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

EVENT_DIR = Path(
    "data/preliminar_analysis/event/qwen-30B"
)

OUTPUT_DIR = Path(
    "data/preliminar_analysis/causal/qwen-30B"
)

MAX_MODEL_LEN = 32768
MAX_NEW_TOKENS = 4096
MAX_RETRIES = 3

ALLOWED_RELATIONS = {
    "causes",
    "enables",
    "motivates",
    "prevents",
}

ALLOWED_EVIDENCE = {
    "direct",
    "inferred",
    "uncertain",
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}: il JSON principale non è un oggetto."
        )

    return data


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def is_complete_output(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        data = read_json(path)
    except Exception:
        return False

    return (
        "error" not in data
        and isinstance(
            data.get("causal_relations"),
            list,
        )
    )


def parse_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError(
            "Risposta vuota dall'LLM."
        )

    cleaned = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start < 0:
        raise ValueError(
            "La risposta non contiene un oggetto JSON."
        )

    candidate = (
        cleaned[start:end + 1]
        if end >= start
        else cleaned[start:]
    )

    try:
        result = json.loads(candidate)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json

        repaired = repair_json(candidate)
        result = json.loads(repaired)

    except Exception as exc:
        raise ValueError(
            "JSON non recuperabile."
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Il risultato recuperato non è un oggetto JSON."
        )

    return result


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        return [value]

    return []


def compact(
    payload: dict[str, Any],
) -> dict[str, Any]:
    events = [
        event
        for event in as_list(
            payload.get("events")
        )
        if isinstance(event, dict)
    ]

    temporal_relations = [
        relation
        for relation in as_list(
            payload.get("temporal_relations")
        )
        if isinstance(relation, dict)
    ]

    return {
        "id_video": payload.get(
            "id_video"
        ),

        "events": [
            {
                "event_id": event.get(
                    "event_id"
                ),

                "description": event.get(
                    "description"
                ),

                "start_time": event.get(
                    "start_time"
                ),

                "end_time": event.get(
                    "end_time"
                ),

                "participants": (
                    event.get("participants")
                    if isinstance(
                        event.get("participants"),
                        list,
                    )
                    else []
                ),

                "evidence_segments": (
                    event.get("evidence_segments")
                    if isinstance(
                        event.get("evidence_segments"),
                        list,
                    )
                    else []
                ),

                "evidence_type": event.get(
                    "evidence_type"
                ),

                "confidence": event.get(
                    "confidence"
                ),
            }
            for event in events
        ],

        "temporal_relations": temporal_relations,
    }


def build_prompt(
    payload: dict[str, Any],
) -> str:
    return (
        "Analizza la rappresentazione consolidata degli eventi "
        "di un video e individua esclusivamente le relazioni "
        "causali supportate dagli eventi forniti.\n\n"

        "Regole:\n"

        "- usa esclusivamente gli eventi e le relazioni temporali "
        "presenti nell'input;\n"

        "- non usare domande, caption, foil o informazioni esterne;\n"

        "- la successione temporale non implica causalità;\n"

        "- crea una relazione soltanto quando gli eventi supportano "
        "una dipendenza causa-effetto, una condizione abilitante, "
        "una motivazione oppure una prevenzione;\n"

        "- cause_event ed effect_event devono essere ID di eventi "
        "presenti nell'input;\n"

        "- relation deve essere uno tra causes, enables, motivates "
        "e prevents;\n"

        "- evidence_type deve essere direct, inferred oppure uncertain;\n"

        "- usa direct quando la relazione è chiaramente supportata "
        "dall'interazione o dal cambiamento osservato;\n"

        "- usa inferred quando è necessaria un'inferenza contestuale "
        "o intenzionale;\n"

        "- usa uncertain soltanto quando esiste un indizio causale "
        "ma l'evidenza è debole;\n"

        "- supporting_events deve contenere esclusivamente ulteriori "
        "eventi presenti nell'input;\n"

        "- non inventare eventi, oggetti, intenzioni o cause;\n"

        "- causal_relations deve essere sempre un array JSON;\n"

        "- ogni elemento di causal_relations deve essere "
        "un oggetto JSON;\n"

        "- se non esistono relazioni sufficientemente supportate, "
        "restituisci causal_relations come lista vuota;\n"

        "- restituisci esclusivamente JSON valido.\n\n"

        "Schema:\n"

        "{\n"
        '  "causal_relations": [\n'
        "    {\n"
        '      "cause_event": "E0001",\n'
        '      "relation": "causes|enables|motivates|prevents",\n'
        '      "effect_event": "E0002",\n'
        '      "evidence_type": "direct|inferred|uncertain",\n'
        '      "supporting_events": ["E0003"],\n'
        '      "explanation": "breve spiegazione basata sugli eventi",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n\n"

        "INPUT:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def confidence_value(
    value: Any,
) -> float:
    try:
        confidence = float(
            str(value).replace(",", ".")
        )
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, confidence),
    )


def normalize_result(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    valid_ids = {
        event["event_id"]
        for event in payload.get(
            "events",
            [],
        )
        if event.get("event_id")
    }

    relations = as_list(
        result.get("causal_relations")
    )

    normalized = []
    seen = set()

    for relation in relations:
        if not isinstance(
            relation,
            dict,
        ):
            continue

        cause = relation.get(
            "cause_event"
        )

        effect = relation.get(
            "effect_event"
        )

        relation_type = str(
            relation.get(
                "relation",
                "",
            )
        ).strip().lower()

        evidence = str(
            relation.get(
                "evidence_type",
                "inferred",
            )
        ).strip().lower()

        if (
            cause not in valid_ids
            or effect not in valid_ids
            or cause == effect
        ):
            continue

        if relation_type not in ALLOWED_RELATIONS:
            continue

        if evidence not in ALLOWED_EVIDENCE:
            evidence = "inferred"

        key = (
            cause,
            relation_type,
            effect,
        )

        if key in seen:
            continue

        seen.add(key)

        supporting = relation.get(
            "supporting_events",
            [],
        )

        if isinstance(
            supporting,
            str,
        ):
            supporting = [supporting]

        if not isinstance(
            supporting,
            list,
        ):
            supporting = []

        supporting = list(
            dict.fromkeys(
                event_id
                for event_id in supporting
                if (
                    event_id in valid_ids
                    and event_id
                    not in {
                        cause,
                        effect,
                    }
                )
            )
        )

        normalized.append(
            {
                "cause_event": cause,

                "relation": relation_type,

                "effect_event": effect,

                "evidence_type": evidence,

                "supporting_events": supporting,

                "explanation": str(
                    relation.get(
                        "explanation",
                        "",
                    )
                ).strip(),

                "confidence": confidence_value(
                    relation.get(
                        "confidence",
                        0.0,
                    )
                ),
            }
        )

    return {
        "causal_relations": normalized
    }


class Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        gpu_utilization: float,
    ) -> None:
        from vllm import LLM, SamplingParams

        print(
            "\n"
            "========================================\n"
            "QWEN3-OMNI CAUSAL ANALYSIS\n"
            "========================================\n"
            f"Model: {model_id}\n"
            f"GPU fisiche: {','.join(GPU_IDS)}\n"
            "Tensor parallel: 4\n"
            "Pipeline parallel: 1\n"
            f"GPU utilization: {gpu_utilization}\n"
            "========================================",
            flush=True,
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",

            tensor_parallel_size=4,

            gpu_memory_utilization=gpu_utilization,

            max_model_len=MAX_MODEL_LEN,

            max_num_seqs=1,

            disable_custom_all_reduce=True,

            enforce_eager=True,

            trust_remote_code=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    def __call__(
        self,
        prompt: str,
    ) -> str:
        outputs = self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if (
            not outputs
            or not outputs[0].outputs
        ):
            raise RuntimeError(
                "vLLM non ha restituito alcun output."
            )

        return (
            outputs[0]
            .outputs[0]
            .text
            .strip()
        )


def infer_json(
    inferencer: Inferencer,
    prompt: str,
    attempts: int = MAX_RETRIES,
) -> dict[str, Any]:
    last_error = None
    raw = ""

    for attempt in range(
        1,
        attempts + 1,
    ):
        current_prompt = prompt

        if attempt > 1:
            current_prompt += (
                "\n\nATTENZIONE: la risposta precedente non "
                "rispettava lo schema richiesto. "
                "Restituisci esclusivamente l'oggetto JSON."
            )

        try:
            raw = inferencer(
                current_prompt
            )

            return parse_json(
                raw
            )

        except Exception as exc:
            last_error = exc

            print(
                f"    Tentativo {attempt}/{attempts} fallito: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    raise RuntimeError(
        f"JSON non ottenuto dopo {attempts} tentativi. "
        f"Ultimo errore: {last_error}. "
        f"Ultima risposta: {raw[:500]!r}"
    )


def process_video(
    path: Path,
    inferencer: Inferencer,
    output_directory: Path,
    model_id: str,
    overwrite: bool,
    index: int,
    total: int,
) -> bool:
    payload = read_json(
        path
    )

    video_id = (
        payload.get("id_video")
        or path.stem.removesuffix(
            "_events"
        )
    )

    output_path = (
        output_directory
        / f"{video_id}_causal.json"
    )

    if (
        not overwrite
        and is_complete_output(
            output_path
        )
    ):
        print(
            f"[{index}/{total}] SKIP {video_id}",
            flush=True,
        )

        return False

    print(
        f"[{index}/{total}] {video_id}",
        flush=True,
    )

    compact_payload = compact(
        payload
    )

    result = infer_json(
        inferencer,
        build_prompt(
            compact_payload
        ),
    )

    result = normalize_result(
        result,
        compact_payload,
    )

    result.update(
        {
            "id_video": video_id,

            "model": model_id,

            "source_event_file": str(
                path
            ),
        }
    )

    write_json(
        output_path,
        result,
    )

    print(
        f"    Salvato: {output_path.name}"
        f" | Relazioni causali: "
        f"{len(result['causal_relations'])}",
        flush=True,
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Causal analysis Qwen3-Omni-30B "
            "con vLLM TP=4."
        )
    )

    parser.add_argument(
        "event_directory",
        nargs="?",
        type=Path,
        default=EVENT_DIR,
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
    )

    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    files = sorted(
        args.event_directory.glob(
            "*_events.json"
        )
    )

    if args.limit_videos > 0:
        files = files[
            :args.limit_videos
        ]

    if not files:
        parser.error(
            "Nessun file *_events.json trovato in "
            f"{args.event_directory}"
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    inferencer = Inferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_utilization=args.gpu_utilization,
    )

    completed = 0
    skipped = 0
    failed = 0

    for index, path in enumerate(
        files,
        start=1,
    ):
        try:
            if process_video(
                path=path,
                inferencer=inferencer,
                output_directory=args.output_directory,
                model_id=args.model,
                overwrite=args.overwrite,
                index=index,
                total=len(files),
            ):
                completed += 1
            else:
                skipped += 1

        except Exception as exc:
            failed += 1

            print(
                f"\n[ERROR] {path.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            traceback.print_exc()

    print(
        "\n"
        "========================================\n"
        "ELABORAZIONE COMPLETATA\n"
        "========================================\n"
        f"Totali:     {len(files)}\n"
        f"Completati: {completed}\n"
        f"Saltati:    {skipped}\n"
        f"Falliti:    {failed}\n"
        "========================================",
        flush=True,
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()