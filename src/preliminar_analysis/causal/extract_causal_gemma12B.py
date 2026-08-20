from __future__ import annotations

import os

# Impostare le variabili d'ambiente PRIMA di importare torch/vLLM.
# Se CUDA_VISIBLE_DEVICES è già definita dalla shell, viene rispettata.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_USE_FLASHINFER", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams


MODEL_ID = "google/gemma-4-12B-it"

EVENT_DIR = Path("data/preliminar_analysis/event/gemma-4-12B")
OUTPUT_DIR = Path("data/preliminar_analysis/causal/gemma-4-12B")

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
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} non contiene un oggetto JSON.")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Risposta vuota.")

    cleaned = (
        text.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("La risposta non contiene un oggetto JSON.")

        candidate = cleaned[start:]

        try:
            result, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as first_error:
            try:
                from json_repair import repair_json
            except ImportError as import_error:
                raise ValueError(
                    "JSON non valido e pacchetto 'json_repair' non installato. "
                    "Installa con: pip install json-repair"
                ) from import_error

            try:
                repaired = repair_json(candidate)
                result = json.loads(repaired)
            except Exception as repair_error:
                raise ValueError(
                    f"Impossibile riparare il JSON: {repair_error}"
                ) from first_error

    if not isinstance(result, dict):
        raise ValueError("Il JSON restituito non è un oggetto.")

    return result


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events", [])
    if not isinstance(events, list):
        events = []

    temporal_relations = payload.get("temporal_relations", [])
    if not isinstance(temporal_relations, list):
        temporal_relations = []

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
            for event in events
            if isinstance(event, dict)
        ],
        "temporal_relations": temporal_relations,
    }


def build_prompt(payload: dict[str, Any]) -> str:
    input_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"""Analizza la rappresentazione consolidata degli eventi di un video e individua esclusivamente le relazioni causali supportate dagli eventi forniti.

Regole:
- usa esclusivamente gli eventi e le relazioni temporali presenti nell'input;
- non usare domande, caption, foil o altre informazioni esterne;
- la semplice successione temporale NON implica causalità;
- crea una relazione soltanto quando il contenuto degli eventi supporta una dipendenza causa-effetto, una condizione abilitante, una motivazione oppure una prevenzione;
- cause_event ed effect_event devono essere ID di eventi presenti nell'input;
- usa relation="causes" per una relazione causa-effetto;
- usa relation="enables" per una condizione che rende possibile un evento successivo;
- usa relation="motivates" per una motivazione/intenzione supportata dal contesto;
- usa relation="prevents" quando un evento impedisce o ostacola un altro evento;
- usa evidence_type="direct" quando la dipendenza è chiaramente supportata dall'interazione o dal cambiamento osservato;
- usa evidence_type="inferred" quando è necessaria un'inferenza contestuale o intenzionale;
- usa evidence_type="uncertain" solo quando esiste un indizio causale ma l'evidenza è debole;
- supporting_events contiene soltanto eventuali ulteriori eventi necessari a sostenere la relazione;
- confidence deve essere un numero tra 0 e 1;
- se non esistono relazioni causali sufficientemente supportate, restituisci una lista vuota;
- non inventare eventi, oggetti, intenzioni o cause non ricavabili dalla rappresentazione fornita;
- restituisci esclusivamente JSON valido, senza markdown e senza testo aggiuntivo.

Schema:
{{
  "causal_relations": [
    {{
      "cause_event": "E0001",
      "relation": "causes",
      "effect_event": "E0002",
      "evidence_type": "direct",
      "supporting_events": ["E0003"],
      "explanation": "breve spiegazione basata esclusivamente sugli eventi forniti",
      "confidence": 0.85
    }}
  ]
}}

INPUT:
{input_json}
"""


def normalize_result(
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    valid_ids = {
        event.get("event_id")
        for event in payload.get("events", [])
        if isinstance(event, dict) and event.get("event_id")
    }

    relations = result.get("causal_relations", [])
    if not isinstance(relations, list):
        relations = []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for relation in relations:
        if not isinstance(relation, dict):
            continue

        cause = relation.get("cause_event")
        effect = relation.get("effect_event")
        relation_type = relation.get("relation")
        evidence = relation.get("evidence_type", "inferred")

        if cause not in valid_ids:
            continue
        if effect not in valid_ids:
            continue
        if cause == effect:
            continue
        if relation_type not in ALLOWED_RELATIONS:
            continue

        if evidence not in ALLOWED_EVIDENCE:
            evidence = "inferred"

        key = (cause, relation_type, effect)
        if key in seen:
            continue
        seen.add(key)

        raw_supporting = relation.get("supporting_events", [])
        if not isinstance(raw_supporting, list):
            raw_supporting = []

        supporting = [
            event_id
            for event_id in raw_supporting
            if (
                event_id in valid_ids
                and event_id not in {cause, effect}
            )
        ]

        try:
            confidence = float(relation.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        normalized.append(
            {
                "cause_event": cause,
                "relation": relation_type,
                "effect_event": effect,
                "evidence_type": evidence,
                "supporting_events": list(dict.fromkeys(supporting)),
                "explanation": str(
                    relation.get("explanation", "")
                ).strip(),
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )

    return {"causal_relations": normalized}


class Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        gpu_utilization: float,
        tensor_parallel_size: int,
        max_model_len: int,
    ) -> None:
        visible_count = torch.cuda.device_count()

        if visible_count == 0:
            raise RuntimeError("Nessuna GPU CUDA visibile.")

        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size deve essere >= 1.")

        if visible_count < tensor_parallel_size:
            raise RuntimeError(
                f"vLLM richiede {tensor_parallel_size} GPU per il tensor parallel, "
                f"ma PyTorch ne vede soltanto {visible_count}. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
            )

        print(
            "[INIT] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
            f"GPU visibili={visible_count} | "
            f"tensor_parallel_size={tensor_parallel_size}"
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=max_model_len,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    def __call__(self, prompt: str) -> str:
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

        if not outputs:
            raise RuntimeError("vLLM non ha restituito alcun output.")

        if not outputs[0].outputs:
            raise RuntimeError("vLLM non ha generato alcuna completion.")

        return outputs[0].outputs[0].text.strip()


def infer_json(
    inferencer: Inferencer,
    prompt: str,
    attempts: int = 2,
) -> dict[str, Any]:
    raw = ""
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        current_prompt = prompt

        if attempt > 1:
            current_prompt += (
                "\n\nATTENZIONE: nel tentativo precedente l'output non era "
                "un oggetto JSON valido. Restituisci esclusivamente il JSON "
                "richiesto, senza markdown, commenti o testo aggiuntivo."
            )

        raw = inferencer(current_prompt)

        try:
            return parse_json(raw)
        except Exception as error:
            last_error = error
            print(
                f"[WARN] Parsing JSON fallito al tentativo "
                f"{attempt}/{attempts}: {error}"
            )

    raise ValueError(
        f"JSON non ottenuto dopo {attempts} tentativi. "
        f"Ultima risposta: {raw[:500]!r}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inferenza causale con Gemma-4-12B a partire dagli "
            "eventi consolidati della preliminary analysis."
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
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if not 0.0 < args.gpu_utilization <= 1.0:
        parser.error("--gpu-utilization deve essere compreso tra 0 e 1.")

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens deve essere > 0.")

    if args.max_model_len <= 0:
        parser.error("--max-model-len deve essere > 0.")

    if args.attempts <= 0:
        parser.error("--attempts deve essere > 0.")

    files = sorted(args.event_directory.glob("*_events.json"))

    if args.limit_videos > 0:
        files = files[: args.limit_videos]

    if not files:
        parser.error(
            f"Nessun file *_events.json trovato in "
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
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    total = len(files)

    for index, path in enumerate(files, start=1):
        video_id = path.stem.removesuffix("_events")
        output_path = (
            args.output_directory
            / f"{video_id}_causal.json"
        )

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] SKIP {video_id}")
            continue

        print(f"[{index}/{total}] Elaborazione {video_id}")

        try:
            payload = read_json(path)
            video_id = payload.get("id_video") or video_id

            output_path = (
                args.output_directory
                / f"{video_id}_causal.json"
            )

            compact_payload = compact(payload)

            if not compact_payload["events"]:
                raise ValueError(
                    "Il file degli eventi non contiene eventi validi."
                )

            raw_result = infer_json(
                inferencer=inferencer,
                prompt=build_prompt(compact_payload),
                attempts=args.attempts,
            )

            result = normalize_result(
                raw_result,
                compact_payload,
            )

            result.update(
                {
                    "id_video": video_id,
                    "model": args.model,
                    "source_event_file": str(path),
                }
            )

            write_json(output_path, result)

            print(
                f"[{index}/{total}] OK {video_id}: "
                f"{len(result['causal_relations'])} relazioni"
            )

        except Exception as error:
            print(
                f"[{index}/{total}] ERRORE {video_id}: "
                f"{type(error).__name__}: {error}"
            )

            result = {
                "id_video": video_id,
                "model": args.model,
                "source_event_file": str(path),
                "causal_relations": [],
                "error": f"{type(error).__name__}: {error}",
            }

            write_json(output_path, result)

    print(f"Output creato in: {args.output_directory}")


if __name__ == "__main__":
    main()