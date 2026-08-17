from __future__ import annotations

import os

# Stesse GPU, ma limitate a 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import traceback
from pathlib import Path

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_ID = "google/gemma-4-12B-it"

EVENT_DIR = Path(
    "data/preliminar_analysis/event/gemma-4-12B"
)
OUTPUT_DIR = Path(
    "data/preliminar_analysis/causal/gemma-4-12B"
)

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


def read_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_json(text):
    if not text or not text.strip():
        raise ValueError("Risposta vuota.")

    cleaned = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")

        if start < 0:
            raise ValueError(
                "La risposta non contiene JSON."
            )

        candidate = cleaned[start:]

        try:
            result, _ = (
                json.JSONDecoder()
                .raw_decode(candidate)
            )

        except json.JSONDecodeError:
            from json_repair import repair_json

            result = json.loads(
                repair_json(candidate)
            )

    if not isinstance(result, dict):
        raise ValueError(
            "Il JSON restituito non è un oggetto."
        )

    return result


def compact(payload):
    return {
        "id_video": payload.get("id_video"),
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
                "participants": event.get(
                    "participants",
                    [],
                ),
                "evidence_segments": event.get(
                    "evidence_segments",
                    [],
                ),
                "evidence_type": event.get(
                    "evidence_type"
                ),
                "confidence": event.get(
                    "confidence"
                ),
            }
            for event in payload.get(
                "events",
                [],
            )
        ],
        "temporal_relations": payload.get(
            "temporal_relations",
            [],
        ),
    }


def build_prompt(payload):
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
- restituisci esclusivamente JSON valido.

Schema:
{{
  "causal_relations": [
    {{
      "cause_event": "E0001",
      "relation": "causes|enables|motivates|prevents",
      "effect_event": "E0002",
      "evidence_type": "direct|inferred|uncertain",
<<<<<<< HEAD
      "supporting_events": ["E0003"],
      "explanation": "breve spiegazione basata esclusivamente sugli eventi forniti",
=======
      "supporting_events": [],
  ]
}}
INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


<<<<<<< HEAD
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
=======
def normalize_result(
    result,
    payload,
):
    valid_ids = {
        event.get("event_id")
        for event in payload.get(
            "events",
            [],
        )
        if event.get("event_id")
    }

    normalized = []
    seen = set()

    for relation in result.get(
        "causal_relations",
        [],
    ):
        if not isinstance(relation, dict):
            continue

        cause = relation.get("cause_event")
        effect = relation.get("effect_event")
        relation_type = relation.get(
            "relation"
        )
        evidence = relation.get(
            "evidence_type",
            "inferred",
        )

        if (
            cause not in valid_ids
            or effect not in valid_ids
            or cause == effect
        ):
            continue

        if (
            relation_type
            not in ALLOWED_RELATIONS
        ):
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

        supporting = [
            event_id
            for event_id in relation.get(
                "supporting_events",
                [],
            )
            if (
                event_id in valid_ids
                and event_id
                not in {cause, effect}
            )
        ]

        try:
            confidence = float(
                relation.get(
                    "confidence",
                    0.0,
                )
            )
>>>>>>> refs/remotes/origin/main
        except (TypeError, ValueError):
            confidence = 0.0

        normalized.append({
            "cause_event": cause,
<<<<<<< HEAD
            "relation": rel,
            "effect_event": effect,
            "evidence_type": evidence,
            "supporting_events": list(dict.fromkeys(supporting)),
            "explanation": str(relation.get("explanation", "")).strip(),
            "confidence": max(0.0, min(1.0, confidence)),
        })

    return {"causal_relations": normalized}


class Inferencer:
    def __init__(self, model_id: str, max_new_tokens: int, gpu_utilization: float) -> None:
        if torch.cuda.device_count() != 4:
            raise RuntimeError(f"Mi aspettavo 4 GPU visibili, ma PyTorch ne vede {torch.cuda.device_count()}.")
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=32768,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def __call__(self, prompt: str) -> str:
        output = self.llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=self.sampling,
            use_tqdm=False,
        )[0]
        return output.outputs[0].text.strip()


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
    parser = argparse.ArgumentParser(description="Causal inference Gemma-3-27B dagli eventi consolidati.")
    parser.add_argument("event_directory", nargs="?", type=Path, default=EVENT_DIR)
    parser.add_argument("output_directory", nargs="?", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.event_directory.glob("*_events.json"))
    if args.limit_videos:
        files = files[:args.limit_videos]
    if not files:
        parser.error(f"Nessun *_events.json trovato in {args.event_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    inferencer = Inferencer(args.model, args.max_new_tokens, args.gpu_utilization)

    for index, path in enumerate(files, 1):
        payload = read_json(path)
        video_id = payload.get("id_video") or path.stem.removesuffix("_events")
        output_path = args.output_directory / f"{video_id}_causal.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] SKIP {video_id}")
            continue

        print(f"[{index}/{len(files)}] {video_id} su GPU fisiche 5,6,1,2 (TP=4)")
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
            print(f"ERRORE {video_id}: {type(error).__name__}: {error}")
            result = {
                "id_video": video_id,
                "model": args.model,
                "source_event_file": str(path),
                "causal_relations": [],
                "error": f"{type(error).__name__}: {error}",
            }

        write_json(output_path, result)

    print(f"Creato output in: {args.output_directory}")


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import os

# # Usa esclusivamente le GPU fisiche 5 e 6.
# # All'interno del processo saranno viste come cuda:0 e cuda:1.
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,0"

# os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
# os.environ["VLLM_USE_FLASHINFER"] = "0"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

# import argparse
# import json
# from pathlib import Path

# import torch
# from vllm import LLM, SamplingParams


# MODEL_ID = "google/gemma-3-27b-it"

# EVENT_DIR = Path("data/preliminar_analysis/event/gemma-27B")
# OUTPUT_DIR = Path("data/preliminar_analysis/causal/gemma-27B")

# ALLOWED_RELATIONS = {"causes", "enables", "motivates", "prevents"}
# ALLOWED_EVIDENCE = {"direct", "inferred", "uncertain"}


# def read_json(path: Path) -> dict:
#     return json.loads(path.read_text(encoding="utf-8"))


# def write_json(path: Path, data: dict) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(
#         json.dumps(data, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )


# def parse_json(text: str) -> dict:
#     text = text.replace("```json", "").replace("```", "").strip()

#     start = text.find("{")
#     end = text.rfind("}")

#     if start < 0 or end < start:
#         raise ValueError("La risposta non contiene un oggetto JSON.")

#     candidate = text[start:end + 1]

#     try:
#         return json.loads(candidate)

#     except json.JSONDecodeError:
#         from json_repair import repair_json
#         return json.loads(repair_json(candidate))


# def compact(payload: dict) -> dict:
#     return {
#         "id_video": payload.get("id_video"),
#         "events": [
#             {
#                 "event_id": event.get("event_id"),
#                 "description": event.get("description"),
#                 "start_time": event.get("start_time"),
#                 "end_time": event.get("end_time"),
#                 "participants": event.get("participants", []),
#                 "evidence_segments": event.get("evidence_segments", []),
#                 "evidence_type": event.get("evidence_type"),
#                 "confidence": event.get("confidence"),
#             }
#             for event in payload.get("events", [])
#         ],
#         "temporal_relations": payload.get("temporal_relations", []),
#     }


# def build_prompt(payload: dict) -> str:
#     return f"""Analizza la rappresentazione consolidata degli eventi di un video e individua esclusivamente le relazioni causali supportate dagli eventi forniti.

# Regole:
# - usa esclusivamente gli eventi e le relazioni temporali presenti nell'input;
# - non usare domande, caption, foil o altre informazioni esterne;
# - la semplice successione temporale NON implica causalità;
# - crea una relazione soltanto quando il contenuto degli eventi supporta una dipendenza causa-effetto, una condizione abilitante, una motivazione oppure una prevenzione;
# - cause_event ed effect_event devono essere ID di eventi presenti nell'input;
# - usa evidence_type="direct" quando la dipendenza è chiaramente supportata dall'interazione o dal cambiamento osservato;
# - usa evidence_type="inferred" quando è necessaria un'inferenza contestuale o intenzionale;
# - usa evidence_type="uncertain" solo quando esiste un indizio causale ma l'evidenza è debole;
# - supporting_events contiene soltanto eventuali ulteriori eventi necessari a sostenere la relazione;
# - non inventare eventi, oggetti, intenzioni o cause non ricavabili dalla rappresentazione fornita;
# - se non esiste evidenza sufficiente, restituisci causal_relations come lista vuota;
# - restituisci ESCLUSIVAMENTE un oggetto JSON valido.

# Schema:
# {{
#   "causal_relations": [
#     {{
#       "cause_event": "E0001",
#       "relation": "causes|enables|motivates|prevents",
#       "effect_event": "E0002",
#       "evidence_type": "direct|inferred|uncertain",
#       "supporting_events": ["E0003"],
#       "explanation": "breve spiegazione basata esclusivamente sugli eventi forniti",
#       "confidence": 0.0
#     }}
#   ]
# }}

# INPUT:
# {json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


# def normalize_result(result: dict, payload: dict) -> dict:
#     valid_ids = {
#         event.get("event_id")
#         for event in payload.get("events", [])
#         if event.get("event_id")
#     }

#     normalized = []
#     seen = set()

#     for relation in result.get("causal_relations", []):
#         if not isinstance(relation, dict):
#             continue

#         cause = relation.get("cause_event")
#         effect = relation.get("effect_event")
#         rel = relation.get("relation")
#         evidence = relation.get("evidence_type", "inferred")

#         if cause not in valid_ids or effect not in valid_ids or cause == effect:
#             continue

#         if rel not in ALLOWED_RELATIONS:
#             continue

#         if evidence not in ALLOWED_EVIDENCE:
#             evidence = "inferred"

#         key = (cause, rel, effect)

#         if key in seen:
#             continue

#         seen.add(key)

#         supporting = [
#             event_id
#             for event_id in relation.get("supporting_events", [])
#             if event_id in valid_ids and event_id not in {cause, effect}
#         ]

#         try:
#             confidence = float(relation.get("confidence", 0.0))
#         except (TypeError, ValueError):
#             confidence = 0.0

#         normalized.append(
#             {
#                 "cause_event": cause,
#                 "relation": rel,
#                 "effect_event": effect,
#                 "evidence_type": evidence,
#                 "supporting_events": list(dict.fromkeys(supporting)),
#                 "explanation": str(
#                     relation.get("explanation", "")
#                 ).strip(),
#                 "confidence": max(0.0, min(1.0, confidence)),
#             }
#         )

#     return {"causal_relations": normalized}


# class Inferencer:
#     def __init__(
#         self,
#         model_id: str,
#         max_new_tokens: int,
#         gpu_utilization: float,
#     ) -> None:

#         gpu_count = torch.cuda.device_count()

#         if gpu_count != 2:
#             raise RuntimeError(
#                 f"Mi aspettavo 2 GPU visibili, "
#                 f"ma PyTorch ne vede {gpu_count}."
#             )

#         print("GPU visibili:")
#         for i in range(gpu_count):
#             print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

#         self.llm = LLM(
#             model=model_id,
#             dtype="bfloat16",
#             tensor_parallel_size=2,
#             gpu_memory_utilization=gpu_utilization,
#             max_model_len=32768,
#         )

#         self.sampling = SamplingParams(
#             temperature=0.0,
#             max_tokens=max_new_tokens,
#         )

#     def __call__(self, prompt: str) -> str:
#         output = self.llm.chat(
#             [
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             sampling_params=self.sampling,
#             use_tqdm=False,
#         )[0]

#         return output.outputs[0].text.strip()


# def infer_json(
#     inferencer: Inferencer,
#     prompt: str,
#     attempts: int = 2,
# ) -> dict:

#     raw = ""
#     error = None

#     for attempt in range(attempts):

#         if attempt == 0:
#             current_prompt = prompt
#         else:
#             current_prompt = (
#                 prompt
#                 + "\n\nLa risposta precedente non era JSON valido. "
#                   "Rispondi soltanto con l'oggetto JSON richiesto."
#             )

#         raw = inferencer(current_prompt)

#         try:
#             return parse_json(raw)

#         except Exception as current_error:
#             error = current_error

#     raise ValueError(
#         f"JSON non ottenuto dopo {attempts} tentativi. "
#         f"Ultima risposta: {raw[:500]!r}"
#     ) from error


# def main() -> None:

#     parser = argparse.ArgumentParser(
#         description=(
#             "Causal inference Gemma-3-27B "
#             "dagli eventi consolidati."
#         )
#     )

#     parser.add_argument(
#         "event_directory",
#         nargs="?",
#         type=Path,
#         default=EVENT_DIR,
#     )

#     parser.add_argument(
#         "output_directory",
#         nargs="?",
#         type=Path,
#         default=OUTPUT_DIR,
#     )

#     parser.add_argument(
#         "--model",
#         default=MODEL_ID,
#     )

#     parser.add_argument(
#         "--max-new-tokens",
#         type=int,
#         default=4096,
#     )

#     parser.add_argument(
#         "--gpu-utilization",
#         type=float,
#         default=0.85,
#     )

#     parser.add_argument(
#         "--limit-videos",
#         type=int,
#         default=0,
#     )

#     parser.add_argument(
#         "--overwrite",
#         action="store_true",
#     )

#     args = parser.parse_args()

#     files = sorted(
#         args.event_directory.glob("*_events.json")
#     )

#     if args.limit_videos:
#         files = files[:args.limit_videos]

#     if not files:
#         parser.error(
#             f"Nessun *_events.json trovato in "
#             f"{args.event_directory}"
#         )

#     args.output_directory.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     inferencer = Inferencer(
#         args.model,
#         args.max_new_tokens,
#         args.gpu_utilization,
#     )

#     for index, path in enumerate(files, 1):

#         payload = read_json(path)

#         video_id = (
#             payload.get("id_video")
#             or path.stem.removesuffix("_events")
#         )

#         output_path = (
#             args.output_directory
#             / f"{video_id}_causal.json"
#         )

#         if output_path.exists() and not args.overwrite:
#             print(
#                 f"[{index}/{len(files)}] "
#                 f"SKIP {video_id}"
#             )
#             continue

#         print(
#             f"[{index}/{len(files)}] "
#             f"{video_id} su GPU fisiche 5,6 (TP=2)"
#         )

#         compact_payload = compact(payload)

#         try:
#             result = infer_json(
#                 inferencer,
#                 build_prompt(compact_payload),
#             )

#             result = normalize_result(
#                 result,
#                 compact_payload,
#             )

#             result.update(
#                 {
#                     "id_video": video_id,
#                     "model": args.model,
#                     "source_event_file": str(path),
#                 }
#             )

#         except Exception as error:

#             print(
#                 f"ERRORE {video_id}: "
#                 f"{type(error).__name__}: {error}"
#             )

#             result = {
#                 "id_video": video_id,
#                 "model": args.model,
#                 "source_event_file": str(path),
#                 "causal_relations": [],
#                 "error": (
#                     f"{type(error).__name__}: {error}"
#                 ),
#             }

#         write_json(
#             output_path,
#             result,
#         )

#     print(
#         f"Creato output in: "
#         f"{args.output_directory}"
#     )


# if __name__ == "__main__":
#     main()
=======
            "relation": relation_type,
            "effect_event": effect,
            "evidence_type": evidence,
            "supporting_events": list(
                dict.fromkeys(supporting)
            ),
            "explanation": str(
                relation.get(
                    "explanation",
                    "",
                )
            ).strip(),
            "confidence": max(
                0.0,
                min(1.0, confidence),
            ),
        })

    return {
        "causal_relations": normalized
    }


class Inferencer:
    def __init__(
        self,
        model_id,
        max_new_tokens,
        gpu_utilization,
    ):
        self.processor = (
            AutoProcessor.from_pretrained(model_id)
        )

        print(
            "Caricamento Gemma-4-12B "
            "su 2 GPU "
            "con TP=2...",
            flush=True,
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=32768,
            max_num_seqs=1,
            limit_mm_per_prompt={
                "image": 0,
                "video": 0,
                "audio": 0,
            },
            trust_remote_code=True,
            enforce_eager=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    def format_prompt(self, prompt):
        return self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def __call__(self, prompt):
        text = self.format_prompt(prompt)

        output = self.llm.generate(
            [text],
            sampling_params=self.sampling,
            use_tqdm=False,
        )[0]

        if not output.outputs:
            return ""

        return output.outputs[0].text.strip()


def infer_json(
    inferencer,
    prompt,
    attempts=2,
):
    raw = ""
    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        current_prompt = prompt

        if attempt > 1:
            current_prompt += (
                "\n\nLa risposta precedente "
                "non era JSON valido. "
                "Rispondi soltanto con "
                "l'oggetto JSON richiesto."
            )

        raw = inferencer(current_prompt)

        try:
            return parse_json(raw)

        except Exception as error:
            last_error = error

    raise ValueError(
        "JSON non ottenuto dopo "
        f"{attempts} tentativi. "
        f"Ultima risposta: {raw[:500]!r}"
    ) from last_error


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Causal inference Gemma-4-12B "
            "dagli eventi consolidati."
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
        default=4096,
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
            "Nessun *_events.json trovato in "
            f"{args.event_directory}"
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    inferencer = Inferencer(
        args.model,
        args.max_new_tokens,
        args.gpu_utilization,
    )

    completed = 0
    skipped = 0
    failed = 0

    for index, path in enumerate(
        files,
        start=1,
    ):
        payload = read_json(path)

        video_id = (
            payload.get("id_video")
            or path.stem.removesuffix(
                "_events"
            )
        )

        output_path = (
            args.output_directory
            / f"{video_id}_causal.json"
        )

        if (
            output_path.exists()
            and not args.overwrite
        ):
            print(
                f"[{index}/{len(files)}] "
                f"SKIP {video_id}",
                flush=True,
            )
            skipped += 1
            continue

        print(
            f"[{index}/{len(files)}] "
            f"{video_id}",
            flush=True,
        )

        compact_payload = compact(payload)

        try:
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

            result.update({
                "id_video": video_id,
                "model": args.model,
                "source_event_file": str(
                    path
                ),
            })

            write_json(
                output_path,
                result,
            )

            completed += 1

            print(
                f"  Salvato: {output_path}",
                flush=True,
            )

        except Exception as error:
            failed += 1

            print(
                f"\n[ERROR] {path.name}",
                flush=True,
            )
            traceback.print_exc()

            write_json(
                output_path,
                {
                    "id_video": video_id,
                    "model": args.model,
                    "source_event_file": str(
                        path
                    ),
                    "causal_relations": [],
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                },
            )

    print(
        "\n"
        "========================================\n"
        "ELABORAZIONE COMPLETATA\n"
        "========================================\n"
        f"Video totali: {len(files)}\n"
        f"Completati:   {completed}\n"
        f"Saltati:      {skipped}\n"
        f"Falliti:      {failed}\n"
        f"Output:       {args.output_directory}\n"
        "========================================",
        flush=True,
    )


if __name__ == "__main__":
    main()
>>>>>>> refs/remotes/origin/main
