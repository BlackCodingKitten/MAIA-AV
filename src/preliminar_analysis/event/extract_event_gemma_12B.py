from __future__ import annotations

import os

# Utilizza esclusivamente le GPU fisiche 4 e 6.
# All'interno del processo verranno viste come cuda:0 e cuda:1.
os.environ["CUDA_VISIBLE_DEVICES"] = "2,1"

# Multiprocessing vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Disabilita FlashInfer sampler
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# Configurazione NCCL
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

# Comunicazione locale vLLM
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# Limita i thread CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import argparse
import json
import traceback
from pathlib import Path

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_ID = "google/gemma-4-12B-it"

SEMANTIC_DIR = Path(
    "data/preliminar_analysis/entity/gemma-4-12B"
)
OUTPUT_DIR = Path(
    "data/preliminar_analysis/event/gemma-4-12B"
)

MAX_MODEL_LEN = 32768
MAX_RETRIES = 3
SAFETY_MARGIN = 512
CHUNK_OVERLAP = 1

SEMANTIC_FIELDS = (
    "entities",
    "actions",
    "events",
    "spatial_relations",
    "state_changes",
    "temporal_relations",
    "causal_hypotheses",
)


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
            try:
                from json_repair import repair_json

                result = json.loads(
                    repair_json(candidate)
                )

            except Exception as error:
                raise ValueError(
                    "JSON non recuperabile.\n\n"
                    f"RAW:\n{cleaned[:3000]}"
                ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "Il JSON restituito non è un oggetto."
        )

    return result


def compact(payload):
    segments = []

    for segment in payload.get("segments", []):
        item = {
            "segment_id": segment.get(
                "segment_id"
            ),
            "start_time": segment.get(
                "start_time"
            ),
            "end_time": segment.get(
                "end_time"
            ),
        }

        item.update({
            key: segment[key]
            for key in SEMANTIC_FIELDS
            if segment.get(key)
        })

        segments.append(item)

    return {
        "id_video": payload.get("id_video"),
        "segments": segments,
    }


def with_segments(payload, segments):
    return {
        "id_video": payload.get("id_video"),
        "segments": segments,
    }


def build_prompt(payload):
    return f"""Consolida i segmenti dell'analisi semantica precedente in una rappresentazione cronologica degli eventi.

Regole:
- usa esclusivamente le informazioni presenti nell'analisi semantica;
- unisci soltanto i duplicati causati dalla sovrapposizione delle finestre;
- mantieni separati eventi realmente distinti o ripetuti;
- ricava i tempi soltanto dai segmenti forniti;
- non inventare intenzioni, cause, emozioni o azioni mancanti;
- usa evidence_type="inferred" soltanto se l'input contiene esplicitamente un'inferenza;
- assegna gli ID E0001, E0002, ... in ordine temporale;
- restituisci esclusivamente JSON valido.

Schema:
{{
  "events": [
    {{
      "event_id": "E0001",
      "description": "string",
      "start_time": 0.0,
      "end_time": 0.0,
      "participants": ["string"],
      "evidence_segments": ["segment_0000"],
      "evidence_type": "observed|inferred|uncertain",
      "confidence": 0.0
    }}
  ],
  "temporal_relations": [
    {{
      "first_event": "E0001",
      "relation": "before|after|overlaps|during|simultaneous",
      "second_event": "E0002",
      "confidence": 0.0
    }}
  ]
}}

INPUT:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}"""


def build_merge_prompt(
    video_id,
    partial_results,
):
    return f"""Fondi le analisi parziali degli eventi dello stesso video in una singola rappresentazione cronologica.

Regole:
- usa esclusivamente gli eventi presenti negli input;
- elimina i duplicati dovuti alla suddivisione dell'input;
- non fondere eventi realmente distinti o ripetuti;
- conserva tempi, partecipanti ed evidenze;
- non inventare nuovi eventi o relazioni;
- ordina gli eventi temporalmente;
- assegna nuovamente ID E0001, E0002, ...;
- aggiorna le relazioni temporali con i nuovi ID;
- restituisci esclusivamente JSON valido.

Schema:
{{
  "events": [
    {{
      "event_id": "E0001",
      "description": "string",
      "start_time": 0.0,
      "end_time": 0.0,
      "participants": ["string"],
      "evidence_segments": ["segment_0000"],
      "evidence_type": "observed|inferred|uncertain",
      "confidence": 0.0
    }}
  ],
  "temporal_relations": [
    {{
      "first_event": "E0001",
      "relation": "before|after|overlaps|during|simultaneous",
      "second_event": "E0002",
      "confidence": 0.0
    }}
  ]
}}

VIDEO:
{video_id}

INPUT:
{json.dumps(partial_results, ensure_ascii=False, separators=(",", ":"))}"""


class Inferencer:
    def __init__(
        self,
        model_id,
        max_new_tokens,
        gpu_utilization,
    ):
        self.max_input_tokens = (
            MAX_MODEL_LEN
            - max_new_tokens
            - SAFETY_MARGIN
        )

        self.processor = (
            AutoProcessor.from_pretrained(model_id)
        )

        print(
            "Caricamento Gemma-4-12B "
            "su 2 GPU con TP=2...",
            flush=True,
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_utilization,
            max_model_len=MAX_MODEL_LEN,
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

    def token_count(self, prompt):
        text = self.format_prompt(prompt)

        return len(
            self.processor.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def fits(self, prompt):
        return (
            self.token_count(prompt)
            <= self.max_input_tokens
        )

    def __call__(self, prompt):
        text = self.format_prompt(prompt)

        outputs = self.llm.generate(
            [text],
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if not outputs or not outputs[0].outputs:
            return ""

        return (
            outputs[0]
            .outputs[0]
            .text
            .strip()
        )


def split_payload(payload, infer):
    segments = payload.get("segments", [])

    if not segments:
        return [payload]

    chunks = []
    current = []

    for segment in segments:
        candidate = with_segments(
            payload,
            current + [segment],
        )

        if infer.fits(
            build_prompt(candidate)
        ):
            current.append(segment)
            continue

        if not current:
            raise ValueError(
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

        chunks.append(
            with_segments(payload, current)
        )

        overlap = (
            current[-CHUNK_OVERLAP:]
            if CHUNK_OVERLAP
            else []
        )

        current = overlap + [segment]

        if not infer.fits(
            build_prompt(
                with_segments(
                    payload,
                    current,
                )
            )
        ):
            current = [segment]

        if not infer.fits(
            build_prompt(
                with_segments(
                    payload,
                    current,
                )
            )
        ):
            raise ValueError(
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

    if current:
        chunks.append(
            with_segments(payload, current)
        )

    return chunks


def save_failed_raw(
    output_directory,
    name,
    attempt,
    raw,
):
    path = (
        output_directory
        / "_failed_raw"
        / f"{name}_attempt_{attempt}.txt"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        raw or "<EMPTY RESPONSE>",
        encoding="utf-8",
    )

    return path


def generate_json(
    infer,
    prompt,
    output_directory,
    name,
):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        raw = ""

        try:
            print(
                f"    Tentativo "
                f"{attempt}/{MAX_RETRIES}",
                flush=True,
            )

            raw = infer(prompt)
            result = parse_json(raw)

            print(
                f"    JSON valido - "
                f"{len(raw)} caratteri",
                flush=True,
            )

            return result

        except Exception as error:
            last_error = error

            print(
                f"    [WARN] "
                f"{type(error).__name__}: "
                f"{error}",
                flush=True,
            )

            path = save_failed_raw(
                output_directory,
                name,
                attempt,
                raw,
            )

            print(
                f"    RAW: {path}",
                flush=True,
            )

    raise RuntimeError(
        "Nessun JSON valido dopo "
        f"{MAX_RETRIES} tentativi: "
        f"{last_error}"
    )


def split_merge_groups(
    video_id,
    results,
    infer,
    ):
    groups = []
    current = []

    for result in results:
        candidate = current + [result]

        if infer.fits(
            build_merge_prompt(
                video_id,
                candidate,
            )
        ):
            current = candidate
            continue

        if not current:
            raise ValueError(
                "Un risultato parziale "
                "supera da solo la context window."
            )

        groups.append(current)
        current = [result]

    if current:
        groups.append(current)

    return groups


def merge_results(
    video_id,
    results,
    infer,
    output_directory,
):
    level = 1

    while len(results) > 1:
        groups = split_merge_groups(
            video_id,
            results,
            infer,
        )

        if (
            len(groups) == len(results)
            and all(
                len(group) == 1
                for group in groups
            )
        ):
            raise ValueError(
                f"Impossibile ridurre il merge "
                f"di {video_id}."
            )

        merged = []

        for index, group in enumerate(
            groups,
            start=1,
        ):
            if len(group) == 1:
                merged.append(group[0])
                continue

            prompt = build_merge_prompt(
                video_id,
                group,
            )

            print(
                f"  Merge livello {level}, "
                f"gruppo {index}/{len(groups)}: "
                f"{infer.token_count(prompt)} token",
                flush=True,
            )

            merged.append(
                generate_json(
                    infer,
                    prompt,
                    output_directory,
                    (
                        f"{video_id}_merge_"
                        f"L{level}_{index:02d}"
                    ),
                )
            )

        results = merged
        level += 1

    return results[0]


def normalize(result):
    result.setdefault("events", [])
    result.setdefault(
        "temporal_relations",
        [],
    )

    if not isinstance(
        result["events"],
        list,
    ):
        raise ValueError(
            "'events' non è una lista."
        )

    if not isinstance(
        result["temporal_relations"],
        list,
    ):
        raise ValueError(
            "'temporal_relations' "
            "non è una lista."
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Event extraction Gemma-4-12B "
            "dalla propria analisi semantica."
        )
    )

    parser.add_argument(
        "semantic_directory",
        nargs="?",
        type=Path,
        default=SEMANTIC_DIR,
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
        args.semantic_directory.glob(
            "*_semantic.json"
        )
    )

    if args.limit_videos > 0:
        files = files[
            :args.limit_videos
        ]

    if not files:
        parser.error(
            "Nessun *_semantic.json trovato in "
            f"{args.semantic_directory}"
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    infer = Inferencer(
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
        try:
            payload = read_json(path)

            video_id = (
                payload.get("id_video")
                or path.stem.removesuffix(
                    "_semantic"
                )
            )

            output_path = (
                args.output_directory
                / f"{video_id}_events.json"
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
                f"\n[{index}/{len(files)}] "
                f"{video_id}",
                flush=True,
            )

            semantic = compact(payload)
            chunks = split_payload(
                semantic,
                infer,
            )

            print(
                f"  Segmenti: "
                f"{len(semantic['segments'])}",
                flush=True,
            )
            print(
                f"  Chunk: {len(chunks)}",
                flush=True,
            )

            partial_results = []

            for chunk_index, chunk in enumerate(
                chunks,
                start=1,
            ):
                prompt = build_prompt(chunk)

                print(
                    f"  Chunk "
                    f"{chunk_index}/{len(chunks)}: "
                    f"{len(chunk['segments'])} "
                    f"segmenti, "
                    f"{infer.token_count(prompt)} "
                    "token",
                    flush=True,
                )

                partial_results.append(
                    generate_json(
                        infer,
                        prompt,
                        args.output_directory,
                        (
                            f"{video_id}_chunk_"
                            f"{chunk_index:02d}"
                        ),
                    )
                )

            result = (
                partial_results[0]
                if len(partial_results) == 1
                else merge_results(
                    video_id,
                    partial_results,
                    infer,
                    args.output_directory,
                )
            )

            result = normalize(result)

            result.update({
                "id_video": video_id,
                "model": args.model,
                "source_semantic_file": str(
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
            print(
                f"  Eventi: "
                f"{len(result['events'])}",
                flush=True,
            )

        except Exception:
            failed += 1

            print(
                f"\n[ERROR] {path.name}",
                flush=True,
            )

            traceback.print_exc()

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