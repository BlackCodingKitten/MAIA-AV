from __future__ import annotations

import os

# ============================================================
# GPU / vLLM CONFIGURATION
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse
import json
import traceback
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

SEMANTIC_DIR = Path(
    "data/preliminar_analysis/entity/entity_semantic/qwen-30B"
)

OUTPUT_DIR = Path(
    "data/preliminar_analysis/event_analysis/qwen-30B"
)

MAX_RETRIES = 3


# ============================================================
# JSON UTILITIES
# ============================================================

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
    """
    Estrae in maniera robusta un oggetto JSON dalla risposta
    del modello.

    Gestisce:
    - JSON puro
    - blocchi ```json ... ```
    - testo prima o dopo il JSON
    - JSON leggermente malformato
    - JSON troncato ma riparabile tramite json_repair
    """

    if text is None:
        raise ValueError(
            "La risposta del modello è None."
        )

    if not isinstance(text, str):
        raise ValueError(
            "La risposta del modello non è una stringa: "
            f"{type(text).__name__}"
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "La risposta del modello è vuota."
        )

    # --------------------------------------------------------
    # Rimuove eventuali Markdown fences
    # --------------------------------------------------------

    cleaned = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # --------------------------------------------------------
    # 1. Prova direttamente json.loads()
    # --------------------------------------------------------

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 2. Cerca un oggetto JSON valido all'interno del testo
    # --------------------------------------------------------

    decoder = json.JSONDecoder()

    for index, char in enumerate(cleaned):

        if char != "{":
            continue

        candidate = cleaned[index:]

        try:
            result, _ = decoder.raw_decode(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            continue

    # --------------------------------------------------------
    # 3. Prova a riparare il JSON
    # --------------------------------------------------------

    start = cleaned.find("{")

    if start >= 0:

        candidate = cleaned[start:]

        try:
            from json_repair import repair_json

            repaired = repair_json(candidate)

            result = json.loads(repaired)

            if isinstance(result, dict):
                return result

        except Exception:
            pass

    # --------------------------------------------------------
    # Nessun JSON recuperabile
    # --------------------------------------------------------

    raise ValueError(
        "La risposta non contiene un oggetto JSON valido.\n\n"
        f"RAW RESPONSE:\n{cleaned[:3000]}"
    )


# ============================================================
# INPUT COMPACTION
# ============================================================

def compact(payload):

    keys = (
        "segment_id",
        "start_time",
        "end_time",
        "entities",
        "actions",
        "events",
        "spatial_relations",
        "state_changes",
        "temporal_relations",
        "causal_hypotheses",
    )

    return {
        "id_video": payload.get("id_video"),
        "segments": [
            {
                key: segment.get(key, [])
                for key in keys
            }
            for segment in payload.get(
                "segments",
                [],
            )
        ],
    }


# ============================================================
# PROMPT
# ============================================================

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


# ============================================================
# INFERENCER
# ============================================================

class Inferencer:

    def __init__(
        self,
        model_id,
        max_new_tokens,
        gpu_utilization,
    ):

        visible_gpus = torch.cuda.device_count()

        if visible_gpus != 2:
            raise RuntimeError(
                "Configurazione GPU non valida: "
                f"PyTorch vede {visible_gpus} GPU, "
                "ma devono essere esattamente 2 "
                "(GPU fisiche 1 e 2)."
            )

        print(
            "GPU visibili a vLLM: "
            f"{os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )

        print(
            "Caricamento Qwen3-Omni-30B "
            "con tensor_parallel_size=2...",
            flush=True,
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",

            # Due GPU: fisiche 1 e 2
            tensor_parallel_size=2,

            gpu_memory_utilization=gpu_utilization,
            max_model_len=32768,

            # Un solo prompt alla volta
            max_num_seqs=1,

            # Questo script lavora esclusivamente
            # sul JSON semantico, non su immagini/audio
            limit_mm_per_prompt={
                "image": 0,
                "audio": 0,
                "video": 0,
            },

            trust_remote_code=True,

            # Evita compilazioni che in passato possono
            # creare problemi con alcune configurazioni
            # Qwen/vLLM.
            enforce_eager=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    def __call__(self, prompt):

        outputs = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if not outputs:
            return ""

        if not outputs[0].outputs:
            return ""

        return (
            outputs[0]
            .outputs[0]
            .text
            .strip()
        )


# ============================================================
# FAILED RAW RESPONSES
# ============================================================

def save_failed_raw(
    output_directory,
    video_id,
    attempt,
    raw,
):

    failed_directory = (
        output_directory
        / "_failed_raw"
    )

    failed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_path = (
        failed_directory
        / f"{video_id}_attempt_{attempt}.txt"
    )

    failed_path.write_text(
        raw if raw else "<EMPTY RESPONSE>",
        encoding="utf-8",
    )

    return failed_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Event extraction Qwen3-Omni-30B "
            "dalla propria analisi semantica "
            "su GPU fisiche 1 e 2."
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

    # --------------------------------------------------------
    # Cerca gli input semantic
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Caricamento del modello UNA SOLA VOLTA
    # --------------------------------------------------------

    infer = Inferencer(
        args.model,
        args.max_new_tokens,
        args.gpu_utilization,
    )

    completed = 0
    skipped = 0
    failed = 0

    total = len(files)

    # --------------------------------------------------------
    # Elaborazione
    # --------------------------------------------------------

    for index, path in enumerate(
        files,
        1,
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

            # ------------------------------------------------
            # Se già elaborato, non rifarlo
            # ------------------------------------------------

            if (
                output_path.exists()
                and not args.overwrite
            ):

                print(
                    f"[{index}/{total}] "
                    f"SKIP {video_id}",
                    flush=True,
                )

                skipped += 1
                continue

            print(
                f"\n[{index}/{total}] "
                f"{video_id} "
                "su GPU fisiche 1,2 (TP=2)",
                flush=True,
            )

            prompt = build_prompt(
                compact(payload)
            )

            result = None
            last_error = None

            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            for attempt in range(
                1,
                MAX_RETRIES + 1,
            ):

                print(
                    f"  Tentativo "
                    f"{attempt}/{MAX_RETRIES}",
                    flush=True,
                )

                raw = ""

                try:

                    raw = infer(prompt)

                    print(
                        "  Lunghezza risposta: "
                        f"{len(raw)} caratteri",
                        flush=True,
                    )

                    result = parse_json(raw)

                    print(
                        "  JSON valido.",
                        flush=True,
                    )

                    break

                except Exception as error:

                    last_error = error

                    print(
                        f"  [WARN] "
                        f"{type(error).__name__}: "
                        f"{error}",
                        flush=True,
                    )

                    # ----------------------------------------
                    # Salva sempre il RAW problematico
                    # ----------------------------------------

                    failed_path = save_failed_raw(
                        args.output_directory,
                        video_id,
                        attempt,
                        raw,
                    )

                    print(
                        "  RAW salvato in: "
                        f"{failed_path}",
                        flush=True,
                    )

            # ------------------------------------------------
            # Nessun JSON valido dopo i retry
            # ------------------------------------------------

            if result is None:

                print(
                    f"  [FAILED] {video_id}: "
                    "nessun JSON valido dopo "
                    f"{MAX_RETRIES} tentativi.",
                    flush=True,
                )

                print(
                    "  Ultimo errore: "
                    f"{last_error}",
                    flush=True,
                )

                failed += 1

                # NON interrompere l'intero job
                continue

            # ------------------------------------------------
            # Normalizzazione struttura
            # ------------------------------------------------

            if "events" not in result:
                result["events"] = []

            if "temporal_relations" not in result:
                result["temporal_relations"] = []

            if not isinstance(
                result["events"],
                list,
            ):
                raise ValueError(
                    f"{video_id}: "
                    "'events' non è una lista."
                )

            if not isinstance(
                result["temporal_relations"],
                list,
            ):
                raise ValueError(
                    f"{video_id}: "
                    "'temporal_relations' "
                    "non è una lista."
                )

            # ------------------------------------------------
            # Metadati
            # ------------------------------------------------

            result.update(
                {
                    "id_video": video_id,
                    "model": args.model,
                    "source_semantic_file": str(
                        path
                    ),
                }
            )

            # ------------------------------------------------
            # Scrittura output
            # ------------------------------------------------

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
                "  Eventi estratti: "
                f"{len(result['events'])}",
                flush=True,
            )

        except Exception as error:

            failed += 1

            print(
                f"\n[ERROR] Impossibile "
                f"elaborare {path.name}",
                flush=True,
            )

            traceback.print_exc()

            # Passa comunque al file successivo
            continue

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n"
        "========================================\n"
        "ELABORAZIONE COMPLETATA\n"
        "========================================\n"
        f"Video totali: {total}\n"
        f"Completati:   {completed}\n"
        f"Saltati:      {skipped}\n"
        f"Falliti:      {failed}\n"
        f"Output:       {args.output_directory}\n"
        "========================================",
        flush=True,
    )


if __name__ == "__main__":
    main()