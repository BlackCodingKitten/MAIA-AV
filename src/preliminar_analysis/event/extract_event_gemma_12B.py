from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
os.environ["VLLM_USE_NCCL_SYMM_MEM"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import argparse
import json
import sys
import traceback

from pathlib import Path
from typing import Any

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
MAX_NEW_TOKENS = 1800
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


class EngineFatalError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
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


def is_engine_dead_exception(
    exc: BaseException,
) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()

    while current is not None:
        current_id = id(current)

        if current_id in visited:
            break

        visited.add(current_id)

        name = current.__class__.__name__
        message = str(current)

        if (
            name == "EngineDeadError"
            or "EngineCore encountered an issue" in message
            or "EngineCore encountered a fatal error" in message
            or "Engine core is dead" in message
            or "EngineDeadError" in message
        ):
            return True

        current = (
            current.__cause__
            or current.__context__
        )

    return False


def extract_json_from_text(
    text: str,
) -> dict[str, Any]:
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

    if end >= start:
        candidate = cleaned[
            start:end + 1
        ]

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    try:
        from json_repair import repair_json

    except ImportError as exc:
        raise ValueError(
            "JSON non valido e json_repair non installato."
        ) from exc

    try:
        repaired = repair_json(
            cleaned[start:]
        )

        result = json.loads(
            repaired
        )

    except Exception as exc:
        raise ValueError(
            "JSON non recuperabile dopo json_repair."
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Il JSON riparato non è un oggetto."
        )

    return result


def normalize_list_field(
    value: Any,
    field_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []

    if value is None:
        return [], warnings

    if isinstance(value, dict):
        return [value], warnings

    if isinstance(value, str):
        text = value.strip()

        if not text or text.lower() == "null":
            return [], warnings

        try:
            parsed = json.loads(text)

        except json.JSONDecodeError:
            warnings.append(
                f"{field_name}: stringa non strutturata scartata: "
                f"{text!r}"
            )
            return [], warnings

        return normalize_list_field(
            parsed,
            field_name,
        )

    if not isinstance(value, list):
        warnings.append(
            f"{field_name}: tipo non valido "
            f"{type(value).__name__}, valore scartato."
        )
        return [], warnings

    normalized: list[dict[str, Any]] = []

    for index, item in enumerate(value):
        if item is None:
            continue

        if isinstance(item, dict):
            normalized.append(item)
            continue

        if isinstance(item, list):
            nested, nested_warnings = normalize_list_field(
                item,
                field_name,
            )

            normalized.extend(
                nested
            )

            warnings.extend(
                nested_warnings
            )

            continue

        if isinstance(item, str):
            text = item.strip()

            if not text or text.lower() == "null":
                continue

            try:
                parsed = json.loads(
                    text
                )

            except json.JSONDecodeError:
                warnings.append(
                    f"{field_name}[{index}]: "
                    f"stringa non strutturata scartata: "
                    f"{text!r}"
                )
                continue

            nested, nested_warnings = normalize_list_field(
                parsed,
                field_name,
            )

            normalized.extend(
                nested
            )

            warnings.extend(
                nested_warnings
            )

            continue

        warnings.append(
            f"{field_name}[{index}]: "
            f"tipo {type(item).__name__} scartato."
        )

    return normalized, warnings


def normalize_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    events, event_warnings = normalize_list_field(
        result.get("events"),
        "events",
    )

    temporal_relations, relation_warnings = normalize_list_field(
        result.get("temporal_relations"),
        "temporal_relations",
    )

    result["events"] = events
    result["temporal_relations"] = temporal_relations

    warnings = (
        event_warnings
        + relation_warnings
    )

    if warnings:
        result["_normalization_warnings"] = warnings
    else:
        result.pop(
            "_normalization_warnings",
            None,
        )

    return result


def core_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "events": result.get(
            "events",
            [],
        ),
        "temporal_relations": result.get(
            "temporal_relations",
            [],
        ),
    }


def save_failed_raw(
    out_dir: Path,
    name: str,
    attempt: int,
    raw_text: str,
) -> Path:
    failed_dir = (
        out_dir
        / "_failed_raw"
    )

    failed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        failed_dir
        / f"{name}_attempt_{attempt}.txt"
    )

    path.write_text(
        raw_text or "<EMPTY RESPONSE>",
        encoding="utf-8",
    )

    return path


class Inferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        gpu_utilization: float,
    ):
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens deve essere maggiore di 0."
            )

        if not 0 < gpu_utilization <= 1:
            raise ValueError(
                "gpu_utilization deve essere compreso tra 0 e 1."
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.max_input_tokens = (
            MAX_MODEL_LEN
            - max_new_tokens
            - SAFETY_MARGIN
        )

        if self.max_input_tokens <= 0:
            raise ValueError(
                "La configurazione non lascia spazio per il prompt."
            )

        print(
            "\n"
            "========================================\n"
            "CARICAMENTO MODELLO\n"
            "========================================\n"
            f"Model:            {model_id}\n"
            "Tensor parallel:  2\n"
            f"Max model length: {MAX_MODEL_LEN}\n"
            f"Max new tokens:   {max_new_tokens}\n"
            f"GPU utilization:  {gpu_utilization}\n"
            "========================================",
            flush=True,
        )

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
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

    def format_prompt(
        self,
        prompt_text: str,
    ) -> str:
        return self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def token_count(
        self,
        prompt_text: str,
    ) -> int:
        formatted = self.format_prompt(
            prompt_text
        )

        encoded = self.processor.tokenizer.encode(
            formatted,
            add_special_tokens=False,
        )

        return len(encoded)

    def fits(
        self,
        prompt_text: str,
    ) -> bool:
        return (
            self.token_count(prompt_text)
            <= self.max_input_tokens
        )

    def generate(
        self,
        prompt_text: str,
    ) -> str:
        formatted = self.format_prompt(
            prompt_text
        )

        outputs = self.llm.generate(
            [formatted],
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if not outputs:
            raise RuntimeError(
                "vLLM non ha restituito risultati."
            )

        if not outputs[0].outputs:
            raise RuntimeError(
                "vLLM non ha restituito una generazione."
            )

        return (
            outputs[0]
            .outputs[0]
            .text
            .strip()
        )


def build_prompt(
    payload: dict[str, Any],
) -> str:
    return (
        "Consolida i segmenti dell'analisi semantica precedente "
        "in una rappresentazione cronologica degli eventi.\n\n"

        "Regole:\n"
        "- usa esclusivamente le informazioni presenti "
        "nell'analisi semantica;\n"
        "- unisci soltanto i duplicati causati dalla "
        "sovrapposizione delle finestre;\n"
        "- mantieni separati eventi realmente distinti "
        "o ripetuti;\n"
        "- ricava i tempi soltanto dai segmenti forniti;\n"
        "- non inventare intenzioni, cause, emozioni "
        "o azioni mancanti;\n"
        "- usa evidence_type=\"inferred\" soltanto se "
        "l'input contiene esplicitamente un'inferenza;\n"
        "- assegna gli ID E0001, E0002, ... "
        "in ordine temporale;\n"
        "- events deve essere sempre un array JSON;\n"
        "- ogni elemento di events deve essere "
        "un oggetto JSON;\n"
        "- temporal_relations deve essere sempre "
        "un array JSON;\n"
        "- ogni elemento di temporal_relations deve essere "
        "un oggetto JSON con first_event, relation, "
        "second_event e confidence;\n"
        "- non rappresentare eventi o relazioni "
        "come stringhe libere;\n"
        "- restituisci esclusivamente JSON valido.\n\n"

        "Schema:\n"
        "{\n"
        '  "events": [\n'
        "    {\n"
        '      "event_id": "E0001",\n'
        '      "description": "string",\n'
        '      "start_time": 0.0,\n'
        '      "end_time": 0.0,\n'
        '      "participants": ["string"],\n'
        '      "evidence_segments": ["segment_0000"],\n'
        '      "evidence_type": "observed|inferred|uncertain",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ],\n"
        '  "temporal_relations": [\n'
        "    {\n"
        '      "first_event": "E0001",\n'
        '      "relation": '
        '"before|after|overlaps|during|simultaneous",\n'
        '      "second_event": "E0002",\n'
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


def build_merge_prompt(
    video_id: str,
    partial_results: list[dict[str, Any]],
) -> str:
    clean_results = [
        core_result(result)
        for result in partial_results
    ]

    return (
        "Fondi le analisi parziali degli eventi dello stesso "
        "video in una singola rappresentazione cronologica.\n\n"

        "Regole:\n"
        "- usa esclusivamente gli eventi presenti negli input;\n"
        "- elimina i duplicati dovuti alla suddivisione "
        "dell'input;\n"
        "- non fondere eventi realmente distinti o ripetuti;\n"
        "- conserva tempi, partecipanti ed evidenze;\n"
        "- non inventare nuovi eventi o relazioni;\n"
        "- ordina gli eventi temporalmente;\n"
        "- assegna nuovamente ID E0001, E0002, ...;\n"
        "- aggiorna le relazioni temporali con i nuovi ID;\n"
        "- events deve essere sempre un array JSON;\n"
        "- ogni elemento di events deve essere "
        "un oggetto JSON;\n"
        "- temporal_relations deve essere sempre "
        "un array JSON;\n"
        "- ogni elemento di temporal_relations deve essere "
        "un oggetto JSON con first_event, relation, "
        "second_event e confidence;\n"
        "- non usare stringhe testuali per rappresentare "
        "eventi o relazioni;\n"
        "- restituisci esclusivamente JSON valido.\n\n"

        f"VIDEO:\n{video_id}\n\n"

        "INPUT:\n"
        + json.dumps(
            clean_results,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def compact_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    segments = payload.get(
        "segments",
        [],
    )

    if segments is None:
        segments = []

    if not isinstance(
        segments,
        list,
    ):
        raise ValueError(
            "'segments' non è una lista."
        )

    compact_segments: list[
        dict[str, Any]
    ] = []

    for index, segment in enumerate(
        segments
    ):
        if not isinstance(
            segment,
            dict,
        ):
            raise ValueError(
                f"segments[{index}] non è un oggetto."
            )

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

        for field in SEMANTIC_FIELDS:
            value = segment.get(field)

            if value not in (
                None,
                "",
                [],
                {},
            ):
                item[field] = value

        compact_segments.append(
            item
        )

    return {
        "id_video": payload.get(
            "id_video"
        ),
        "segments": compact_segments,
    }


def create_payload_chunks(
    payload: dict[str, Any],
    infer: Inferencer,
) -> list[dict[str, Any]]:
    segments = payload["segments"]

    if not segments:
        return []

    video_id = payload.get(
        "id_video"
    )

    chunks: list[
        dict[str, Any]
    ] = []

    current: list[
        dict[str, Any]
    ] = []

    for segment in segments:
        candidate = {
            "id_video": video_id,
            "segments": (
                current
                + [segment]
            ),
        }

        if infer.fits(
            build_prompt(
                candidate
            )
        ):
            current.append(
                segment
            )
            continue

        if not current:
            raise ValueError(
                f"Il segmento "
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

        chunks.append(
            {
                "id_video": video_id,
                "segments": current,
            }
        )

        overlap = (
            current[-CHUNK_OVERLAP:]
            if CHUNK_OVERLAP > 0
            else []
        )

        current = (
            overlap
            + [segment]
        )

        candidate = {
            "id_video": video_id,
            "segments": current,
        }

        if not infer.fits(
            build_prompt(
                candidate
            )
        ):
            current = [
                segment
            ]

        candidate = {
            "id_video": video_id,
            "segments": current,
        }

        if not infer.fits(
            build_prompt(
                candidate
            )
        ):
            raise ValueError(
                f"Il segmento "
                f"{segment.get('segment_id')} "
                "supera da solo la context window."
            )

    if current:
        chunks.append(
            {
                "id_video": video_id,
                "segments": current,
            }
        )

    return chunks


def generate_with_retry(
    infer: Inferencer,
    prompt: str,
    out_dir: Path,
    name: str,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        raw = ""

        print(
            f"    Tentativo "
            f"{attempt}/{MAX_RETRIES}",
            flush=True,
        )

        try:
            raw = infer.generate(
                prompt
            )

        except Exception as exc:
            if is_engine_dead_exception(
                exc
            ):
                raise EngineFatalError(
                    "EngineCore vLLM terminato. "
                    "La stessa istanza non può essere riutilizzata."
                ) from exc

            last_error = exc

            print(
                f"    [WARN] "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            failed_path = save_failed_raw(
                out_dir,
                name,
                attempt,
                raw,
            )

            print(
                f"    RAW salvato in: "
                f"{failed_path}",
                flush=True,
            )

            continue

        try:
            parsed = extract_json_from_text(
                raw
            )

            result = normalize_result(
                parsed
            )

        except Exception as exc:
            last_error = exc

            print(
                f"    [WARN] "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            failed_path = save_failed_raw(
                out_dir,
                name,
                attempt,
                raw,
            )

            print(
                f"    RAW salvato in: "
                f"{failed_path}",
                flush=True,
            )

            continue

        warnings = result.get(
            "_normalization_warnings",
            [],
        )

        print(
            f"    JSON valido "
            f"({len(raw)} chars)"
            f" | eventi="
            f"{len(result['events'])}"
            f" | relazioni="
            f"{len(result['temporal_relations'])}",
            flush=True,
        )

        for warning in warnings:
            print(
                f"    [NORMALIZE] "
                f"{warning}",
                flush=True,
            )

        return result

    raise RuntimeError(
        f"Fallito dopo "
        f"{MAX_RETRIES} tentativi. "
        f"Ultimo errore: "
        f"{last_error}"
    )


def merge_hierarchical(
    video_id: str,
    results: list[dict[str, Any]],
    infer: Inferencer,
    out_dir: Path,
) -> dict[str, Any]:
    if not results:
        return {
            "events": [],
            "temporal_relations": [],
        }

    if len(results) == 1:
        return results[0]

    current_results = results
    level = 1

    while len(
        current_results
    ) > 1:
        groups: list[
            list[dict[str, Any]]
        ] = []

        current_group: list[
            dict[str, Any]
        ] = []

        for result in current_results:
            candidate = (
                current_group
                + [result]
            )

            prompt = build_merge_prompt(
                video_id,
                candidate,
            )

            if infer.fits(
                prompt
            ):
                current_group.append(
                    result
                )
                continue

            if not current_group:
                raise ValueError(
                    f"Un risultato parziale "
                    f"di {video_id} supera "
                    "da solo la context window."
                )

            groups.append(
                current_group
            )

            current_group = [
                result
            ]

        if current_group:
            groups.append(
                current_group
            )

        if (
            len(groups)
            == len(current_results)
            and all(
                len(group) == 1
                for group in groups
            )
        ):
            raise ValueError(
                f"Impossibile ridurre "
                f"il merge per {video_id}: "
                "ogni risultato occupa "
                "un gruppo indipendente."
            )

        next_results: list[
            dict[str, Any]
        ] = []

        for group_idx, group in enumerate(
            groups,
            start=1,
        ):
            if len(group) == 1:
                next_results.append(
                    group[0]
                )
                continue

            print(
                f"  Merge livello {level}, "
                f"gruppo "
                f"{group_idx}/{len(groups)}",
                flush=True,
            )

            merged = generate_with_retry(
                infer,
                build_merge_prompt(
                    video_id,
                    group,
                ),
                out_dir,
                (
                    f"{video_id}"
                    f"_merge_L{level}"
                    f"_{group_idx:02d}"
                ),
            )

            next_results.append(
                merged
            )

        current_results = (
            next_results
        )

        level += 1

    return current_results[0]


def empty_result(
    video_id: str,
    model: str,
    source: Path,
) -> dict[str, Any]:
    return {
        "events": [],
        "temporal_relations": [],
        "id_video": video_id,
        "model": model,
        "source_semantic_file": str(
            source
        ),
    }


def collect_warnings(
    results: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []

    for result in results:
        warnings.extend(
            result.get(
                "_normalization_warnings",
                [],
            )
        )

    return warnings


def process_video(
    path: Path,
    infer: Inferencer,
    args: argparse.Namespace,
    index: int,
    total: int,
) -> bool:
    payload = read_json(
        path
    )

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
            f"[{index}/{total}] "
            f"SKIP {video_id}",
            flush=True,
        )

        return False

    print(
        f"\n[{index}/{total}] "
        f"{video_id}",
        flush=True,
    )

    semantic = compact_payload(
        payload
    )

    if not semantic["segments"]:
        write_json(
            output_path,
            empty_result(
                video_id,
                args.model,
                path,
            ),
        )

        print(
            f"  Salvato: "
            f"{output_path.name}"
            " | Eventi: 0",
            flush=True,
        )

        return True

    chunks = create_payload_chunks(
        semantic,
        infer,
    )

    print(
        f"  Segmenti: "
        f"{len(semantic['segments'])}"
        f" | Chunk elaborati: "
        f"{len(chunks)}",
        flush=True,
    )

    partial_results: list[
        dict[str, Any]
    ] = []

    for chunk_idx, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"  Chunk "
            f"{chunk_idx}/{len(chunks)} "
            f"({len(chunk['segments'])} segs)",
            flush=True,
        )

        partial = generate_with_retry(
            infer,
            build_prompt(
                chunk
            ),
            args.output_directory,
            (
                f"{video_id}"
                f"_chunk_{chunk_idx:02d}"
            ),
        )

        partial_results.append(
            partial
        )

    partial_warnings = collect_warnings(
        partial_results
    )

    final_result = merge_hierarchical(
        video_id,
        partial_results,
        infer,
        args.output_directory,
    )

    final_result = normalize_result(
        final_result
    )

    all_warnings = (
        partial_warnings
        + final_result.get(
            "_normalization_warnings",
            [],
        )
    )

    if all_warnings:
        final_result[
            "_normalization_warnings"
        ] = list(
            dict.fromkeys(
                all_warnings
            )
        )
    else:
        final_result.pop(
            "_normalization_warnings",
            None,
        )

    final_result.update(
        {
            "id_video": video_id,
            "model": args.model,
            "source_semantic_file": str(
                path
            ),
        }
    )

    write_json(
        output_path,
        final_result,
    )

    print(
        f"  Salvato: "
        f"{output_path.name}"
        f" | Eventi: "
        f"{len(final_result['events'])}"
        f" | Relazioni: "
        f"{len(final_result['temporal_relations'])}"
        f" | Warning: "
        f"{len(final_result.get('_normalization_warnings', []))}",
        flush=True,
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Event extraction della Preliminary Analysis "
            "con Gemma 4 12B."
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
            "Nessun file *_semantic.json "
            f"trovato in {args.semantic_directory}"
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    infer = Inferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_utilization=args.gpu_utilization,
    )

    stats = {
        "completed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for idx, path in enumerate(
        files,
        start=1,
    ):
        try:
            completed = process_video(
                path,
                infer,
                args,
                idx,
                len(files),
            )

            stats[
                "completed"
                if completed
                else "skipped"
            ] += 1

        except EngineFatalError:
            stats["failed"] += 1

            print(
                "\n"
                "========================================\n"
                "ENGINE vLLM TERMINATO\n"
                "========================================\n"
                f"File corrente: {path.name}\n\n"
                "EngineCore è morto e non può essere "
                "riutilizzato nello stesso processo.\n\n"
                "Gli output completati sono già salvati.\n"
                "Rieseguendo senza --overwrite verranno "
                "saltati automaticamente.\n"
                "========================================",
                flush=True,
            )

            traceback.print_exc()

            sys.exit(2)

        except Exception:
            stats["failed"] += 1

            print(
                "\n"
                "[ERROR] Elaborazione fallita per: "
                f"{path.name}",
                flush=True,
            )

            traceback.print_exc()

    print(
        "\n"
        "========================================\n"
        "ELABORAZIONE COMPLETATA\n"
        "========================================\n"
        f"Video totali: {len(files)}\n"
        f"Completati:   {stats['completed']}\n"
        f"Saltati:      {stats['skipped']}\n"
        f"Falliti:      {stats['failed']}\n"
        "========================================",
        flush=True,
    )


if __name__ == "__main__":
    main()