from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
os.environ["VLLM_USE_FLASHINFER_SAMPLING"] = "0"
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
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import parse_model_json, write_csv, write_json

DEFAULT_MODEL = "google/gemma-3-27b-it"


def compact_semantic_input(payload: dict[str, Any]) -> dict[str, Any]:
    keys = ("segment_id", "start_time", "end_time", "entities", "actions", "events", "state_changes", "temporal_relations", "spatial_relations")
    return {
        "id_video": payload.get("id_video"),
        "segments": [{key: segment.get(key, [] if key not in {"segment_id", "start_time", "end_time"} else None) for key in keys} for segment in payload.get("segments", [])],
    }


def build_event_prompt(payload: dict[str, Any]) -> str:
    schema = {
        "events": [{
            "event_id": "E0001",
            "description": "string",
            "start_time": 0.0,
            "end_time": 0.0,
            "participants": ["string"],
            "evidence_segments": ["segment_0000"],
            "evidence_type": "observed|inferred|uncertain",
            "confidence": 0.0,
        }],
        "temporal_relations": [{
            "first_event": "E0001",
            "relation": "before|after|overlaps|simultaneous|during",
            "second_event": "E0002",
            "confidence": 0.0,
        }],
    }
    return (
        "Consolida l'analisi semantica di un singolo video in eventi cronologici. "
        "Unisci i duplicati dovuti alle finestre sovrapposte, mantieni separati gli eventi realmente distinti, "
        "usa solo tempi, partecipanti e relazioni supportati dall'input, non inventare cause o intenzioni. "
        "Assegna gli ID E0001, E0002, ... in ordine temporale. "
        "Restituisci esclusivamente un oggetto JSON valido conforme a questo schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\nINPUT:\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


class EventInferencer:
    def __init__(self, model_id: str, max_new_tokens: int, gpu_memory_utilization: float) -> None:
        if torch.cuda.device_count() != 4:
            raise RuntimeError(f"Mi aspettavo 4 GPU visibili, ma PyTorch ne vede {torch.cuda.device_count()}.")
        print(f"Caricamento {model_id} su GPU fisiche {os.environ['CUDA_VISIBLE_DEVICES']} (TP=4)...", flush=True)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def __call__(self, prompt: str) -> str:
        outputs = self.llm.chat([{"role": "user", "content": prompt}], sampling_params=self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""


def analyze_video(path: Path, output_directory: Path, inferencer: EventInferencer, model_id: str, overwrite: bool, keep_raw: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    video_id = str(payload.get("id_video") or path.stem.removesuffix("_semantic"))
    output_path = output_directory / f"{video_id}_events.json"
    if output_path.exists() and not overwrite:
        return {"id_video": video_id, "status": "skipped", "numero_eventi": None, "output": str(output_path)}

    raw = inferencer(build_event_prompt(compact_semantic_input(payload)))
    result = parse_model_json(raw)
    result.update({"id_video": video_id, "model": model_id, "source_semantic_file": str(path)})
    if keep_raw:
        result["raw_response"] = raw
    write_json(output_path, result)
    return {"id_video": video_id, "status": "completed", "numero_eventi": len(result.get("events", [])), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Event detection con Gemma-3-27B via vLLM su GPU 2,3,4,5.")
    parser.add_argument("semantic_directory", nargs="?", type=Path, default="data/preliminar_analysis/entity/entity_semantic/gemma-27B")
    parser.add_argument("output_directory", nargs="?", type=Path, default="data/preliminar_analysis/event_analysis/gemma-27B")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    files = sorted(args.semantic_directory.resolve().glob("*_semantic.json"))
    if args.limit_videos > 0:
        files = files[: args.limit_videos]
    if not files:
        parser.error(f"Nessun *_semantic.json trovato in {args.semantic_directory}")

    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    inferencer = EventInferencer(args.model, args.max_new_tokens, args.gpu_utilization)
    rows = []

    for index, path in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {path.name}", flush=True)
        try:
            rows.append(analyze_video(path, args.output_directory, inferencer, args.model, args.overwrite, args.keep_raw))
        except Exception as error:
            traceback.print_exc()
            rows.append({"id_video": path.stem, "status": "failed", "numero_eventi": None, "output": f"{type(error).__name__}: {error}"})

    write_csv(args.output_directory / "riepilogo_eventi.csv", rows)
    print(f"\nCompletato: {args.output_directory}")


if __name__ == "__main__":
    main()