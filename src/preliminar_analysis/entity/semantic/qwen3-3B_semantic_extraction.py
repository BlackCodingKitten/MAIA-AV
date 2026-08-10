from __future__ import annotations

import os
import sys
from pathlib import Path
# ==============================================================================
# DISABILITA FLASHINFER PER EVITARE CRASH DI COMPILAZIONE E IMPORTAZIONE
# Deve essere fatto PRIMA di importare vLLM o Torch.
# ==============================================================================
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ==============================================================================
# FORZATURA GPU: Isola le GPU fisiche 5 e 6.
# vLLM le legherà insieme in Tensor Parallelism (TP=2).
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
from PIL import Image

try:
    from src.vllm import LLM, SamplingParams
except ImportError as error:
    raise RuntimeError(
        "Manca la libreria vLLM. Installala con: pip install vllm"
    ) from error

from utils import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class Qwen30VLLMInferencer:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        max_frames: int = 8,
    ) -> None:
        print("\n" + "=" * 80)
        print(" Inizializzazione Qwen3-VL-30B con Tensor Parallelism (TP=2)")
        print(" Le GPU fisiche 5 e 6 lavoreranno SIMULTANEAMENTE.")
        print("=" * 80 + "\n")

        self.max_new_tokens = max_new_tokens

        # tensor_parallel_size=2 spezza i calcoli dei layer simultaneamente su GPU 5 e 6
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=2,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            limit_mm_per_prompt={"image": max_frames},
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministico per output JSON
            max_tokens=self.max_new_tokens,
        )

    def __call__(self, frame_paths: tuple[Path, ...], prompt: str) -> str:
        images = [Image.open(path).convert("RGB") for path in frame_paths]

        try:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Sei un annotatore scientifico di video. "
                        "Restituisci esclusivamente JSON valido, "
                        "conciso e verificabile."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]

            outputs = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            return outputs[0].outputs[0].text.strip()

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi semantica dei dense frame con Qwen3-VL-30B in "
            "Tensor Parallelism su GPU 5 e 6."
        )
    )

    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/preprocessing",
        help="Directory contenente una cartella per video con dense_frames/.",
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default="data/preliminar_analysis/entity/entity_semantic/qwen-30B",
    )

    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=10000)

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video; 0 = tutti.",
    )

    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Mantenuto per compatibilità CLI (vLLM gestisce l'attenzione nativamente).",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")

    args = parser.parse_args()

    video_directories = discover_video_directories(
        args.preprocessing_directory
    )

    if args.limit_videos > 0:
        video_directories = video_directories[: args.limit_videos]

    if not video_directories:
        parser.error("Non sono state trovate cartelle video contenenti dense_frames.")

    args.output_directory.mkdir(parents=True, exist_ok=True)

    inferencer = Qwen30VLLMInferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
    )

    rows = []

    for index, video_directory in enumerate(video_directories, start=1):
        print(
            f"[{index}/{len(video_directories)}] "
            f"Analisi PARALLELA Qwen30: {video_directory.name}",
            flush=True,
        )

        try:
            row = process_video_with_inferencer(
                model_name=args.model,
                video_directory=video_directory,
                output_directory=args.output_directory,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                max_frames=args.max_frames,
                inferencer=inferencer,
                overwrite=args.overwrite,
                keep_raw=args.keep_raw,
            )
            rows.append(row)

        except Exception as error:
            print(
                f"Errore durante {video_directory.name}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            rows.append(
                {
                    "id_video": video_directory.name,
                    "status": "failed",
                    "numero_segmenti": None,
                    "numero_elementi": None,
                    "numero_errori": 1,
                    "output": f"{type(error).__name__}: {error}",
                }
            )

    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(f"\nCompletato! Risultati salvati in: {args.output_directory}")


if __name__ == "__main__":
    main()