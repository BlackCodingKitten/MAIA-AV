from __future__ import annotations

import os

# Impostate per limitare a 2 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import time
import traceback
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)


DEFAULT_MODEL = "google/gemma-4-12B-it"
DEFAULT_OUTPUT = Path(
    "data/preliminar_analysis/entity/gemma-4-12B"
)


class Gemma4VLLMInferencer:
    """
    Entity/Semantic extraction per la preliminary analysis.

    Questa fase rimane intenzionalmente VISUAL-ONLY:
    riceve esclusivamente i frame prodotti dal preprocessing, esattamente
    come la precedente pipeline Gemma-27B.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 10000,
        gpu_memory_utilization: float = 0.85,
        max_frames: int = 32,
    ) -> None:
        self.model_id = model_id
        self.max_frames = max_frames
        self.window_counter = 0

        print("\n" + "=" * 80, flush=True)
        print(f"Caricamento modello: {model_id}", flush=True)
        print(
            f"GPU visibili: {os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        print("Tensor Parallelism: 2 GPU", flush=True)
        print("=" * 80 + "\n", flush=True)

        start = time.time()

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
            max_num_seqs=1,
            limit_mm_per_prompt={
                "image": max_frames,
            },
            # Per semantic extraction serve più dettaglio che nel VSV binario.
            mm_processor_kwargs={
                "max_soft_tokens": 280,
            },
            trust_remote_code=True,
            enforce_eager=True,
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        print(
            f"Modello caricato in {time.time() - start:.2f}s.",
            flush=True,
        )

    def __call__(
        self,
        frame_paths: tuple[Path, ...],
        prompt: str,
    ) -> str:
        self.window_counter += 1
        window_id = self.window_counter

        valid_paths = []

        for frame_path in frame_paths:
            frame_path = Path(frame_path)

            if frame_path.exists() and frame_path.is_file():
                valid_paths.append(frame_path)

        if not valid_paths:
            raise ValueError(
                "La finestra non contiene alcun frame valido."
            )

        valid_paths = valid_paths[:self.max_frames]

        print(
            f"--> [Finestra #{window_id}] "
            f"{len(valid_paths)} frame.",
            flush=True,
        )

        images = []

        try:
            for frame_path in valid_paths:
                with Image.open(frame_path) as image:
                    image.load()
                    images.append(
                        image.convert("RGB").copy()
                    )

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image"}
                            for _ in images
                        ],
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            formatted_prompt = (
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

            outputs = self.llm.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {
                        "image": images,
                    },
                },
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            if not outputs or not outputs[0].outputs:
                return ""

            return outputs[0].outputs[0].text.strip()

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi Entity/Semantic visual-only con "
            "Gemma-4-12B Unified."
        )
    )

    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
        default=Path(
            "data/preliminar_analysis/preprocessing"
        ),
        type=Path,
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        default=DEFAULT_OUTPUT,
        type=Path,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=10000,
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

    parser.add_argument(
        "--keep-raw",
        action="store_true",
    )

    args = parser.parse_args()

    args.preprocessing_directory = (
        args.preprocessing_directory.resolve()
    )
    args.output_directory = (
        args.output_directory.resolve()
    )

    if not args.preprocessing_directory.exists():
        parser.error(
            "Directory di preprocessing inesistente: "
            f"{args.preprocessing_directory}"
        )

    video_directories = discover_video_directories(
        args.preprocessing_directory
    )

    if args.limit_videos > 0:
        video_directories = video_directories[
            :args.limit_videos
        ]

    if not video_directories:
        parser.error(
            "Non sono state trovate cartelle dense_frames."
        )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    inferencer = Gemma4VLLMInferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_utilization,
        max_frames=args.max_frames,
    )

    rows = []
    total_videos = len(video_directories)

    for index, video_directory in enumerate(
        video_directories,
        start=1,
    ):
        file_output = (
            args.output_directory
            / f"{video_directory.name}_semantic.json"
        )

        if (
            file_output.exists()
            and not args.overwrite
        ):
            print(
                f"[{index}/{total_videos}] "
                f"SKIP {video_directory.name}",
                flush=True,
            )
            continue

        print(
            f"\n[{index}/{total_videos}] "
            f"Analisi: {video_directory.name}",
            flush=True,
        )

        start = time.time()

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

            print(
                f"SUCCESS in {time.time() - start:.2f}s",
                flush=True,
            )

        except Exception as error:
            traceback.print_exc()

            rows.append({
                "id_video": video_directory.name,
                "status": "failed",
                "numero_segmenti": None,
                "numero_elementi": None,
                "numero_errori": 1,
                "output": (
                    f"{type(error).__name__}: {error}"
                ),
            })

    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(
        f"\nCompletato. Output: "
        f"{args.output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()