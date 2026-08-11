from __future__ import annotations

import os

# Configurazione prima di importare torch/vLLM.
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,1,2"
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
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import discover_video_directories, process_video_with_inferencer, write_csv

DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class QwenVLLMInferencer:
    def __init__(self, model_id: str, max_new_tokens: int = 10000, gpu_memory_utilization: float = 0.85, max_frames: int = 32) -> None:
        if torch.cuda.device_count() != 4:
            raise RuntimeError(f"Mi aspettavo 4 GPU visibili, ma PyTorch ne vede {torch.cuda.device_count()}.")

        print(f"\nCaricamento {model_id} su GPU fisiche {os.environ['CUDA_VISIBLE_DEVICES']} (TP=4)...", flush=True)
        start = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max_frames},
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.max_frames = max_frames
        print(f"Modello caricato in {time.time() - start:.2f}s.\n", flush=True)

    def __call__(self, frame_paths: tuple[Path, ...], prompt: str) -> str:
        paths = [Path(path) for path in frame_paths[: self.max_frames]]
        if not paths:
            raise ValueError("La finestra non contiene frame.")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Frame inesistente: {path}")

        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())

            messages = [{
                "role": "user",
                "content": [
                    *({"type": "image", "image": str(path)} for path in paths),
                    {"type": "text", "text": prompt},
                ],
            }]
            formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.llm.generate(
                {"prompt": formatted_prompt, "multi_modal_data": {"image": images}},
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else ""
        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisi semantica con Qwen3-VL-30B-A3B-Instruct via vLLM su GPU 2,3,4,5.")
    parser.add_argument("preprocessing_directory", nargs="?", type=Path, default="data/preliminar_analysis/preprocessing")
    parser.add_argument("output_directory", nargs="?", type=Path, default="data/preliminar_analysis/entity/entity_semantic/qwen-30B")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    args.preprocessing_directory = args.preprocessing_directory.resolve()
    args.output_directory = args.output_directory.resolve()
    if not args.preprocessing_directory.exists():
        parser.error(f"Directory di preprocessing inesistente: {args.preprocessing_directory}")

    videos = discover_video_directories(args.preprocessing_directory)
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]
    if not videos:
        parser.error("Non sono state trovate cartelle dense_frames.")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    inferencer = QwenVLLMInferencer(args.model, args.max_new_tokens, args.gpu_utilization, args.max_frames)
    rows = []

    for index, video_directory in enumerate(videos, 1):
        output_path = args.output_directory / f"{video_directory.name}_semantic.json"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(videos)}] SALTO {video_directory.name}", flush=True)
            continue

        print(f"[{index}/{len(videos)}] {video_directory.name}", flush=True)
        try:
            rows.append(process_video_with_inferencer(
                model_name=args.model,
                video_directory=video_directory,
                output_directory=args.output_directory,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                max_frames=args.max_frames,
                inferencer=inferencer,
                overwrite=args.overwrite,
                keep_raw=args.keep_raw,
            ))
        except Exception as error:
            traceback.print_exc()
            rows.append({
                "id_video": video_directory.name,
                "status": "failed",
                "numero_segmenti": None,
                "numero_elementi": None,
                "numero_errori": 1,
                "output": f"{type(error).__name__}: {error}",
            })

    write_csv(args.output_directory / "riepilogo_video.csv", rows)
    print(f"\nCompletato: {args.output_directory}")


if __name__ == "__main__":
    main()