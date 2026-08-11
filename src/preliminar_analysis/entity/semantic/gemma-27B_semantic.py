from __future__ import annotations

import os

# ==============================================================================
# CONFIGURAZIONE AMBIENTE
# DEVE ESSERE FATTA PRIMA DI IMPORTARE vLLM / torch
# ==============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,0"

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


# ==============================================================================
# IMPORT
# ==============================================================================

import sys
import argparse
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# ==============================================================================
# PROJECT PATH
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)


DEFAULT_MODEL = "google/gemma-3-27b-it"


# ==============================================================================
# GEMMA 3 + vLLM
# ==============================================================================

class GemmaVLLMInferencer:

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 10000,
        gpu_memory_utilization: float = 0.85,
        max_frames: int = 32,
    ) -> None:

        print("\n" + "=" * 80, flush=True)
        print(f"Caricamento modello: {model_id}", flush=True)
        print(
            f"GPU visibili: {os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
        print(
            f"Tensor Parallelism: {torch.cuda.device_count()} GPU",
            flush=True,
        )
        print("=" * 80 + "\n", flush=True)

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.window_counter = 0

        if torch.cuda.device_count() != 4:
            raise RuntimeError(
                f"Mi aspettavo 4 GPU visibili, ma PyTorch ne vede "
                f"{torch.cuda.device_count()}."
            )

        start = time.time()

        # Processor ufficiale Gemma 3
        self.processor = AutoProcessor.from_pretrained(
            model_id,
        )

        # vLLM ufficiale.
        #
        # limit_mm_per_prompt è FONDAMENTALE quando vengono passate
        # più immagini nello stesso prompt.
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
            limit_mm_per_prompt={
                "image": max_frames,
            },
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        elapsed = time.time() - start

        print(
            f"\nModello caricato correttamente in {elapsed:.2f}s.\n",
            flush=True,
        )


    def __call__(
        self,
        frame_paths: tuple[Path, ...],
        prompt: str,
    ) -> str:

        self.window_counter += 1
        window_id = self.window_counter

        print(
            f"--> [Finestra #{window_id}] "
            f"Ricevuti {len(frame_paths)} frame.",
            flush=True,
        )

        # ----------------------------------------------------------------------
        # CONTROLLO ESPLICITO DEI PATH
        # ----------------------------------------------------------------------

        valid_paths: list[Path] = []

        for i, frame_path in enumerate(frame_paths):

            if frame_path is None:
                raise ValueError(
                    f"frame_paths[{i}] è None."
                )

            frame_path = Path(frame_path)

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Frame inesistente: {frame_path}"
                )

            if not frame_path.is_file():
                raise ValueError(
                    f"Il frame non è un file: {frame_path}"
                )

            valid_paths.append(frame_path)

        if not valid_paths:
            raise ValueError(
                "La finestra non contiene alcun frame valido."
            )

        if len(valid_paths) > self.max_frames:
            valid_paths = valid_paths[:self.max_frames]

        print(
            f"--> [Finestra #{window_id}] "
            f"Caricamento di {len(valid_paths)} immagini.",
            flush=True,
        )

        # ----------------------------------------------------------------------
        # CARICAMENTO PIL
        # ----------------------------------------------------------------------

        images: list[Image.Image] = []

        try:

            for frame_path in valid_paths:

                # IMPORTANTE:
                # facciamo una copia completamente caricata in memoria.
                #
                # vLLM riceverà direttamente gli oggetti PIL;
                # non dovrà ricostruire alcun Path da image.filename.

                with Image.open(frame_path) as image:
                    image.load()
                    images.append(
                        image.convert("RGB").copy()
                    )

            # ------------------------------------------------------------------
            # TEMPLATE GEMMA 3
            #
            # È la struttura prevista dall'esempio ufficiale vLLM:
            #
            #   {"type": "image", "image": ...}
            #
            # per ogni immagine.
            # ------------------------------------------------------------------

            image_placeholders = [
                {
                    "type": "image",
                    "image": str(frame_path),
                }
                for frame_path in valid_paths
            ]

            messages = [
                {
                    "role": "user",
                    "content": [
                        *image_placeholders,
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            formatted_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            print(
                f"--> [Finestra #{window_id}] "
                f"Prompt costruito correttamente.",
                flush=True,
            )

            start = time.time()

            # ------------------------------------------------------------------
            # vLLM MULTIMODALE
            #
            # Questa è la parte importante:
            #
            # prompt -> stringa ottenuta dal template
            # image  -> lista reale di PIL.Image
            # ------------------------------------------------------------------

            outputs = self.llm.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {
                        "image": images,
                    },
                },
                sampling_params=self.sampling_params,
            )

            elapsed = time.time() - start

            print(
                f"<-- [Finestra #{window_id}] "
                f"Generazione completata in {elapsed:.2f}s.",
                flush=True,
            )

            if not outputs:
                return ""

            if not outputs[0].outputs:
                return ""

            return outputs[0].outputs[0].text.strip()

        finally:

            for image in images:
                image.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analisi semantica video con Gemma-3-27B "
            "tramite vLLM su GPU 2,3,4,5."
        )
    )

    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
        default="data/preliminar_analysis/preprocessing",
        type=Path,
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
        default=("data/preliminar_analysis/entity/gemma-27B"),
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
        help="Numero massimo di video da elaborare (0 = tutti).",
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

    # --------------------------------------------------------------------------
    # VALIDAZIONE INPUT
    # --------------------------------------------------------------------------

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

    print(
        f"Preprocessing directory: "
        f"{args.preprocessing_directory}",
        flush=True,
    )

    print(
        f"Output directory: "
        f"{args.output_directory}",
        flush=True,
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

    # --------------------------------------------------------------------------
    # MODELLO
    # --------------------------------------------------------------------------

    inferencer = GemmaVLLMInferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_utilization,
        max_frames=args.max_frames,
    )

    rows = []

    total_videos = len(video_directories)

    print(
        f"\nInizio elaborazione di "
        f"{total_videos} video...",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------------------------

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
                f"\n[{index}/{total_videos}] "
                f"SALTO: {file_output.name} "
                f"già presente.",
                flush=True,
            )

            continue

        print(
            "\n========================================================",
            flush=True,
        )

        print(
            f"[{index}/{total_videos}] "
            f"Analisi video: {video_directory.name}",
            flush=True,
        )

        print(
            "========================================================",
            flush=True,
        )

        video_start = time.time()

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

            elapsed = time.time() - video_start

            print(
                f"SUCCESS: '{video_directory.name}' "
                f"elaborato in {elapsed:.2f}s",
                flush=True,
            )

        except Exception as error:

            print(
                f"\nERRORE durante "
                f"{video_directory.name}",
                flush=True,
            )

            # NON NASCONDIAMO PIÙ IL TRACEBACK
            traceback.print_exc()

            rows.append(
                {
                    "id_video": video_directory.name,
                    "status": "failed",
                    "numero_segmenti": None,
                    "numero_elementi": None,
                    "numero_errori": 1,
                    "output": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

    # --------------------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------------------

    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(
        "\nCOMPLETATO! "
        f"Risultati salvati in: "
        f"{args.output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()