from __future__ import annotations

import os

<<<<<<< HEAD
# ==============================================================================
# CONFIGURAZIONE AMBIENTE
# DEVE ESSERE FATTA PRIMA DI IMPORTARE vLLM / torch
# ==============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

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
=======
# Impostate per limitare a 2 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
>>>>>>> refs/remotes/origin/main
import time
import traceback
from pathlib import Path

<<<<<<< HEAD
import torch
=======
>>>>>>> refs/remotes/origin/main
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


<<<<<<< HEAD
# ==============================================================================
# PROJECT PATH
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
=======
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
>>>>>>> refs/remotes/origin/main

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

<<<<<<< HEAD

=======
>>>>>>> refs/remotes/origin/main
from utils import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

<<<<<<< HEAD
DEFAULT_MODEL = "google/gemma-4-12B-it"


# ==============================================================================
# GEMMA 3 + vLLM
# ==============================================================================

class GemmaVLLMInferencer:
=======

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
>>>>>>> refs/remotes/origin/main

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 10000,
        gpu_memory_utilization: float = 0.85,
        max_frames: int = 32,
    ) -> None:
<<<<<<< HEAD
=======
        self.model_id = model_id
        self.max_frames = max_frames
        self.window_counter = 0
>>>>>>> refs/remotes/origin/main

        print("\n" + "=" * 80, flush=True)
        print(f"Caricamento modello: {model_id}", flush=True)
        print(
            f"GPU visibili: {os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )
<<<<<<< HEAD
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
=======
        print("Tensor Parallelism: 2 GPU", flush=True)
        print("=" * 80 + "\n", flush=True)

        start = time.time()

        self.processor = AutoProcessor.from_pretrained(model_id)

>>>>>>> refs/remotes/origin/main
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=2,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,
<<<<<<< HEAD
            limit_mm_per_prompt={
                "image": max_frames,
            },
=======
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
>>>>>>> refs/remotes/origin/main
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

<<<<<<< HEAD
        elapsed = time.time() - start

        print(
            f"\nModello caricato correttamente in {elapsed:.2f}s.\n",
            flush=True,
        )


=======
        print(
            f"Modello caricato in {time.time() - start:.2f}s.",
            flush=True,
        )

>>>>>>> refs/remotes/origin/main
    def __call__(
        self,
        frame_paths: tuple[Path, ...],
        prompt: str,
    ) -> str:
<<<<<<< HEAD

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
=======
        self.window_counter += 1
        window_id = self.window_counter

        valid_paths = []

        for frame_path in frame_paths:
            frame_path = Path(frame_path)

            if frame_path.exists() and frame_path.is_file():
                valid_paths.append(frame_path)
>>>>>>> refs/remotes/origin/main

        if not valid_paths:
            raise ValueError(
                "La finestra non contiene alcun frame valido."
            )

<<<<<<< HEAD
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

=======
        valid_paths = valid_paths[:self.max_frames]

        print(
            f"--> [Finestra #{window_id}] "
            f"{len(valid_paths)} frame.",
            flush=True,
        )

        images = []

        try:
            for frame_path in valid_paths:
>>>>>>> refs/remotes/origin/main
                with Image.open(frame_path) as image:
                    image.load()
                    images.append(
                        image.convert("RGB").copy()
                    )

<<<<<<< HEAD
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

=======
>>>>>>> refs/remotes/origin/main
            messages = [
                {
                    "role": "user",
                    "content": [
<<<<<<< HEAD
                        *image_placeholders,
=======
                        *[
                            {"type": "image"}
                            for _ in images
                        ],
>>>>>>> refs/remotes/origin/main
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

<<<<<<< HEAD
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

=======
            formatted_prompt = (
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

>>>>>>> refs/remotes/origin/main
            outputs = self.llm.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {
                        "image": images,
                    },
                },
                sampling_params=self.sampling_params,
<<<<<<< HEAD
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
=======
                use_tqdm=False,
            )

            if not outputs or not outputs[0].outputs:
>>>>>>> refs/remotes/origin/main
                return ""

            return outputs[0].outputs[0].text.strip()

        finally:
<<<<<<< HEAD

=======
>>>>>>> refs/remotes/origin/main
            for image in images:
                image.close()


<<<<<<< HEAD
# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analisi semantica video con Gemma-4-12B "
=======
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi Entity/Semantic visual-only con "
            "Gemma-4-12B Unified."
>>>>>>> refs/remotes/origin/main
        )
    )

    parser.add_argument(
        "preprocessing_directory",
        nargs="?",
<<<<<<< HEAD
        default="data/preliminar_analysis/preprocessing",
=======
        default=Path(
            "data/preliminar_analysis/preprocessing"
        ),
>>>>>>> refs/remotes/origin/main
        type=Path,
    )

    parser.add_argument(
        "output_directory",
        nargs="?",
<<<<<<< HEAD
        default=("data/preliminar_analysis/entity/gemma-12B"),
=======
        default=DEFAULT_OUTPUT,
>>>>>>> refs/remotes/origin/main
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
<<<<<<< HEAD
        help="Numero massimo di video da elaborare (0 = tutti).",
=======
>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
    # --------------------------------------------------------------------------
    # VALIDAZIONE INPUT
    # --------------------------------------------------------------------------

    args.preprocessing_directory = (
        args.preprocessing_directory.resolve()
    )

=======
    args.preprocessing_directory = (
        args.preprocessing_directory.resolve()
    )
>>>>>>> refs/remotes/origin/main
    args.output_directory = (
        args.output_directory.resolve()
    )

    if not args.preprocessing_directory.exists():
        parser.error(
            "Directory di preprocessing inesistente: "
            f"{args.preprocessing_directory}"
        )

<<<<<<< HEAD
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

=======
>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
    # --------------------------------------------------------------------------
    # MODELLO
    # --------------------------------------------------------------------------

    inferencer = GemmaVLLMInferencer(
=======
    inferencer = Gemma4VLLMInferencer(
>>>>>>> refs/remotes/origin/main
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_utilization,
        max_frames=args.max_frames,
    )

    rows = []
<<<<<<< HEAD

    total_videos = len(video_directories)

    print(
        f"\nInizio elaborazione di "
        f"{total_videos} video...",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------------------------------

=======
    total_videos = len(video_directories)

>>>>>>> refs/remotes/origin/main
    for index, video_directory in enumerate(
        video_directories,
        start=1,
    ):
<<<<<<< HEAD

=======
>>>>>>> refs/remotes/origin/main
        file_output = (
            args.output_directory
            / f"{video_directory.name}_semantic.json"
        )

        if (
            file_output.exists()
            and not args.overwrite
        ):
<<<<<<< HEAD

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

=======
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
>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
            elapsed = time.time() - video_start

            print(
                f"SUCCESS: '{video_directory.name}' "
                f"elaborato in {elapsed:.2f}s",
=======
            print(
                f"SUCCESS in {time.time() - start:.2f}s",
>>>>>>> refs/remotes/origin/main
                flush=True,
            )

        except Exception as error:
<<<<<<< HEAD

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
=======
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
>>>>>>> refs/remotes/origin/main

    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(
<<<<<<< HEAD
        "\nCOMPLETATO! "
        f"Risultati salvati in: "
=======
        f"\nCompletato. Output: "
>>>>>>> refs/remotes/origin/main
        f"{args.output_directory}",
        flush=True,
    )


if __name__ == "__main__":
    main()