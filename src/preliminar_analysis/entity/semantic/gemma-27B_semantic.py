from __future__ import annotations



import sys
import argparse
import time
from pathlib import Path
from PIL import Image
import os

# Disabilita FlashInfer sia per il sampling che per l'attention
os.environ["VLLM_USE_FLASHINFER_SAMPLING"] = "0"


# ==============================================================================
# DISABILITA FLASHINFER PER EVITARE L'ERRORE DI COMPILAZIONE CUDA CUB
# ==============================================================================
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
# Isola le 4 GPU desiderate (2, 3, 4, 5)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

# Prevenzione Deadlock NCCL, Timeouts e stalli di Shared Memory
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["VLLM_CUSTOM_ALL_REDUCE"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["VLLM_RPC_TIMEOUT"] = "300"

# Limitazione Threading CPU per evitare contesa
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"



# Importa VllmModel dal modulo in src/vllm.py
try:
    from src.vllm import VllmModel
except ImportError:
    try:
        from vllm import VllmModel
    except ImportError as error:
        raise RuntimeError(
            "Impossibile importare VllmModel. Assicurati che 'src/vllm.py' sia presente nel progetto."
        ) from error

from utils import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

DEFAULT_MODEL = "google/gemma-3-27b-it"


class GemmaVLLMInferencer:
    """
    Adapter per l'inferenza di Gemma che sfrutta il wrapper VllmModel (src/vllm.py).
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 10000,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        print("\n" + "=" * 80, flush=True)
        print(f" [1/2] Inizializzazione {model_id} tramite VllmModel...", flush=True)
        print(" Configurazione: Tensor Parallelism automatico sulle GPU visibili (TP=4)", flush=True)
        print("=" * 80 + "\n", flush=True)

        self.max_new_tokens = max_new_tokens
        self.window_counter = 0

        start_load = time.time()

        # VllmModel calcola automaticamente tensor_parallel_size in base alle GPU visibili
        self.vllm_model = VllmModel(
            model_id=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_tokens=self.max_new_tokens,
            temperature=0.0,  # Deterministico per JSON
            verbose=True,
        )

        load_duration = time.time() - start_load
        print(f"\n [2/2] Modello caricato con successo in {load_duration:.2f}s!\n", flush=True)

    def __call__(self, frame_paths: tuple[Path, ...], prompt: str) -> str:
        self.window_counter += 1
        current_window = self.window_counter

        print(
            f"   --> [Finestra #{current_window}] Caricamento {len(frame_paths)} frame...",
            flush=True,
        )

        start_time = time.time()
        images = [Image.open(path).convert("RGB") for path in frame_paths]

        try:
            print(
                f"   --> [Finestra #{current_window}] Generazione risposta su vLLM...",
                flush=True,
            )

            # Invocazione via VllmModel.generate_continuation
            results = self.vllm_model.generate_continuation(
                prompts=[prompt],
                images=[images],
                max_tokens=self.max_new_tokens,
                temperature=0.0,
            )

            elapsed = time.time() - start_time
            print(
                f"   <-- [Finestra #{current_window}] Completata in {elapsed:.2f}s.",
                flush=True,
            )

            output_text = results[0] if results else ""
            return output_text.strip()

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisi semantica video con Gemma tramite VllmModel su 4 GPU."
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
        default="data/preliminar_analysis/entity/entity_semantic/gemma-27B",
        type=Path,
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--gpu-utilization", type=float, default=0.85)

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video da elaborare (0 = tutti).",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")

    args = parser.parse_args()

    video_directories = discover_video_directories(args.preprocessing_directory)

    if args.limit_videos > 0:
        video_directories = video_directories[: args.limit_videos]

    if not video_directories:
        parser.error("Non sono state trovate cartelle dense_frames.")

    args.output_directory.mkdir(parents=True, exist_ok=True)

    inferencer = GemmaVLLMInferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_utilization,
    )

    rows = []
    total_videos = len(video_directories)

    print(f"\nInizio elaborazione di {total_videos} video...", flush=True)

    for index, video_directory in enumerate(video_directories, start=1):
        file_output = args.output_directory / f"{video_directory.name}_semantic.json"
        if file_output.exists() and not args.overwrite:
            print(f"\n[{index}/{total_videos}] SALTO: File {file_output.name} già presente.", flush=True)
            continue

        print(f"\n========================================================", flush=True)
        print(f"[{index}/{total_videos}] Inizio analisi video: {video_directory.name}", flush=True)
        print(f"========================================================", flush=True)

        video_start_time = time.time()

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

            video_elapsed = time.time() - video_start_time
            print(
                f"SUCCESS: Video '{video_directory.name}' elaborato in {video_elapsed:.2f}s",
                flush=True,
            )

        except Exception as error:
            print(
                f"ERRORE durante {video_directory.name}: {type(error).__name__}: {error}",
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

    write_csv(args.output_directory / "riepilogo_video.csv", rows)
    print(f"\nCOMPLETATO TOTALE! Risultati salvati in: {args.output_directory}", flush=True)


if __name__ == "__main__":
    main()