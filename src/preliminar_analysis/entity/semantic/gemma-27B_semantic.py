from __future__ import annotations

import os
import time

# ==============================================================================
# FORZATURA GPU: Isola le GPU fisiche da 2 a 5 (2, 3, 4, 5 -> 4 GPU totali).
# Fissa anche i thread CPU a 1 per evitare contesa e zittire i warning.
# ==============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

# Risoluzione Timeout e Deadlock
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["VLLM_CUSTOM_ALL_REDUCE"] = "0" 
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["VLLM_RPC_TIMEOUT"] = "300"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from PIL import Image

try:
    from vllm import LLM, SamplingParams
except ImportError as error:
    raise RuntimeError(
        "Manca la libreria vLLM. Installala con: pip install vllm"
    ) from error

from semantic_common import (
    discover_video_directories,
    process_video_with_inferencer,
    write_csv,
)

DEFAULT_MODEL = "google/gemma-3-27b-it"


class Gemma27VLLMInferencer:
    def __init__(self, model_id: str, max_new_tokens: int, max_frames: int = 8) -> None:
        print("\n" + "=" * 80, flush=True)
        print(" [1/2] Inizializzazione Gemma 3 27B su 4 GPU (2, 3, 4, 5 -> 4 GPU totali)...", flush=True)
        print(" Configurazione: Tensor Parallelism (TP=2) x Pipeline Parallelism (PP=4)", flush=True)
        print(" Caricamento pesi e allocazione VRAM in corso...", flush=True)
        print("=" * 80 + "\n", flush=True)

        self.max_new_tokens = max_new_tokens
        self.window_counter = 0

        start_load = time.time()
        
        # TP=2 e PP=2 distribuiscono il carico su 4 GPU
        # distributed_executor_backend="mp" velocizza l'avvio su macchine a singolo nodo
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=4,   # <-- Spalma ogni livello su tutte e 4 le GPU
            pipeline_parallel_size=1, # <-- Rimuove l'effetto "catena di montaggio" sequenziale
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            limit_mm_per_prompt={"image": max_frames},
        )
        load_duration = time.time() - start_load
        print(f"\n [2/2] Modello caricato con successo in {load_duration:.2f}s!\n", flush=True)

        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministico per output JSON
            max_tokens=self.max_new_tokens,
        )

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

            print(
                f"   --> [Finestra #{current_window}] Generazione risposta su vLLM...",
                flush=True,
            )

            outputs = self.llm.chat(
                messages,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            elapsed = time.time() - start_time
            print(
                f"   <-- [Finestra #{current_window}] Completata in {elapsed:.2f}s.",
                flush=True,
            )

            return outputs[0].outputs[0].text.strip()

        finally:
            for image in images:
                image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analisi semantica con Gemma-3-27B in parallelo "
            "su 4 GPU (2, 3, 4, 5)."
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
        default="data/preliminar_analysis/entity/entity_semantic/gemma-27B",
        type=Path,
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=10000)

    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Numero massimo di video; 0 = tutti.",
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
        parser.error("Non sono state trovate cartelle dense_frames.")

    args.output_directory.mkdir(parents=True, exist_ok=True)

    inferencer = Gemma27VLLMInferencer(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        max_frames=args.max_frames,
    )

    rows = []
    total_videos = len(video_directories)

    print(f"\nInizio elaborazione di {total_videos} video...", flush=True)

    for index, video_directory in enumerate(video_directories, start=1):
        
        # --- CONTROLLO FILE GIA' ESISTENTE ---
        file_output = args.output_directory / f"{video_directory.name}_semantic.json"
        if file_output.exists() and not args.overwrite:
            print(f"\n[{index}/{total_videos}] SALTO: File {file_output.name} già presente.", flush=True)
            continue
        # -------------------------------------

        print(
            f"\n========================================================",
            flush=True,
        )
        print(
            f"[{index}/{total_videos}] Inizio analisi video: {video_directory.name}",
            flush=True,
        )
        print(
            f"========================================================",
            flush=True,
        )

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
                f"ERRORE durante {video_directory.name}: "
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

    # Scrive il CSV con i risultati della sessione corrente
    write_csv(
        args.output_directory / "riepilogo_video.csv",
        rows,
    )

    print(f"\nCOMPLETATO TOTALE! Risultati salvati in: {args.output_directory}", flush=True)


if __name__ == "__main__":
    main()