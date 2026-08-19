from __future__ import annotations

import os

# ==============================================================================
# CONFIGURAZIONE AMBIENTE
#
# DEVE ESSERE ESEGUITA PRIMA DI IMPORTARE:
# - torch
# - transformers
# - vllm
# ==============================================================================


# ------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------

# Se vuoi utilizzare le GPU fisiche 4 e 6:
#
#   GPU fisica 4 -> cuda:0 nel processo
#   GPU fisica 6 -> cuda:1 nel processo
#
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"


# ------------------------------------------------------------------------------
# MULTIPROCESSING vLLM
# ------------------------------------------------------------------------------

# Necessario per evitare fork dopo l'inizializzazione CUDA.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Comunicazione esclusivamente locale.
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
os.environ["VLLM_LOOPBACK_IP"] = "127.0.0.1"


# ------------------------------------------------------------------------------
# DISABILITAZIONE SHARED / SYMMETRIC MEMORY
# ------------------------------------------------------------------------------

# Disabilita PyTorch symmetric-memory all-reduce.
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"

# Disabilita FlashInfer all-reduce.
os.environ["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"

# Disabilita NCCL symmetric memory.
os.environ["VLLM_USE_NCCL_SYMM_MEM"] = "0"


# ------------------------------------------------------------------------------
# FLASHINFER
# ------------------------------------------------------------------------------

# Disabilita il sampler FlashInfer.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


# ------------------------------------------------------------------------------
# NCCL
# ------------------------------------------------------------------------------

# Non utilizzare P2P diretto tra le GPU.
os.environ["NCCL_P2P_DISABLE"] = "1"

# Non utilizzare /dev/shm come trasporto NCCL.
os.environ["NCCL_SHM_DISABLE"] = "1"

# Non utilizzare InfiniBand.
os.environ["NCCL_IB_DISABLE"] = "1"

# Comunicazione NCCL tramite socket locali.
os.environ["NCCL_SOCKET_IFNAME"] = "lo"


# ------------------------------------------------------------------------------
# THREAD CPU
# ------------------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ==============================================================================
# IMPORT
# ==============================================================================

import traceback

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import (
    SYSTEM,
    arguments,
    evaluate,
    extract_audio,
)

from media_utils import (
    AUDIO_SAMPLE_RATE,
    close_images,
    load_audio_waveform,
    load_video_frames,
)


# ==============================================================================
# CONFIGURAZIONE MODELLO
# ==============================================================================

MODEL_ID = "google/gemma-4-12B-it"
MODEL_NAME = "gemma-4-12B"

N_FRAMES = 32


# ==============================================================================
# MODEL
# ==============================================================================


class Model:
    def __init__(self):

        # ----------------------------------------------------------------------
        # PROCESSOR
        # ----------------------------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        # ----------------------------------------------------------------------
        # vLLM
        # ----------------------------------------------------------------------

        self.llm = LLM(
            model=MODEL_ID,

            dtype="bfloat16",

            # --------------------------------------------------------------
            # PARALLELISMO
            # --------------------------------------------------------------

            # Le due GPU visibili sono cuda:0 e cuda:1.
            tensor_parallel_size=2,

            # Multiprocessing locale, senza Ray.
            distributed_executor_backend="mp",

            # --------------------------------------------------------------
            # MEMORIA
            # --------------------------------------------------------------

            gpu_memory_utilization=0.85,

            # 32 frame + prompt + eventuale audio.
            max_model_len=32768,

            # Elaboriamo un solo item alla volta.
            max_num_seqs=1,

            # --------------------------------------------------------------
            # MULTIMODAL INPUT
            # --------------------------------------------------------------

            limit_mm_per_prompt={
                "image": N_FRAMES,
                "audio": 1,
            },

            # Disabilita completamente la cache del processor multimodale.
            # In particolare evita il relativo meccanismo IPC/cache condivisa.
            mm_processor_cache_gb=0,

            # --------------------------------------------------------------
            # COMUNICAZIONE GPU
            # --------------------------------------------------------------

            # Non usa il custom all-reduce di vLLM.
            # Il collective viene quindi gestito attraverso NCCL.
            disable_custom_all_reduce=True,

            # --------------------------------------------------------------
            # ALTRO
            # --------------------------------------------------------------

            trust_remote_code=True,

            # Niente CUDA graphs.
            # È più conservativo e stabile per questo workload.
            enforce_eager=True,
        )

        # ----------------------------------------------------------------------
        # GENERATION
        # ----------------------------------------------------------------------

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=16,
        )

    # ==========================================================================
    # INFERENCE
    # ==========================================================================

    def __call__(
        self,
        mode,
        row,
        prompt,
        paths,
    ):
        images = []
        audio = None

        try:

            # ==================================================================
            # 1. NO INPUT
            #
            # Caption + foil / prompt testuale soltanto.
            #
            # Nessun:
            # - video
            # - frame
            # - audio
            # - trascrizione aggiuntiva
            # ==================================================================

            if mode in ("no_input", "only_transcription"):
                # no_input: solo prompt caption/foil.
                # only_transcription: la trascrizione è già incorporata nel prompt
                # da common.py, quindi non si caricano immagini o audio.
                pass

            # ==================================================================
            # 2. ONLY VIDEO
            #
            # Usa il video MUTO.
            # ==================================================================

            elif mode == "only_video":

                images = load_video_frames(
                    paths["mute"],
                    N_FRAMES,
                )

            # ==================================================================
            # 3. ONLY AUDIO
            #
            # L'audio viene estratto dal video originale.
            #
            # Non utilizziamo paths["audio"], perché nel progetto l'audio
            # proviene dal file video originale.
            # ==================================================================

            elif mode == "only_audio":

                audio_path = extract_audio(
                    paths["video"]
                )

                audio = load_audio_waveform(
                    audio_path,
                    AUDIO_SAMPLE_RATE,
                    max_seconds=30.0,
                )

            # ==================================================================
            # 4. VIDEO + AUDIO
            #
            # Video originale:
            #
            #   visual -> 32 frame
            #   audio  -> waveform estratta dal medesimo MP4
            #
            # ==================================================================

            elif mode == "video_audio":

                images = load_video_frames(
                    paths["video"],
                    N_FRAMES,
                )

                audio_path = extract_audio(
                    paths["video"]
                )

                audio = load_audio_waveform(
                    audio_path,
                    AUDIO_SAMPLE_RATE,
                    max_seconds=30.0,
                )

            # ==================================================================
            # 5. TRANSCRIPT + VIDEO
            #
            # Il video è quello MUTO.
            #
            # La trascrizione NON viene caricata qui:
            # common.py la incorpora già nel prompt specifico della modalità.
            # ==================================================================

            elif mode == "transcript_video":

                images = load_video_frames(
                    paths["mute"],
                    N_FRAMES,
                )

            # ==================================================================
            # MODALITÀ ERRATA
            # ==================================================================

            else:
                raise ValueError(
                    f"Modalità non supportata: {mode}"
                )

            # ==================================================================
            # COSTRUZIONE DEL CONTENT MULTIMODALE
            # ==================================================================

            content = []

            # ------------------------------------------------------------------
            # FRAME VIDEO
            # ------------------------------------------------------------------

            for _ in images:
                content.append(
                    {
                        "type": "image",
                    }
                )

            # ------------------------------------------------------------------
            # AUDIO
            # ------------------------------------------------------------------

            if audio is not None:
                content.append(
                    {
                        "type": "audio",
                    }
                )

            # ------------------------------------------------------------------
            # PROMPT
            #
            # Lo mettiamo dopo gli input multimodali in modo che la domanda
            # faccia riferimento alle evidenze appena fornite.
            # ------------------------------------------------------------------

            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            # ==================================================================
            # CHAT
            # ==================================================================

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]

            # ==================================================================
            # CHAT TEMPLATE GEMMA 4
            # ==================================================================

            formatted_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,

                # Non vogliamo che i 16 token disponibili vengano consumati
                # dal reasoning interno.
                enable_thinking=False,
            )

            # ==================================================================
            # REQUEST vLLM
            # ==================================================================

            request = {
                "prompt": formatted_prompt,
            }

            # ------------------------------------------------------------------
            # DATI MULTIMODALI
            # ------------------------------------------------------------------

            multi_modal_data = {}

            if images:
                multi_modal_data["image"] = images

            if audio is not None:
                multi_modal_data["audio"] = (
                    audio,
                    AUDIO_SAMPLE_RATE,
                )

            # no_input:
            #
            # multi_modal_data == {}
            #
            # quindi la chiave non viene proprio aggiunta alla request.
            if multi_modal_data:
                request["multi_modal_data"] = multi_modal_data

            # ==================================================================
            # GENERAZIONE
            # ==================================================================

            results = self.llm.generate(
                request,
                self.sampling,
                use_tqdm=False,
            )

            if not results:
                raise RuntimeError(
                    "vLLM non ha restituito alcun risultato."
                )

            if not results[0].outputs:
                raise RuntimeError(
                    "vLLM non ha restituito alcun output."
                )

            output = (
                results[0]
                .outputs[0]
                .text
                .strip()
            )

            return output

        # ======================================================================
        # ERROR HANDLING
        # ======================================================================

        except Exception:

            print(
                f"\n"
                f"[ERROR] Gemma 4 12B\n"
                f"mode={mode}\n"
                f"prompt={prompt[:200]!r}\n"
            )

            traceback.print_exc()

            raise

        # ======================================================================
        # CLEANUP
        # ======================================================================

        finally:

            # Chiude i frame PIL / risorse equivalenti create da media_utils.
            #
            # In no_input e only_audio images == [], quindi è innocuo.
            close_images(images)


# ==============================================================================
# MAIN
# ==============================================================================


if __name__ == "__main__":

    args = arguments()

    evaluate(
        MODEL_NAME,
        Model(),
        args.modes,
        args.limit,
        args.overwrite,
    )