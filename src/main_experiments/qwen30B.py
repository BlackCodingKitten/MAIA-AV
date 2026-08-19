from __future__ import annotations

import os

# ==============================================================================
# CONFIGURAZIONE AMBIENTE
#
# DEVE ESSERE ESEGUITA PRIMA DI IMPORTARE:
# - torch
# - transformers
# - qwen_omni_utils
# ==============================================================================


# ------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------

# GPU fisiche utilizzate:
#
#   GPU fisica 5 -> cuda:0
#   GPU fisica 6 -> cuda:1
#   GPU fisica 1 -> cuda:2
#   GPU fisica 2 -> cuda:3
#
os.environ["CUDA_VISIBLE_DEVICES"] = "5,1,2,3"


# ------------------------------------------------------------------------------
# CONFIGURAZIONE GIÀ UTILIZZATA PER QWEN
# ------------------------------------------------------------------------------

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

import traceback

import torch

from qwen_omni_utils import process_mm_info

from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from common import (
    SYSTEM,
    arguments,
    evaluate,
    extract_audio,
)

from media_utils import (
    close_images,
    ensure_wav,
    load_video_frames,
)


# ==============================================================================
# CONFIGURAZIONE MODELLO
# ==============================================================================

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen-30B"

N_FRAMES = 32


SUPPORTED_MODES = {
    "no_input",
    "only_transcription",
    "only_video",
    "only_audio",
    "video_audio",
    "transcript_video",
}


# ==============================================================================
# MODEL
# ==============================================================================


class Model:

    def __init__(self):

        # ======================================================================
        # PROCESSOR
        # ======================================================================

        self.processor = (
            Qwen3OmniMoeProcessor
            .from_pretrained(
                MODEL_ID,
            )
        )

        # ======================================================================
        # MODELLO
        # ======================================================================

        print(
            "\n"
            "============================================================\n"
            "Caricamento Qwen3-Omni-30B su 4 GPU\n"
            "============================================================\n"
            "GPU fisiche: 5, 6, 1, 2\n"
            "GPU logiche: cuda:0, cuda:1, cuda:2, cuda:3\n"
            "Device map:  balanced\n"
            "============================================================",
            flush=True,
        )

        self.model = (
            Qwen3OmniMoeForConditionalGeneration
            .from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                device_map="balanced",
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            .eval()
        )

        # ======================================================================
        # TALKER
        # ======================================================================
        #
        # L'esperimento richiede esclusivamente output testuale.
        #
        # Disabilitare il Talker evita di mantenere in memoria la parte
        # necessaria alla generazione vocale.
        # ======================================================================

        self.model.disable_talker()

        # ======================================================================
        # DEBUG DEVICE MAP
        # ======================================================================

        if hasattr(
            self.model,
            "hf_device_map",
        ):
            print(
                "\nDevice map Qwen3-Omni:",
                flush=True,
            )

            used_devices = set()

            for module_name, device in (
                self.model.hf_device_map.items()
            ):

                device_str = str(device)

                if device_str.startswith("cuda:"):
                    used_devices.add(device_str)

                elif isinstance(device, int):
                    used_devices.add(
                        f"cuda:{device}"
                    )

            print(
                f"GPU utilizzate dal modello: "
                f"{sorted(used_devices)}",
                flush=True,
            )

            if len(used_devices) < 4:
                print(
                    "[WARN] Il device map non sta utilizzando "
                    "tutte e quattro le GPU.",
                    flush=True,
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

        frames = []

        try:

            # ==================================================================
            # CONTROLLO MODALITÀ
            # ==================================================================

            if mode not in SUPPORTED_MODES:

                raise ValueError(
                    f"Modalità non supportata: {mode}"
                )

            # ==================================================================
            # COSTRUZIONE CONTENT
            # ==================================================================

            content = []

            # ==================================================================
            # 1. NO INPUT
            #
            # Solo prompt.
            #
            # Nessun:
            # - video
            # - audio
            # - trascrizione aggiuntiva
            # ==================================================================

            if mode == "no_input":

                pass

            # ==================================================================
            # 2. ONLY TRANSCRIPTION
            #
            # Modalità completamente text-only.
            #
            # La trascrizione viene già inserita nel prompt da common.py.
            #
            # Non vengono caricati:
            # - video
            # - frame
            # - audio
            # ==================================================================

            elif mode == "only_transcription":

                pass

            # ==================================================================
            # 3. ONLY VIDEO
            #
            # Utilizza esclusivamente il video muto.
            # ==================================================================

            elif mode == "only_video":

                frames = load_video_frames(
                    paths["mute"],
                    N_FRAMES,
                )

                content.append(
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 1.0,
                    }
                )

            # ==================================================================
            # 4. ONLY AUDIO
            #
            # Utilizza esclusivamente il file audio.
            # ==================================================================

            elif mode == "only_audio":

                audio_path = ensure_wav(
                    paths["audio"]
                )

                content.append(
                    {
                        "type": "audio",
                        "audio": str(
                            audio_path.resolve()
                        ),
                    }
                )

            # ==================================================================
            # 5. VIDEO + AUDIO
            #
            # Il video originale viene utilizzato per:
            #
            #   1. estrarre i 32 frame
            #   2. estrarre separatamente la traccia audio
            #
            # Non chiediamo a process_mm_info di estrarre automaticamente
            # l'audio dal video.
            # ==================================================================

            elif mode == "video_audio":

                frames = load_video_frames(
                    paths["video"],
                    N_FRAMES,
                )

                content.append(
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 1.0,
                    }
                )

                audio_path = ensure_wav(
                    extract_audio(
                        paths["video"]
                    )
                )

                content.append(
                    {
                        "type": "audio",
                        "audio": str(
                            audio_path.resolve()
                        ),
                    }
                )

            # ==================================================================
            # 6. TRANSCRIPTION + VIDEO
            #
            # Video:
            #   usa il file muto.
            #
            # Trascrizione:
            #   è già contenuta nel prompt preparato da common.py.
            # ==================================================================

            elif mode == "transcript_video":

                frames = load_video_frames(
                    paths["mute"],
                    N_FRAMES,
                )

                content.append(
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 1.0,
                    }
                )

            # ==================================================================
            # PROMPT TESTUALE
            #
            # Viene aggiunto sempre.
            #
            # Per only_transcription e transcript_video contiene già
            # la trascrizione.
            # ==================================================================

            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            # ==================================================================
            # CONVERSATION
            # ==================================================================

            conversation = [
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
            # CHAT TEMPLATE
            # ==================================================================

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            # ==================================================================
            # AUDIO NEL VIDEO
            #
            # Deve restare False.
            #
            # only_video:
            #   video muto
            #
            # transcript_video:
            #   video muto
            #
            # video_audio:
            #   audio fornito separatamente
            # ==================================================================

            use_audio_in_video = False

            # ==================================================================
            # PROCESSAMENTO MULTIMODALE
            # ==================================================================

            audios, images, videos = process_mm_info(
                conversation,
                use_audio_in_video=use_audio_in_video,
            )

            # ==================================================================
            # PROCESSOR
            # ==================================================================

            inputs = self.processor(
                text=text,

                audio=audios,
                images=images,
                videos=videos,

                return_tensors="pt",

                padding=True,

                use_audio_in_video=use_audio_in_video,
            )

            # ==================================================================
            # INPUT DEVICE
            #
            # Il modello è sharded su quattro GPU.
            #
            # Gli input iniziali vengono messi sul device del modello.
            # Accelerate gestisce successivamente il passaggio tra i device
            # assegnati ai diversi moduli.
            # ==================================================================

            inputs = (
                inputs
                .to(self.model.device)
                .to(self.model.dtype)
            )

            # ==================================================================
            # GENERAZIONE
            # ==================================================================

            with torch.inference_mode():

                text_ids, _ = self.model.generate(
                    **inputs,

                    # ----------------------------------------------------------
                    # SOLO OUTPUT TESTUALE
                    # ----------------------------------------------------------

                    return_audio=False,

                    # ----------------------------------------------------------
                    # THINKER
                    # ----------------------------------------------------------

                    thinker_return_dict_in_generate=True,

                    thinker_max_new_tokens=4,

                    thinker_do_sample=False,

                    # ----------------------------------------------------------
                    # AUDIO
                    # ----------------------------------------------------------

                    use_audio_in_video=use_audio_in_video,
                )

            # ==================================================================
            # RIMOZIONE DEL PROMPT DALLA GENERAZIONE
            # ==================================================================

            generated = text_ids.sequences[
                :,
                inputs["input_ids"].shape[1]:,
            ]

            # ==================================================================
            # DECODIFICA
            # ==================================================================

            output = self.processor.batch_decode(
                generated,

                skip_special_tokens=True,

                clean_up_tokenization_spaces=False,
            )[0].strip()

            return output

        # ======================================================================
        # ERROR HANDLING
        # ======================================================================

        except Exception:

            print(
                "\n"
                "============================================================\n"
                "ERRORE QWEN3-OMNI 30B\n"
                "============================================================\n"
                f"mode: {mode}\n"
                f"prompt: {prompt[:200]!r}\n"
                "============================================================",
                flush=True,
            )

            traceback.print_exc()

            raise

        # ======================================================================
        # CLEANUP FRAME
        # ======================================================================

        finally:

            close_images(
                frames
            )


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