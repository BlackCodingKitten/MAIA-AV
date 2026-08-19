from __future__ import annotations

import os

# Deve stare prima di importare vLLM.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,5,7"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
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

import traceback

import numpy as np
from vllm import LLM, SamplingParams

from common import SYSTEM, arguments, evaluate, extract_audio
from media_utils import (
    AUDIO_SAMPLE_RATE,
    close_images,
    ensure_wav,
    load_audio_waveform,
    load_video_frames,
)


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

AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"


def qwen_prompt(text: str, audio: bool = False, video: bool = False) -> str:
    media = (AUDIO_TOKEN if audio else "") + (VIDEO_TOKEN if video else "")

    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{media}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_audio(path):
    wav = ensure_wav(path)

    return (
        load_audio_waveform(
            wav,
            AUDIO_SAMPLE_RATE,
            max_seconds=30.0,
        ),
        AUDIO_SAMPLE_RATE,
    )


class Model:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",

            # 4 GPU visibili: 5,6,1,2
            tensor_parallel_size=4,

            gpu_memory_utilization=0.85,
            max_model_len=32768,
            max_num_seqs=1,

            # Un singolo video e/o audio per richiesta.
            limit_mm_per_prompt={
                "video": 1,
                "audio": 1,
            },

            # La configurazione che avevamo già stabilizzato.
            disable_custom_all_reduce=True,
            enforce_eager=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=4,
        )

    def __call__(self, mode, row, prompt, paths):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Modalità non supportata: {mode}")

        frames = []

        try:
            use_video = mode in {
                "only_video",
                "video_audio",
                "transcript_video",
            }

            use_audio = mode in {
                "only_audio",
                "video_audio",
            }

            request = {
                "prompt": qwen_prompt(
                    prompt,
                    audio=use_audio,
                    video=use_video,
                )
            }

            multimodal = {}

            if use_video:
                video_path = (
                    paths["video"]
                    if mode == "video_audio"
                    else paths["mute"]
                )

                frames = load_video_frames(
                    video_path,
                    N_FRAMES,
                )

                # vLLM video input: sequenza di frame NumPy RGB.
                multimodal["video"] = [
                    np.asarray(frame.convert("RGB"))
                    for frame in frames
                ]

            if use_audio:
                audio_path = (
                    paths["audio"]
                    if mode == "only_audio"
                    else extract_audio(paths["video"])
                )

                multimodal["audio"] = load_audio(audio_path)

            if multimodal:
                request["multi_modal_data"] = multimodal

            result = self.llm.generate(
                request,
                sampling_params=self.sampling,
                use_tqdm=False,
            )

            if not result or not result[0].outputs:
                raise RuntimeError("vLLM non ha restituito output.")

            return result[0].outputs[0].text.strip()

        except Exception:
            print(
                f"\n[ERROR] Qwen3-Omni 30B"
                f"\nmode={mode}"
                f"\nprompt={prompt[:200]!r}\n",
                flush=True,
            )

            traceback.print_exc()
            raise

        finally:
            close_images(frames)


if __name__ == "__main__":
    a = arguments()

    evaluate(
        MODEL_NAME,
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
    )