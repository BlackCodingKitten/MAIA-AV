from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,1,7"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_FLASHINFER_SAMPLING"] = "0"
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

from functools import lru_cache
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from common import SYSTEM, arguments, evaluate
from media_utils import (
    AUDIO_SAMPLE_RATE,
    close_images,
    load_audio_waveform,
    load_video_frames,
)


MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen-30B"
N_FRAMES = 32

VIDEO_MODES = {
    "only_video",
    "video_audio",
    "transcript_video",
}

AUDIO_MODES = {
    "only_audio",
    "video_audio",
}

SUPPORTED_MODES = {
    "no_input",
    "only_audio",
    "only_video",
    "only_transcription",
    "video_audio",
    "transcript_video",
}

AUDIO_TOKEN = "<|audio_start|><|audio_pad|><|audio_end|>"
VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"


def format_prompt(
    text: str,
    use_audio: bool,
    use_video: bool,
) -> str:
    media = ""

    if use_audio:
        media += AUDIO_TOKEN

    if use_video:
        media += VIDEO_TOKEN

    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{media}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@lru_cache(maxsize=1)
def load_video(path: Path) -> np.ndarray:
    frames = load_video_frames(path, N_FRAMES)

    try:
        video = np.stack(
            [
                np.asarray(frame, dtype=np.uint8)
                for frame in frames
            ],
            axis=0,
        )
    finally:
        close_images(frames)

    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            f"Formato video non valido per {path}: "
            f"shape={video.shape}"
        )

    return video


@lru_cache(maxsize=1)
def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio = load_audio_waveform(
        path,
        sample_rate=AUDIO_SAMPLE_RATE,
        max_seconds=30.0,
    )

    return audio, AUDIO_SAMPLE_RATE


class Model:
    def __init__(self):
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            max_num_seqs=1,
            limit_mm_per_prompt={
                "video": 1,
                "audio": 1,
            },
            disable_custom_all_reduce=True,
            enforce_eager=True,
            trust_remote_code=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=4,
        )

    def __call__(
        self,
        mode,
        row,
        prompt,
        paths,
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Modalità non supportata: {mode}"
            )

        use_video = mode in VIDEO_MODES
        use_audio = mode in AUDIO_MODES

        request = {
            "prompt": format_prompt(
                prompt,
                use_audio=use_audio,
                use_video=use_video,
            )
        }

        multimodal = {}

        if use_video:
            video_path = (
                paths["video"]
                if mode == "video_audio"
                else paths["mute"]
            )

            multimodal["video"] = load_video(
                video_path
            )

        if use_audio:
            audio_path = (
                paths["audio"]
                if mode == "only_audio"
                else paths["video"]
            )

            multimodal["audio"] = load_audio(
                audio_path
            )

        if multimodal:
            request["multi_modal_data"] = multimodal

        outputs = self.llm.generate(
            request,
            sampling_params=self.sampling,
            use_tqdm=False,
        )

        if not outputs or not outputs[0].outputs:
            raise RuntimeError(
                "vLLM non ha restituito alcun output."
            )

        return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    args = arguments()

    evaluate(
        MODEL_NAME,
        Model(),
        args.modes,
        args.limit,
        args.overwrite,
    )