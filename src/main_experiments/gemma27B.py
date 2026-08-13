import os
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

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import SYSTEM, arguments, evaluate

MODEL_ID = "google/gemma-3-27b-it"
N_FRAMES = 32


def frames(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ids = [round(i * (n - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
    images = []

    for i in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    if not images:
        raise RuntimeError(f"Nessun frame letto da {path}")
    return images


class Model:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=4,
            gpu_memory_utilization=.85,
            max_model_len=32768,
            limit_mm_per_prompt={"image": N_FRAMES},
        )
        self.sampling = SamplingParams(temperature=0, max_tokens=4)

    def __call__(self, mode, row, prompt, paths):
        images = [] if mode == "no_input" else frames(paths["mute"])
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        request = {"prompt": text}
        if images:
            request["multi_modal_data"] = {"image": images}

        output = self.llm.generate(request, self.sampling, use_tqdm=False)[0].outputs[0].text.strip()
        for image in images:
            image.close()
        return output


if __name__ == "__main__":
    a = arguments()
    evaluate(
        "gemma-27B",
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
        unsupported=("only_audio", "video_audio"),
    )