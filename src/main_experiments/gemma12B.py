import os

# 1. Modificato: Esponiamo solo 2 GPU (ad esempio la 5 e la 6)
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import traceback

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import SYSTEM, arguments, evaluate, extract_audio
from media_utils import (
    AUDIO_SAMPLE_RATE,
    close_images,
    load_audio_waveform,
    load_video_frames,
)


MODEL_ID = "google/gemma-4-12B-it"
MODEL_NAME = "gemma-4-12B"
N_FRAMES = 32


class Model:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            # 2. Modificato: Impostiamo la parallelizzazione su 2 GPU
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            max_num_seqs=1,
            limit_mm_per_prompt={
                "image": N_FRAMES,
                "audio": 1,
            },
            trust_remote_code=True,
            enforce_eager=True,
        )

        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=16,
        )

    def __call__(self, mode, row, prompt, paths):
        images = []
        audio = None

        try:
            if mode in ("only_video", "transcript_video"):
                images = load_video_frames(
                    paths["mute"],
                    N_FRAMES,
                )

            elif mode == "only_audio":
                audio = load_audio_waveform(
                    paths["audio"],
                    AUDIO_SAMPLE_RATE,
                    max_seconds=30.0,
                )

            elif mode == "video_audio":
                images = load_video_frames(
                    paths["video"],
                    N_FRAMES,
                )

                audio = load_audio_waveform(
                    extract_audio(paths["video"]),
                    AUDIO_SAMPLE_RATE,
                    max_seconds=30.0,
                )

            # Per il video usiamo i frame come immagini già decodificate.
            # Così Gemma 4 non apre mai direttamente l'MP4.
            content = []

            content.extend(
                {"type": "image"}
                for _ in images
            )

            content.append({
                "type": "text",
                "text": prompt,
            })

            if audio is not None:
                content.append({"type": "audio"})

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

            formatted_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            request = {
                "prompt": formatted_prompt,
            }

            multi_modal_data = {}

            if images:
                multi_modal_data["image"] = images

            if audio is not None:
                multi_modal_data["audio"] = (
                    audio,
                    AUDIO_SAMPLE_RATE,
                )

            if multi_modal_data:
                request["multi_modal_data"] = multi_modal_data

            output = self.llm.generate(
                request,
                self.sampling,
                use_tqdm=False,
            )[0].outputs[0].text.strip()

            return output

        except Exception:
            traceback.print_exc()
            raise

        finally:
            close_images(images)


if __name__ == "__main__":
    a = arguments()

    evaluate(
        MODEL_NAME,
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
    )