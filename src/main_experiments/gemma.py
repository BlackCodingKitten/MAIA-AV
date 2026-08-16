import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import traceback

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

from common import SYSTEM, arguments, evaluate, extract_audio
from media_utils import (
    AUDIO_SAMPLE_RATE,
    close_images,
    load_audio_waveform,
    load_video_frames,
)


MODEL_ID = "google/gemma-3n-E4B-it"
NUM_FRAMES = 32


class Model:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()

    def __call__(self, mode, row, prompt, paths):
        images = []
        audio = None

        try:
            if mode in ("only_video", "transcript_video"):
                images = load_video_frames(
                    paths["mute"],
                    NUM_FRAMES,
                )

            elif mode == "only_audio":
                audio = load_audio_waveform(
                    paths["audio"],
                    AUDIO_SAMPLE_RATE,
                )

            elif mode == "video_audio":
                images = load_video_frames(
                    paths["video"],
                    NUM_FRAMES,
                )
                audio = load_audio_waveform(
                    extract_audio(paths["video"]),
                    AUDIO_SAMPLE_RATE,
                )

            # no_input e only_transcription rimangono text-only.
            content = []

            content.extend(
                {"type": "image"}
                for _ in images
            )

            if audio is not None:
                content.append({"type": "audio"})

            content.append({
                "type": "text",
                "text": prompt,
            })

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

            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            processor_kwargs = {
                "text": text,
                "images": images if images else None,
                "audio": audio,
                "return_tensors": "pt",
            }

            if audio is not None:
                processor_kwargs["sampling_rate"] = AUDIO_SAMPLE_RATE

            inputs = self.processor(
                **processor_kwargs,
            ).to(self.model.device)

            n = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                )[0][n:]

            return self.processor.decode(
                output,
                skip_special_tokens=True,
            ).strip()

        except Exception:
            traceback.print_exc()
            raise

        finally:
            close_images(images)


if __name__ == "__main__":
    a = arguments()

    evaluate(
        "gemma",
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
    )
