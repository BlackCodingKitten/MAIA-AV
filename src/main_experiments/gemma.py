import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

from common import SYSTEM, arguments, evaluate, extract_audio

MODEL_ID = "google/gemma-3n-E4B-it"


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
        content = []

        if mode == "only_audio":
            content.append({"type": "audio", "audio": str(paths["audio"].resolve())})
        elif mode in ("only_video", "transcript_video"):
            content.append({"type": "video", "video": str(paths["mute"].resolve())})
        elif mode == "video_audio":
            content += [
                {"type": "video", "video": str(paths["video"].resolve())},
                {"type": "audio", "audio": str(extract_audio(paths["video"]).resolve())},
            ]

        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        n = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)[0][n:]
        return self.processor.decode(output, skip_special_tokens=True).strip()


if __name__ == "__main__":
    a = arguments()
    evaluate("gemma", Model(), a.modes, a.limit, a.overwrite)