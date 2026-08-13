import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from common import SYSTEM, arguments, evaluate

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"


class Model:
    def __init__(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        self.model.disable_talker()

    def __call__(self, mode, row, prompt, paths):
        content, use_audio_in_video = [], mode == "video_audio"

        if mode == "only_audio":
            content.append({"type": "audio", "audio": str(paths["audio"].resolve())})
        elif mode in ("only_video", "transcript_video"):
            content.append({"type": "video", "video": str(paths["mute"].resolve())})
        elif mode == "video_audio":
            content.append({"type": "video", "video": str(paths["video"].resolve())})

        content.append({"type": "text", "text": prompt})
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        ).to(self.model.device).to(self.model.dtype)

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                return_audio=False,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=4,
                do_sample=False,
            )

        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


if __name__ == "__main__":
    a = arguments()
    evaluate("qwen", Model(), a.modes, a.limit, a.overwrite)