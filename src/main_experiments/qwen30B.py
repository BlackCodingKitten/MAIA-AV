import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from common import SYSTEM, arguments, evaluate

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


class Model:
    def __init__(self):
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        self.model.disable_talker()

    def __call__(self, mode, row, prompt, paths):
        content, use_audio_in_video = [], mode == "video_audio"

        if mode == "only_audio":
            content.append({"type": "audio", "audio": str(paths["audio"].resolve())})
        elif mode in ("only_video", "transcript_video"):
            content.append({"type": "video", "video": str(paths["mute"].resolve())})
        elif mode == "only_transcription":
            pass
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
            text_ids, _ = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=use_audio_in_video,
                max_new_tokens=4,
                do_sample=False,
            )

        generated = text_ids.sequences[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


if __name__ == "__main__":
    a = arguments()
    evaluate("qwen-30B", Model(), a.modes, a.limit, a.overwrite)