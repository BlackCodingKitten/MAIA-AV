import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import traceback

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from common import SYSTEM, arguments, evaluate, extract_audio
from media_utils import (
    close_images,
    ensure_wav,
    load_video_frames,
)


MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
N_FRAMES = 32


class Model:
    def __init__(self):
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_ID
        )

        self.model = (
            Qwen3OmniMoeForConditionalGeneration
            .from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            .eval()
        )

        self.model.disable_talker()

    def __call__(self, mode, row, prompt, paths):
        frames = []

        try:
            content = []

            if mode in (
                "only_video",
                "transcript_video",
                "video_audio",
            ):
                video_path = (
                    paths["video"]
                    if mode == "video_audio"
                    else paths["mute"]
                )

                # Frame già estratti via ffmpeg: niente TorchCodec/cv2.
                frames = load_video_frames(
                    video_path,
                    N_FRAMES,
                )

                content.append({
                    "type": "video",
                    "video": frames,
                    "fps": 1.0,
                })

            if mode == "only_audio":
                audio_path = ensure_wav(
                    paths["audio"]
                )

                content.append({
                    "type": "audio",
                    "audio": str(audio_path.resolve()),
                })

            elif mode == "video_audio":
                audio_path = ensure_wav(
                    extract_audio(paths["video"])
                )

                content.append({
                    "type": "audio",
                    "audio": str(audio_path.resolve()),
                })

            # no_input e only_transcription sono text-only.
            content.append({
                "type": "text",
                "text": prompt,
            })

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

            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            use_audio_in_video = False

            audios, images, videos = process_mm_info(
                conversation,
                use_audio_in_video=use_audio_in_video,
            )

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

            generated = text_ids.sequences[
                :,
                inputs["input_ids"].shape[1]:,
            ]

            return self.processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

        except Exception:
            traceback.print_exc()
            raise

        finally:
            close_images(frames)


if __name__ == "__main__":
    a = arguments()

    evaluate(
        "qwen-30B",
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
    )