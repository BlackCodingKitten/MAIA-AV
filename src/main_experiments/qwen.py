import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import traceback

import torch
from qwen_omni_utils import process_mm_info
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from common import SYSTEM, arguments, evaluate, extract_audio
from media_utils import (
    close_images,
    ensure_wav,
    load_video_frames,
)


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
N_FRAMES = 32


class Model:
    def __init__(self):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            MODEL_ID
        )

        self.model = (
            Qwen2_5OmniForConditionalGeneration
            .from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
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

                # Il video viene già decodificato da ffmpeg in media_utils.
                # process_mm_info riceve una lista di PIL.Image e non apre l'MP4.
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
                # Audio separato dal video:
                # ffmpeg -> WAV 16 kHz -> audio encoder Qwen.
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

            # L'audio del video è già stato estratto e aggiunto come modalità
            # audio indipendente, quindi non chiediamo al loader di riaprire
            # il file video per estrarne l'audio.
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
                generated = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=use_audio_in_video,
                    max_new_tokens=4,
                    do_sample=False,
                )

            generated = generated[
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
        "qwen",
        Model(),
        a.modes,
        a.limit,
        a.overwrite,
    )