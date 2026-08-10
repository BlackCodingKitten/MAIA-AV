import os
import io
import base64
from typing import Optional, List, Union
from PIL import Image
import torch
from vllm import LLM, SamplingParams
from html import escape
from pathlib import Path
import html as _h

# ---------------------------
# Utilities
# ---------------------------

def format_image(img: Union[Image.Image, os.PathLike]) -> str:
    """
    Converts a PIL Image or image path to a base64-encoded PNG data URI.
    """
    if isinstance(img, os.PathLike):
        img = Image.open(img)
    img = img.convert("RGB")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def _get_vllm_engine(
    model_id: str,
    dtype: torch.dtype,
    allowed_local_media_path: os.PathLike = "",
    hf_cache_dir: Optional[os.PathLike] = os.environ.get("HF_HOME", None),
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.7,
    enforce_eager: bool = False,
    **kwargs
) -> LLM:
    """
    Helper to create a vLLM LLM engine with model-specific params.
    Generation params adapted from vLLM docs:
    https://docs.vllm.ai/en/latest/examples/offline_inference/vision_language.html
    """
    model_id_lower = model_id.lower()
    common_kwargs = dict(
        model=model_id,
        allowed_local_media_path=allowed_local_media_path,
        dtype=dtype,
        download_dir=(str(hf_cache_dir) + "/hub") if hf_cache_dir else None,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )

    # Model-specific overrides
    if "gemma" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=65536,
            max_num_seqs=2,
            mm_processor_kwargs={"do_pan_and_scan": False},
        )
    elif "qwen" in model_id_lower and "2.5" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=65536,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
        )
    elif "llava" in model_id_lower:
        specific_kwargs = dict(max_model_len=4096)
    elif "mistral-small" in model_id_lower:
        specific_kwargs = dict(
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=65536,
            max_num_seqs=2,
            disable_mm_preprocessor_cache=True,
        )
    elif "aya-vision" in model_id_lower:
        specific_kwargs = dict(
            max_model_len=65536,
            max_num_seqs=2,
            mm_processor_kwargs={"crop_to_patches": True},
        )
    elif "molmo" in model_id_lower:
        specific_kwargs = dict(trust_remote_code=True)
    elif "internvl" in model_id_lower:
        specific_kwargs = dict(
            trust_remote_code=True,
            max_model_len=65536,
            #mm_processor_kwargs={"max_dynamic_patch": 4},
            #limit_mm_per_prompt={"image": 32}
        )
    elif "glm-4.1v" in model_id_lower:
        specific_kwargs = dict(
            trust_remote_code=True,
            max_num_seqs=2,
            max_num_batched_tokens=8192
        )
    else:
        raise NotImplementedError(f"{model_id} not implemented")

    return LLM(**common_kwargs, **specific_kwargs)


# ---------------------------
# Main class
# ---------------------------

class VllmModel:
    """
    Vision-Language inference wrapper around vLLM.
    """

    def __init__(
        self,
        model_id: str,
        img_tag: str = "{image}",  # kept for backward-compat, unused now
        hf_cache_dir: Optional[os.PathLike] = os.environ.get("HF_HOME", None),
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gpu_memory_utilization: float = 0.7,
        allowed_local_media_path: os.PathLike = "",
        verbose: bool = False,
        enforce_eager: bool = False,
        **kwargs
    ):
        os.environ["VLLM_LOGGING_LEVEL"] = "INFO" if verbose else "ERROR"

        self.model_id = model_id
        self.img_tag = img_tag
        self.seed = seed
        self.verbose = verbose
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Determine tensor parallel size
        tensor_parallel_size = 1
        if self.device.type == "cuda":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpu_ids = [g for g in visible.split(",") if g != ""] if visible else list(range(torch.cuda.device_count()))
            print(f"vLLM will use visible GPUs: {gpu_ids}")
        tensor_parallel_size = max(1, len(gpu_ids))

        self.dtype = dtype or (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)

        if self.verbose:
            print(f"Huggingface cache dir: {hf_cache_dir}")
        print(f"Loading model {self.model_id}")

        self.model = _get_vllm_engine(
            model_id=self.model_id,
            allowed_local_media_path=allowed_local_media_path,
            gpu_memory_utilization=gpu_memory_utilization,
            hf_cache_dir=hf_cache_dir,
            dtype=self.dtype,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self.model.get_tokenizer()
        print(f"Model {self.model_id} loaded on device {self.device} with {self.dtype} precision.")

        # Default sampling params (override per-call via kwargs)
        self.sampling_params = SamplingParams(
            seed=self.seed,
            max_tokens=kwargs.pop("max_tokens", 512),
            min_tokens=kwargs.pop("min_tokens", 1),
            n=kwargs.pop("num_return_sequences", 1),
            best_of=kwargs.pop("num_return_sequences", 1),
            temperature=kwargs.pop("temperature", 0.8),
            top_p=kwargs.pop("top_p", 0.95),
            top_k=kwargs.pop("top_k", 50),
        )

    def __del__(self):
        # Clean up vLLM engine
        try:
            if hasattr(self, "model") and hasattr(self.model, "llm_engine"):
                self.model.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        # Clean up torch distributed if used
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                import torch.distributed as dist
                dist.destroy_process_group()
        except Exception:
            pass

    # ---------------------------
    # New: user-only conversation builder
    # ---------------------------
    def _build_conversation(
        self,
        prompt: str,
        image_list: Optional[List[Union[Image.Image, os.PathLike]]] = None
    ) -> List[dict]:
        """
        Builds a single-turn conversation with only a USER message.
        Images (if any) are attached before the text as 'image_url' parts (base64 data URIs).
        """
        content_parts = []
        if image_list:
            for img in image_list:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": format_image(img)}
                })
        # Always append the prompt as text
        content_parts.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content_parts}]
    
    def save_conversation_to_html(self, conversations, continuations, html_path) -> None:

        p = Path(html_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        out = [
            '<!doctype html>',
            '<html lang="en"><head><meta charset="utf-8"><title>Batch chat</title>',
            '<style>body{font-family:system-ui;max-width:900px;margin:24px auto;padding:0 8px}'
            '.c{border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}'
            'img{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px;margin:6px 0}'
            '.txt{white-space:pre-wrap;word-break:break-word;margin:6px 0}</style></head><body>'
        ]

        for i, (conv, gen) in enumerate(zip(conversations, continuations)):
            parts = conv[0].get("content", []) if conv else []
            out.append(f'<div class="c"><h3># {i}</h3><div><strong>USER</strong></div>')
            for part in parts:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    out.append(f'<img src="{_h.escape(url, quote=True)}">')
                elif part.get("type") == "text":
                    out.append(f'<div class="txt">{_h.escape(part.get("text",""))}</div>')
            out.append('<div><strong>MODEL</strong></div>')
            out.append(f'<div class="txt">{_h.escape(gen or "")}</div></div>')

        out.append('</body></html>')
        p.write_text("\n".join(out), encoding="utf-8")


    # ---------------------------
    # Public API
    # ---------------------------
    def generate_continuation(
        self,
        prompts: Union[str, List[str]],
        images: Optional[List[List[Union[Image.Image, os.PathLike]]]] = None,
        iteration: int = 0, 
        experiment_dir: Optional[os.PathLike] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate continuations for one or more plain-text prompts, optionally with images.

        Args:
            prompts: str or list[str] — treated as questions/plain text.
            images: Optional[list[list[Image or path]]] — parallel to prompts.
                    Each inner list contains zero or more images for that prompt.
            iteration: int — current batch iteration (for logging purposes).

        Returns:
            list[str]: one generated continuation per prompt.
        """
        # Normalize prompts to list[str]
        if isinstance(prompts, str):
            prompts_list = [prompts]
        else:
            prompts_list = prompts

        # Validate / normalize images alignment
        if images is not None and len(images) != len(prompts_list):
            raise ValueError(
                f"'images' length ({len(images)}) must match number of prompts ({len(prompts_list)})."
            )

        # Build batch of conversations (one per prompt)
        all_messages = []
        for i, p in enumerate(prompts_list):
            imgs_for_p = images[i] if images is not None else None
            conv = self._build_conversation(prompt=p, image_list=imgs_for_p)
            all_messages.append(conv)

        # Stop at EOS if tokenizer provides one
        stop_ids = None
        try:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                stop_ids = [eos_id]
        except Exception:
            pass

        # Per-call sampling params (override defaults)
        sampling_params = SamplingParams(
            seed=kwargs.get("seed", self.sampling_params.seed),
            max_tokens=kwargs.get("max_tokens", self.sampling_params.max_tokens),
            temperature=kwargs.get("temperature", self.sampling_params.temperature),
            top_p=kwargs.get("top_p", self.sampling_params.top_p),
            top_k=kwargs.get("top_k", self.sampling_params.top_k),
            n=kwargs.get("num_return_sequences", self.sampling_params.n),
            best_of=kwargs.get("num_return_sequences", self.sampling_params.best_of),
            stop_token_ids=stop_ids,
        )

        continuations: List[str] = []
        try:
            outputs = self.model.chat(
                all_messages,
                sampling_params=sampling_params,
                use_tqdm=self.verbose
            )
        except Exception as e:
            print(f"IT: {iteration}. Error during model generation: {e}")
            continuations = ["<ERROR>"] * len(prompts_list)
            self.save_conversation_to_html(all_messages, continuations, os.path.join(experiment_dir, f"iter_{iteration}_error.html"))
            # Return empty continuations on error
            return continuations

        for out in outputs:
            # take first sequence for each prompt
            continuations.append(out.outputs[0].text)

        if iteration % 100 == 0:
            self.save_conversation_to_html(all_messages, continuations, os.path.join(experiment_dir, f"iter_{iteration}.html"))

        return continuations

if __name__ == "__main__":
    pass
