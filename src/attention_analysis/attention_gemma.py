from __future__ import annotations

# ============================================================
# IMPORTANT:
# CUDA_VISIBLE_DEVICES must be defined before importing torch.
#
# Physical GPU -> logical CUDA mapping:
#
#   GPU 5 -> cuda:0
#
# Only physical GPU 5 is visible to PyTorch.
# ============================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import (
    AttentionInterface,
    AutoModelForMultimodalLM,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Gemma3nForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward,
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

N_FRAMES = 32
MAX_NEW_TOKENS = 64

VIDEOS_DIR = Path("data/input/video")
TRANSCRIPTIONS_FILE = Path("data/input/transcription/transcription.csv")
QUESTIONS_FILE = Path("data/vsv/question_classification.csv")
OUTPUT_DIR = Path("data/attention_analysis/gemma-3n-E4B")


# Exact sequence requested.
#
# There are 15 positions but 13 unique videos because
# video 75 and video 16 occur twice.
VIDEO_IDS = [
    91, 42, 16, 75, 17,
    37, 49, 11, 50, 35,
    1, 75, 70, 4, 16,
]


MODELS = {
    'gemma-3n-E4B': {
        "model_id": 'google/gemma-3n-E4B-it',
        "family": 'gemma3n',
        "supports_audio": True,
    },
}

CONDITIONS = [
    "audio_video",
    "transcription_video",
]


# Only physical GPU 5 is visible.
# Inside the process it is therefore cuda:0.
MAX_MEMORY = {
    0: "40GiB",
}


# ============================================================
# PROMPT
# ============================================================

QUESTION_PROMPT = """Rispondi alla seguente domanda sulla base delle informazioni disponibili.

Domanda:
{question}

Risposta:"""


TRANSCRIPTION_PROMPT = """Trascrizione del contenuto audio:
{transcription}

Rispondi alla seguente domanda sulla base delle informazioni disponibili.

Domanda:
{question}

Risposta:"""


# ============================================================
# ATTENTION RECORDER
# ============================================================

class AttentionRecorder:
    """
    Records only the attention produced by the final query
    position of each attention call.

    This avoids storing the complete [seq_len x seq_len]
    attention matrices.

    For generation:
        prefill:
            last prompt query -> first generated token

        next calls:
            q_len = 1
            one attention vector for each generated token
    """

    def __init__(self) -> None:
        self.active = False

        self.base_length = 0

        self.modality_masks: dict[str, torch.Tensor] = {}

        self.rows: list[dict[str, Any]] = []

    def start(
        self,
        base_length: int,
        modality_masks: dict[str, torch.Tensor],
    ) -> None:

        self.active = True

        self.base_length = base_length

        self.modality_masks = modality_masks

        self.rows = []

    def stop(self) -> list[dict[str, Any]]:

        self.active = False

        rows = self.rows

        self.rows = []

        self.modality_masks = {}

        return rows

    def record(
        self,
        module: torch.nn.Module,
        probabilities: torch.Tensor,
        key_length: int,
    ) -> None:

        if not self.active:
            return

        module_name = getattr(
            module,
            "_maia_module_name",
            module.__class__.__name__,
        )

        module_name_lower = module_name.lower()

        # We only want the language decoder.
        #
        # Do not collect internal attention from vision
        # or audio encoders.
        excluded = (
            "vision",
            "audio",
            "encoder",
            "projector",
            "perceiver",
        )

        if any(
            name in module_name_lower
            for name in excluded
        ):
            return

        # Decoder KV length must contain the complete
        # multimodal input sequence.
        if key_length < self.base_length:
            return

        # probabilities:
        #
        # [batch, num_heads, 1, key_length]

        if probabilities.ndim != 4:
            return

        probabilities = probabilities[
            0, :, 0, :
        ]

        num_heads = probabilities.shape[0]

        # step 0:
        # attention responsible for the first generated token.
        #
        # step 1:
        # attention responsible for the second token.
        #
        generation_step = max(
            0,
            key_length - self.base_length,
        )

        modality_attention: dict[
            str,
            torch.Tensor,
        ] = {}

        for modality, base_mask in (
            self.modality_masks.items()
        ):

            full_mask = torch.zeros(
                key_length,
                dtype=torch.bool,
                device=probabilities.device,
            )

            valid_length = min(
                len(base_mask),
                key_length,
            )

            full_mask[:valid_length] = (
                base_mask[:valid_length]
                .to(probabilities.device)
            )

            if full_mask.any():

                values = probabilities[
                    :,
                    full_mask,
                ].sum(dim=-1)

            else:

                values = torch.zeros(
                    num_heads,
                    device=probabilities.device,
                )

            modality_attention[
                modality
            ] = values

        # Attention assigned to input tokens.
        input_attention = torch.zeros(
            num_heads,
            device=probabilities.device,
        )

        for values in modality_attention.values():
            input_attention += values

        # At later generation steps some attention goes
        # to previously generated answer tokens.
        generated_history_attention = (
            1.0 - input_attention
        ).clamp(
            min=0.0,
            max=1.0,
        )

        layer = extract_layer_number(
            module_name
        )

        for head in range(num_heads):

            row: dict[str, Any] = {
                "generation_step": generation_step,
                "module": module_name,
                "layer": layer,
                "head": head,
                "input_attention": float(
                    input_attention[head].item()
                ),
                "generated_history_attention": float(
                    generated_history_attention[
                        head
                    ].item()
                ),
            }

            denominator = float(
                input_attention[head].item()
            )

            for modality, values in (
                modality_attention.items()
            ):

                raw = float(
                    values[head].item()
                )

                # Raw attention as fraction of the complete
                # attention distribution.
                row[
                    f"{modality}_attention"
                ] = raw

                # Share restricted to original input sources.
                #
                # This removes attention to earlier generated
                # answer tokens.
                row[
                    f"{modality}_source_share"
                ] = (
                    raw / denominator
                    if denominator > 0
                    else 0.0
                )

            self.rows.append(row)


RECORDER = AttentionRecorder()


# ============================================================
# ATTENTION UTILITIES
# ============================================================

def extract_layer_number(
    module_name: str,
) -> int:

    patterns = [
        r"\.layers\.(\d+)\.",
        r"\.layer\.(\d+)\.",
        r"\.blocks\.(\d+)\.",
        r"\.block\.(\d+)\.",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            module_name,
        )

        if match:
            return int(match.group(1))

    return -1


def repeat_kv(
    tensor: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:

    num_kv_heads = tensor.shape[1]

    if num_kv_heads == num_query_heads:
        return tensor

    if num_query_heads % num_kv_heads != 0:
        return tensor

    repetitions = (
        num_query_heads // num_kv_heads
    )

    return tensor.repeat_interleave(
        repetitions,
        dim=1,
    )


# ============================================================
# CUSTOM ATTENTION
# ============================================================

def compute_last_query_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None,
    **kwargs: Any,
) -> None:

    if not RECORDER.active:
        return

    module_name = getattr(
        module,
        "_maia_module_name",
        "",
    ).lower()

    # Avoid modality encoders.
    if any(
        x in module_name
        for x in (
            "vision",
            "audio",
            "encoder",
            "projector",
            "perceiver",
        )
    ):
        return

    key_length = key.shape[-2]

    if key_length < RECORDER.base_length:
        return

    # Only last query position.
    #
    # query:
    # [B, H, Q, D]
    #
    query_last = query[
        ...,
        -1:,
        :,
    ]

    key_for_scores = repeat_kv(
        key,
        query_last.shape[1],
    )

    if scaling is None:
        scaling = (
            query_last.shape[-1] ** -0.5
        )

    scores = torch.matmul(
        query_last.float(),
        key_for_scores.float().transpose(
            -2,
            -1,
        ),
    )

    scores *= scaling

    # Gemma may use attention logit soft-capping.
    softcap = (
        kwargs.get(
            "attn_logit_softcapping"
        )
        or kwargs.get("softcap")
    )

    if softcap:

        scores = (
            torch.tanh(
                scores / softcap
            )
            * softcap
        )

    if attention_mask is not None:

        mask = attention_mask

        # Keep only the mask corresponding to
        # the final query.
        if mask.ndim == 4:

            mask = mask[
                ...,
                -1:,
                :key_length,
            ]

        if mask.dtype == torch.bool:

            scores = scores.masked_fill(
                ~mask,
                torch.finfo(
                    scores.dtype
                ).min,
            )

        else:

            scores = scores + mask.float()

    probabilities = F.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    )

    RECORDER.record(
        module=module,
        probabilities=probabilities,
        key_length=key_length,
    )


def maia_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
):

    # --------------------------------------------------------
    # Actual model inference
    # --------------------------------------------------------
    #
    # SDPA remains responsible for the real model output.
    #
    # Therefore we do NOT materialize the normal complete
    # attention matrix.
    # --------------------------------------------------------

    output, _ = sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )

    # --------------------------------------------------------
    # Diagnostic computation
    # --------------------------------------------------------
    #
    # Recompute only ONE attention row:
    #
    # last query -> all previous tokens
    #
    # This is what is required to attribute the next
    # generated token.
    # --------------------------------------------------------

    compute_last_query_attention(
        module=module,
        query=query,
        key=key,
        attention_mask=attention_mask,
        scaling=scaling,
        **kwargs,
    )

    return output, None


# Register custom attention backend before loading models.
AttentionInterface.register(
    "maia_attention",
    maia_attention_forward,
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    model_name: str,
):

    info = MODELS[model_name]

    model_id = info["model_id"]

    family = info["family"]

    print()
    print("=" * 80)
    print(f"Loading: {model_name}")
    print(f"Model ID: {model_id}")
    print("=" * 80)

    common_kwargs = {
        # CUDA_VISIBLE_DEVICES="5" makes physical GPU 5
        # appear inside the process as cuda:0.
        #
        # Force the entire model onto that single GPU.
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "maia_attention",
        "low_cpu_mem_usage": True,
    }

    # --------------------------------------------------------
    # Gemma 3 12B
    # --------------------------------------------------------

    if family == "gemma3":

        model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                model_id,
                **common_kwargs,
            )
        )

        processor = (
            AutoProcessor.from_pretrained(
                model_id,
            )
        )

    # --------------------------------------------------------
    # Gemma 3n E4B
    # --------------------------------------------------------

    elif family == "gemma3n":

        model = (
            Gemma3nForConditionalGeneration
            .from_pretrained(
                model_id,
                **common_kwargs,
            )
        )

        processor = (
            AutoProcessor.from_pretrained(
                model_id,
            )
        )

    # --------------------------------------------------------
    # Qwen 2.5 Omni 3B
    # --------------------------------------------------------

    elif family == "qwen2_5_omni":

        model = (
            Qwen2_5OmniForConditionalGeneration
            .from_pretrained(
                model_id,
                **common_kwargs,
            )
        )

        processor = (
            Qwen2_5OmniProcessor
            .from_pretrained(
                model_id,
            )
        )

        if hasattr(
            model,
            "disable_talker",
        ):
            model.disable_talker()

    # --------------------------------------------------------
    # Qwen 3 Omni 30B
    # --------------------------------------------------------

    elif family == "qwen3_omni":

        model = (
            AutoModelForMultimodalLM
            .from_pretrained(
                model_id,
                **common_kwargs,
            )
        )

        processor = (
            AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
        )

        if hasattr(
            model,
            "disable_talker",
        ):
            model.disable_talker()

    else:

        raise ValueError(
            f"Unknown model family: {family}"
        )

    model.eval()

    register_module_names(model)

    print("Device map:")

    if hasattr(
        model,
        "hf_device_map",
    ):

        for name, device in (
            model.hf_device_map.items()
        ):
            print(
                f"  {name}: {device}"
            )

    return model, processor


def register_module_names(
    model: torch.nn.Module,
) -> None:

    """
    Store the complete module path directly on each attention
    module so that the custom attention function knows which
    layer generated the signal.
    """

    for name, module in (
        model.named_modules()
    ):

        if "attention" in (
            module.__class__.__name__.lower()
        ) or "attn" in name.lower():

            module._maia_module_name = name


# ============================================================
# VIDEO DISCOVERY
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}


def extract_video_id(
    path: Path,
) -> int | None:

    numbers = re.findall(
        r"\d+",
        path.stem,
    )

    if not numbers:
        return None

    return int(numbers[-1])


def discover_videos(
    videos_dir: Path,
) -> dict[int, Path]:

    result = {}

    for path in videos_dir.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in VIDEO_EXTENSIONS
        ):
            continue

        video_id = extract_video_id(
            path
        )

        if video_id is None:
            continue

        if video_id not in result:
            result[video_id] = path

    return result


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def load_video_frames(
    video_path: Path,
    n_frames: int = N_FRAMES,
):

    import av

    container = av.open(
        str(video_path)
    )

    frames = [
        frame.to_image()
        for frame in container.decode(
            video=0
        )
    ]

    container.close()

    if not frames:

        raise RuntimeError(
            f"No video frames in {video_path}"
        )

    indexes = np.linspace(
        0,
        len(frames) - 1,
        n_frames,
    ).round().astype(int)

    return [
        frames[index]
        for index in indexes
    ]

def normalize_video_id(value: Any) -> int:
    match = re.search(r"\d+", Path(str(value)).stem)
    if not match:
        raise ValueError(f"Cannot extract video id from: {value}")
    return int(match.group())


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path: Path,
    output_dir: Path,
) -> Path:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        output_dir
        / f"{video_path.stem}.wav"
    )

    if output.exists():
        return output

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output),
    ]

    subprocess.run(
        command,
        check=True,
    )

    return output


# ============================================================
# CSV LOADING
# ============================================================

def load_transcriptions(path: Path) -> dict[int, str]:
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["video_name", "transcription"],
        encoding="utf-8-sig",
    )
    df["video_id"] = df["video_name"].map(normalize_video_id)
    return dict(zip(df["video_id"], df["transcription"].fillna("").astype(str).str.strip()))


def load_questions(path: Path) -> dict[int, list[dict[str, Any]]]:
    df = pd.read_csv(
        path,
        sep=";",
        usecols=["video_name", "question_order", "question_text"],
        encoding="utf-8-sig",
    )
    df["video_id"] = df["video_name"].map(normalize_video_id)
    df = df.sort_values(["video_id", "question_order"])

    return {
        int(video_id): [
            {
                "row_index": int(index),
                "question_order": int(row.question_order),
                "question": str(row.question_text).strip(),
            }
            for index, row in group.iterrows()
        ]
        for video_id, group in df.groupby("video_id", sort=False)
    }


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(
    question: str,
    transcription: str,
    condition: str,
) -> str:

    if (
        condition
        == "transcription_video"
    ):

        return TRANSCRIPTION_PROMPT.format(
            transcription=transcription,
            question=question,
        )

    return QUESTION_PROMPT.format(
        question=question,
    )


# ============================================================
# QWEN INPUT
# ============================================================

def prepare_qwen_inputs(
    processor,
    video_path: Path,
    prompt: str,
    condition: str,
):

    from qwen_omni_utils import (
        process_mm_info,
    )

    use_audio = (
        condition == "audio_video"
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(
                        video_path
                    ),
                    "nframes": N_FRAMES,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = (
        processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
    )

    audios, images, videos = (
        process_mm_info(
            conversation,
            use_audio_in_video=use_audio,
        )
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio,
    )

    return inputs, use_audio


# ============================================================
# GEMMA 3n INPUT
# ============================================================

def prepare_gemma3n_inputs(
    processor,
    video_path: Path,
    prompt: str,
    condition: str,
    audio_dir: Path,
):

    frames = load_video_frames(
        video_path,
        N_FRAMES,
    )

    content = []

    # Represent the video using exactly 32 uniformly
    # sampled visual observations.
    for frame in frames:

        content.append(
            {
                "type": "image",
                "image": frame,
            }
        )

    if condition == "audio_video":

        audio_path = extract_audio(
            video_path,
            audio_dir,
        )

        content.append(
            {
                "type": "audio",
                "audio": str(
                    audio_path
                ),
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    return (
        inputs,
        condition == "audio_video",
    )


# ============================================================
# GEMMA 3 12B INPUT
# ============================================================

def prepare_gemma3_inputs(
    processor,
    video_path: Path,
    prompt: str,
):

    frames = load_video_frames(
        video_path,
        N_FRAMES,
    )

    content = [
        {
            "type": "image",
            "image": frame,
        }
        for frame in frames
    ]

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    return inputs, False


# ============================================================
# MOVE INPUTS
# ============================================================

def model_input_device(
    model,
) -> torch.device:

    # With device_map="auto", the embeddings / first decoder
    # module normally indicate the correct input device.

    for parameter in model.parameters():

        if parameter.device.type == "cuda":
            return parameter.device

    return torch.device("cpu")


def move_inputs_to_device(
    inputs,
    model,
):

    device = model_input_device(
        model
    )

    try:
        dtype = model.dtype
    except Exception:
        dtype = torch.bfloat16

    if dtype is None:
        dtype = torch.bfloat16

    for key, value in list(
        inputs.items()
    ):

        if not torch.is_tensor(value):
            continue

        if torch.is_floating_point(value):
            inputs[key] = value.to(
                device=device,
                dtype=dtype,
            )
        else:
            inputs[key] = value.to(
                device=device
            )

    return inputs


# ============================================================
# SUBSEQUENCE IDENTIFICATION
# ============================================================

def find_subsequence(
    sequence: list[int],
    subsequence: list[int],
) -> tuple[int, int] | None:

    if not subsequence:
        return None

    length = len(
        subsequence
    )

    for start in range(
        len(sequence)
        - length
        + 1
    ):

        end = start + length

        if (
            sequence[start:end]
            == subsequence
        ):

            return start, end

    return None


def text_span_mask(
    tokenizer,
    input_ids: list[int],
    text: str,
) -> torch.Tensor:

    mask = torch.zeros(
        len(input_ids),
        dtype=torch.bool,
    )

    if not text:
        return mask

    # Tokenization can differ slightly depending on
    # whitespace immediately preceding the span.
    #
    # Try a few context variants.
    variants = [
        text,
        " " + text,
        "\n" + text,
        "\n\n" + text,
    ]

    for variant in variants:

        token_ids = tokenizer.encode(
            variant,
            add_special_tokens=False,
        )

        span = find_subsequence(
            input_ids,
            token_ids,
        )

        if span is None:
            continue

        start, end = span

        mask[start:end] = True

        return mask

    return mask


# ============================================================
# SPECIAL TOKEN IDENTIFICATION
# ============================================================

def recursively_find_token_ids(
    value: Any,
    keyword: str,
) -> set[int]:

    result: set[int] = set()

    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            key_lower = (
                str(key).lower()
            )

            if (
                keyword in key_lower
                and "token" in key_lower
                and isinstance(
                    child,
                    int,
                )
            ):

                result.add(
                    child
                )

            result.update(
                recursively_find_token_ids(
                    child,
                    keyword,
                )
            )

    elif isinstance(
        value,
        (list, tuple),
    ):

        for child in value:

            result.update(
                recursively_find_token_ids(
                    child,
                    keyword,
                )
            )

    return result


# ============================================================
# MODALITY MASKS
# ============================================================

def build_modality_masks(
    model,
    processor,
    inputs,
    question: str,
    transcription: str,
    condition: str,
) -> dict[str, torch.Tensor]:

    input_ids_tensor = inputs[
        "input_ids"
    ]

    input_ids = (
        input_ids_tensor[0]
        .detach()
        .cpu()
        .tolist()
    )

    tokenizer = (
        processor.tokenizer
    )

    config = (
        model.config.to_dict()
    )

    visual_token_ids = set()

    for keyword in [
        "image",
        "video",
        "vision",
    ]:

        visual_token_ids.update(
            recursively_find_token_ids(
                config,
                keyword,
            )
        )

    audio_token_ids = (
        recursively_find_token_ids(
            config,
            "audio",
        )
    )

    tokens = (
        tokenizer.convert_ids_to_tokens(
            input_ids
        )
    )

    video_mask = torch.zeros(
        len(input_ids),
        dtype=torch.bool,
    )

    audio_mask = torch.zeros(
        len(input_ids),
        dtype=torch.bool,
    )

    for index, (
        token_id,
        token,
    ) in enumerate(
        zip(
            input_ids,
            tokens,
        )
    ):

        token_text = (
            str(token).lower()
        )

        # ----------------------------
        # VIDEO / IMAGE
        # ----------------------------

        if (
            token_id
            in visual_token_ids

            or "image_soft_token"
            in token_text

            or "image_pad"
            in token_text

            or "video_pad"
            in token_text

            or "video_token"
            in token_text

            or "vision_token"
            in token_text
        ):

            video_mask[
                index
            ] = True

        # ----------------------------
        # AUDIO
        # ----------------------------

        if (
            token_id
            in audio_token_ids

            or "audio_soft_token"
            in token_text

            or "audio_pad"
            in token_text

            or "audio_token"
            in token_text
        ):

            audio_mask[
                index
            ] = True

    question_mask = (
        text_span_mask(
            tokenizer,
            input_ids,
            question,
        )
    )

    if (
        condition
        == "transcription_video"
        and transcription
    ):

        transcription_mask = (
            text_span_mask(
                tokenizer,
                input_ids,
                transcription,
            )
        )

    else:

        transcription_mask = (
            torch.zeros(
                len(input_ids),
                dtype=torch.bool,
            )
        )

    # Do not allow overlap.
    question_mask &= ~(
        video_mask
        | audio_mask
    )

    transcription_mask &= ~(
        video_mask
        | audio_mask
        | question_mask
    )

    # Actual valid tokens.
    if (
        "attention_mask"
        in inputs
    ):

        valid_tokens = (
            inputs[
                "attention_mask"
            ][0]
            .detach()
            .cpu()
            .bool()
        )

    else:

        valid_tokens = (
            torch.ones(
                len(input_ids),
                dtype=torch.bool,
            )
        )

    occupied = (
        video_mask
        | audio_mask
        | transcription_mask
        | question_mask
    )

    other_text_mask = (
        valid_tokens
        & ~occupied
    )

    return {
        "video": video_mask,
        "audio": audio_mask,
        "transcription": transcription_mask,
        "question": question_mask,
        "other_text": other_text_mask,
    }


# ============================================================
# MASK DIAGNOSTICS
# ============================================================

def modality_mask_stats(
    masks: dict[str, torch.Tensor],
) -> dict[str, int]:

    return {
        modality: int(
            mask.sum().item()
        )
        for modality, mask
        in masks.items()
    }


# ============================================================
# GENERATION
# ============================================================

def normalize_generation_output(
    output,
):

    if isinstance(
        output,
        tuple,
    ):

        output = output[0]

    if hasattr(
        output,
        "sequences",
    ):

        output = (
            output.sequences
        )

    return output


def run_generation(
    model,
    processor,
    inputs,
    family: str,
    use_audio: bool,
):

    input_length = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    kwargs = {
        "max_new_tokens":
            MAX_NEW_TOKENS,
        "do_sample":
            False,
        "use_cache":
            True,
    }

    if family in {
        "qwen2_5_omni",
        "qwen3_omni",
    }:

        kwargs.update(
            {
                "return_audio":
                    False,
                "use_audio_in_video":
                    use_audio,
            }
        )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            **kwargs,
        )

    output = (
        normalize_generation_output(
            output
        )
    )

    # Most HF causal generation outputs:
    #
    # prompt + generated tokens
    #
    if (
        output.shape[-1]
        >= input_length
    ):

        prefix = output[
            0,
            :input_length,
        ]

        if torch.equal(
            prefix,
            inputs[
                "input_ids"
            ][0],
        ):

            generated_ids = (
                output[
                    0,
                    input_length:,
                ]
            )

        else:

            generated_ids = (
                output[0]
            )

    else:

        generated_ids = (
            output[0]
        )

    generated_ids_list = (
        generated_ids
        .detach()
        .cpu()
        .tolist()
    )

    generated_text = (
        processor.decode(
            generated_ids_list,
            skip_special_tokens=True,
        )
    )

    return (
        generated_ids_list,
        generated_text.strip(),
    )


# ============================================================
# ATTENTION ROW ANNOTATION
# ============================================================

def annotate_attention_rows(
    rows: list[dict[str, Any]],
    processor,
    generated_ids: list[int],
    metadata: dict[str, Any],
):

    tokenizer = (
        processor.tokenizer
    )

    result = []

    for row in rows:

        step = int(
            row[
                "generation_step"
            ]
        )

        token_id = (
            generated_ids[step]
            if step
            < len(generated_ids)
            else None
        )

        if token_id is not None:

            token_text = (
                tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
            )

        else:

            token_text = ""

        complete_row = {
            **metadata,
            **row,
            "generated_token_id":
                token_id,
            "generated_token":
                token_text,
        }

        result.append(
            complete_row
        )

    return result


# ============================================================
# CSV WRITING
# ============================================================

def append_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SINGLE ANALYSIS
# ============================================================

def analyse_single_item(
    model,
    processor,
    model_name: str,
    info: dict[str, Any],
    sample_position: int,
    video_id: int,
    question_index: int,
    question: str,
    video_path: Path,
    transcription: str,
    condition: str,
    audio_dir: Path,
):

    family = info[
        "family"
    ]

    prompt = build_prompt(
        question=question,
        transcription=transcription,
        condition=condition,
    )

    # --------------------------------------------------------
    # INPUT PREPARATION
    # --------------------------------------------------------

    if family in {
        "qwen2_5_omni",
        "qwen3_omni",
    }:

        inputs, use_audio = (
            prepare_qwen_inputs(
                processor=processor,
                video_path=video_path,
                prompt=prompt,
                condition=condition,
            )
        )

    elif family == "gemma3n":

        inputs, use_audio = (
            prepare_gemma3n_inputs(
                processor=processor,
                video_path=video_path,
                prompt=prompt,
                condition=condition,
                audio_dir=audio_dir,
            )
        )

    elif family == "gemma3":

        inputs, use_audio = (
            prepare_gemma3_inputs(
                processor=processor,
                video_path=video_path,
                prompt=prompt,
            )
        )

    else:

        raise ValueError(
            family
        )

    inputs = move_inputs_to_device(
        inputs,
        model,
    )

    # --------------------------------------------------------
    # IDENTIFY MODALITIES IN INPUT TOKEN SEQUENCE
    # --------------------------------------------------------

    masks = build_modality_masks(
        model=model,
        processor=processor,
        inputs=inputs,
        question=question,
        transcription=transcription,
        condition=condition,
    )

    stats = modality_mask_stats(
        masks
    )

    print(
        "    token masks:",
        stats,
    )

    base_length = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    # --------------------------------------------------------
    # ATTENTION RECORDING
    # --------------------------------------------------------

    RECORDER.start(
        base_length=base_length,
        modality_masks=masks,
    )

    try:

        generated_ids, generated_text = (
            run_generation(
                model=model,
                processor=processor,
                inputs=inputs,
                family=family,
                use_audio=use_audio,
            )
        )

    finally:

        attention_rows = (
            RECORDER.stop()
        )

    metadata = {
        "model": model_name,
        "model_id": info[
            "model_id"
        ],
        "sample_position":
            sample_position,
        "video_id":
            video_id,
        "question_index":
            question_index,
        "condition":
            condition,
    }

    attention_rows = (
        annotate_attention_rows(
            attention_rows,
            processor,
            generated_ids,
            metadata,
        )
    )

    generation_row = {
        **metadata,
        "question":
            question,
        "generated_text":
            generated_text,
        "input_tokens":
            base_length,
        "generated_tokens":
            len(generated_ids),

        "video_tokens":
            stats["video"],

        "audio_tokens":
            stats["audio"],

        "transcription_tokens":
            stats[
                "transcription"
            ],

        "question_tokens":
            stats["question"],

        "other_text_tokens":
            stats["other_text"],
    }

    return (
        attention_rows,
        generation_row,
    )


# ============================================================
# AGGREGATION
# ============================================================

SOURCE_SHARE_COLUMNS = [
    "video_source_share",
    "audio_source_share",
    "transcription_source_share",
    "question_source_share",
    "other_text_source_share",
]


RAW_ATTENTION_COLUMNS = [
    "video_attention",
    "audio_attention",
    "transcription_attention",
    "question_attention",
    "other_text_attention",
    "generated_history_attention",
]


def create_summaries(
    detail_path: Path,
    output_dir: Path,
):

    if not detail_path.exists():
        return

    data = pd.read_csv(
    detail_path,
    sep=None,
    engine="python",
    encoding="utf-8-sig",
)

    if data.empty:
        return

    numeric_columns = (
        SOURCE_SHARE_COLUMNS
        + RAW_ATTENTION_COLUMNS
    )

    # ========================================================
    # BY GENERATED TOKEN
    # ========================================================

    token_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "sample_position",
                "video_id",
                "question_index",
                "generation_step",
                "generated_token_id",
                "generated_token",
            ],
            as_index=False,
            dropna=False,
        )[numeric_columns]
        .mean()
    )

    token_summary.to_csv(
        output_dir
        / "attention_by_token.csv",
        index=False,
    )

    # ========================================================
    # BY LAYER
    # ========================================================

    layer_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "layer",
            ],
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    layer_summary.to_csv(
        output_dir
        / "attention_by_layer.csv",
        index=False,
    )

    # ========================================================
    # BY VIDEO
    # ========================================================

    video_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "sample_position",
                "video_id",
            ],
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    video_summary.to_csv(
        output_dir
        / "attention_by_video.csv",
        index=False,
    )

    # ========================================================
    # MODEL × CONDITION
    # ========================================================

    model_summary = (
        data.groupby(
            [
                "model",
                "condition",
            ],
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    model_summary.to_csv(
        output_dir
        / "attention_summary.csv",
        index=False,
    )

    # ========================================================
    # LAYER × HEAD
    # ========================================================

    head_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "layer",
                "head",
            ],
            as_index=False,
        )[numeric_columns]
        .mean()
    )

    head_summary.to_csv(
        output_dir
        / "attention_by_head.csv",
        index=False,
    )


# ============================================================
# ATTENTION PERCENTAGES
# ============================================================

def create_percentage_outputs(
    detail_path: Path,
    output_dir: Path,
):

    if not detail_path.exists():
        return

    data = pd.read_csv(
        detail_path,
        encoding="utf-8-sig",
    )

    if data.empty:
        return

    source_columns = [
        "video_source_share",
        "audio_source_share",
        "transcription_source_share",
        "question_source_share",
        "other_text_source_share",
    ]

    group_columns = [
        "model",
        "condition",
        "sample_position",
        "video_id",
        "question_index",
    ]

    percentages = (
        data.groupby(
            group_columns,
            as_index=False,
        )[source_columns]
        .mean()
    )

    rename = {
        "video_source_share": "video_pct",
        "audio_source_share": "audio_pct",
        "transcription_source_share": "transcription_pct",
        "question_source_share": "question_pct",
        "other_text_source_share": "other_text_pct",
    }

    percentages = percentages.rename(
        columns=rename
    )

    percentage_columns = list(
        rename.values()
    )

    percentages[percentage_columns] = (
        percentages[percentage_columns] * 100.0
    )

    percentages["total_input_pct"] = (
        percentages[percentage_columns]
        .sum(axis=1)
    )

    percentages.to_csv(
        output_dir / "attention_percentages.csv",
        index=False,
    )

    summary = (
        percentages.groupby(
            ["model", "condition"],
            as_index=False,
        )[percentage_columns]
        .mean()
    )

    summary.to_csv(
        output_dir / "attention_percentage_summary.csv",
        index=False,
    )

    print()
    print("Attention percentages:")
    print(summary.round(2).to_string(index=False))


# ============================================================
# MEMORY CLEANUP
# ============================================================

def cleanup_gpu() -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)

    parser.add_argument("--transcriptions", type=Path, default=TRANSCRIPTIONS_FILE)

    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)

    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(
            MODELS.keys()
        ),
        default=list(
            MODELS.keys()
        ),
    )

    parser.add_argument(
        "--questions-per-video",
        type=int,
        default=0,
        help=(
            "0 = use every question associated with "
            "each selected video. "
            "N > 0 = use only the first N questions."
        ),
    )

    args = parser.parse_args()

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = (
        output_dir
        / "attention_detail.csv"
    )

    generations_path = (
        output_dir
        / "generations.csv"
    )

    status_path = (
        output_dir
        / "attention_status.csv"
    )

    audio_dir = (
        output_dir
        / "audio_cache"
    )

    # New run: remove previous files.
    for path in [
        detail_path,
        generations_path,
        status_path,
    ]:

        if path.exists():
            path.unlink()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "Discovering videos..."
    )

    all_videos = discover_videos(
        args.videos_dir
    )

    print(
        f"Found {len(all_videos)} videos."
    )

    missing_videos = [
        video_id
        for video_id in set(
            VIDEO_IDS
        )
        if video_id
        not in all_videos
    ]

    if missing_videos:

        raise FileNotFoundError(
            "Missing requested videos: "
            f"{missing_videos}"
        )

    print(
        "Loading transcriptions..."
    )

    transcriptions = (
        load_transcriptions(
            args.transcriptions
        )
    )

    print(
        "Loading questions..."
    )

    questions = (
        load_questions(
            args.questions
        )
    )

    status_rows = []

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    for model_name in (
        args.models
    ):

        info = MODELS[
            model_name
        ]

        model, processor = (
            load_model(
                model_name
            )
        )

        # ----------------------------------------------------
        # 15 requested sample positions
        # ----------------------------------------------------

        for (
            sample_position,
            video_id,
        ) in enumerate(
            VIDEO_IDS,
            start=1,
        ):

            video_path = (
                all_videos[
                    video_id
                ]
            )

            transcription = (
                transcriptions.get(
                    video_id,
                    "",
                )
            )

            video_questions = (
                questions.get(
                    video_id,
                    [],
                )
            )

            if not video_questions:

                print(
                    f"[WARNING] No questions "
                    f"for video {video_id}"
                )

                continue

            if (
                args.questions_per_video
                > 0
            ):

                video_questions = (
                    video_questions[
                        :args.questions_per_video
                    ]
                )

            # ------------------------------------------------
            # TWO CONDITIONS
            # ------------------------------------------------

            for condition in CONDITIONS:

                # Gemma 3 12B:
                # no native audio modality.
                if (
                    condition
                    == "audio_video"

                    and not info[
                        "supports_audio"
                    ]
                ):

                    print(
                        f"[SKIP] "
                        f"{model_name} | "
                        f"sample={sample_position} | "
                        f"video={video_id} | "
                        f"{condition}: "
                        f"raw audio unsupported"
                    )

                    status_rows.append(
                        {
                            "model":
                                model_name,

                            "sample_position":
                                sample_position,

                            "video_id":
                                video_id,

                            "condition":
                                condition,

                            "question_index":
                                "",

                            "status":
                                "not_supported",

                            "reason":
                                (
                                    "Model does not "
                                    "accept native "
                                    "raw-audio input."
                                ),
                        }
                    )

                    continue

                # --------------------------------------------
                # QUESTIONS
                # --------------------------------------------

                for (
                    question_index,
                    item,
                ) in enumerate(
                    video_questions,
                    start=1,
                ):

                    question = (
                        item[
                            "question"
                        ]
                    )

                    print()
                    print(
                        f"[RUN] "
                        f"{model_name} | "
                        f"{condition} | "
                        f"sample "
                        f"{sample_position}/15 | "
                        f"video={video_id} | "
                        f"question="
                        f"{question_index}/"
                        f"{len(video_questions)}"
                    )

                    try:

                        (
                            attention_rows,
                            generation_row,
                        ) = (
                            analyse_single_item(
                                model=model,
                                processor=processor,
                                model_name=model_name,
                                info=info,
                                sample_position=sample_position,
                                video_id=video_id,
                                question_index=question_index,
                                question=question,
                                video_path=video_path,
                                transcription=transcription,
                                condition=condition,
                                audio_dir=audio_dir,
                            )
                        )

                        append_rows(
                            detail_path,
                            attention_rows,
                        )

                        append_rows(
                            generations_path,
                            [
                                generation_row
                            ],
                        )

                        status_rows.append(
                            {
                                "model":
                                    model_name,

                                "sample_position":
                                    sample_position,

                                "video_id":
                                    video_id,

                                "condition":
                                    condition,

                                "question_index":
                                    question_index,

                                "status":
                                    "ok",

                                "reason":
                                    "",
                            }
                        )

                        print(
                            "    -> OK"
                        )

                        print(
                            "    -> response:",
                            generation_row[
                                "generated_text"
                            ][:200],
                        )

                    except Exception as error:

                        print(
                            "    -> ERROR:",
                            repr(error),
                        )

                        status_rows.append(
                            {
                                "model":
                                    model_name,

                                "sample_position":
                                    sample_position,

                                "video_id":
                                    video_id,

                                "condition":
                                    condition,

                                "question_index":
                                    question_index,

                                "status":
                                    "error",

                                "reason":
                                    repr(error),
                            }
                        )

                    # Write status continuously.
                    pd.DataFrame(
                        status_rows
                    ).to_csv(
                        status_path,
                        index=False,
                    )

                    cleanup_gpu()

        # ----------------------------------------------------
        # UNLOAD CURRENT MODEL
        # ----------------------------------------------------

        print()
        print(
            f"Unloading {model_name}..."
        )

        del model
        del processor

        cleanup_gpu()

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    pd.DataFrame(
        status_rows
    ).to_csv(
        status_path,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARIES
    # --------------------------------------------------------

    print()
    print(
        "Creating attention summaries..."
    )

    create_summaries(
        detail_path,
        output_dir,
    )

    print()
    print(
        "Calculating attention percentages..."
    )

    create_percentage_outputs(
        detail_path,
        output_dir,
    )

    print()
    print("=" * 80)
    print(
        "ATTENTION ANALYSIS COMPLETED"
    )
    print("=" * 80)

    print(
        f"Results: {output_dir}"
    )


if __name__ == "__main__":
    main()