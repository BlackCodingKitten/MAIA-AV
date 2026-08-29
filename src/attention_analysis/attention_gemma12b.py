from __future__ import annotations

# ============================================================
# IMPORTANT
# ============================================================
#
# CUDA_VISIBLE_DEVICES must be set BEFORE importing torch.
#
# Physical GPU -> logical CUDA mapping:
#
#   GPU 6 -> cuda:0
#
# The complete Gemma 4 12B model is loaded on this single GPU.
# ============================================================

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import re
import subprocess
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
)

from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward,
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "gemma-4-12B"
MODEL_ID = "google/gemma-4-12B-it"


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

N_FRAMES = 32
MAX_NEW_TOKENS = 64

VIDEOS_DIR = Path("data/input/video")
TRANSCRIPTIONS_FILE = Path(
    "data/input/transcription/transcription.csv"
)
QUESTIONS_FILE = Path(
    "data/vsv/question_classification.csv"
)

OUTPUT_DIR = Path(
    "data/attention_analysis/gemma-4-12B"
)

AUDIO_DIR = (
    OUTPUT_DIR
    / "audio_cache"
)


# Exact same 15 sample positions used in the other analyses.
#
# There are 15 positions but 13 unique videos because
# video 75 and video 16 occur twice.
VIDEO_IDS = [
    91, 42, 16, 75, 17,
    37, 49, 11, 50, 35,
    1, 75, 70, 4, 16,
]


CONDITIONS = [
    "audio_video",
    "transcription_video",
]


# ============================================================
# PROMPTS
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
    position of each decoder attention call.

    This avoids materializing and storing the complete
    [sequence_length x sequence_length] attention matrices.

    During generation:

        generation_step = 0
            final prompt query -> first generated token

        generation_step = 1
            first generated token -> second generated token

        etc.
    """

    def __init__(self) -> None:

        self.active = False

        self.base_length = 0

        self.modality_masks: dict[
            str,
            torch.Tensor,
        ] = {}

        self.rows: list[
            dict[str, Any]
        ] = []

    def start(
        self,
        base_length: int,
        modality_masks: dict[
            str,
            torch.Tensor,
        ],
    ) -> None:

        self.active = True

        self.base_length = (
            base_length
        )

        self.modality_masks = (
            modality_masks
        )

        self.rows = []

    def stop(
        self,
    ) -> list[dict[str, Any]]:

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

        module_name_lower = (
            module_name.lower()
        )

        # ----------------------------------------------------
        # KEEP ONLY LANGUAGE DECODER ATTENTION
        # ----------------------------------------------------
        #
        # Gemma 4 Unified is encoder-free for audio and vision,
        # but this exclusion remains useful and harmless.
        # ----------------------------------------------------

        excluded = (
            "vision",
            "audio",
            "encoder",
            "projector",
            "perceiver",
            "embed_vision",
            "embed_audio",
        )

        if any(
            name in module_name_lower
            for name in excluded
        ):

            return

        # The decoder KV sequence must contain at least the
        # complete original prompt sequence.
        #
        # If a future implementation exposes a physically
        # truncated sliding-window KV tensor, that call is
        # skipped because its absolute token positions cannot
        # be aligned safely with the complete prompt mask.
        if (
            key_length
            < self.base_length
        ):

            return

        # Expected diagnostic shape:
        #
        # [batch, num_heads, 1, key_length]
        if (
            probabilities.ndim
            != 4
        ):

            return

        probabilities = (
            probabilities[
                0,
                :,
                0,
                :,
            ]
        )

        num_heads = (
            probabilities.shape[0]
        )

        generation_step = max(
            0,
            key_length
            - self.base_length,
        )

        modality_attention: dict[
            str,
            torch.Tensor,
        ] = {}

        # ----------------------------------------------------
        # SUM ATTENTION OVER TOKENS BELONGING TO EACH SOURCE
        # ----------------------------------------------------

        for (
            modality,
            base_mask,
        ) in self.modality_masks.items():

            full_mask = torch.zeros(
                key_length,
                dtype=torch.bool,
                device=probabilities.device,
            )

            valid_length = min(
                len(base_mask),
                key_length,
            )

            full_mask[
                :valid_length
            ] = (
                base_mask[
                    :valid_length
                ]
                .to(
                    probabilities.device
                )
            )

            if full_mask.any():

                values = (
                    probabilities[
                        :,
                        full_mask,
                    ]
                    .sum(
                        dim=-1
                    )
                )

            else:

                values = torch.zeros(
                    num_heads,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )

            modality_attention[
                modality
            ] = values

        # ----------------------------------------------------
        # ATTENTION TO ORIGINAL INPUT
        # ----------------------------------------------------

        input_attention = torch.zeros(
            num_heads,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )

        for values in (
            modality_attention.values()
        ):

            input_attention += (
                values
            )

        # At later generation steps the model can attend to
        # tokens already generated in the answer.
        generated_history_attention = (
            1.0
            - input_attention
        ).clamp(
            min=0.0,
            max=1.0,
        )

        layer = (
            extract_layer_number(
                module_name
            )
        )

        # ----------------------------------------------------
        # ONE ROW PER HEAD
        # ----------------------------------------------------

        for head in range(
            num_heads
        ):

            denominator = float(
                input_attention[
                    head
                ].item()
            )

            row: dict[
                str,
                Any,
            ] = {

                "generation_step":
                    generation_step,

                "module":
                    module_name,

                "layer":
                    layer,

                "head":
                    head,

                "input_attention":
                    denominator,

                "generated_history_attention":
                    float(
                        generated_history_attention[
                            head
                        ].item()
                    ),
            }

            for (
                modality,
                values,
            ) in modality_attention.items():

                raw = float(
                    values[
                        head
                    ].item()
                )

                # Raw share of the complete attention
                # distribution.
                row[
                    f"{modality}_attention"
                ] = raw

                # Share normalized only over the original
                # prompt/input sources.
                row[
                    f"{modality}_source_share"
                ] = (
                    raw
                    / denominator

                    if denominator > 0

                    else 0.0
                )

            self.rows.append(
                row
            )


RECORDER = (
    AttentionRecorder()
)


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

            return int(
                match.group(1)
            )

    return -1


def repeat_kv(
    tensor: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:

    """
    Expand grouped-query KV heads to the number of query heads
    for the diagnostic attention computation.
    """

    num_kv_heads = (
        tensor.shape[1]
    )

    if (
        num_kv_heads
        == num_query_heads
    ):

        return tensor

    if (
        num_query_heads
        % num_kv_heads
        != 0
    ):

        return tensor

    repetitions = (
        num_query_heads
        // num_kv_heads
    )

    return (
        tensor.repeat_interleave(
            repetitions,
            dim=1,
        )
    )


# ============================================================
# CUSTOM ATTENTION BACKEND
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

    # Keep decoder self-attention only.
    if any(
        name in module_name
        for name in (
            "vision",
            "audio",
            "encoder",
            "projector",
            "perceiver",
            "embed_vision",
            "embed_audio",
        )
    ):

        return

    key_length = (
        key.shape[-2]
    )

    if (
        key_length
        < RECORDER.base_length
    ):

        return

    # --------------------------------------------------------
    # FINAL QUERY POSITION ONLY
    # --------------------------------------------------------

    query_last = (
        query[
            ...,
            -1:,
            :,
        ]
    )

    key_for_scores = (
        repeat_kv(
            key,
            query_last.shape[1],
        )
    )

    if scaling is None:

        scaling = (
            query_last.shape[-1]
            ** -0.5
        )

    # --------------------------------------------------------
    # DIAGNOSTIC SCORES IN FP32
    # --------------------------------------------------------
    #
    # This does not alter the model forward output.
    # --------------------------------------------------------

    scores = torch.matmul(
        query_last.float(),
        key_for_scores
        .float()
        .transpose(
            -2,
            -1,
        ),
    )

    scores *= scaling

    # Compatibility with attention implementations that expose
    # optional logit soft-capping.
    softcap = (
        kwargs.get(
            "attn_logit_softcapping"
        )
        or kwargs.get(
            "softcap"
        )
    )

    if softcap:

        scores = (
            torch.tanh(
                scores
                / softcap
            )
            * softcap
        )

    # --------------------------------------------------------
    # APPLY EXACT ATTENTION MASK USED BY THE MODEL
    # --------------------------------------------------------

    if (
        attention_mask
        is not None
    ):

        mask = (
            attention_mask
        )

        if (
            mask.ndim
            == 4
        ):

            mask = (
                mask[
                    ...,
                    -1:,
                    :key_length,
                ]
            )

        if (
            mask.dtype
            == torch.bool
        ):

            scores = (
                scores.masked_fill(
                    ~mask,
                    torch.finfo(
                        scores.dtype
                    ).min,
                )
            )

        else:

            scores = (
                scores
                + mask.float()
            )

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
    # REAL MODEL ATTENTION
    # --------------------------------------------------------

    output, _ = (
        sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    )

    # --------------------------------------------------------
    # DIAGNOSTIC ATTENTION
    # --------------------------------------------------------
    #
    # Recompute only:
    #
    #     final query -> all available keys
    #
    # rather than materializing the complete attention matrix.
    # --------------------------------------------------------

    compute_last_query_attention(
        module=module,
        query=query,
        key=key,
        attention_mask=attention_mask,
        scaling=scaling,
        **kwargs,
    )

    return (
        output,
        None,
    )


# Register BEFORE loading Gemma 4.
AttentionInterface.register(
    "maia_attention",
    maia_attention_forward,
)


# ============================================================
# MODEL LOADING
# ============================================================

def register_module_names(
    model: torch.nn.Module,
) -> None:

    """
    Store the full module path directly on attention modules.
    """

    for (
        name,
        module,
    ) in model.named_modules():

        class_name = (
            module
            .__class__
            .__name__
            .lower()
        )

        if (
            "attention"
            in class_name
            or "attn"
            in name.lower()
        ):

            module._maia_module_name = (
                name
            )


def load_model():

    print()
    print(
        "=" * 80
    )
    print(
        f"Loading: {MODEL_NAME}"
    )
    print(
        f"Model ID: {MODEL_ID}"
    )
    print(
        "Physical GPU: 6"
    )
    print(
        "Logical device: cuda:0"
    )
    print(
        "=" * 80
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
        )
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            MODEL_ID,

            # Only physical GPU 6 is visible, so GPU index 0
            # inside this process is physical GPU 6.
            device_map={
                "": 0
            },

            dtype=torch.bfloat16,

            attn_implementation=
                "maia_attention",

            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    register_module_names(
        model
    )

    model_type = getattr(
        model.config,
        "model_type",
        "",
    )

    print(
        f"Loaded class: "
        f"{model.__class__.__name__}"
    )

    print(
        f"Config model_type: "
        f"{model_type}"
    )

    # This catches accidental loading of Gemma 3 or a different
    # Gemma 4 implementation immediately.
    if (
        model_type
        != "gemma4_unified"
    ):

        raise RuntimeError(
            "Expected Gemma 4 Unified "
            f"(model_type='gemma4_unified'), "
            f"but loaded model_type={model_type!r}."
        )

    print(
        f"Model dtype: "
        f"{model.dtype}"
    )

    return (
        model,
        processor,
    )


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

    return int(
        numbers[-1]
    )


def normalize_video_id(
    value: Any,
) -> int:

    match = re.search(
        r"\d+",
        Path(
            str(value)
        ).stem,
    )

    if not match:

        raise ValueError(
            "Cannot extract video id "
            f"from: {value}"
        )

    return int(
        match.group()
    )


def discover_videos(
    videos_dir: Path,
) -> dict[
    int,
    Path,
]:

    result: dict[
        int,
        Path,
    ] = {}

    for path in (
        videos_dir.rglob("*")
    ):

        if not path.is_file():

            continue

        if (
            path.suffix.lower()
            not in VIDEO_EXTENSIONS
        ):

            continue

        video_id = (
            extract_video_id(
                path
            )
        )

        if (
            video_id
            is None
        ):

            continue

        result.setdefault(
            video_id,
            path,
        )

    return result


# ============================================================
# VIDEO FRAME SAMPLING
# ============================================================

def load_video_frames(
    video_path: Path,
    n_frames: int = N_FRAMES,
):

    """
    Decode the source video and return exactly N_FRAMES
    uniformly sampled PIL images.
    """

    import av

    container = av.open(
        str(
            video_path
        )
    )

    frames = [
        frame.to_image()

        for frame
        in container.decode(
            video=0
        )
    ]

    container.close()

    if not frames:

        raise RuntimeError(
            "No video frames found in "
            f"{video_path}"
        )

    indexes = (
        np.linspace(
            0,
            len(frames) - 1,
            n_frames,
        )
        .round()
        .astype(int)
    )

    return [
        frames[
            index
        ]

        for index
        in indexes
    ]


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path: Path,
    output_dir: Path,
) -> Path:

    """
    Extract the audio track embedded in the source video.

    Output:
        mono WAV
        16 kHz

    Gemma 4 12B receives this waveform as native audio input.
    """

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
        str(
            video_path
        ),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(
            output
        ),
    ]

    subprocess.run(
        command,
        check=True,
    )

    return output


# ============================================================
# CSV LOADING
# ============================================================

def load_transcriptions(
    path: Path,
) -> dict[
    int,
    str,
]:

    df = pd.read_csv(
        path,
        sep=";",
        usecols=[
            "video_name",
            "transcription",
        ],
        encoding="utf-8-sig",
    )

    df[
        "video_id"
    ] = (
        df[
            "video_name"
        ]
        .map(
            normalize_video_id
        )
    )

    return dict(
        zip(
            df[
                "video_id"
            ],

            df[
                "transcription"
            ]
            .fillna("")
            .astype(str)
            .str.strip(),
        )
    )


def load_questions(
    path: Path,
) -> dict[
    int,
    list[
        dict[str, Any]
    ],
]:

    df = pd.read_csv(
        path,
        sep=";",
        usecols=[
            "video_name",
            "question_order",
            "question_text",
        ],
        encoding="utf-8-sig",
    )

    df[
        "video_id"
    ] = (
        df[
            "video_name"
        ]
        .map(
            normalize_video_id
        )
    )

    df = (
        df.sort_values(
            [
                "video_id",
                "question_order",
            ]
        )
    )

    return {

        int(
            video_id
        ): [

            {
                "row_index":
                    int(
                        index
                    ),

                "question_order":
                    int(
                        row.question_order
                    ),

                "question":
                    str(
                        row.question_text
                    ).strip(),
            }

            for (
                index,
                row,
            ) in group.iterrows()
        ]

        for (
            video_id,
            group,
        ) in df.groupby(
            "video_id",
            sort=False,
        )
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

        return (
            TRANSCRIPTION_PROMPT
            .format(
                transcription=
                    transcription,

                question=
                    question,
            )
        )

    if (
        condition
        == "audio_video"
    ):

        return (
            QUESTION_PROMPT
            .format(
                question=
                    question,
            )
        )

    raise ValueError(
        f"Unknown condition: "
        f"{condition}"
    )


# ============================================================
# GEMMA 4 INPUT PREPARATION
# ============================================================

def prepare_gemma4_inputs(
    processor,
    video_path: Path,
    prompt: str,
    condition: str,
    audio_dir: Path,
):

    """
    Construct the real experimental inputs.

    audio_video:
        32 video frames
        + question text
        + native raw audio extracted from the same video

    transcription_video:
        32 video frames
        + textual transcription
        + question text

    Gemma 4 best-practice modality ordering is respected:
        image(s) -> text -> audio
    """

    frames = load_video_frames(
        video_path,
        N_FRAMES,
    )

    content = []

    # --------------------------------------------------------
    # 32 VISUAL OBSERVATIONS
    # --------------------------------------------------------

    for frame in frames:

        content.append(
            {
                "type":
                    "image",

                "image":
                    frame,
            }
        )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    content.append(
        {
            "type":
                "text",

            "text":
                prompt,
        }
    )

    # --------------------------------------------------------
    # NATIVE AUDIO
    # --------------------------------------------------------
    #
    # Gemma 4 documentation recommends placing audio after
    # text. The waveform is extracted from the same source MP4.
    # --------------------------------------------------------

    if (
        condition
        == "audio_video"
    ):

        audio_path = (
            extract_audio(
                video_path,
                audio_dir,
            )
        )

        content.append(
            {
                "type":
                    "audio",

                "audio":
                    str(
                        audio_path
                    ),
            }
        )

    messages = [
        {
            "role":
                "user",

            "content":
                content,
        }
    ]

    inputs = (
        processor
        .apply_chat_template(
            messages,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",

            add_generation_prompt=True,

            # Disable explicit reasoning output so that
            # attention is measured over the answer generation
            # rather than a long reasoning trace.
            enable_thinking=False,
        )
    )

    return inputs


# ============================================================
# MOVE INPUTS TO GPU 6
# ============================================================

def move_inputs_to_device(
    inputs,
    model,
):

    """
    The entire model is on the only visible GPU:
        physical GPU 6 -> cuda:0

    Floating multimodal tensors are cast to the model dtype.
    Integer and boolean tensors preserve their dtype.
    """

    device = torch.device(
        "cuda:0"
    )

    dtype = (
        model.dtype
        if model.dtype is not None
        else torch.bfloat16
    )

    for (
        key,
        value,
    ) in list(
        inputs.items()
    ):

        if not torch.is_tensor(
            value
        ):

            continue

        if torch.is_floating_point(
            value
        ):

            inputs[
                key
            ] = value.to(
                device=device,
                dtype=dtype,
            )

        else:

            inputs[
                key
            ] = value.to(
                device=device,
            )

    return inputs


# ============================================================
# INPUT DIAGNOSTICS
# ============================================================

def print_input_diagnostics(
    inputs,
) -> None:

    print(
        "    input tensors:"
    )

    for (
        key,
        value,
    ) in inputs.items():

        if torch.is_tensor(
            value
        ):

            print(
                f"      {key}: "
                f"shape={tuple(value.shape)} | "
                f"dtype={value.dtype} | "
                f"device={value.device}"
            )


# ============================================================
# TOKEN SUBSEQUENCE UTILITIES
# ============================================================

def find_subsequence(
    sequence: list[int],
    subsequence: list[int],
) -> tuple[
    int,
    int,
] | None:

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

        end = (
            start
            + length
        )

        if (
            sequence[
                start:end
            ]
            == subsequence
        ):

            return (
                start,
                end,
            )

    return None


def text_span_mask_fallback(
    tokenizer,
    input_ids: list[int],
    text: str,
) -> torch.Tensor:

    """
    Fallback for tokenizers that do not expose offset mappings.
    """

    mask = torch.zeros(
        len(
            input_ids
        ),
        dtype=torch.bool,
    )

    if not text:

        return mask

    variants = [
        text,
        " " + text,
        "\n" + text,
        "\n\n" + text,
    ]

    for variant in variants:

        token_ids = (
            tokenizer.encode(
                variant,
                add_special_tokens=False,
            )
        )

        span = (
            find_subsequence(
                input_ids,
                token_ids,
            )
        )

        if (
            span
            is None
        ):

            continue

        start, end = (
            span
        )

        mask[
            start:end
        ] = True

        return mask

    return mask


def build_prompt_text_masks(
    tokenizer,
    input_ids: list[int],
    prompt: str,
    question: str,
    transcription: str,
    condition: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    """
    Locate QUESTION and TRANSCRIPTION inside the exact complete
    prompt rather than tokenizing each span in isolation.

    This avoids zero masks caused by BPE boundary differences
    introduced by surrounding whitespace/chat-template context.
    """

    question_mask = (
        torch.zeros(
            len(
                input_ids
            ),
            dtype=torch.bool,
        )
    )

    transcription_mask = (
        torch.zeros(
            len(
                input_ids
            ),
            dtype=torch.bool,
        )
    )

    try:

        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        prompt_ids = (
            encoded[
                "input_ids"
            ]
        )

        offsets = (
            encoded[
                "offset_mapping"
            ]
        )

    except Exception:

        return (
            text_span_mask_fallback(
                tokenizer,
                input_ids,
                question,
            ),

            text_span_mask_fallback(
                tokenizer,
                input_ids,
                transcription,
            )

            if (
                condition
                == "transcription_video"
                and transcription
            )

            else transcription_mask,
        )

    prompt_span = (
        find_subsequence(
            input_ids,
            prompt_ids,
        )
    )

    if (
        prompt_span
        is None
    ):

        return (
            text_span_mask_fallback(
                tokenizer,
                input_ids,
                question,
            ),

            text_span_mask_fallback(
                tokenizer,
                input_ids,
                transcription,
            )

            if (
                condition
                == "transcription_video"
                and transcription
            )

            else transcription_mask,
        )

    prompt_start, _ = (
        prompt_span
    )

    def mark_char_span(
        target_mask: torch.Tensor,
        char_start: int,
        char_end: int,
    ) -> None:

        if (
            char_start < 0
            or char_end <= char_start
        ):

            return

        for (
            local_index,
            offset,
        ) in enumerate(
            offsets
        ):

            token_start = int(
                offset[0]
            )

            token_end = int(
                offset[1]
            )

            if (
                token_end
                <= token_start
            ):

                continue

            if (
                token_end
                > char_start

                and token_start
                < char_end
            ):

                global_index = (
                    prompt_start
                    + local_index
                )

                if (
                    0
                    <= global_index
                    < len(
                        target_mask
                    )
                ):

                    target_mask[
                        global_index
                    ] = True

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------
    #
    # rfind() avoids accidentally matching an identical phrase
    # that may also occur inside the transcription.
    # --------------------------------------------------------

    question_start = (
        prompt.rfind(
            question
        )
    )

    if (
        question_start
        >= 0
    ):

        mark_char_span(
            question_mask,
            question_start,
            question_start
            + len(
                question
            ),
        )

    # --------------------------------------------------------
    # TRANSCRIPTION
    # --------------------------------------------------------

    if (
        condition
        == "transcription_video"

        and transcription
    ):

        transcription_start = (
            prompt.find(
                transcription
            )
        )

        if (
            transcription_start
            >= 0
        ):

            mark_char_span(
                transcription_mask,
                transcription_start,
                transcription_start
                + len(
                    transcription
                ),
            )

    return (
        question_mask,
        transcription_mask,
    )


# ============================================================
# MODALITY MASKS
# ============================================================

def build_modality_masks(
    processor,
    inputs,
    prompt: str,
    question: str,
    transcription: str,
    condition: str,
) -> dict[
    str,
    torch.Tensor,
]:

    input_ids = (
        inputs[
            "input_ids"
        ][0]
        .detach()
        .cpu()
        .tolist()
    )

    tokenizer = (
        processor.tokenizer
    )

    sequence_length = (
        len(
            input_ids
        )
    )

    video_mask = torch.zeros(
        sequence_length,
        dtype=torch.bool,
    )

    audio_mask = torch.zeros(
        sequence_length,
        dtype=torch.bool,
    )

    # --------------------------------------------------------
    # PRIMARY METHOD: GEMMA 4 MM TOKEN TYPE IDS
    # --------------------------------------------------------
    #
    # Hugging Face multimodal token type convention:
    #
    #   0 = text
    #   1 = image
    #   2 = video
    #   3 = audio
    #
    # We use 32 images to represent the source video, therefore
    # both image and video type ids are counted as VIDEO.
    # --------------------------------------------------------

    if (
        "mm_token_type_ids"
        in inputs
    ):

        mm_types = (
            inputs[
                "mm_token_type_ids"
            ][0]
            .detach()
            .cpu()
        )

        video_mask = (
            (mm_types == 1)
            | (mm_types == 2)
        )

        audio_mask = (
            mm_types == 3
        )

    elif (
        "token_type_ids"
        in inputs
    ):

        # Compatibility fallback if the processor exposes the
        # multimodal ids under token_type_ids.
        mm_types = (
            inputs[
                "token_type_ids"
            ][0]
            .detach()
            .cpu()
        )

        video_mask = (
            (mm_types == 1)
            | (mm_types == 2)
        )

        audio_mask = (
            mm_types == 3
        )

    else:

        # ----------------------------------------------------
        # SECONDARY FALLBACK: SPECIAL TOKEN IDS
        # ----------------------------------------------------

        image_token_id = getattr(
            tokenizer,
            "image_token_id",
            None,
        )

        audio_token_id = getattr(
            tokenizer,
            "audio_token_id",
            None,
        )

        video_token_id = getattr(
            processor,
            "video_token_id",
            None,
        )

        for (
            index,
            token_id,
        ) in enumerate(
            input_ids
        ):

            if (
                image_token_id
                is not None
                and token_id
                == image_token_id
            ):

                video_mask[
                    index
                ] = True

            if (
                video_token_id
                is not None
                and token_id
                == video_token_id
            ):

                video_mask[
                    index
                ] = True

            if (
                audio_token_id
                is not None
                and token_id
                == audio_token_id
            ):

                audio_mask[
                    index
                ] = True

    # --------------------------------------------------------
    # QUESTION + TRANSCRIPTION
    # --------------------------------------------------------

    (
        question_mask,
        transcription_mask,
    ) = build_prompt_text_masks(
        tokenizer=
            tokenizer,

        input_ids=
            input_ids,

        prompt=
            prompt,

        question=
            question,

        transcription=
            transcription,

        condition=
            condition,
    )

    # --------------------------------------------------------
    # NO OVERLAP
    # --------------------------------------------------------

    question_mask &= ~(
        video_mask
        | audio_mask
    )

    transcription_mask &= ~(
        video_mask
        | audio_mask
        | question_mask
    )

    # --------------------------------------------------------
    # VALID TOKENS
    # --------------------------------------------------------

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
                sequence_length,
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

        "video":
            video_mask,

        "audio":
            audio_mask,

        "transcription":
            transcription_mask,

        "question":
            question_mask,

        "other_text":
            other_text_mask,
    }


def modality_mask_stats(
    masks: dict[
        str,
        torch.Tensor,
    ],
) -> dict[
    str,
    int,
]:

    return {

        modality:
            int(
                mask.sum().item()
            )

        for (
            modality,
            mask,
        ) in masks.items()
    }


def validate_modality_masks(
    stats: dict[
        str,
        int,
    ],
    condition: str,
) -> None:

    """
    Stop immediately if an expected experimental modality was
    not identified. This prevents silently generating invalid
    attention percentages.
    """

    if (
        stats[
            "video"
        ]
        <= 0
    ):

        raise RuntimeError(
            "No visual tokens were identified "
            "in the input sequence."
        )

    if (
        stats[
            "question"
        ]
        <= 0
    ):

        raise RuntimeError(
            "No question tokens were identified "
            "in the input sequence."
        )

    if (
        condition
        == "audio_video"

        and stats[
            "audio"
        ]
        <= 0
    ):

        raise RuntimeError(
            "audio_video was requested but no "
            "audio tokens were identified."
        )

    if (
        condition
        == "transcription_video"

        and stats[
            "transcription"
        ]
        <= 0
    ):

        raise RuntimeError(
            "transcription_video was requested but no "
            "transcription tokens were identified."
        )


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

        output = (
            output[
                0
            ]
        )

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
):

    input_length = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    with torch.inference_mode():

        output = (
            model.generate(
                **inputs,

                max_new_tokens=
                    MAX_NEW_TOKENS,

                do_sample=
                    False,

                use_cache=
                    True,
            )
        )

    output = (
        normalize_generation_output(
            output
        )
    )

    # --------------------------------------------------------
    # EXTRACT GENERATED TOKENS
    # --------------------------------------------------------

    if (
        output.shape[-1]
        >= input_length
    ):

        prefix = (
            output[
                0,
                :input_length,
            ]
        )

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
                output[
                    0
                ]
            )

    else:

        generated_ids = (
            output[
                0
            ]
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
        .strip()
    )

    return (
        generated_ids_list,
        generated_text,
    )


# ============================================================
# ATTENTION ROW ANNOTATION
# ============================================================

def annotate_attention_rows(
    rows: list[
        dict[str, Any]
    ],
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
            generated_ids[
                step
            ]

            if (
                step
                < len(
                    generated_ids
                )
            )

            else None
        )

        if (
            token_id
            is not None
        ):

            token_text = (
                tokenizer.decode(
                    [
                        token_id
                    ],
                    skip_special_tokens=False,
                )
            )

        else:

            token_text = ""

        result.append(
            {
                **metadata,
                **row,

                "generated_token_id":
                    token_id,

                "generated_token":
                    token_text,
            }
        )

    return result


# ============================================================
# CSV WRITING
# ============================================================

def append_rows(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:

    if not rows:

        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = (
        path.exists()
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=list(
                    rows[
                        0
                    ].keys()
                ),
            )
        )

        if not exists:

            writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SINGLE ITEM ANALYSIS
# ============================================================

def analyse_single_item(
    model,
    processor,
    sample_position: int,
    video_id: int,
    question_index: int,
    question: str,
    video_path: Path,
    transcription: str,
    condition: str,
):

    prompt = build_prompt(
        question=
            question,

        transcription=
            transcription,

        condition=
            condition,
    )

    # --------------------------------------------------------
    # CREATE REAL GEMMA 4 MULTIMODAL INPUT
    # --------------------------------------------------------

    inputs = (
        prepare_gemma4_inputs(
            processor=
                processor,

            video_path=
                video_path,

            prompt=
                prompt,

            condition=
                condition,

            audio_dir=
                AUDIO_DIR,
        )
    )

    inputs = (
        move_inputs_to_device(
            inputs,
            model,
        )
    )

    print_input_diagnostics(
        inputs
    )

    # --------------------------------------------------------
    # BUILD SOURCE MASKS
    # --------------------------------------------------------

    masks = (
        build_modality_masks(
            processor=
                processor,

            inputs=
                inputs,

            prompt=
                prompt,

            question=
                question,

            transcription=
                transcription,

            condition=
                condition,
        )
    )

    stats = (
        modality_mask_stats(
            masks
        )
    )

    print(
        "    token masks:",
        stats,
    )

    validate_modality_masks(
        stats,
        condition,
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
        base_length=
            base_length,

        modality_masks=
            masks,
    )

    generated_ids = []
    generated_text = ""

    try:

        (
            generated_ids,
            generated_text,
        ) = run_generation(
            model=
                model,

            processor=
                processor,

            inputs=
                inputs,
        )

    finally:

        attention_rows = (
            RECORDER.stop()
        )

    if not attention_rows:

        raise RuntimeError(
            "Generation completed but no decoder "
            "attention rows were recorded."
        )

    metadata = {

        "model":
            MODEL_NAME,

        "model_id":
            MODEL_ID,

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
            len(
                generated_ids
            ),

        "video_tokens":
            stats[
                "video"
            ],

        "audio_tokens":
            stats[
                "audio"
            ],

        "transcription_tokens":
            stats[
                "transcription"
            ],

        "question_tokens":
            stats[
                "question"
            ],

        "other_text_tokens":
            stats[
                "other_text"
            ],
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
) -> None:

    if not detail_path.exists():

        return

    data = pd.read_csv(
        detail_path,
        encoding="utf-8-sig",
    )

    if data.empty:

        return

    numeric_columns = (
        SOURCE_SHARE_COLUMNS
        + RAW_ATTENTION_COLUMNS
    )

    # --------------------------------------------------------
    # BY GENERATED TOKEN
    # --------------------------------------------------------

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
        )[
            numeric_columns
        ]
        .mean()
    )

    token_summary.to_csv(
        output_dir
        / "attention_by_token.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BY LAYER
    # --------------------------------------------------------

    layer_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "layer",
            ],
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    layer_summary.to_csv(
        output_dir
        / "attention_by_layer.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BY VIDEO / SAMPLE POSITION
    # --------------------------------------------------------

    video_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "sample_position",
                "video_id",
            ],
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    video_summary.to_csv(
        output_dir
        / "attention_by_video.csv",
        index=False,
    )

    # --------------------------------------------------------
    # BY HEAD
    # --------------------------------------------------------

    head_summary = (
        data.groupby(
            [
                "model",
                "condition",
                "layer",
                "head",
            ],
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    head_summary.to_csv(
        output_dir
        / "attention_by_head.csv",
        index=False,
    )

    # --------------------------------------------------------
    # MODEL x CONDITION
    # --------------------------------------------------------

    model_summary = (
        data.groupby(
            [
                "model",
                "condition",
            ],
            as_index=False,
        )[
            numeric_columns
        ]
        .mean()
    )

    model_summary.to_csv(
        output_dir
        / "attention_summary.csv",
        index=False,
    )


# ============================================================
# ATTENTION PERCENTAGES
# ============================================================

def create_percentage_outputs(
    detail_path: Path,
    output_dir: Path,
) -> None:

    """
    Convert source_share values from [0, 1] to percentages.

    The resulting percentages are normalized over the original
    input sources:

        video
        audio
        transcription
        question
        other_text

    Therefore they sum to approximately 100% for each analyzed
    question.
    """

    if not detail_path.exists():

        return

    data = pd.read_csv(
        detail_path,
        encoding="utf-8-sig",
    )

    if data.empty:

        return

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
        )[
            SOURCE_SHARE_COLUMNS
        ]
        .mean()
    )

    rename = {

        "video_source_share":
            "video_pct",

        "audio_source_share":
            "audio_pct",

        "transcription_source_share":
            "transcription_pct",

        "question_source_share":
            "question_pct",

        "other_text_source_share":
            "other_text_pct",
    }

    percentages = (
        percentages.rename(
            columns=
                rename
        )
    )

    percentage_columns = (
        list(
            rename.values()
        )
    )

    percentages[
        percentage_columns
    ] = (
        percentages[
            percentage_columns
        ]
        * 100.0
    )

    percentages[
        "total_input_pct"
    ] = (
        percentages[
            percentage_columns
        ]
        .sum(
            axis=1
        )
    )

    percentages.to_csv(
        output_dir
        / "attention_percentages.csv",
        index=False,
    )

    # --------------------------------------------------------
    # GLOBAL MODEL x CONDITION
    # --------------------------------------------------------

    summary = (
        percentages.groupby(
            [
                "model",
                "condition",
            ],
            as_index=False,
        )[
            percentage_columns
        ]
        .mean()
    )

    summary.to_csv(
        output_dir
        / "attention_percentage_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PERCENTAGES BY VIDEO / SAMPLE
    # --------------------------------------------------------

    by_video = (
        percentages.groupby(
            [
                "model",
                "condition",
                "sample_position",
                "video_id",
            ],
            as_index=False,
        )[
            percentage_columns
        ]
        .mean()
    )

    by_video.to_csv(
        output_dir
        / "attention_percentage_by_video.csv",
        index=False,
    )

    print()
    print(
        "Attention percentages:"
    )

    print(
        summary
        .round(
            2
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# GPU CLEANUP
# ============================================================

def cleanup_gpu(
) -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


# ============================================================
# MAIN
# ============================================================

def main(
) -> None:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=VIDEOS_DIR,
    )

    parser.add_argument(
        "--transcriptions",
        type=Path,
        default=TRANSCRIPTIONS_FILE,
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--questions-per-video",
        type=int,
        default=0,
        help=(
            "0 = all questions for each selected video. "
            "N > 0 = first N questions only."
        ),
    )

    args = (
        parser.parse_args()
    )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    global AUDIO_DIR

    AUDIO_DIR = (
        output_dir
        / "audio_cache"
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

    summary_files = [

        output_dir
        / "attention_by_token.csv",

        output_dir
        / "attention_by_layer.csv",

        output_dir
        / "attention_by_video.csv",

        output_dir
        / "attention_by_head.csv",

        output_dir
        / "attention_summary.csv",

        output_dir
        / "attention_percentages.csv",

        output_dir
        / "attention_percentage_summary.csv",

        output_dir
        / "attention_percentage_by_video.csv",
    ]

    # --------------------------------------------------------
    # NEW RUN
    # --------------------------------------------------------

    for path in [
        detail_path,
        generations_path,
        status_path,
        *summary_files,
    ]:

        if path.exists():

            path.unlink()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "Discovering videos..."
    )

    all_videos = (
        discover_videos(
            args.videos_dir
        )
    )

    print(
        f"Found "
        f"{len(all_videos)} "
        f"videos."
    )

    missing_videos = [

        video_id

        for video_id
        in set(
            VIDEO_IDS
        )

        if (
            video_id
            not in all_videos
        )
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    (
        model,
        processor,
    ) = load_model()

    status_rows = []

    # --------------------------------------------------------
    # SAME 15 SAMPLE POSITIONS
    # --------------------------------------------------------

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
                f"[WARNING] "
                f"No questions "
                f"for video "
                f"{video_id}"
            )

            continue

        if (
            args.questions_per_video
            > 0
        ):

            video_questions = (
                video_questions[
                    :
                    args.questions_per_video
                ]
            )

        # ----------------------------------------------------
        # BOTH CONDITIONS
        # ----------------------------------------------------

        for condition in (
            CONDITIONS
        ):

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
                    f"{MODEL_NAME} | "
                    f"{condition} | "
                    f"sample "
                    f"{sample_position}/15 | "
                    f"video="
                    f"{video_id} | "
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
                            model=
                                model,

                            processor=
                                processor,

                            sample_position=
                                sample_position,

                            video_id=
                                video_id,

                            question_index=
                                question_index,

                            question=
                                question,

                            video_path=
                                video_path,

                            transcription=
                                transcription,

                            condition=
                                condition,
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
                                MODEL_NAME,

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
                        ][
                            :200
                        ],
                    )

                except Exception as error:

                    print(
                        "    -> ERROR:",
                        repr(
                            error
                        ),
                    )

                    status_rows.append(
                        {
                            "model":
                                MODEL_NAME,

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
                                repr(
                                    error
                                ),
                        }
                    )

                # Save status continuously.
                pd.DataFrame(
                    status_rows
                ).to_csv(
                    status_path,
                    index=False,
                )

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
    print(
        "=" * 80
    )

    print(
        "GEMMA 4 12B ATTENTION ANALYSIS COMPLETED"
    )

    print(
        "=" * 80
    )

    print(
        f"Results: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
