from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
from PIL import Image


DEFAULT_NUM_FRAMES = 32
AUDIO_SAMPLE_RATE = 16000


def load_video_frames(path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> list[Image.Image]:
    """
    Legge il video con OpenCV e restituisce frame RGB campionati uniformemente.

    Il file video viene decodificato qui, prima di essere passato al modello.
    In questo modo i modelli non dipendono da TorchCodec/torchvision per la
    lettura degli MP4.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Numero di frame non valido per: {path}")

    count = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, count, dtype=int)

    images = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))

    cap.release()

    if not images:
        raise RuntimeError(f"Nessun frame letto da: {path}")

    return images


def close_images(images) -> None:
    for image in images or []:
        try:
            image.close()
        except Exception:
            pass


def ensure_wav(path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> Path:
    """
    Converte un file audio qualsiasi in WAV mono PCM a 16 kHz.
    È utile soprattutto per Qwen, così il loader audio riceve sempre un formato
    semplice e stabile.
    """
    path = Path(path).resolve()

    key = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    out = (
        Path(tempfile.gettempdir())
        / "maia_av_audio_cache"
        / f"{path.stem}_{key}_{sample_rate}.wav"
    )

    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        return out

    subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
    )

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Conversione audio fallita: {path}")

    return out


def load_audio_waveform(
    path: Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    max_seconds: float | None = None,
) -> np.ndarray:
    """
    Restituisce waveform mono float32 al sample rate richiesto.
    """
    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=True,
        duration=max_seconds,
    )

    if audio.size == 0:
        raise RuntimeError(f"Audio vuoto: {path}")

    return audio.astype(np.float32, copy=False)
