from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from PIL import Image


DEFAULT_NUM_FRAMES = 32
AUDIO_SAMPLE_RATE = 16000


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _duration_seconds(path: Path) -> float:
    """
    Legge la durata del file con ffprobe.
    Non richiede OpenCV, TorchCodec, torchvision o decord.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        duration = float(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(
            f"Durata video non leggibile per {path}: {result.stdout!r}"
        ) from error

    if duration <= 0:
        raise RuntimeError(f"Durata video non valida per: {path}")

    return duration


def load_video_frames(
    path: Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> list[Image.Image]:
    """
    Estrae frame RGB con ffmpeg.

    Il video viene decodificato PRIMA di essere passato al modello.
    Questo elimina la dipendenza da cv2/TorchCodec per la lettura degli MP4.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    duration = _duration_seconds(path)

    stat = path.stat()
    key_source = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{num_frames}|{duration:.6f}"
    )
    key = hashlib.sha1(
        key_source.encode("utf-8")
    ).hexdigest()[:16]

    cache_dir = (
        Path(tempfile.gettempdir())
        / "maia_av_frame_cache"
        / key
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        cache_dir / f"frame_{i:03d}.jpg"
        for i in range(1, num_frames + 1)
    ]

    if not all(p.exists() and p.stat().st_size > 0 for p in expected):
        for old in cache_dir.glob("frame_*.jpg"):
            old.unlink(missing_ok=True)

        # Campionamento uniforme sull'intera durata.
        output_fps = num_frames / duration

        command = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-i", str(path),
            "-an",
            "-vf", f"fps={output_fps:.12f}",
            "-frames:v", str(num_frames),
            "-q:v", "2",
            str(cache_dir / "frame_%03d.jpg"),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg non riesce a estrarre i frame da {path}:\n"
                f"{result.stderr}"
            )

    frame_paths = sorted(cache_dir.glob("frame_*.jpg"))

    if not frame_paths:
        raise RuntimeError(f"Nessun frame estratto da: {path}")

    images: list[Image.Image] = []

    for frame_path in frame_paths[:num_frames]:
        with Image.open(frame_path) as image:
            images.append(
                image.convert("RGB").copy()
            )

    if not images:
        raise RuntimeError(f"Nessun frame RGB caricato da: {path}")

    return images


def close_images(images) -> None:
    for image in images or []:
        try:
            image.close()
        except Exception:
            pass


def ensure_wav(
    path: Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path:
    """
    Converte l'audio in WAV PCM mono 16 kHz con ffmpeg.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    stat = path.stat()
    key_source = (
        f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{sample_rate}"
    )
    key = hashlib.sha1(
        key_source.encode("utf-8")
    ).hexdigest()[:16]

    out = (
        Path(tempfile.gettempdir())
        / "maia_av_audio_cache"
        / f"{path.stem}_{key}.wav"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > 0:
        return out

    result = subprocess.run(
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg non riesce a convertire l'audio {path}:\n"
            f"{result.stderr}"
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
    Restituisce una waveform mono float32.
    """
    wav = ensure_wav(path, sample_rate)

    audio, _ = librosa.load(
        str(wav),
        sr=sample_rate,
        mono=True,
        duration=max_seconds,
    )

    if audio.size == 0:
        raise RuntimeError(f"Audio vuoto: {path}")

    return audio.astype(np.float32, copy=False)
