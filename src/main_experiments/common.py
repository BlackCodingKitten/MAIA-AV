import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

CAPTION_FOIL = Path("data/vsv/caption-foil/caption_foil.csv")
AUDIO_DIR = Path("data/input/audio")
MUTE_DIR = Path("data/input/mute")
VIDEO_DIR = Path("data/input/video")
TRANSCRIPTIONS = Path("data/input/transcription/transcriptions.csv")
OUTPUT_DIR = Path("data/output")

MODES = ("no_input", "only_audio", "only_video", "video_audio", "transcript_video")

SYSTEM = "Sei un assistente binario per un task di Visual Statement Verification. Rispondi esclusivamente con 0 oppure 1."

PROMPTS = {
    "no_input": (
        # "Non ti viene fornito alcun contenuto visivo o acustico. "
        "Basandoti esclusivamente sulle due alternative testuali, scegli.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "only_audio": (
        "Ascolta esclusivamente l'audio fornito."
        "Scegli quale delle due descrizioni è corretta sulla base delle informazioni ricavabili esclusivamente dall'audio.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "only_video": (
        "Osserva il video fornito"
        "Scegli quale delle due descrizioni è corretta sulla base esclusivamente delle informazioni visive disponibili.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "video_audio": (
        "Osserva il video e ascolta il relativo audio. "
        "Scegli quale delle due descrizioni è corretta sulla base dell'insieme delle informazioni visive e acustiche disponibili.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "transcript_video": (
        "Osserva il video fornito e la trascrizione del suo audio"
        "Scegli quale delle due descrizioni è corretta sulla base dell'insieme delle informazioni visive e testuali disponibili.\n\n"
        "Trascrizione dell'audio: {transcript}\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),
}


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def rows(limit=0):
    with CAPTION_FOIL.open(encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))
    return data[:limit] if limit else data


def transcriptions():
    with TRANSCRIPTIONS.open(encoding="utf-8-sig", newline="") as f:
        return {r["video_name"]: r["transcription"] for r in csv.DictReader(f, delimiter=";")}


def paths(row):
    stem = Path(row["video_name"]).stem
    return {
        "audio": AUDIO_DIR / f"{stem}.mp3",
        "mute": MUTE_DIR / f"{stem}.mp4",
        "video": VIDEO_DIR / f"{stem}.mp4",
    }


def prompt(row, mode, transcript=""):
    return PROMPTS[mode].format(
        answer1=row["answer1"],
        answer2=row["answer2"],
        transcript=transcript,
    )


def prediction(text):
    m = re.search(r"(?<!\d)[01](?!\d)", str(text))
    return int(m.group()) if m else None


def extract_audio(video):
    out = Path(tempfile.gettempdir()) / "maia_av_audio_cache" / f"{video.stem}.wav"
    out.parent.mkdir(exist_ok=True)
    if not out.exists():
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", str(video),
                "-vn", "-ac", "1", "-ar", "16000",
                str(out),
            ],
            check=True,
        )
    return out


FIELDS = [
    "id", "video_id", "video_name", "pool_id", "pool_item", "question_category",
    "question", "caption", "foil", "answer1", "answer2", "target",
    "model", "mode", "raw_model_output", "predicted_label",
    "is_correct", "result", "error",
]


def evaluate(model_name, infer, modes, limit=0, overwrite=False, unsupported=()):
    data, transcripts = rows(limit), transcriptions()

    for mode in modes:
        out_dir = OUTPUT_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{model_name}.csv"

        if overwrite and out.exists():
            out.unlink()

        done = set()
        if out.exists():
            with out.open(encoding="utf-8-sig", newline="") as f:
                done = {r["id"] for r in csv.DictReader(f)}

        with out.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if not out.stat().st_size:
                writer.writeheader()

            for i, row in enumerate(data, 1):
                if row["id"] in done:
                    continue

                transcript = transcripts.get(row["video_name"], "") if mode == "transcript_video" else ""
                raw, pred, error = "", None, ""

                if mode in unsupported:
                    result = "unsupported"
                else:
                    try:
                        raw = infer(mode, row, prompt(row, mode, transcript), paths(row))
                        pred = prediction(raw)
                        result = (
                            "correct" if pred == int(row["target"])
                            else "wrong" if pred is not None
                            else "invalid"
                        )
                    except Exception as e:
                        result, error = "error", f"{type(e).__name__}: {e}"

                writer.writerow({
                    **{k: row.get(k, "") for k in FIELDS},
                    "model": model_name,
                    "mode": mode,
                    "raw_model_output": raw,
                    "predicted_label": "" if pred is None else pred,
                    "is_correct": int(pred == int(row["target"])) if pred is not None else 0,
                    "result": result,
                    "error": error,
                })
                f.flush()
                print(f"[{model_name}] [{mode}] {i}/{len(data)} {row['id']} -> {result}")