import argparse
import csv
import re
import subprocess
import tempfile

from pathlib import Path


CAPTION_FOIL = Path(
    "data/vsv/caption-foil/caption_foil.csv"
)

AUDIO_DIR = Path("data/input/audio")
MUTE_DIR = Path("data/input/mute")
VIDEO_DIR = Path("data/input/video")

TRANSCRIPTIONS = Path(
    "data/input/transcription/transcriptions.csv"
)

OUTPUT_DIR = Path("data/output")


MODES = (
    "no_input",
    "only_audio",
    "only_video",
    "only_transcription",
    "video_audio",
    "transcript_video",
)


SYSTEM = (
    "Sei un assistente binario per un task di "
    "Visual Statement Verification. "
    "Rispondi esclusivamente con 0 oppure 1."
)


PROMPTS = {
    "no_input": (
        "Basandoti esclusivamente sulle due alternative testuali, scegli.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "only_audio": (
        "Ascolta esclusivamente l'audio fornito. "
        "Scegli quale delle due descrizioni è corretta sulla base "
        "delle informazioni ricavabili esclusivamente dall'audio.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "only_video": (
        "Osserva il video fornito. "
        "Scegli quale delle due descrizioni è corretta sulla base "
        "esclusivamente delle informazioni visive disponibili.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "only_transcription": (
        "Leggi esclusivamente la trascrizione dell'audio fornita. "
        "Scegli quale delle due descrizioni è corretta sulla base "
        "esclusivamente delle informazioni testuali disponibili "
        "nella trascrizione.\n\n"
        "Trascrizione dell'audio: {transcript}\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "video_audio": (
        "Osserva il video e ascolta il relativo audio. "
        "Scegli quale delle due descrizioni è corretta sulla base "
        "dell'insieme delle informazioni visive e acustiche disponibili.\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),

    "transcript_video": (
        "Osserva il video fornito e la trascrizione del suo audio. "
        "Scegli quale delle due descrizioni è corretta sulla base "
        "dell'insieme delle informazioni visive e testuali disponibili.\n\n"
        "Trascrizione dell'audio: {transcript}\n\n"
        "0: {answer1}\n"
        "1: {answer2}\n\n"
        "Rispondi esclusivamente con 0 oppure 1."
    ),
}


FIELDS = [
    "id",
    "video_id",
    "video_name",
    "pool_id",
    "pool_item",
    "question_category",
    "question",
    "caption",
    "foil",
    "answer1",
    "answer2",
    "target",
    "model",
    "mode",
    "raw_model_output",
    "predicted_label",
    "is_correct",
    "result",
    "error",
]


COMPLETED_RESULTS = {
    "correct",
    "wrong",
    "invalid",
    "unsupported",
}


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def rows(limit=0):
    with CAPTION_FOIL.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        data = list(
            csv.DictReader(file)
        )

    return (
        data[:limit]
        if limit
        else data
    )


def transcriptions():
    with TRANSCRIPTIONS.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return {
            row["video_name"]: row["transcription"]
            for row in csv.DictReader(
                file,
                delimiter=";",
            )
        }


def paths(row):
    stem = Path(
        row["video_name"]
    ).stem

    return {
        "audio": AUDIO_DIR / f"{stem}.mp3",
        "mute": MUTE_DIR / f"{stem}.mp4",
        "video": VIDEO_DIR / f"{stem}.mp4",
    }


def prompt(
    row,
    mode,
    transcript="",
):
    return PROMPTS[mode].format(
        answer1=row["answer1"],
        answer2=row["answer2"],
        transcript=transcript,
    )


def prediction(text):
    match = re.search(
        r"(?<!\d)[01](?!\d)",
        str(text),
    )

    return (
        int(match.group())
        if match
        else None
    )


def extract_audio(video):
    out = (
        Path(tempfile.gettempdir())
        / "maia_av_audio_cache"
        / f"{video.stem}.wav"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not out.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(out),
            ],
            check=True,
        )

    return out


def clean_existing_output(path):
    if not path.exists():
        return set()

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        existing = list(
            csv.DictReader(file)
        )

    completed = {}

    for row in existing:
        row_id = row.get("id")

        if (
            row_id
            and row.get("result") in COMPLETED_RESULTS
        ):
            completed[row_id] = row

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for row in completed.values():
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in FIELDS
                }
            )

    return set(
        completed.keys()
    )


def evaluate(
    model_name,
    infer,
    modes,
    limit=0,
    overwrite=False,
    unsupported=(),
):
    data = rows(limit)
    transcripts = transcriptions()

    for mode in modes:
        out_dir = (
            OUTPUT_DIR
            / mode
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        out = (
            out_dir
            / f"{model_name}.csv"
        )

        if overwrite and out.exists():
            out.unlink()

        done = clean_existing_output(
            out
        )

        with out.open(
            "a",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=FIELDS,
            )

            if out.stat().st_size == 0:
                writer.writeheader()

            for index, row in enumerate(
                data,
                start=1,
            ):
                if row["id"] in done:
                    continue

                raw = ""
                pred = None
                error = ""

                transcript = ""

                if mode in {
                    "only_transcription",
                    "transcript_video",
                }:
                    if row["video_name"] not in transcripts:
                        result = "error"
                        error = (
                            "Trascrizione non trovata per "
                            f"{row['video_name']}"
                        )

                        writer.writerow(
                            build_output_row(
                                row,
                                model_name,
                                mode,
                                raw,
                                pred,
                                result,
                                error,
                            )
                        )

                        file.flush()

                        print(
                            f"[{model_name}] "
                            f"[{mode}] "
                            f"{index}/{len(data)} "
                            f"{row['id']} -> error"
                        )

                        continue

                    transcript = transcripts[
                        row["video_name"]
                    ]

                if mode in unsupported:
                    result = "unsupported"

                else:
                    try:
                        raw = infer(
                            mode,
                            row,
                            prompt(
                                row,
                                mode,
                                transcript,
                            ),
                            paths(row),
                        )

                        pred = prediction(
                            raw
                        )

                        if pred is None:
                            result = "invalid"

                        elif pred == int(
                            row["target"]
                        ):
                            result = "correct"

                        else:
                            result = "wrong"

                    except Exception as exc:
                        result = "error"

                        error = (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                writer.writerow(
                    build_output_row(
                        row,
                        model_name,
                        mode,
                        raw,
                        pred,
                        result,
                        error,
                    )
                )

                file.flush()

                print(
                    f"[{model_name}] "
                    f"[{mode}] "
                    f"{index}/{len(data)} "
                    f"{row['id']} -> {result}"
                )


def build_output_row(
    row,
    model_name,
    mode,
    raw,
    pred,
    result,
    error,
):
    target = int(
        row["target"]
    )

    return {
        **{
            field: row.get(
                field,
                "",
            )
            for field in FIELDS
        },

        "model": model_name,
        "mode": mode,
        "raw_model_output": raw,

        "predicted_label": (
            ""
            if pred is None
            else pred
        ),

        "is_correct": (
            int(pred == target)
            if pred is not None
            else 0
        ),

        "result": result,
        "error": error,
    }