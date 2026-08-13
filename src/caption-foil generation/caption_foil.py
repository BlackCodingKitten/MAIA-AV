import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI, APIError, AuthenticationError, RateLimitError


MODEL = "gpt-4o-2024-08-06"

RAW_ANSWERS_DIR = Path("data/vsv/raw_human_answers")
QUESTIONS_FILE = Path("data/vsv/question_classification.csv")
OUTPUT_DIR = Path("data/vsv/caption-foil")

CHECKPOINT_FILE = OUTPUT_DIR / "caption_foil_checkpoint.csv"
OUTPUT_FILE = OUTPUT_DIR / "caption_foil.csv"

RANDOM_SEED = 42

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def clean_generation(text, prefix):
    text = text.strip()
    if text.startswith(f"{prefix}: "):
        text = text[3:]
    elif text.startswith(f"{prefix}:"):
        text = text[2:]
    return text.strip()


def call_openai(messages, max_tokens=100, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            if attempt == retries - 1:
                raise
            time.sleep(30)

        except AuthenticationError:
            raise RuntimeError(
                "Errore di autenticazione: controlla OPENAI_API_KEY."
            )

        except APIError:
            if attempt == retries - 1:
                raise
            time.sleep(10)


def generate_caption(question, answer):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that creates statements in Italian about videos."
                "Given a question Q and an answer A concerning a video, you must create a statement S based on A."
                "While generating S, try not to alter the words composing A."
                "If A includes first-person verbs or phrases (e.g., 'I think,' 'I believe'), rephrase S to be impersonal, avoiding a first-person perspective."
                "The statement should be a concise, declarative sentence."
                "Here are some examples:\n\n"
                "   Q1: Cosa sta facendo il gatto sul tavolo?\n"
                "   A1: Non c'è nessun felino nel video.\n"
                "   S1: Nel video non c'è nessun felino sul tavolo.\n\n"
                "   Q2: Da quanto tempo i ragazzi stanno ballando la salsa?\n"
                "   A2: Probabilmente le persone nel filmato ballano da qualche ora ma è difficile dirlo con sicurezza in base a ciò che si vede.\n"
                "   S2: Nel filmato probabilmente le persone stanno ballando la salsa da qualche ora, ma è difficile stabilirlo con sicurezza.\n\n"
                "   Q3: Chi è il padre della bambina che corre?\n"
                "   A3: Non è specificato\n"
                "   S3: Dal video non è deducibile chi sia il padre della bambina che corre\n\n"
                "   Q4: Cosa succederebbe se venisse a piovere?\n"
                "   A4: Penso che tutto l'evento finirebbe all'istante.\n"
                "   S4: Se venisse a piovere probabilmente l'evento finirebbe all'istante.\n\n"
            ),
        },
        {
            "role": "user",
            "content": f"Q: {question}\nA: {answer}",
        },
        {
            "role": "assistant",
            "content": "S:",
        },
    ]

    return clean_generation(call_openai(messages), "S")


def generate_foil(caption, macroarea):
    macroarea = macroarea.lower().strip()

    prompts = {
        "spaziale": (
            "Given an Italian caption (C) regarding the position or location of someone or something, "
            "your task is to create its foil (F) by changing only the spatial information.\n"
            "Don't add other information respect to what is stated in C.\n"
            "Here is an example to guide you:\n"
            "C: La donna nel video si trova in un campo di papaveri.\n"
            "F: La donna nel video si trova in una scuola"
        ),
        "temporale": (
            "Given an Italian caption (C) regarding events and when they happen, "
            "your task is to create its foil (F) by changing only the temporal information.\n"
            "Don't add other information respect to what is stated in C.\n"
            "Here is an example to guide you:\n"
            "C: Mentre la donna scriveva il libro, il bimbo ha iniziato a piangere.\n"
            "F: Dopo che la donna è andata in cucina, il bimbo ha iniziato a piangere."
        ),
        "causale": (
            "Given an Italian caption (C) regarding the causes or the effects of events, "
            "your task is to create its foil (F) by changing the causes or the effect of the main event.\n"
            "Don't add other information respect to what is stated in C.\n"
            "Here is an example to guide you:\n"
            "C: L'uomo ha iniziato a correre perchè pioveva a dirotto.\n"
            "F: L'uomo ha iniziato a correre perchè aveva paura."
        ),
    }

    if macroarea not in prompts:
        raise ValueError(
            f"Macroarea non supportata: {macroarea}. "
            f"Valori ammessi: {', '.join(prompts)}"
        )

    messages = [
        {
            "role": "system",
            "content": "You are an assistant designed to create foils based on captions.",
        },
        {
            "role": "user",
            "content": f"{prompts[macroarea]}\nC: {caption}\n",
        },
    ]

    return clean_generation(call_openai(messages), "F")


def normalize_video_name(value):
    value = str(value).strip()

    match = re.search(r"(\d+)", value)
    if not match:
        raise ValueError(f"Impossibile ricavare il numero del video da: {value}")

    return f"video{int(match.group(1)):03d}.mp4"


def load_questions():
    questions = pd.read_csv(
        QUESTIONS_FILE,
        sep=";",
        encoding="utf-8-sig",
    )

    questions["video_name"] = questions["video_name"].map(normalize_video_name)
    questions["principal_dimension"] = (
        questions["principal_dimension"].str.lower().str.strip()
    )

    duplicated = questions.duplicated(
        ["video_name", "principal_dimension"],
        keep=False,
    )

    if duplicated.any():
        duplicates = questions.loc[
            duplicated,
            ["video_name", "principal_dimension", "question_text"],
        ]
        raise ValueError(
            "Sono presenti più domande per la stessa coppia video/macroarea:\n"
            f"{duplicates.to_string(index=False)}"
        )

    return questions[
        ["video_name", "principal_dimension", "question_text"]
    ].rename(columns={"principal_dimension": "macroarea"})


def load_human_answers():
    files = sorted(RAW_ANSWERS_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {RAW_ANSWERS_DIR}"
        )

    frames = []

    for file in files:
        df = pd.read_csv(
            file,
            sep=";",
            encoding="utf-8-sig",
        )

        # Supporta sia "risposta" sia "risposte".
        if "risposta" not in df.columns and "risposte" in df.columns:
            df = df.rename(columns={"risposte": "risposta"})

        required = {"titolo_video", "macroarea", "risposta"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{file} non contiene le colonne richieste: {sorted(missing)}"
            )

        df = df[["titolo_video", "macroarea", "risposta"]].copy()
        df["video_name"] = df["titolo_video"].map(normalize_video_name)
        df["macroarea"] = df["macroarea"].str.lower().str.strip()
        df["annotator"] = file.stem.removeprefix("risposte_")

        frames.append(df)

    answers = pd.concat(frames, ignore_index=True)

    answers = answers.dropna(subset=["risposta"])
    answers["risposta"] = answers["risposta"].astype(str).str.strip()
    answers = answers[answers["risposta"] != ""]

    return answers[
        ["video_name", "macroarea", "annotator", "risposta"]
    ]


def prepare_source_data():
    questions = load_questions()
    answers = load_human_answers()

    data = answers.merge(
        questions,
        on=["video_name", "macroarea"],
        how="left",
        validate="many_to_one",
    )

    missing_questions = data[data["question_text"].isna()]

    if not missing_questions.empty:
        missing = (
            missing_questions[["video_name", "macroarea"]]
            .drop_duplicates()
            .to_string(index=False)
        )
        raise ValueError(
            "Non trovo la domanda corrispondente per:\n"
            f"{missing}"
        )

    data["video_id"] = data["video_name"].str.removesuffix(".mp4")
    data["pool_id"] = data["video_id"] + "_" + data["macroarea"]

    data = data.sort_values(
        ["video_id", "macroarea", "annotator"],
        kind="stable",
    ).reset_index(drop=True)

    data["pool_item"] = data.groupby("pool_id").cumcount() + 1
    data["id"] = (
        data["pool_id"]
        + "_"
        + data["pool_item"].astype(str).str.zfill(2)
    )

    return data


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        return {
            row["id"]: row
            for _, row in checkpoint.iterrows()
        }

    return {}


def generate_caption_foil_pairs(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    generated_rows = []

    total = len(data)

    for index, row in data.iterrows():
        item_id = row["id"]

        if item_id in checkpoint:
            saved = checkpoint[item_id]

            generated_rows.append({
                "id": item_id,
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "pool_id": row["pool_id"],
                "pool_item": row["pool_item"],
                "question_category": row["macroarea"],
                "question": row["question_text"],
                "annotator": row["annotator"],
                "human_answer": row["risposta"],
                "caption": saved["caption"],
                "foil": saved["foil"],
            })

            print(f"[{index + 1}/{total}] {item_id} già presente: skip")
            continue

        print(f"[{index + 1}/{total}] {item_id}")

        caption = generate_caption(
            row["question_text"],
            row["risposta"],
        )

        foil = generate_foil(
            caption,
            row["macroarea"],
        )

        generated_rows.append({
            "id": item_id,
            "video_id": row["video_id"],
            "video_name": row["video_name"],
            "pool_id": row["pool_id"],
            "pool_item": row["pool_item"],
            "question_category": row["macroarea"],
            "question": row["question_text"],
            "annotator": row["annotator"],
            "human_answer": row["risposta"],
            "caption": caption,
            "foil": foil,
        })

        pd.DataFrame(generated_rows).to_csv(
            CHECKPOINT_FILE,
            index=False,
        )

    return pd.DataFrame(generated_rows)


def randomize_caption_foil_order(df):
    """
    Randomizza la posizione caption/foil senza introdurre un bias di posizione.

    Il bilanciamento viene fatto separatamente per macroarea:
    circa metà delle righe ha la caption in answer1 (target=0),
    l'altra metà ha la caption in answer2 (target=1).
    """
    rng = random.Random(RANDOM_SEED)

    df = df.copy()
    targets = pd.Series(index=df.index, dtype="int64")

    for _, indices in df.groupby("question_category").groups.items():
        indices = list(indices)
        n = len(indices)

        labels = [0] * (n // 2) + [1] * (n // 2)

        if n % 2:
            labels.append(rng.randint(0, 1))

        rng.shuffle(labels)

        for idx, target in zip(indices, labels):
            targets.loc[idx] = target

    df["target"] = targets.astype(int)

    df["answer1"] = df.apply(
        lambda row: row["caption"] if row["target"] == 0 else row["foil"],
        axis=1,
    )

    df["answer2"] = df.apply(
        lambda row: row["foil"] if row["target"] == 0 else row["caption"],
        axis=1,
    )

    return df


def main():
    data = prepare_source_data()

    print(
        f"Trovate {len(data)} risposte umane in "
        f"{data['pool_id'].nunique()} pool."
    )

    pool_sizes = data.groupby("pool_id").size()

    print(
        "Dimensione pool: "
        f"min={pool_sizes.min()}, "
        f"max={pool_sizes.max()}, "
        f"media={pool_sizes.mean():.2f}"
    )

    generated = generate_caption_foil_pairs(data)
    final_df = randomize_caption_foil_order(generated)

    columns = [
        "id",
        "video_id",
        "video_name",
        "pool_id",
        "pool_item",
        "question_category",
        "question",
        "annotator",
        "human_answer",
        "caption",
        "foil",
        "answer1",
        "answer2",
        "target",
    ]

    final_df[columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nFile finale salvato in: {OUTPUT_FILE}")
    print("\nDistribuzione target:")
    print(
        final_df.groupby(
            ["question_category", "target"]
        ).size()
    )


if __name__ == "__main__":
    main()