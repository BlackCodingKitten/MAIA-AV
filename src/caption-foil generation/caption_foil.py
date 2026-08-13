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
EXPECTED_MACROAREAS = {"spaziale", "temporale", "causale"}
EXPECTED_ANNOTATORS = 4

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

    required = {"video_name", "principal_dimension", "question_text"}
    missing = required - set(questions.columns)

    if missing:
        raise ValueError(
            f"{QUESTIONS_FILE} non contiene le colonne richieste: {sorted(missing)}"
        )

    questions = questions[
        ["video_name", "principal_dimension", "question_text"]
    ].copy()

    if questions.isna().any().any():
        raise ValueError(
            f"{QUESTIONS_FILE} contiene valori mancanti nelle colonne richieste."
        )

    questions["video_name"] = questions["video_name"].map(normalize_video_name)
    questions["principal_dimension"] = (
        questions["principal_dimension"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    questions["question_text"] = questions["question_text"].astype(str).str.strip()

    unsupported = set(questions["principal_dimension"]) - EXPECTED_MACROAREAS
    if unsupported:
        raise ValueError(
            "Macroaree non supportate nel file delle domande: "
            f"{sorted(unsupported)}"
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
            "Sono presenti più domande per la stessa coppia video/macroarea:"
            f"{duplicates.to_string(index=False)}"
        )

    macroareas_by_video = questions.groupby("video_name")[
        "principal_dimension"
    ].agg(set)

    invalid_videos = macroareas_by_video[
        macroareas_by_video != EXPECTED_MACROAREAS
    ]

    if not invalid_videos.empty:
        details = "".join(
            f"{video}: {sorted(macroareas)}"
            for video, macroareas in invalid_videos.items()
        )
        raise ValueError(
            "Ogni video deve avere esattamente una domanda spaziale, "
            "una temporale e una causale."
            f"{details}"
        )

    return questions.rename(
        columns={"principal_dimension": "macroarea"}
    )
def load_human_answers():
    files = sorted(RAW_ANSWERS_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in {RAW_ANSWERS_DIR}"
        )

    if len(files) != EXPECTED_ANNOTATORS:
        raise ValueError(
            f"Attesi {EXPECTED_ANNOTATORS} file di annotatori in "
            f"{RAW_ANSWERS_DIR}, trovati {len(files)}: "
            f"{', '.join(file.name for file in files)}"
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
        df["macroarea"] = (
            df["macroarea"].astype(str).str.lower().str.strip()
        )
        df["annotator"] = file.stem.removeprefix("risposte_")

        unsupported = set(df["macroarea"]) - EXPECTED_MACROAREAS
        if unsupported:
            raise ValueError(
                f"{file} contiene macroaree non supportate: "
                f"{sorted(unsupported)}"
            )

        df = df.dropna(subset=["risposta"])
        df["risposta"] = df["risposta"].astype(str).str.strip()
        df = df[df["risposta"] != ""]

        duplicated = df.duplicated(
            ["video_name", "macroarea"],
            keep=False,
        )

        if duplicated.any():
            duplicates = df.loc[
                duplicated,
                ["video_name", "macroarea"],
            ]
            raise ValueError(
                f"{file} contiene più risposte per la stessa "
                "coppia video/macroarea:"
                f"{duplicates.to_string(index=False)}"
            )

        frames.append(
            df[["video_name", "macroarea", "annotator", "risposta"]]
        )

    return pd.concat(frames, ignore_index=True)
def prepare_source_data():
    questions = load_questions()
    answers = load_human_answers()

    question_keys = set(
        map(
            tuple,
            questions[["video_name", "macroarea"]].itertuples(
                index=False,
                name=None,
            ),
        )
    )

    for annotator, group in answers.groupby("annotator"):
        answer_keys = set(
            map(
                tuple,
                group[["video_name", "macroarea"]].itertuples(
                    index=False,
                    name=None,
                ),
            )
        )

        missing = question_keys - answer_keys
        extra = answer_keys - question_keys

        if missing or extra:
            message = [f"Risposte non allineate per l'annotatore {annotator}."]

            if missing:
                message.append(
                    "Domande senza risposta: "
                    + ", ".join(
                        f"{video}/{macroarea}"
                        for video, macroarea in sorted(missing)
                    )
                )

            if extra:
                message.append(
                    "Risposte senza domanda corrispondente: "
                    + ", ".join(
                        f"{video}/{macroarea}"
                        for video, macroarea in sorted(extra)
                    )
                )

            raise ValueError("".join(message))

    data = answers.merge(
        questions,
        on=["video_name", "macroarea"],
        how="left",
        validate="many_to_one",
    )

    data["video_id"] = data["video_name"].str.removesuffix(".mp4")
    data["pool_id"] = data["video_id"] + "_" + data["macroarea"]

    data = data.sort_values(
        ["video_id", "macroarea", "annotator"],
        kind="stable",
    ).reset_index(drop=True)

    pool_sizes = data.groupby("pool_id").size()
    invalid_pools = pool_sizes[pool_sizes != EXPECTED_ANNOTATORS]

    if not invalid_pools.empty:
        raise ValueError(
            "Ogni pool deve contenere esattamente "
            f"{EXPECTED_ANNOTATORS} risposte."
            f"{invalid_pools.to_string()}"
        )

    data["pool_item"] = data.groupby("pool_id").cumcount() + 1
    data["id"] = (
        data["pool_id"]
        + "_"
        + data["pool_item"].astype(str).str.zfill(2)
    )

    return data
def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return {}

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        encoding="utf-8-sig",
    )

    required = {
        "id",
        "question",
        "annotator",
        "human_answer",
        "caption",
        "foil",
    }
    missing = required - set(checkpoint.columns)

    if missing:
        raise ValueError(
            f"{CHECKPOINT_FILE} non contiene le colonne richieste: "
            f"{sorted(missing)}"
        )

    checkpoint = checkpoint.drop_duplicates("id", keep="last")

    return {
        row["id"]: row
        for _, row in checkpoint.iterrows()
    }
def generate_caption_foil_pairs(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    generated_rows = []

    total = len(data)

    for index, row in data.iterrows():
        item_id = row["id"]
        saved = checkpoint.get(item_id)

        checkpoint_is_valid = (
            saved is not None
            and str(saved["question"]) == str(row["question_text"])
            and str(saved["annotator"]) == str(row["annotator"])
            and str(saved["human_answer"]) == str(row["risposta"])
            and pd.notna(saved["caption"])
            and pd.notna(saved["foil"])
        )

        if checkpoint_is_valid:
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

        if saved is not None:
            print(
                f"[{index + 1}/{total}] {item_id} checkpoint non coerente: "
                "rigenerazione"
            )
        else:
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
            encoding="utf-8-sig",
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
    targets = {}

    for category, indices in df.groupby("question_category").groups.items():
        indices = list(indices)
        n = len(indices)

        labels = [0] * (n // 2) + [1] * (n // 2)

        if n % 2:
            labels.append(rng.randint(0, 1))

        rng.shuffle(labels)

        for idx, target in zip(indices, labels):
            targets[idx] = target

    df["target"] = df.index.map(targets).astype(int)

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
        encoding="utf-8-sig",
    )

    print(f"File finale salvato in: {OUTPUT_FILE}")
    print("Distribuzione target:")
    print(
        final_df.groupby(
            ["question_category", "target"]
        ).size()
    )


if __name__ == "__main__":
    main()