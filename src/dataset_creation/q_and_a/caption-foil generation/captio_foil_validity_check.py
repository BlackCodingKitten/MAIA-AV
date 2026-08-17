from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/vsv/caption-foil/caption_foil.csv")

OUTPUT_DIR = Path("data/vsv/caption-foil/validation")

STRUCTURAL_OUTPUT = OUTPUT_DIR / "caption_foil_structural_validation.csv"
FINAL_OUTPUT = OUTPUT_DIR / "caption_foil_validation.csv"

DEFAULT_MODEL = "gpt-4o-2024-08-06"

VALID_CATEGORIES = {
    "causale",
    "temporale",
    "spaziale",
}

REQUIRED_COLUMNS = {
    "id",
    "question_category",
    "caption",
    "foil",
}

MAX_RETRIES = 5


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY non trovata.\n"
            "Imposta la variabile d'ambiente prima di eseguire lo script."
        )

    return OpenAI()


# ============================================================
# UTILITY
# ============================================================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_category(category: str) -> str:
    return clean_text(category).lower()


def normalize_structural_label(text: str) -> str:
    """
    Normalizza le possibili variazioni dell'output del modello.
    Restituisce:
        correct
        not correct
        invalid
    """

    value = clean_text(text).lower()

    if value.startswith("output:"):
        value = value[len("output:"):].strip()

    value = value.rstrip(".").strip()

    # Controllare prima "not correct", perché contiene "correct".
    if value == "not correct":
        return "not correct"

    if value == "correct":
        return "correct"

    return "invalid"


def normalize_nli_label(text: str) -> str:
    """
    Restituisce:
        Contradiction
        Neutral
        Entailment
        Invalid
    """

    value = clean_text(text)

    if value.lower().startswith("output:"):
        value = value[len("output:"):].strip()

    value = value.rstrip(".").strip().lower()

    if value == "contradiction":
        return "Contradiction"

    if value == "neutral":
        return "Neutral"

    if value == "entailment":
        return "Entailment"

    return "Invalid"


def call_model(
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
) -> str:

    last_error = None

    for attempt in range(max_retries):

        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=prompt,
                temperature=0,
                max_output_tokens=16,
            )

            return response.output_text.strip()

        except Exception as exc:
            last_error = exc

            wait_time = min(2 ** attempt, 30)

            print(
                f"\nAPI error "
                f"(tentativo {attempt + 1}/{max_retries}): {exc}"
            )

            if attempt < max_retries - 1:
                print(f"Nuovo tentativo tra {wait_time}s...")
                time.sleep(wait_time)

    raise RuntimeError(
        f"Chiamata API fallita dopo {max_retries} tentativi."
    ) from last_error


# ============================================================
# STRUCTURAL CHECK PROMPTS
# ============================================================

STRUCTURAL_SYSTEM_PROMPT = (
    "You are an assistant designed to validate the correctness "
    "of foils based on captions. "
    "Follow the requested semantic category strictly. "
    "Return only 'correct' or 'not correct'."
)


def get_structural_prompt(
    caption: str,
    foil: str,
    category: str,
) -> str:

    category = normalize_category(category)

    # --------------------------------------------------------
    # CAUSAL
    # --------------------------------------------------------

    if category == "causale":

        return (
            "Given an Italian caption (C) dealing with causal "
            "relations between events and its foil (F), your task "
            "is to assess whether F is a valid causal foil of C.\n\n"

            "To be valid, F must modify the cause, the effect, or "
            "the causal relation expressed in C while preserving "
            "the remaining relevant content as much as possible.\n"

            "The difference between C and F must therefore concern "
            "causal information rather than an unrelated entity, "
            "location, temporal relation, or event.\n\n"

            "If F is a valid causal foil, output 'correct'. "
            "Otherwise output 'not correct'.\n\n"

            "Example:\n"
            "C: Il ragazzo è inciampato perché sul pavimento "
            "c'era una buca.\n"
            "F: Il ragazzo è inciampato perché aveva i lacci "
            "sciolti.\n"
            "Output: correct\n\n"

            f"C: {caption}\n"
            f"F: {foil}\n"
            "Output:"
        )

    # --------------------------------------------------------
    # TEMPORAL
    # --------------------------------------------------------

    if category == "temporale":

        return (
            "Given an Italian caption (C) dealing with temporal "
            "information about events and its foil (F), your task "
            "is to assess whether F is a valid temporal foil of C.\n\n"

            "To be valid, F must modify the temporal information "
            "expressed in C, such as the ordering of events, "
            "before/after relations, the moment at which an event "
            "occurs, or another temporally relevant relation.\n"

            "The remaining relevant semantic content should be "
            "preserved as much as possible.\n\n"

            "If F is a valid temporal foil, output 'correct'. "
            "Otherwise output 'not correct'.\n\n"

            "Example:\n"
            "C: Nel video la donna esce prima che il ragazzo "
            "entri in casa.\n"
            "F: Nel video la donna esce dopo che il ragazzo "
            "è entrato in casa.\n"
            "Output: correct\n\n"

            f"C: {caption}\n"
            f"F: {foil}\n"
            "Output:"
        )

    # --------------------------------------------------------
    # SPATIAL
    # --------------------------------------------------------

    if category == "spaziale":

        return (
            "Given an Italian caption (C) dealing with spatial "
            "information and its foil (F), your task is to assess "
            "whether F is a valid spatial foil of C.\n\n"

            "The spatial category considered here includes both "
            "locations and spatial relations, including cases in "
            "which spatial information refers to a specific moment "
            "or phase of the video.\n\n"

            "To be valid, F must modify spatial information "
            "expressed in C, such as the location, position, "
            "direction, or spatial relation of a person, object, "
            "or event.\n"

            "Examples of relevant contrasts include inside/outside, "
            "above/below, in front of/behind, near/far, different "
            "locations, or different positions reached during "
            "the video.\n"

            "The remaining relevant semantic content should be "
            "preserved as much as possible.\n\n"

            "If F is a valid spatial foil, output 'correct'. "
            "Otherwise output 'not correct'.\n\n"

            "Example 1:\n"
            "C: Alla fine del video, il ragazzo si trova dietro "
            "il bancone.\n"
            "F: Alla fine del video, il ragazzo si trova davanti "
            "al bancone.\n"
            "Output: correct\n\n"

            "Example 2:\n"
            "C: La ragazza si trova a terra sulla spiaggia.\n"
            "F: La ragazza si trova a terra nel parco.\n"
            "Output: correct\n\n"

            f"C: {caption}\n"
            f"F: {foil}\n"
            "Output:"
        )

    raise ValueError(
        f"Categoria non supportata: {category}"
    )


def structural_validation(
    client: OpenAI,
    model: str,
    caption: str,
    foil: str,
    category: str,
) -> str:

    prompt = get_structural_prompt(
        caption=caption,
        foil=foil,
        category=category,
    )

    raw_output = call_model(
        client=client,
        model=model,
        instructions=STRUCTURAL_SYSTEM_PROMPT,
        prompt=prompt,
    )

    return normalize_structural_label(raw_output)


# ============================================================
# NLI VALIDATION
# ============================================================

NLI_SYSTEM_PROMPT = (
    "You are an NLI classifier. "
    "Return only one of the following labels: "
    "Entailment, Contradiction, Neutral."
)


def nli_validation(
    client: OpenAI,
    model: str,
    caption: str,
    foil: str,
) -> str:

    prompt = (
        "Your task is to determine the natural language inference "
        "(NLI) relationship between S1 and S2.\n\n"

        "The possible labels are:\n"
        "- Entailment: S2 logically follows from S1.\n"
        "- Contradiction: S2 contradicts S1.\n"
        "- Neutral: S1 and S2 are related, but S2 neither follows "
        "from nor contradicts S1.\n\n"

        "Provide only one label as output: "
        "Entailment, Contradiction, or Neutral.\n\n"

        "Example:\n"
        "S1: The capital of France is Paris.\n"
        "S2: The capital of France is Berlin.\n"
        "Output: Contradiction\n\n"

        f"S1: {caption}\n"
        f"S2: {foil}\n"
        "Output:"
    )

    raw_output = call_model(
        client=client,
        model=model,
        instructions=NLI_SYSTEM_PROMPT,
        prompt=prompt,
    )

    return normalize_nli_label(raw_output)


# ============================================================
# FINAL STATUS
# ============================================================

def get_validation_status(
    structural_eval: str,
    nli_eval: str,
) -> str:

    if structural_eval != "correct":
        return "FAIL_STRUCTURAL"

    if nli_eval == "Contradiction":
        return "PASS"

    if nli_eval == "Neutral":
        return "REVIEW"

    if nli_eval == "Entailment":
        return "REVIEW_HIGH_PRIORITY"

    return "ERROR"


# ============================================================
# DATASET CHECK
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Mancano le seguenti colonne nel CSV: "
            + ", ".join(sorted(missing_columns))
        )

    categories = {
        normalize_category(category)
        for category in df["question_category"].dropna().unique()
    }

    unsupported = categories - VALID_CATEGORIES

    if unsupported:
        raise ValueError(
            "Categorie non supportate trovate nel dataset: "
            + ", ".join(sorted(unsupported))
        )


# ============================================================
# STAGE 1: STRUCTURAL CHECK
# ============================================================

def run_structural_check(
    client: OpenAI,
    model: str,
    force: bool = False,
) -> pd.DataFrame:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if STRUCTURAL_OUTPUT.exists() and not force:

        print(
            f"Ripresa structural check da:\n"
            f"  {STRUCTURAL_OUTPUT}"
        )

        df = pd.read_csv(STRUCTURAL_OUTPUT)

    else:

        print(
            f"Caricamento dataset:\n"
            f"  {INPUT_PATH}"
        )

        df = pd.read_csv(INPUT_PATH)

        df["structural_eval"] = ""
        df["structural_error"] = ""

    validate_dataframe(df)

    total = len(df)

    print("\n========================================")
    print("STAGE 1 - STRUCTURAL CHECK")
    print("========================================")
    print(f"Totale coppie: {total}")
    print(f"Modello: {model}")
    print()

    for index, row in df.iterrows():

        existing = clean_text(
            row.get("structural_eval", "")
        )

        if (
            not force
            and existing in {"correct", "not correct"}
        ):
            continue

        item_id = clean_text(row["id"])
        category = normalize_category(
            row["question_category"]
        )
        caption = clean_text(row["caption"])
        foil = clean_text(row["foil"])

        print(
            f"[{index + 1}/{total}] "
            f"{item_id} | {category}"
        )

        if not caption or not foil:

            df.at[index, "structural_eval"] = "invalid"
            df.at[index, "structural_error"] = (
                "Missing caption or foil"
            )

            df.to_csv(
                STRUCTURAL_OUTPUT,
                index=False,
            )

            continue

        try:

            result = structural_validation(
                client=client,
                model=model,
                caption=caption,
                foil=foil,
                category=category,
            )

            df.at[index, "structural_eval"] = result
            df.at[index, "structural_error"] = ""

            if result != "correct":
                print(
                    f"  -> {result.upper()}"
                )

        except Exception as exc:

            print(f"  -> ERROR: {exc}")

            df.at[index, "structural_eval"] = "error"
            df.at[index, "structural_error"] = str(exc)

        # Salvataggio incrementale.
        # In caso di interruzione non perdiamo il lavoro già svolto.
        df.to_csv(
            STRUCTURAL_OUTPUT,
            index=False,
        )

    return df


# ============================================================
# STAGE 2: NLI CHECK
# ============================================================

def run_nli_check(
    client: OpenAI,
    model: str,
    structural_df: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:

    if FINAL_OUTPUT.exists() and not force:

        print(
            f"\nRipresa NLI check da:\n"
            f"  {FINAL_OUTPUT}"
        )

        df = pd.read_csv(FINAL_OUTPUT)

    else:

        df = structural_df.copy()

        df["nli_eval"] = ""
        df["nli_error"] = ""
        df["validation_status"] = ""

    total = len(df)

    print("\n========================================")
    print("STAGE 2 - CAPTION/FOIL NLI")
    print("========================================")
    print()

    for index, row in df.iterrows():

        structural_eval = clean_text(
            row.get("structural_eval", "")
        )

        existing_nli = clean_text(
            row.get("nli_eval", "")
        )

        existing_status = clean_text(
            row.get("validation_status", "")
        )

        # Già processato.
        if (
            not force
            and existing_nli
            in {
                "Contradiction",
                "Neutral",
                "Entailment",
                "SKIPPED",
            }
            and existing_status
        ):
            continue

        item_id = clean_text(row["id"])
        category = normalize_category(
            row["question_category"]
        )

        caption = clean_text(row["caption"])
        foil = clean_text(row["foil"])

        # Il secondo stage viene applicato soltanto alle
        # coppie che hanno superato lo structural check.
        if structural_eval != "correct":

            df.at[index, "nli_eval"] = "SKIPPED"
            df.at[index, "validation_status"] = (
                "FAIL_STRUCTURAL"
            )

            df.to_csv(
                FINAL_OUTPUT,
                index=False,
            )

            continue

        print(
            f"[{index + 1}/{total}] "
            f"{item_id} | {category}"
        )

        try:

            result = nli_validation(
                client=client,
                model=model,
                caption=caption,
                foil=foil,
            )

            df.at[index, "nli_eval"] = result
            df.at[index, "nli_error"] = ""

            status = get_validation_status(
                structural_eval=structural_eval,
                nli_eval=result,
            )

            df.at[index, "validation_status"] = status

            if status != "PASS":
                print(
                    f"  -> {result} | {status}"
                )

        except Exception as exc:

            print(f"  -> ERROR: {exc}")

            df.at[index, "nli_eval"] = "ERROR"
            df.at[index, "nli_error"] = str(exc)
            df.at[index, "validation_status"] = "ERROR"

        df.to_csv(
            FINAL_OUTPUT,
            index=False,
        )

    return df


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df: pd.DataFrame) -> None:

    print("\n")
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("\nStructural check:")
    print(
        df["structural_eval"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

    print("\nNLI:")
    print(
        df["nli_eval"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

    print("\nFinal status:")
    print(
        df["validation_status"]
        .fillna("missing")
        .value_counts()
        .to_string()
    )

    print("\nFinal status by category:")

    table = pd.crosstab(
        df["question_category"],
        df["validation_status"],
    )

    print(table.to_string())

    # --------------------------------------------------------
    # Review file
    # --------------------------------------------------------

    review_mask = df["validation_status"].isin(
        [
            "REVIEW",
            "REVIEW_HIGH_PRIORITY",
            "FAIL_STRUCTURAL",
            "ERROR",
        ]
    )

    review_df = df.loc[review_mask].copy()

    review_path = (
        OUTPUT_DIR
        / "caption_foil_manual_review.csv"
    )

    review_df.to_csv(
        review_path,
        index=False,
    )

    print(
        f"\nElementi da controllare manualmente: "
        f"{len(review_df)}"
    )

    print(
        f"File per manual review:\n"
        f"  {review_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage validation of caption-foil pairs: "
            "structural validation followed by NLI."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "OpenAI model used for validation. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute all evaluations even if output "
            "files already exist."
        ),
    )

    args = parser.parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input non trovato: {INPUT_PATH}"
        )

    client = get_client()

    structural_df = run_structural_check(
        client=client,
        model=args.model,
        force=args.force,
    )

    final_df = run_nli_check(
        client=client,
        model=args.model,
        structural_df=structural_df,
        force=args.force,
    )

    print_summary(final_df)

    print(
        f"\nDataset validato salvato in:\n"
        f"  {FINAL_OUTPUT}"
    )


if __name__ == "__main__":
    main()