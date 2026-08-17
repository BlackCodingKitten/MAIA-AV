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

INPUT_PATH = Path(
    "data/vsv/caption-foil/caption_foil.csv"
)

OUTPUT_DIR = Path(
    "data/vsv/caption-foil/validation"
)

STRUCTURAL_OUTPUT = (
    OUTPUT_DIR
    / "caption_foil_structural_validation.csv"
)

FINAL_OUTPUT = (
    OUTPUT_DIR
    / "caption_foil_validation.csv"
)

MANUAL_REVIEW_OUTPUT = (
    OUTPUT_DIR
    / "caption_foil_manual_review.csv"
)

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

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY non trovata.\n"
            "Imposta la variabile d'ambiente prima "
            "di eseguire lo script."
        )

    return OpenAI(
        api_key=api_key
    )


# ============================================================
# DATAFRAME UTILITIES
# ============================================================

def ensure_text_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Garantisce che le colonne utilizzate per risultati,
    errori e label possano contenere stringhe.

    Quando un CSV contiene una colonna completamente vuota,
    pandas può inferirla come float64 perché i valori mancanti
    vengono rappresentati come NaN.

    Questa funzione forza tali colonne a dtype object e converte
    i valori mancanti in stringhe vuote.
    """

    for column in columns:

        if column not in df.columns:

            df[column] = pd.Series(
                [""] * len(df),
                index=df.index,
                dtype="object",
            )

        else:

            df[column] = (
                df[column]
                .astype("object")
                .where(
                    pd.notna(df[column]),
                    "",
                )
            )

    return df


def clean_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_category(category: str) -> str:

    return clean_text(
        category
    ).lower()


# ============================================================
# OUTPUT NORMALIZATION
# ============================================================

def normalize_structural_label(
    text: str,
) -> str:
    """
    Normalizza l'output dello structural check.

    Possibili risultati:
        correct
        not correct
        invalid
    """

    value = clean_text(
        text
    ).lower()

    if value.startswith("output:"):
        value = value[
            len("output:"):
        ].strip()

    value = (
        value
        .rstrip(".")
        .strip()
    )

    # Deve essere controllato prima perché
    # "not correct" contiene la stringa "correct".
    if value == "not correct":
        return "not correct"

    if value == "correct":
        return "correct"

    return "invalid"


def normalize_nli_label(
    text: str,
) -> str:
    """
    Normalizza l'output del classificatore NLI.

    Possibili risultati:
        Contradiction
        Neutral
        Entailment
        Invalid
    """

    value = clean_text(
        text
    )

    if value.lower().startswith(
        "output:"
    ):
        value = value[
            len("output:"):
        ].strip()

    value = (
        value
        .rstrip(".")
        .strip()
        .lower()
    )

    if value == "contradiction":
        return "Contradiction"

    if value == "neutral":
        return "Neutral"

    if value == "entailment":
        return "Entailment"

    return "Invalid"


# ============================================================
# OPENAI CALL
# ============================================================

def call_model(
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
    max_retries: int = MAX_RETRIES,
) -> str:

    last_error = None

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=prompt,
                    temperature=0,
                    max_output_tokens=16,
                )
            )

            return (
                response
                .output_text
                .strip()
            )

        except Exception as exc:

            last_error = exc

            wait_time = min(
                2 ** attempt,
                30,
            )

            print(
                "\nAPI error "
                f"(tentativo "
                f"{attempt + 1}/"
                f"{max_retries}): "
                f"{exc}"
            )

            if (
                attempt
                < max_retries - 1
            ):

                print(
                    "Nuovo tentativo "
                    f"tra {wait_time}s..."
                )

                time.sleep(
                    wait_time
                )

    raise RuntimeError(
        "Chiamata API fallita "
        f"dopo {max_retries} tentativi."
    ) from last_error


# ============================================================
# STRUCTURAL CHECK
# ============================================================

STRUCTURAL_SYSTEM_PROMPT = (
    "You are an assistant designed to validate "
    "the correctness of foils based on captions. "
    "Follow the requested semantic category strictly. "
    "Return only 'correct' or 'not correct'."
)


def get_structural_prompt(
    caption: str,
    foil: str,
    category: str,
) -> str:

    category = normalize_category(
        category
    )

    # ========================================================
    # CAUSAL
    # ========================================================

    if category == "causale":

        return (
            "Given an Italian caption (C) dealing with causal "
            "relations between events and its foil (F), your task "
            "is to assess whether F is a valid causal foil of C.\n\n"

            "To be valid, F must modify the cause, the effect, "
            "or the causal relation expressed in C while "
            "preserving the remaining relevant content as much "
            "as possible.\n\n"

            "The contrast must concern causal information. "
            "Changing only an entity, location, object, action, "
            "or temporal relation without modifying the causal "
            "content does not constitute a valid causal foil.\n\n"

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

    # ========================================================
    # TEMPORAL
    # ========================================================

    if category == "temporale":

        return (
            "Given an Italian caption (C) dealing with temporal "
            "information about events and its foil (F), your task "
            "is to assess whether F is a valid temporal foil of C.\n\n"

            "To be valid, F must modify the temporal relationship "
            "expressed in C, such as event ordering, before/after "
            "relations, the moment at which an event occurs, "
            "or another temporally relevant relation.\n\n"

            "The participants, actions, objects, and remaining "
            "semantic content should be preserved as much as "
            "possible. Replacing an entity or an event without "
            "actually modifying the temporal relation is not "
            "a valid temporal foil.\n\n"

            "If F is a valid temporal foil, output 'correct'. "
            "Otherwise output 'not correct'.\n\n"

            "Example:\n"
            "C: L'uomo con la maglia bianca saluta prima "
            "dell'uomo con la maglia verde.\n"
            "F: L'uomo con la maglia bianca saluta dopo "
            "l'uomo con la maglia verde.\n"
            "Output: correct\n\n"

            f"C: {caption}\n"
            f"F: {foil}\n"
            "Output:"
        )

    # ========================================================
    # SPATIAL
    # ========================================================

    if category == "spaziale":

        return (
            "Given an Italian caption (C) dealing with spatial "
            "information and its foil (F), your task is to assess "
            "whether F is a valid spatial foil of C.\n\n"

            "Spatial information includes:\n"
            "- absolute locations;\n"
            "- relative locations;\n"
            "- positions of people or objects;\n"
            "- body-part locations such as ear, nose, hand, "
            "foot, arm, or head;\n"
            "- containment relations such as inside/outside;\n"
            "- source and destination locations;\n"
            "- directional relations such as above/below, "
            "in front of/behind, near/far;\n"
            "- spatial configurations that hold at a specific "
            "moment or phase of the video.\n\n"

            "Changing the physical site or body part where an "
            "action occurs counts as a spatial modification, "
            "provided that the action, participants, and other "
            "semantic content remain substantially unchanged.\n\n"

            "To be valid, F must modify the relevant spatial "
            "information while preserving the remaining content "
            "as much as possible.\n\n"

            "Changing only the identity of an object or device "
            "without changing its location or spatial relation "
            "does not constitute a valid spatial foil.\n\n"

            "Example 1:\n"
            "C: Il beccuccio viene inserito nell'orecchio "
            "del cane.\n"
            "F: Il beccuccio viene inserito nel naso "
            "del cane.\n"
            "Output: correct\n\n"

            "Example 2:\n"
            "C: Il ragazzo si trova davanti al bancone.\n"
            "F: Il ragazzo si trova dietro al bancone.\n"
            "Output: correct\n\n"

            "Example 3:\n"
            "C: L'oggetto viene messo dentro la scatola.\n"
            "F: L'oggetto viene messo fuori dalla scatola.\n"
            "Output: correct\n\n"

            f"C: {caption}\n"
            f"F: {foil}\n"
            "Output:"
        )

    raise ValueError(
        "Categoria non supportata: "
        f"{category}"
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

    return normalize_structural_label(
        raw_output
    )


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
        "Your task is to determine the natural language "
        "inference (NLI) relationship between S1 and S2.\n\n"

        "The possible labels are:\n"

        "- Entailment: S2 logically follows from S1.\n"

        "- Contradiction: S2 contradicts S1.\n"

        "- Neutral: S1 and S2 are related, but S2 neither "
        "follows from nor contradicts S1.\n\n"

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

    return normalize_nli_label(
        raw_output
    )


# ============================================================
# FINAL STATUS
# ============================================================

def get_validation_status(
    structural_eval: str,
    nli_eval: str,
) -> str:

    if (
        structural_eval
        != "correct"
    ):
        return "FAIL_STRUCTURAL"

    if (
        nli_eval
        == "Contradiction"
    ):
        return "PASS"

    if (
        nli_eval
        == "Neutral"
    ):
        return "REVIEW"

    if (
        nli_eval
        == "Entailment"
    ):
        return (
            "REVIEW_HIGH_PRIORITY"
        )

    return "ERROR"


# ============================================================
# DATASET CHECK
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
) -> None:

    missing_columns = (
        REQUIRED_COLUMNS
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Mancano le seguenti "
            "colonne nel CSV: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    categories = {
        normalize_category(
            category
        )
        for category
        in df[
            "question_category"
        ]
        .dropna()
        .unique()
    }

    unsupported = (
        categories
        - VALID_CATEGORIES
    )

    if unsupported:

        raise ValueError(
            "Categorie non supportate "
            "trovate nel dataset: "
            + ", ".join(
                sorted(
                    unsupported
                )
            )
        )


# ============================================================
# STAGE 1
# STRUCTURAL CHECK
# ============================================================

def run_structural_check(
    client: OpenAI,
    model: str,
    force: bool = False,
) -> pd.DataFrame:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # RESUME FROM EXISTING CHECKPOINT
    # ========================================================

    if (
        STRUCTURAL_OUTPUT.exists()
        and not force
    ):

        print(
            "Ripresa structural check da:\n"
            f"  {STRUCTURAL_OUTPUT}"
        )

        df = pd.read_csv(
            STRUCTURAL_OUTPUT
        )

        # IMPORTANT:
        # colonne vuote lette dal CSV possono essere
        # interpretate come float64.
        df = ensure_text_columns(
            df,
            [
                "structural_eval",
                "structural_error",
            ],
        )

    # ========================================================
    # NEW RUN
    # ========================================================

    else:

        print(
            "Caricamento dataset:\n"
            f"  {INPUT_PATH}"
        )

        df = pd.read_csv(
            INPUT_PATH
        )

        df = ensure_text_columns(
            df,
            [
                "structural_eval",
                "structural_error",
            ],
        )

    validate_dataframe(
        df
    )

    total = len(
        df
    )

    print()
    print(
        "========================================"
    )
    print(
        "STAGE 1 - STRUCTURAL CHECK"
    )
    print(
        "========================================"
    )
    print(
        f"Totale coppie: {total}"
    )
    print(
        f"Modello: {model}"
    )
    print()

    for index, row in df.iterrows():

        existing = clean_text(
            row.get(
                "structural_eval",
                "",
            )
        ).lower()

        # Se già processato correttamente
        # viene saltato.
        if (
            not force
            and existing
            in {
                "correct",
                "not correct",
            }
        ):
            continue

        item_id = clean_text(
            row["id"]
        )

        category = normalize_category(
            row[
                "question_category"
            ]
        )

        caption = clean_text(
            row["caption"]
        )

        foil = clean_text(
            row["foil"]
        )

        print(
            f"[{index + 1}/{total}] "
            f"{item_id} | "
            f"{category}"
        )

        # ====================================================
        # MISSING INPUT
        # ====================================================

        if (
            not caption
            or not foil
        ):

            df.at[
                index,
                "structural_eval",
            ] = "invalid"

            df.at[
                index,
                "structural_error",
            ] = (
                "Missing caption or foil"
            )

            df.to_csv(
                STRUCTURAL_OUTPUT,
                index=False,
                encoding="utf-8-sig",
            )

            print(
                "  -> INVALID: "
                "missing caption or foil"
            )

            continue

        # ====================================================
        # MODEL VALIDATION
        # ====================================================

        try:

            result = (
                structural_validation(
                    client=client,
                    model=model,
                    caption=caption,
                    foil=foil,
                    category=category,
                )
            )

            df.at[
                index,
                "structural_eval",
            ] = result

            # Ora questa assegnazione è sicura
            # perché la colonna è dtype object.
            df.at[
                index,
                "structural_error",
            ] = ""

            if (
                result
                != "correct"
            ):
                print(
                    f"  -> "
                    f"{result.upper()}"
                )

        except Exception as exc:

            print(
                f"  -> ERROR: {exc}"
            )

            df.at[
                index,
                "structural_eval",
            ] = "error"

            df.at[
                index,
                "structural_error",
            ] = str(
                exc
            )

        # ====================================================
        # INCREMENTAL CHECKPOINT
        # ====================================================

        df.to_csv(
            STRUCTURAL_OUTPUT,
            index=False,
            encoding="utf-8-sig",
        )

    return df


# ============================================================
# STAGE 2
# NLI CHECK
# ============================================================

def run_nli_check(
    client: OpenAI,
    model: str,
    structural_df: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:

    # ========================================================
    # RESUME EXISTING NLI FILE
    # ========================================================

    if (
        FINAL_OUTPUT.exists()
        and not force
    ):

        print(
            "\nRipresa NLI check da:\n"
            f"  {FINAL_OUTPUT}"
        )

        df = pd.read_csv(
            FINAL_OUTPUT
        )

        df = ensure_text_columns(
            df,
            [
                "structural_eval",
                "structural_error",
                "nli_eval",
                "nli_error",
                "validation_status",
            ],
        )

    # ========================================================
    # NEW NLI RUN
    # ========================================================

    else:

        df = (
            structural_df
            .copy()
        )

        df = ensure_text_columns(
            df,
            [
                "structural_eval",
                "structural_error",
                "nli_eval",
                "nli_error",
                "validation_status",
            ],
        )

    total = len(
        df
    )

    print()
    print(
        "========================================"
    )
    print(
        "STAGE 2 - CAPTION/FOIL NLI"
    )
    print(
        "========================================"
    )
    print()

    for index, row in df.iterrows():

        structural_eval = clean_text(
            row.get(
                "structural_eval",
                "",
            )
        ).lower()

        existing_nli = clean_text(
            row.get(
                "nli_eval",
                "",
            )
        )

        existing_status = clean_text(
            row.get(
                "validation_status",
                "",
            )
        )

        # ====================================================
        # ALREADY PROCESSED
        # ====================================================

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

        item_id = clean_text(
            row["id"]
        )

        category = normalize_category(
            row[
                "question_category"
            ]
        )

        caption = clean_text(
            row["caption"]
        )

        foil = clean_text(
            row["foil"]
        )

        # ====================================================
        # STRUCTURAL FAILURE
        # ====================================================

        if (
            structural_eval
            != "correct"
        ):

            df.at[
                index,
                "nli_eval",
            ] = "SKIPPED"

            df.at[
                index,
                "nli_error",
            ] = ""

            df.at[
                index,
                "validation_status",
            ] = (
                "FAIL_STRUCTURAL"
            )

            df.to_csv(
                FINAL_OUTPUT,
                index=False,
                encoding="utf-8-sig",
            )

            continue

        print(
            f"[{index + 1}/{total}] "
            f"{item_id} | "
            f"{category}"
        )

        # ====================================================
        # NLI
        # ====================================================

        try:

            result = (
                nli_validation(
                    client=client,
                    model=model,
                    caption=caption,
                    foil=foil,
                )
            )

            df.at[
                index,
                "nli_eval",
            ] = result

            df.at[
                index,
                "nli_error",
            ] = ""

            status = (
                get_validation_status(
                    structural_eval=(
                        structural_eval
                    ),
                    nli_eval=result,
                )
            )

            df.at[
                index,
                "validation_status",
            ] = status

            if (
                status
                != "PASS"
            ):

                print(
                    f"  -> {result} "
                    f"| {status}"
                )

        except Exception as exc:

            print(
                f"  -> ERROR: {exc}"
            )

            df.at[
                index,
                "nli_eval",
            ] = "ERROR"

            df.at[
                index,
                "nli_error",
            ] = str(
                exc
            )

            df.at[
                index,
                "validation_status",
            ] = "ERROR"

        # ====================================================
        # INCREMENTAL SAVE
        # ====================================================

        df.to_csv(
            FINAL_OUTPUT,
            index=False,
            encoding="utf-8-sig",
        )

    return df


# ============================================================
# SUMMARY + MANUAL REVIEW
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print()
    print()
    print(
        "=" * 60
    )
    print(
        "VALIDATION SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        "\nStructural check:"
    )

    print(
        df[
            "structural_eval"
        ]
        .fillna(
            "missing"
        )
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nNLI:"
    )

    print(
        df[
            "nli_eval"
        ]
        .fillna(
            "missing"
        )
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nFinal status:"
    )

    print(
        df[
            "validation_status"
        ]
        .fillna(
            "missing"
        )
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nFinal status by category:"
    )

    table = pd.crosstab(
        df[
            "question_category"
        ],
        df[
            "validation_status"
        ],
    )

    print(
        table.to_string()
    )

    # ========================================================
    # MANUAL REVIEW DATASET
    # ========================================================

    review_mask = (
        df[
            "validation_status"
        ]
        .isin(
            [
                "REVIEW",
                "REVIEW_HIGH_PRIORITY",
                "FAIL_STRUCTURAL",
                "ERROR",
            ]
        )
    )

    review_df = (
        df.loc[
            review_mask
        ]
        .copy()
    )

    # Colonne dedicate alla successiva
    # revisione manuale.
    if (
        "manual_status"
        not in review_df.columns
    ):

        review_df[
            "manual_status"
        ] = ""

    if (
        "manual_notes"
        not in review_df.columns
    ):

        review_df[
            "manual_notes"
        ] = ""

    if (
        "corrected_caption"
        not in review_df.columns
    ):

        review_df[
            "corrected_caption"
        ] = ""

    if (
        "corrected_foil"
        not in review_df.columns
    ):

        review_df[
            "corrected_foil"
        ] = ""

    review_df = ensure_text_columns(
        review_df,
        [
            "manual_status",
            "manual_notes",
            "corrected_caption",
            "corrected_foil",
        ],
    )

    review_df.to_csv(
        MANUAL_REVIEW_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nElementi da controllare "
        f"manualmente: "
        f"{len(review_df)}"
    )

    print(
        "\nFile per manual review:"
    )

    print(
        f"  {MANUAL_REVIEW_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage validation of "
            "caption-foil pairs: "
            "structural validation "
            "followed by NLI."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "OpenAI model used for "
            "validation. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute all evaluations "
            "even if checkpoint files "
            "already exist."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Input non trovato: "
            f"{INPUT_PATH}"
        )

    # ========================================================
    # CLIENT
    # ========================================================

    client = get_client()

    # ========================================================
    # STAGE 1
    # ========================================================

    structural_df = (
        run_structural_check(
            client=client,
            model=args.model,
            force=args.force,
        )
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    final_df = (
        run_nli_check(
            client=client,
            model=args.model,
            structural_df=(
                structural_df
            ),
            force=args.force,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_summary(
        final_df
    )

    print()
    print(
        "Dataset validato salvato in:"
    )

    print(
        f"  {FINAL_OUTPUT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()