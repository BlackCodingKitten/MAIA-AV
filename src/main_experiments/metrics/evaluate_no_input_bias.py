from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = {
    "id",
    "question",
    "answer1",
    "answer2",
    "target",
    "predicted_label",
    "is_correct",
    "model",
    "mode",
}

CHANCE_LEVEL = 0.50


# ============================================================
# UTILITIES
# ============================================================

def normalize_mode(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def normalize_binary(series: pd.Series) -> pd.Series:
    """
    Converts binary values to {0, 1}.
    Invalid or missing values become NaN.
    """

    def convert(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, bool):
            return int(value)

        try:
            number = int(float(value))
            if number in (0, 1):
                return number
        except (ValueError, TypeError):
            pass

        value = str(value).strip().lower()

        if value in {"true", "correct", "yes"}:
            return 1

        if value in {"false", "wrong", "no"}:
            return 0

        return np.nan

    return series.apply(convert)


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer used for length measurements.
    """
    return re.findall(
        r"\b[\wÀ-ÖØ-öø-ÿ]+\b",
        str(text).lower(),
        flags=re.UNICODE,
    )


# ============================================================
# LOAD CSV
# ============================================================

def read_no_input_csv(path: Path) -> pd.DataFrame:

    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=";")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["mode"] = df["mode"].apply(normalize_mode)

    # Keep no-input rows only.
    df = df[
        df["mode"].isin(
            {
                "no_input",
                "noinput",
                "none",
            }
        )
    ].copy()

    if df.empty:
        raise ValueError(
            f"{path}: no rows with mode='no_input' found."
        )

    df["target"] = normalize_binary(df["target"])
    df["predicted_label"] = normalize_binary(
        df["predicted_label"]
    )
    df["is_correct"] = normalize_binary(
        df["is_correct"]
    )

    return df


# ============================================================
# 1. RAW ACCURACY
# ============================================================

def raw_accuracy(df: pd.DataFrame) -> float:

    valid = df["is_correct"].dropna()

    if len(valid) == 0:
        return np.nan

    return float(valid.mean())


# ============================================================
# 2. LANGUAGE-ONLY ADVANTAGE
# ============================================================

def language_only_advantage(
    accuracy: float,
) -> float:

    if np.isnan(accuracy):
        return np.nan

    return accuracy - CHANCE_LEVEL


# ============================================================
# 3. POSITION BIAS
# ============================================================

def position_bias(df: pd.DataFrame) -> dict:

    predictions = (
        df["predicted_label"]
        .dropna()
        .astype(int)
    )

    if len(predictions) == 0:
        return {
            "p_choose_answer1": np.nan,
            "p_choose_answer2": np.nan,
            "position_bias": np.nan,
            "preferred_position": None,
        }

    # predicted_label == 0 -> answer1
    # predicted_label == 1 -> answer2

    p_answer1 = float(
        (predictions == 0).mean()
    )

    p_answer2 = float(
        (predictions == 1).mean()
    )

    # Magnitude of departure from perfect balance.
    bias = abs(p_answer1 - 0.5)

    if p_answer1 > p_answer2:
        preferred = "answer1"

    elif p_answer2 > p_answer1:
        preferred = "answer2"

    else:
        preferred = "balanced"

    return {
        "p_choose_answer1": p_answer1,
        "p_choose_answer2": p_answer2,
        "position_bias": bias,
        "preferred_position": preferred,
    }


# ============================================================
# 4. MATTHEWS CORRELATION COEFFICIENT
# ============================================================

def compute_mcc(df: pd.DataFrame) -> float:

    subset = df[
        ["target", "predicted_label"]
    ].dropna()

    if len(subset) == 0:
        return np.nan

    y_true = subset["target"].astype(int)
    y_pred = subset["predicted_label"].astype(int)

    # MCC is still defined by sklearn in many degenerate
    # cases and returns 0 when appropriate.
    return float(
        matthews_corrcoef(
            y_true,
            y_pred,
        )
    )


# ============================================================
# 5. LEXICAL ATTRACTION
# ============================================================

def compute_lexical_similarity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes TF-IDF cosine similarity between:

        question <-> answer1
        question <-> answer2

    The vectorizer is fitted on the entire no-input set
    for the model so that IDF weights remain consistent.
    """

    result = df.copy()

    corpus = pd.concat(
        [
            result["question"],
            result["answer1"],
            result["answer2"],
        ],
        ignore_index=True,
    ).fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 1),
    )

    vectorizer.fit(corpus)

    questions = vectorizer.transform(
        result["question"]
        .fillna("")
        .astype(str)
    )

    answers1 = vectorizer.transform(
        result["answer1"]
        .fillna("")
        .astype(str)
    )

    answers2 = vectorizer.transform(
        result["answer2"]
        .fillna("")
        .astype(str)
    )

    sim1 = np.array(
        [
            cosine_similarity(
                questions[i],
                answers1[i],
            )[0, 0]
            for i in range(len(result))
        ]
    )

    sim2 = np.array(
        [
            cosine_similarity(
                questions[i],
                answers2[i],
            )[0, 0]
            for i in range(len(result))
        ]
    )

    result["lexical_similarity_answer1"] = sim1
    result["lexical_similarity_answer2"] = sim2

    result["lexically_preferred_position"] = np.where(
        sim1 > sim2,
        0,
        np.where(
            sim2 > sim1,
            1,
            np.nan,
        ),
    )

    return result


def lexical_attraction(df: pd.DataFrame) -> dict:
    """
    Lexical Attraction Rate:

        P(
            model chooses the answer with
            greater lexical similarity to the question
        )

    Ties are excluded.
    """

    lexical_df = compute_lexical_similarity(df)

    valid = lexical_df[
        lexical_df[
            "lexically_preferred_position"
        ].notna()
        &
        lexical_df[
            "predicted_label"
        ].notna()
    ].copy()

    if len(valid) == 0:
        return {
            "lexical_attraction_rate": np.nan,
            "n_lexically_distinguishable": 0,
            "correct_option_lexical_advantage": np.nan,
        }

    attracted = (
        valid["predicted_label"].astype(int)
        ==
        valid[
            "lexically_preferred_position"
        ].astype(int)
    )

    lexical_attraction_rate = float(
        attracted.mean()
    )

    # --------------------------------------------------------
    # Dataset-level diagnostic:
    #
    # Is the gold answer itself more lexically similar
    # to the question?
    # --------------------------------------------------------

    target_matches_lexical = (
        valid["target"].astype(int)
        ==
        valid[
            "lexically_preferred_position"
        ].astype(int)
    )

    correct_option_lexical_advantage = float(
        target_matches_lexical.mean()
    )

    return {
        "lexical_attraction_rate":
            lexical_attraction_rate,

        "n_lexically_distinguishable":
            len(valid),

        "correct_option_lexical_advantage":
            correct_option_lexical_advantage,
    }


# ============================================================
# 6. LENGTH BIAS
# ============================================================

def length_bias(df: pd.DataFrame) -> dict:

    data = df.copy()

    data["answer1_length"] = (
        data["answer1"]
        .fillna("")
        .apply(lambda x: len(tokenize(x)))
    )

    data["answer2_length"] = (
        data["answer2"]
        .fillna("")
        .apply(lambda x: len(tokenize(x)))
    )

    # Which alternative is longer?
    data["longer_position"] = np.where(
        data["answer1_length"]
        >
        data["answer2_length"],
        0,
        np.where(
            data["answer2_length"]
            >
            data["answer1_length"],
            1,
            np.nan,
        ),
    )

    valid = data[
        data["longer_position"].notna()
        &
        data["predicted_label"].notna()
    ].copy()

    if len(valid) == 0:
        return {
            "longer_option_selection_rate":
                np.nan,

            "n_different_length_pairs":
                0,

            "correct_option_longer_rate":
                np.nan,

            "mean_answer1_length":
                float(
                    data["answer1_length"].mean()
                ),

            "mean_answer2_length":
                float(
                    data["answer2_length"].mean()
                ),
        }

    chooses_longer = (
        valid["predicted_label"].astype(int)
        ==
        valid["longer_position"].astype(int)
    )

    correct_is_longer = (
        valid["target"].astype(int)
        ==
        valid["longer_position"].astype(int)
    )

    return {
        "longer_option_selection_rate":
            float(chooses_longer.mean()),

        "n_different_length_pairs":
            len(valid),

        # Useful dataset diagnostic:
        # is the gold answer systematically longer?
        "correct_option_longer_rate":
            float(correct_is_longer.mean()),

        "mean_answer1_length":
            float(
                data["answer1_length"].mean()
            ),

        "mean_answer2_length":
            float(
                data["answer2_length"].mean()
            ),
    }


# ============================================================
# COMPLETE EVALUATION FOR ONE MODEL
# ============================================================

def evaluate_model(
    df: pd.DataFrame,
    source_file: str,
) -> dict:

    models = df["model"].dropna().unique()

    if len(models) != 1:
        raise ValueError(
            f"{source_file}: expected exactly one model, "
            f"found {models}"
        )

    model = str(models[0])

    accuracy = raw_accuracy(df)

    pos = position_bias(df)
    lex = lexical_attraction(df)
    length = length_bias(df)

    valid_predictions = int(
        df["predicted_label"]
        .notna()
        .sum()
    )

    invalid_predictions = int(
        df["predicted_label"]
        .isna()
        .sum()
    )

    return {
        "model": model,
        "source_file": source_file,

        "n_items": len(df),
        "n_valid_predictions": valid_predictions,
        "n_invalid_predictions": invalid_predictions,

        # ----------------------------------------------------
        # NO-INPUT METRICS
        # ----------------------------------------------------

        "raw_accuracy": accuracy,

        "language_only_advantage":
            language_only_advantage(
                accuracy
            ),

        "mcc":
            compute_mcc(df),

        "p_choose_answer1":
            pos["p_choose_answer1"],

        "p_choose_answer2":
            pos["p_choose_answer2"],

        "position_bias":
            pos["position_bias"],

        "preferred_position":
            pos["preferred_position"],

        "lexical_attraction_rate":
            lex[
                "lexical_attraction_rate"
            ],

        "n_lexically_distinguishable":
            lex[
                "n_lexically_distinguishable"
            ],

        "correct_option_lexical_advantage":
            lex[
                "correct_option_lexical_advantage"
            ],

        "longer_option_selection_rate":
            length[
                "longer_option_selection_rate"
            ],

        "n_different_length_pairs":
            length[
                "n_different_length_pairs"
            ],

        "correct_option_longer_rate":
            length[
                "correct_option_longer_rate"
            ],

        "mean_answer1_length":
            length[
                "mean_answer1_length"
            ],

        "mean_answer2_length":
            length[
                "mean_answer2_length"
            ],
    }


# ============================================================
# CATEGORY BREAKDOWN
# ============================================================

def category_breakdown(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    This is NOT interpreted as comprehension.

    It simply measures whether linguistic/structural
    biases are stronger in some question categories.
    """

    rows = []

    for category, group in df.groupby(
        "question_category"
    ):

        accuracy = raw_accuracy(group)

        rows.append(
            {
                "model":
                    group["model"].iloc[0],

                "question_category":
                    category,

                "n_items":
                    len(group),

                "raw_accuracy":
                    accuracy,

                "language_only_advantage":
                    language_only_advantage(
                        accuracy
                    ),

                "mcc":
                    compute_mcc(group),

                **position_bias(group),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "No-input linguistic and structural bias "
            "analysis for MAIA-AV."
        )
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help=(
            "One or more no_input CSV files. "
            "Example: gemma.csv qwen.csv ..."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/output/no_input/metrics"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    breakdown_frames = []

    for path in args.csv_files:

        print(
            f"\nProcessing: {path}"
        )

        df = read_no_input_csv(path)

        summary_rows.append(
            evaluate_model(
                df,
                source_file=str(path),
            )
        )

        breakdown_frames.append(
            category_breakdown(df)
        )

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        args.output_dir
        / "no_input_bias_metrics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # CATEGORY BREAKDOWN
    # ========================================================

    breakdown = pd.concat(
        breakdown_frames,
        ignore_index=True,
    )

    breakdown_path = (
        args.output_dir
        / "no_input_category_breakdown.csv"
    )

    breakdown.to_csv(
        breakdown_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        220,
    )

    print(
        "\n"
        "=================================================="
    )
    print(
        "NO-INPUT LINGUISTIC / STRUCTURAL BIAS ANALYSIS"
    )
    print(
        "==================================================\n"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {breakdown_path}"
    )


if __name__ == "__main__":
    main()