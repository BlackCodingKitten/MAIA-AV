from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


DEFAULT_INPUT_ROOT = Path("data/output")
DEFAULT_OUTPUT_DIR = Path("data/output/metrics")
EXPECTED_POOL_SIZE = 4

REQUIRED_COLUMNS = {
    "id",
    "pool_id",
    "model",
    "mode",
    "is_correct",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))

    for old, new in {
        "+": "_",
        "-": "_",
        " ": "_",
        "/": "_",
    }.items():
        value = value.replace(old, new)

    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")


def canonicalize_mode(mode: str) -> str:
    """Map the actual MAIA-AV mode names to canonical labels."""
    m = normalize_text(mode)

    aliases = {
        # no input
        "no_input": "no_input",
        "noinput": "no_input",
        "none": "no_input",
        "nessun_input": "no_input",

        # audio only
        "audio": "audio_only",
        "only_audio": "audio_only",
        "solo_audio": "audio_only",
        "audio_only": "audio_only",

        # transcription only
        "transcription": "transcription_only",
        "transcript": "transcription_only",
        "trascrizione": "transcription_only",
        "only_transcription": "transcription_only",
        "transcription_only": "transcription_only",
        "only_transcript": "transcription_only",
        "transcript_only": "transcription_only",
        "solo_trascrizione": "transcription_only",

        # video only
        "video": "video_only",
        "only_video": "video_only",
        "solo_video": "video_only",
        "video_only": "video_only",
        "mute": "video_only",

        # aligned audio + video
        "video_audio": "audio_video",
        "audio_video": "audio_video",
        "video_and_audio": "audio_video",
        "audio_and_video": "audio_video",

        # aligned transcript + video
        "transcript_video": "video_transcription",
        "video_transcript": "video_transcription",
        "transcription_video": "video_transcription",
        "video_transcription": "video_transcription",
        "trascrizione_video": "video_transcription",
        "video_trascrizione": "video_transcription",
    }

    # Detect perturbation experiments before the simple alias lookup.
    if "audio" in m and "video" in m and any(
        token in m for token in ("misaligned", "mismatch", "shifted", "disallineato")
    ):
        return "audio_video_misaligned"

    if (
        "video" in m
        and any(token in m for token in ("transcription", "transcript", "trascrizione"))
        and any(token in m for token in ("misaligned", "mismatch", "shifted", "disallineato"))
    ):
        return "video_transcription_misaligned"

    return aliases.get(m, m)


def normalize_is_correct(series: pd.Series) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value) == 1.0:
                return 1
            if float(value) == 0.0:
                return 0

        value = str(value).strip().lower()
        if value in {"1", "true", "correct", "yes", "y"}:
            return 1
        if value in {"0", "false", "wrong", "no", "n"}:
            return 0
        return np.nan

    return series.apply(convert)


# ============================================================
# DATA LOADING
# ============================================================

def read_experiment_csv(path: Path) -> pd.DataFrame:
    """Read one MAIA-AV output CSV, handling BOM and delimiter variation."""
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")

    # Protect against BOM or accidental whitespace in headers.
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)}; "
            f"found {df.columns.tolist()}"
        )

    df = df.copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["pool_id"] = df["pool_id"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["mode_original"] = df["mode"].astype(str).str.strip()
    df["mode"] = df["mode_original"].apply(canonicalize_mode)
    df["is_correct"] = normalize_is_correct(df["is_correct"])
    return df


def load_all_experiments(input_root: Path) -> pd.DataFrame:
    """
    Load all compatible CSVs, INCLUDING no_input.

    no_input is retained only because it is the A=0,V=0 control cell for
    the Audio x Video interaction model. It is removed later from ordinary
    performance metrics.
    """
    input_root = input_root.resolve()
    print(f"\nSearching experimental CSVs in:\n  {input_root}")

    if not input_root.exists():
        raise RuntimeError(f"Input directory does not exist: {input_root}")

    csv_files = sorted(input_root.rglob("*.csv"))
    print(f"\nCSV files found: {len(csv_files)}")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_root}")

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    for path in csv_files:
        # Avoid recursively ingesting the script's own output files.
        if "metrics" in path.parts:
            continue

        try:
            df = read_experiment_csv(path)
        except Exception as exc:
            skipped.append((str(path), str(exc)))
            print(f"[SKIP] {path.name}\n       {exc}")
            continue

        df["source_file"] = str(path)
        frames.append(df)

        modes = sorted(df["mode"].dropna().unique())
        models = sorted(df["model"].dropna().unique())
        print(f"[OK]   {path.name}\n       model={models}, mode={modes}, rows={len(df)}")

    if not frames:
        details = "\n".join(f"- {p}: {e}" for p, e in skipped)
        raise RuntimeError(
            "\nNo compatible experimental CSV found.\n"
            "All candidate CSV files were rejected.\n\n"
            f"{details}"
        )

    data = pd.concat(frames, ignore_index=True)

    invalid = data["is_correct"].isna()
    if invalid.any():
        print(
            f"\n[WARNING] {int(invalid.sum())} rows have invalid is_correct "
            "values and will be removed."
        )
        data = data.loc[~invalid].copy()

    data["is_correct"] = data["is_correct"].astype(int)

    duplicated = data.duplicated(subset=["model", "mode", "id"], keep=False)
    if duplicated.any():
        examples = (
            data.loc[duplicated, ["model", "mode", "id", "source_file"]]
            .sort_values(["model", "mode", "id"])
            .head(30)
        )
        raise ValueError(
            "\nDuplicate items found for (model, mode, id):\n\n"
            + examples.to_string(index=False)
        )

    print("\n==============================================")
    print("LOADED EXPERIMENTAL DATA")
    print("==============================================")
    summary = (
        data.groupby(["model", "mode"])
        .size()
        .reset_index(name="n_items")
        .sort_values(["model", "mode"])
    )
    print(summary.to_string(index=False))

    return data


def get_performance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude no_input from all ordinary performance/comprehension metrics.
    It remains available only in all_data for the interaction analysis.
    """
    return data.loc[data["mode"] != "no_input"].copy()


# ============================================================
# 1. ITEM ACCURACY
# ============================================================

def item_accuracy(df: pd.DataFrame) -> float:
    return float(df["is_correct"].mean()) if not df.empty else np.nan


# ============================================================
# 2. AGGREGATE ACCURACY
# ============================================================

def pool_results(df: pd.DataFrame, expected_pool_size: int) -> pd.DataFrame:
    grouped = (
        df.groupby("pool_id", as_index=False)
        .agg(n_items=("id", "size"), n_correct=("is_correct", "sum"))
    )
    grouped["pool_correct"] = (
        (grouped["n_items"] == expected_pool_size)
        & (grouped["n_correct"] == expected_pool_size)
    ).astype(int)
    return grouped


def aggregate_accuracy(df: pd.DataFrame, expected_pool_size: int) -> float:
    pools = pool_results(df, expected_pool_size)
    return float(pools["pool_correct"].mean()) if not pools.empty else np.nan


def compute_per_experiment_metrics(
    data: pd.DataFrame, expected_pool_size: int
) -> pd.DataFrame:
    rows = []
    for (model, mode), group in data.groupby(["model", "mode"], sort=True):
        pools = pool_results(group, expected_pool_size)
        rows.append(
            {
                "model": model,
                "experiment": mode,
                "n_items": len(group),
                "n_correct_items": int(group["is_correct"].sum()),
                "item_accuracy": item_accuracy(group),
                "n_pools": len(pools),
                "n_correct_pools": int(pools["pool_correct"].sum()),
                "aggregate_accuracy": aggregate_accuracy(group, expected_pool_size),
                "n_incomplete_pools": int(
                    (pools["n_items"] != expected_pool_size).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# PAIRED MATCHING
# ============================================================

def pair_conditions(
    data: pd.DataFrame, model: str, base_mode: str, new_mode: str
) -> pd.DataFrame:
    base = (
        data.loc[
            (data["model"] == model) & (data["mode"] == base_mode),
            ["id", "pool_id", "is_correct"],
        ]
        .rename(
            columns={"pool_id": "base_pool_id", "is_correct": "base_correct"}
        )
    )

    new = (
        data.loc[
            (data["model"] == model) & (data["mode"] == new_mode),
            ["id", "pool_id", "is_correct"],
        ]
        .rename(columns={"pool_id": "new_pool_id", "is_correct": "new_correct"})
    )

    paired = base.merge(new, on="id", how="inner", validate="one_to_one")
    if paired.empty:
        return paired

    if (paired["base_pool_id"] != paired["new_pool_id"]).any():
        raise ValueError(
            f"Inconsistent pool_id for model {model}: {base_mode} vs {new_mode}"
        )

    paired["pool_id"] = paired["base_pool_id"]
    return paired


# ============================================================
# 3. CONDITIONAL MODALITY GAIN
# 4. REPAIR / DAMAGE
# ============================================================

def compute_pairwise_metrics(paired: pd.DataFrame) -> dict:
    if paired.empty:
        return {
            "n_paired_items": 0,
            "base_accuracy": np.nan,
            "new_accuracy": np.nan,
            "conditional_modality_gain": np.nan,
            "repair_rate": np.nan,
            "damage_rate": np.nan,
            "n_stable_correct": 0,
            "n_damage": 0,
            "n_repair": 0,
            "n_stable_wrong": 0,
        }

    base = paired["base_correct"].astype(int)
    new = paired["new_correct"].astype(int)

    n11 = int(((base == 1) & (new == 1)).sum())
    n10 = int(((base == 1) & (new == 0)).sum())
    n01 = int(((base == 0) & (new == 1)).sum())
    n00 = int(((base == 0) & (new == 0)).sum())

    base_acc = float(base.mean())
    new_acc = float(new.mean())
    n_base_wrong = n00 + n01
    n_base_correct = n11 + n10

    return {
        "n_paired_items": len(paired),
        "base_accuracy": base_acc,
        "new_accuracy": new_acc,
        "conditional_modality_gain": new_acc - base_acc,
        "repair_rate": n01 / n_base_wrong if n_base_wrong else np.nan,
        "damage_rate": n10 / n_base_correct if n_base_correct else np.nan,
        "n_stable_correct": n11,
        "n_damage": n10,
        "n_repair": n01,
        "n_stable_wrong": n00,
    }


def compute_all_modality_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("video_only", "audio_video", "audio_added_to_video"),
        (
            "video_only",
            "video_transcription",
            "transcription_added_to_video",
        ),
    ]

    rows = []
    for model in sorted(data["model"].unique()):
        available = set(data.loc[data["model"] == model, "mode"])
        for base_mode, new_mode, label in comparisons:
            if base_mode not in available or new_mode not in available:
                continue
            paired = pair_conditions(data, model, base_mode, new_mode)
            rows.append(
                {
                    "model": model,
                    "comparison": label,
                    "base_experiment": base_mode,
                    "new_experiment": new_mode,
                    **compute_pairwise_metrics(paired),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 5. AUDIO x VIDEO INTERACTION EFFECT
# ============================================================

def prepare_interaction_data(
    data: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    Complete 2x2 design:

                       Video=0       Video=1
        Audio=0        no_input      video_only
        Audio=1        audio_only    audio_video

    Only item IDs present in ALL FOUR cells are retained.
    """
    required_modes = ["no_input", "audio_only", "video_only", "audio_video"]
    model_data = data.loc[data["model"] == model].copy()
    available = set(model_data["mode"].unique())
    missing = [m for m in required_modes if m not in available]
    if missing:
        return pd.DataFrame(), missing

    ids_by_mode = {
        mode: set(model_data.loc[model_data["mode"] == mode, "id"])
        for mode in required_modes
    }
    common_ids = set.intersection(*ids_by_mode.values())
    if not common_ids:
        return pd.DataFrame(), []

    interaction_data = model_data.loc[
        model_data["mode"].isin(required_modes)
        & model_data["id"].isin(common_ids)
    ].copy()

    counts = interaction_data.groupby("id")["mode"].nunique()
    valid_ids = set(counts[counts == 4].index)
    interaction_data = interaction_data.loc[
        interaction_data["id"].isin(valid_ids)
    ].copy()

    pool_counts = interaction_data.groupby("id")["pool_id"].nunique()
    if (pool_counts != 1).any():
        raise ValueError(
            f"{model}: some item IDs have different pool_id values across conditions."
        )

    coding = {
        "no_input": (0, 0),
        "audio_only": (1, 0),
        "video_only": (0, 1),
        "audio_video": (1, 1),
    }
    interaction_data["audio_present"] = interaction_data["mode"].map(
        lambda x: coding[x][0]
    )
    interaction_data["video_present"] = interaction_data["mode"].map(
        lambda x: coding[x][1]
    )

    return interaction_data, []


def compute_audio_video_interaction(data: pd.DataFrame) -> pd.DataFrame:
    """
    For each model fit:

        logit(P(correct)) = beta0 + beta_A*A + beta_V*V + beta_AV*(A*V)

    beta_AV is the behavioural Audio x Video interaction effect.
    Cluster-robust standard errors are computed by pool_id.

    no_input participates ONLY here as the A=0,V=0 control cell.
    """
    rows = []

    for model in sorted(data["model"].unique()):
        df, missing = prepare_interaction_data(data, model)

        if missing:
            rows.append(
                {
                    "model": model,
                    "status": "not_computable",
                    "missing_modes": ", ".join(missing),
                    "n_complete_items": 0,
                    "n_observations": 0,
                    "interaction_beta": np.nan,
                    "interaction_std_error": np.nan,
                    "interaction_z": np.nan,
                    "interaction_p_value": np.nan,
                    "interaction_ci95_low": np.nan,
                    "interaction_ci95_high": np.nan,
                    "interaction_odds_ratio": np.nan,
                    "interaction_or_ci95_low": np.nan,
                    "interaction_or_ci95_high": np.nan,
                }
            )
            continue

        if df.empty:
            rows.append(
                {
                    "model": model,
                    "status": "no_common_items",
                    "missing_modes": "",
                    "n_complete_items": 0,
                    "n_observations": 0,
                    "interaction_beta": np.nan,
                    "interaction_std_error": np.nan,
                    "interaction_z": np.nan,
                    "interaction_p_value": np.nan,
                    "interaction_ci95_low": np.nan,
                    "interaction_ci95_high": np.nan,
                    "interaction_odds_ratio": np.nan,
                    "interaction_or_ci95_low": np.nan,
                    "interaction_or_ci95_high": np.nan,
                }
            )
            continue

        try:
            fit = smf.glm(
                "is_correct ~ audio_present * video_present",
                data=df,
                family=sm.families.Binomial(),
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": df["pool_id"]},
            )

            term = "audio_present:video_present"
            beta = float(fit.params[term])
            se = float(fit.bse[term])
            z = float(fit.tvalues[term])
            p = float(fit.pvalues[term])
            ci_low, ci_high = map(float, fit.conf_int().loc[term].tolist())
            cell_acc = df.groupby("mode")["is_correct"].mean().to_dict()

            rows.append(
                {
                    "model": model,
                    "status": "ok",
                    "missing_modes": "",
                    "n_complete_items": int(df["id"].nunique()),
                    "n_observations": len(df),
                    "accuracy_no_input": cell_acc.get("no_input", np.nan),
                    "accuracy_audio_only": cell_acc.get("audio_only", np.nan),
                    "accuracy_video_only": cell_acc.get("video_only", np.nan),
                    "accuracy_audio_video": cell_acc.get("audio_video", np.nan),
                    "interaction_beta": beta,
                    "interaction_std_error": se,
                    "interaction_z": z,
                    "interaction_p_value": p,
                    "interaction_ci95_low": ci_low,
                    "interaction_ci95_high": ci_high,
                    "interaction_odds_ratio": float(np.exp(beta)),
                    "interaction_or_ci95_low": float(np.exp(ci_low)),
                    "interaction_or_ci95_high": float(np.exp(ci_high)),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model,
                    "status": "fit_failed",
                    "missing_modes": "",
                    "n_complete_items": int(df["id"].nunique()),
                    "n_observations": len(df),
                    "interaction_beta": np.nan,
                    "interaction_std_error": np.nan,
                    "interaction_z": np.nan,
                    "interaction_p_value": np.nan,
                    "interaction_ci95_low": np.nan,
                    "interaction_ci95_high": np.nan,
                    "interaction_odds_ratio": np.nan,
                    "interaction_or_ci95_low": np.nan,
                    "interaction_or_ci95_high": np.nan,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 6A. JOINT-MODALITY ACCURACY
# ============================================================

def read_requirements_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "pool_id" not in df.columns:
        required = {"video_name", "principal_dimension"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Cannot reconstruct pool_id: video_name and principal_dimension are required."
            )

        video_stem = (
            df["video_name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        )
        df["pool_id"] = video_stem + "_" + df["principal_dimension"].astype(str)

    return df


def normalize_requirement(value: str) -> str:
    value = normalize_text(value)
    aliases = {
        "av_required": "audio_visual_required",
        "audio_video_required": "audio_visual_required",
        "audio_visual_required": "audio_visual_required",
        "audio_visual": "audio_visual_required",
        "vt_required": "video_transcript_required",
        "video_transcription_required": "video_transcript_required",
        "video_transcript_required": "video_transcript_required",
    }
    return aliases.get(value, value)


def compute_joint_modality_accuracy(
    data: pd.DataFrame,
    requirements: pd.DataFrame,
    expected_pool_size: int,
) -> pd.DataFrame:
    if "modality_requirement" not in requirements.columns:
        return pd.DataFrame()

    requirements = requirements.copy()
    requirements["modality_requirement"] = requirements[
        "modality_requirement"
    ].apply(normalize_requirement)

    setups = [
        ("audio_visual_required", "audio_video", "audio_visual"),
        (
            "video_transcript_required",
            "video_transcription",
            "video_transcription",
        ),
    ]

    rows = []
    for requirement, mode, label in setups:
        required_pools = set(
            requirements.loc[
                requirements["modality_requirement"] == requirement, "pool_id"
            ].astype(str)
        )
        if not required_pools:
            continue

        for model in sorted(data["model"].unique()):
            subset = data.loc[
                (data["model"] == model)
                & (data["mode"] == mode)
                & data["pool_id"].isin(required_pools)
            ].copy()
            if subset.empty:
                continue

            pools = pool_results(subset, expected_pool_size)
            rows.append(
                {
                    "model": model,
                    "joint_modality_type": label,
                    "experiment": mode,
                    "n_required_questions": len(required_pools),
                    "n_evaluated_questions": len(pools),
                    "joint_modality_accuracy": float(pools["pool_correct"].mean()),
                    "n_correct_required_questions": int(
                        pools["pool_correct"].sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 6B. ALIGNMENT SENSITIVITY
# ============================================================

def compute_alignment_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("audio_video_misaligned", "audio_video", "audio_visual_alignment"),
        (
            "video_transcription_misaligned",
            "video_transcription",
            "video_transcription_alignment",
        ),
    ]

    rows = []
    for model in sorted(data["model"].unique()):
        available = set(data.loc[data["model"] == model, "mode"])

        for misaligned, aligned, label in comparisons:
            if aligned not in available or misaligned not in available:
                continue

            paired = pair_conditions(data, model, misaligned, aligned)
            if paired.empty:
                continue

            aligned_acc = float(paired["new_correct"].mean())
            misaligned_acc = float(paired["base_correct"].mean())
            rows.append(
                {
                    "model": model,
                    "alignment_type": label,
                    "aligned_experiment": aligned,
                    "misaligned_experiment": misaligned,
                    "n_paired_items": len(paired),
                    "aligned_accuracy": aligned_acc,
                    "misaligned_accuracy": misaligned_acc,
                    "alignment_sensitivity": aligned_acc - misaligned_acc,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# OUTPUT
# ============================================================

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the approved MAIA-AV multimodal metrics. no_input is "
            "excluded from ordinary performance metrics and used only as the "
            "A=0,V=0 control cell for the Audio x Video interaction model."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--requirements-csv",
        type=Path,
        default=None,
        help="Optional classification CSV containing modality_requirement.",
    )
    parser.add_argument("--pool-size", type=int, default=EXPECTED_POOL_SIZE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load every condition, including no_input.
    all_data = load_all_experiments(args.input_root)

    # Exclude no_input from ordinary performance/comprehension metrics.
    performance_data = get_performance_data(all_data)

    performance_metrics = compute_per_experiment_metrics(
        performance_data, expected_pool_size=args.pool_size
    )
    save_dataframe(
        performance_metrics,
        args.output_dir / "performance_per_experiment_model.csv",
    )

    contribution_metrics = compute_all_modality_comparisons(performance_data)
    save_dataframe(
        contribution_metrics,
        args.output_dir / "modality_contribution_per_model.csv",
    )

    # no_input is used ONLY in this calculation.
    interaction_metrics = compute_audio_video_interaction(all_data)
    save_dataframe(
        interaction_metrics,
        args.output_dir / "audio_video_interaction_effect.csv",
    )

    if args.requirements_csv is not None:
        requirements = read_requirements_csv(args.requirements_csv)
        joint_metrics = compute_joint_modality_accuracy(
            performance_data,
            requirements,
            expected_pool_size=args.pool_size,
        )
        save_dataframe(
            joint_metrics,
            args.output_dir / "joint_modality_accuracy.csv",
        )

    alignment_metrics = compute_alignment_sensitivity(performance_data)
    save_dataframe(
        alignment_metrics,
        args.output_dir / "alignment_sensitivity.csv",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n====================================================")
    print("PERFORMANCE METRICS - NO_INPUT EXCLUDED")
    print("====================================================")
    print(performance_metrics.to_string(index=False))

    if not contribution_metrics.empty:
        print("\n====================================================")
        print("MODALITY CONTRIBUTION")
        print("====================================================")
        print(contribution_metrics.to_string(index=False))

    print("\n====================================================")
    print("AUDIO x VIDEO INTERACTION - NO_INPUT AS CONTROL ONLY")
    print("====================================================")
    print(interaction_metrics.to_string(index=False))

    if not alignment_metrics.empty:
        print("\n====================================================")
        print("ALIGNMENT SENSITIVITY")
        print("====================================================")
        print(alignment_metrics.to_string(index=False))

    print(f"\nResults saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()