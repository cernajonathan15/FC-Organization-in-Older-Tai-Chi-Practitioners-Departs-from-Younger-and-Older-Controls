"""Stage 3: phenotype inference, robustness, and behavior correlations."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import analysis_config as cfg
from analysis_utils import (
    cliffs_delta,
    ensure_output_directories,
    fit_axis,
    hedges_g,
    percentile_interval,
    score_axis,
    winsorize_older_positions,
    winsorize_axis_data,
)


METRICS = (
    "position",
    "deviation",
    "relative_distance_youth",
    "rsa_fisher_difference",
)


def load_data() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    feature_path = cfg.DATASET_DIR / "participant_features.csv"
    score_path = cfg.PHENOTYPE_DIR / "participant_phenotype_scores.csv"
    if not feature_path.exists() or not score_path.exists():
        raise FileNotFoundError("Run stages 1 and 2 before stage 3")
    features = pd.read_csv(feature_path, dtype={"participant": str})
    scores = pd.read_csv(score_path, dtype={"participant": str})
    metadata = pd.read_csv(cfg.DATASET_DIR / "feature_metadata.csv")
    feature_names = metadata["feature_name"].tolist()
    values = features[feature_names].to_numpy(float)
    groups = features["group"].to_numpy(str)
    return features, feature_names, values, groups


def score_groups(
    young: np.ndarray,
    old: np.ndarray,
    tcoa: np.ndarray,
    estimator: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if cfg.ANALYSIS_METHOD == "robust_median":
        _, adjusted = winsorize_axis_data(young, old, tcoa)
        adjusted_young, adjusted_old, adjusted_tcoa = adjusted
    else:
        adjusted_young, adjusted_old, adjusted_tcoa = young, old, tcoa
    model = fit_axis(adjusted_young, adjusted_old, estimator=estimator)
    tcoa_scores = score_axis(adjusted_tcoa, model)
    old_scores = score_axis(adjusted_old, model)
    if cfg.ANALYSIS_METHOD == "position_winsor":
        _, adjusted_old_position, adjusted_tcoa_position, _, _ = (
            winsorize_older_positions(old_scores["position"], tcoa_scores["position"])
        )
        old_scores["position"] = adjusted_old_position
        tcoa_scores["position"] = adjusted_tcoa_position
    return tcoa_scores, old_scores


def location_name(metric: str) -> str:
    if cfg.ANALYSIS_METHOD == "robust_median" and metric in {"position", "deviation"}:
        return "median"
    return "mean"


def metric_statistics(
    young: np.ndarray,
    old: np.ndarray,
    tcoa: np.ndarray,
    estimator: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    tcoa_scores, old_scores = score_groups(young, old, tcoa, estimator)
    primary: dict[str, float] = {}
    mean_sensitivity: dict[str, float] = {}
    for metric in METRICS:
        reducer = np.median if location_name(metric) == "median" else np.mean
        primary[metric] = float(reducer(tcoa_scores[metric]) - reducer(old_scores[metric]))
        mean_sensitivity[metric] = float(
            np.mean(tcoa_scores[metric]) - np.mean(old_scores[metric])
        )
    return primary, mean_sensitivity, tcoa_scores, old_scores


def older_group_permutation(
    young: np.ndarray,
    old: np.ndarray,
    tcoa: np.ndarray,
    rng: np.random.Generator,
    estimator: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    observed, observed_means, tcoa_scores, old_scores = metric_statistics(
        young,
        old,
        tcoa,
        estimator,
    )
    older = np.vstack([old, tcoa])
    n_old = len(old)
    null = {metric: np.empty(cfg.N_PERMUTATIONS, dtype=float) for metric in METRICS}
    mean_null = {metric: np.empty(cfg.N_PERMUTATIONS, dtype=float) for metric in METRICS}
    for replication in range(cfg.N_PERMUTATIONS):
        order = rng.permutation(len(older))
        perm_old = older[order[:n_old]]
        perm_tcoa = older[order[n_old:]]
        perm_stats, perm_means, _, _ = metric_statistics(
            young,
            perm_old,
            perm_tcoa,
            estimator,
        )
        for metric in METRICS:
            null[metric][replication] = perm_stats[metric]
            mean_null[metric][replication] = perm_means[metric]
        if (replication + 1) % max(cfg.N_PERMUTATIONS // 20, 1) == 0:
            print(
                f"\rOlder-group permutations: {replication + 1}/{cfg.N_PERMUTATIONS}",
                end="",
                flush=True,
            )
    print()

    alternatives = {
        "position": "TCOA position > OAC position",
        "deviation": "TCOA and OAC deviation differ (two-sided)",
        "relative_distance_youth": "TCOA relative youth proximity > OAC",
        "rsa_fisher_difference": "TCOA relative YAC-pattern similarity > OAC",
    }
    result_rows: list[dict[str, object]] = []
    for metric in METRICS:
        if metric == "deviation":
            p_value = (1 + np.count_nonzero(np.abs(null[metric]) >= abs(observed[metric]))) / (
                cfg.N_PERMUTATIONS + 1
            )
        else:
            p_value = (1 + np.count_nonzero(null[metric] >= observed[metric])) / (
                cfg.N_PERMUTATIONS + 1
            )
        result_rows.append(
            {
                "axis_estimator": estimator,
                "is_primary_estimator": estimator == cfg.PRIMARY_AXIS_ESTIMATOR,
                "metric": metric,
                "location_statistic": location_name(metric),
                "alternative": alternatives[metric],
                "oac_mean": float(np.mean(old_scores[metric])),
                "tcoa_mean": float(np.mean(tcoa_scores[metric])),
                "oac_median": float(np.median(old_scores[metric])),
                "tcoa_median": float(np.median(tcoa_scores[metric])),
                "observed_tcoa_minus_oac": observed[metric],
                "hedges_g_tcoa_minus_oac": hedges_g(
                    tcoa_scores[metric], old_scores[metric]
                ),
                "cliffs_delta_tcoa_minus_oac": cliffs_delta(
                    tcoa_scores[metric], old_scores[metric]
                ),
                "permutations": cfg.N_PERMUTATIONS,
                "permutation_p": p_value,
            }
        )
    all_results = pd.DataFrame(result_rows)
    primary = all_results[all_results["metric"].isin(["position", "deviation"])].copy()
    robustness = all_results[
        all_results["metric"].isin(
            ["relative_distance_youth", "rsa_fisher_difference"]
        )
    ].copy()
    mean_rows: list[dict[str, object]] = []
    if cfg.ANALYSIS_METHOD == "robust_median":
        for metric in ("position", "deviation"):
            if metric == "deviation":
                mean_p = (1 + np.count_nonzero(np.abs(mean_null[metric]) >= abs(observed_means[metric]))) / (cfg.N_PERMUTATIONS + 1)
            else:
                mean_p = (1 + np.count_nonzero(mean_null[metric] >= observed_means[metric])) / (cfg.N_PERMUTATIONS + 1)
            mean_rows.append({
                "axis_estimator": estimator,
                "is_primary_estimator": estimator == cfg.PRIMARY_AXIS_ESTIMATOR,
                "metric": metric,
                "location_statistic": "mean sensitivity",
                "oac_mean": float(np.mean(old_scores[metric])),
                "tcoa_mean": float(np.mean(tcoa_scores[metric])),
                "observed_tcoa_minus_oac": observed_means[metric],
                "hedges_g_tcoa_minus_oac": hedges_g(tcoa_scores[metric], old_scores[metric]),
                "permutations": cfg.N_PERMUTATIONS,
                "permutation_p": mean_p,
            })
    return primary, robustness, pd.DataFrame(mean_rows), null


def bootstrap_axis(
    young: np.ndarray,
    old: np.ndarray,
    tcoa: np.ndarray,
    rng: np.random.Generator,
    estimator: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int]] = []
    for replication in range(cfg.N_BOOTSTRAPS):
        sample_young = young[rng.integers(0, len(young), len(young))]
        sample_old = old[rng.integers(0, len(old), len(old))]
        sample_tcoa = tcoa[rng.integers(0, len(tcoa), len(tcoa))]
        scored, _ = score_groups(
            sample_young, sample_old, sample_tcoa, estimator
        )
        row: dict[str, float | int | bool | str] = {
            "replication": replication,
            "axis_estimator": estimator,
            "is_primary_estimator": estimator == cfg.PRIMARY_AXIS_ESTIMATOR,
            "mean_tcoa_position": float(np.mean(scored["position"])),
            "mean_tcoa_deviation": float(np.mean(scored["deviation"])),
            "mean_tcoa_relative_distance_youth": float(np.mean(scored["relative_distance_youth"])),
            "mean_tcoa_rsa_fisher_difference": float(np.mean(scored["rsa_fisher_difference"])),
        }
        if cfg.ANALYSIS_METHOD == "robust_median":
            row["median_tcoa_position"] = float(np.median(scored["position"]))
            row["median_tcoa_deviation"] = float(np.median(scored["deviation"]))
        rows.append(row)
    replicates = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    for column in replicates.columns:
        if column in {
            "replication",
            "axis_estimator",
            "is_primary_estimator",
        }:
            continue
        lower, upper = percentile_interval(replicates[column].to_numpy())
        summaries.append(
            {
                "axis_estimator": estimator,
                "is_primary_estimator": estimator == cfg.PRIMARY_AXIS_ESTIMATOR,
                "metric": column,
                "bootstrap_mean": float(replicates[column].mean()),
                "ci_2_5": lower,
                "ci_97_5": upper,
                "bootstrap_replicates": cfg.N_BOOTSTRAPS,
            }
        )
    summary = pd.DataFrame(summaries)
    position_metric = (
        "median_tcoa_position"
        if cfg.ANALYSIS_METHOD == "robust_median"
        else "mean_tcoa_position"
    )
    position = replicates[position_metric].to_numpy()
    midpoint_p = (1 + np.count_nonzero(position <= 0.5)) / (cfg.N_BOOTSTRAPS + 1)
    summary["one_sided_bootstrap_p_vs_0_5"] = np.nan
    summary.loc[
        summary["metric"].eq(position_metric),
        "one_sided_bootstrap_p_vs_0_5",
    ] = midpoint_p
    return replicates, summary


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    tiny = 3e-14
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, 201):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < tiny:
            break
    return result


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return I_x(a, b), needed for the exact Student-t tail probability."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    leading_term = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return leading_term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (
        leading_term * _beta_continued_fraction(b, a, 1.0 - x) / b
    )


def pearson_positive_direction(
    predictor: np.ndarray,
    outcome: np.ndarray,
) -> tuple[float, float, int, float]:
    """Pearson r and the one-sided p-value for H1: correlation > 0."""
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Pearson inputs must be one-dimensional and equally sized")
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pearson correlation requires at least three finite pairs")
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = math.sqrt(float(centered_x @ centered_x) * float(centered_y @ centered_y))
    if denominator <= 0:
        raise ValueError("Pearson correlation is undefined for a constant variable")
    correlation = float(centered_x @ centered_y) / denominator
    correlation = float(np.clip(correlation, -1.0, 1.0))
    degrees_freedom = len(x) - 2
    if abs(correlation) == 1.0:
        t_statistic = math.copysign(math.inf, correlation)
        two_sided_p = 0.0
    else:
        t_statistic = correlation * math.sqrt(
            degrees_freedom / (1.0 - correlation**2)
        )
        beta_argument = degrees_freedom / (
            degrees_freedom + t_statistic**2
        )
        two_sided_p = _regularized_incomplete_beta(
            beta_argument,
            degrees_freedom / 2.0,
            0.5,
        )
    one_sided_p = (
        two_sided_p / 2.0
        if correlation >= 0
        else 1.0 - two_sided_p / 2.0
    )
    return correlation, t_statistic, degrees_freedom, one_sided_p


def behavior_correlations(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run directional Pearson correlations in the TCOA group only."""
    if not cfg.BEHAVIOR_FILE.exists():
        raise FileNotFoundError(f"Behavior workbook is missing: {cfg.BEHAVIOR_FILE}")
    behavior = pd.read_excel(cfg.BEHAVIOR_FILE, sheet_name="Sheet1")
    required_behavior_columns = {
        "ID",
        "final_score_RBANS",
        "final_score_miniBEST",
        "final_score_GDS",
    }
    missing_columns = required_behavior_columns.difference(behavior.columns)
    if missing_columns:
        raise ValueError(
            "Behavior workbook is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    numeric_ids = pd.to_numeric(behavior["ID"], errors="coerce")
    if numeric_ids.isna().any():
        raise ValueError("Behavior workbook contains a missing or nonnumeric ID")
    behavior = behavior.copy()
    behavior["participant"] = numeric_ids.astype(int).astype(str)
    if behavior["participant"].duplicated().any():
        duplicates = behavior.loc[
            behavior["participant"].duplicated(keep=False), "participant"
        ].tolist()
        raise ValueError(f"Duplicate participant IDs in behavior workbook: {duplicates}")

    tcoa = scores.loc[
        scores["group"].eq("TCOA"),
        ["participant", "group", "analysis_deviation"],
    ].copy()
    tcoa["participant"] = tcoa["participant"].astype(str)
    merged = tcoa.merge(
        behavior[
            [
                "participant",
                "final_score_RBANS",
                "final_score_miniBEST",
                "final_score_GDS",
            ]
        ],
        on="participant",
        how="left",
        validate="one_to_one",
    )
    expected_n = len(cfg.GROUPS["TCOA"])
    if len(merged) != expected_n:
        raise ValueError(f"Expected {expected_n} TCOA scores; found {len(merged)}")
    numeric_columns = [
        "analysis_deviation",
        "final_score_RBANS",
        "final_score_miniBEST",
        "final_score_GDS",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged[numeric_columns].isna().any().any():
        missing = merged.loc[
            merged[numeric_columns].isna().any(axis=1),
            ["participant", *numeric_columns],
        ]
        raise ValueError(f"Missing TCOA phenotype/behavior data:\n{missing}")

    merged["RBANS_healthy"] = merged["final_score_RBANS"]
    merged["MiniBEST_healthy"] = merged["final_score_miniBEST"]
    merged["GDS_healthy"] = -merged["final_score_GDS"]
    merged = merged.sort_values("participant").reset_index(drop=True)

    outcomes = (
        ("RBANS", "RBANS_healthy", "final_score_RBANS", "higher RBANS is better"),
        (
            "Mini-BESTest",
            "MiniBEST_healthy",
            "final_score_miniBEST",
            "higher Mini-BESTest is better",
        ),
        (
            "GDS (reverse-scored)",
            "GDS_healthy",
            "final_score_GDS",
            "higher reverse-scored GDS is better (lower raw GDS)",
        ),
    )
    predictor = merged["analysis_deviation"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for outcome, healthy_column, source_column, outcome_direction in outcomes:
        correlation, t_statistic, degrees_freedom, p_value = (
            pearson_positive_direction(
                predictor,
                merged[healthy_column].to_numpy(float),
            )
        )
        rows.append(
            {
                "sample": "TCOA only",
                "axis_estimator": cfg.PRIMARY_AXIS_ESTIMATOR,
                "predictor": "analysis_deviation (off-axis deviation)",
                "outcome": outcome,
                "source_column": source_column,
                "outcome_direction": outcome_direction,
                "alternative_hypothesis": "positive correlation with healthier function",
                "n": len(merged),
                "pearson_r": correlation,
                "t_statistic": t_statistic,
                "degrees_freedom": degrees_freedom,
                "one_sided_p": p_value,
                "multiple_testing_adjustment": "none",
            }
        )
    return merged, pd.DataFrame(rows)


def run_inference() -> None:
    features, _, values, groups = load_data()
    for legacy_name in (
        "axis_validation_result.csv",
        "reference_jackknife_sensitivity.csv",
    ):
        (cfg.INFERENCE_DIR / legacy_name).unlink(missing_ok=True)
    young = values[groups == "YAC"]
    old = values[groups == "OAC"]
    tcoa = values[groups == "TCOA"]
    primary_results: list[pd.DataFrame] = []
    robustness_results: list[pd.DataFrame] = []
    mean_sensitivity_results: list[pd.DataFrame] = []
    bootstrap_replicates_all: list[pd.DataFrame] = []
    bootstrap_summaries: list[pd.DataFrame] = []
    older_nulls: dict[str, dict[str, np.ndarray]] = {}

    for estimator in cfg.AXIS_ESTIMATORS:
        # Resetting to the same seed gives each estimator identical label
        # permutations and participant bootstrap indices for fair comparison.
        print(f"Running inference for axis estimator: {estimator}")
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        primary, robustness, mean_sensitivity, older_null = older_group_permutation(
            young,
            old,
            tcoa,
            rng,
            estimator,
        )
        bootstrap_replicates, bootstrap_summary = bootstrap_axis(
            young,
            old,
            tcoa,
            rng,
            estimator,
        )
        primary_results.append(primary)
        robustness_results.append(robustness)
        if not mean_sensitivity.empty:
            mean_sensitivity_results.append(mean_sensitivity)
        bootstrap_replicates_all.append(bootstrap_replicates)
        bootstrap_summaries.append(bootstrap_summary)
        older_nulls[estimator] = older_null

    primary_all = pd.concat(primary_results, ignore_index=True)
    robustness_all = pd.concat(robustness_results, ignore_index=True)
    mean_sensitivity_all = (
        pd.concat(mean_sensitivity_results, ignore_index=True)
        if mean_sensitivity_results
        else pd.DataFrame()
    )
    bootstrap_replicates_all_frame = pd.concat(
        bootstrap_replicates_all,
        ignore_index=True,
    )
    bootstrap_summary_all = pd.concat(bootstrap_summaries, ignore_index=True)
    primary_all.to_csv(
        cfg.INFERENCE_DIR / "primary_phenotype_inference.csv",
        index=False,
    )
    robustness_all.to_csv(
        cfg.INFERENCE_DIR / "robustness_inference.csv",
        index=False,
    )
    bootstrap_replicates_all_frame.to_csv(
        cfg.INFERENCE_DIR / "bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )
    bootstrap_summary_all.to_csv(
        cfg.INFERENCE_DIR / "bootstrap_summary.csv",
        index=False,
    )
    if not mean_sensitivity_all.empty:
        mean_sensitivity_all.to_csv(
            cfg.INFERENCE_DIR / "mean_sensitivity_inference.csv", index=False
        )
    primary_scores = pd.read_csv(
        cfg.PHENOTYPE_DIR / "participant_phenotype_scores.csv",
        dtype={"participant": str},
    )
    behavior_data, behavior_results = behavior_correlations(primary_scores)
    behavior_data.to_csv(
        cfg.INFERENCE_DIR / "behavior_analysis_data.csv",
        index=False,
    )
    behavior_results.to_csv(
        cfg.INFERENCE_DIR / "behavior_correlation_results.csv",
        index=False,
    )
    for obsolete_behavior_file in (
        "behavior_regression_results.csv",
        "behavior_huber_sensitivity.csv",
    ):
        (cfg.INFERENCE_DIR / obsolete_behavior_file).unlink(missing_ok=True)

    primary_estimator = cfg.PRIMARY_AXIS_ESTIMATOR
    primary_null = older_nulls[primary_estimator]
    np.savez_compressed(
        cfg.INFERENCE_DIR / "permutation_distributions.npz",
        position=primary_null["position"],
        deviation=primary_null["deviation"],
        relative_distance_youth=primary_null["relative_distance_youth"],
        rsa_fisher_difference=primary_null["rsa_fisher_difference"],
    )
    all_null_arrays: dict[str, np.ndarray] = {}
    for estimator in cfg.AXIS_ESTIMATORS:
        for metric in METRICS:
            all_null_arrays[f"{estimator}__{metric}"] = older_nulls[estimator][
                metric
            ]
    np.savez_compressed(
        cfg.INFERENCE_DIR / "permutation_distributions_by_estimator.npz",
        **all_null_arrays,
    )
    print(f"Phenotype inference results saved to {cfg.INFERENCE_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cfg.validate_config()
    ensure_output_directories()
    run_inference()


if __name__ == "__main__":
    main()
