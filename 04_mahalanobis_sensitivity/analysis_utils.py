"""Shared calculations for the FC age-phenotype pipeline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import analysis_config as cfg


@dataclass(frozen=True)
class AxisModel:
    center: np.ndarray
    scale: np.ndarray
    young_endpoint: np.ndarray
    old_endpoint: np.ndarray
    direction: np.ndarray
    young_endpoint_standardized: np.ndarray
    old_endpoint_standardized: np.ndarray
    direction_standardized: np.ndarray
    projection_weight_standardized: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    whitener: np.ndarray
    dewhitener: np.ndarray
    direction_norm_squared: float
    scale_floor: float
    n_scaled_features_floored: int
    estimator: str
    center_method: str
    scale_method: str
    endpoint_method: str
    distance_metric: str
    young_endpoint_iterations: int
    old_endpoint_iterations: int
    young_endpoint_converged: bool
    old_endpoint_converged: bool
    ledoit_wolf_shrinkage: float
    covariance_condition_number: float
    covariance_eigenvalue_floor: float

    @property
    def young_centroid(self) -> np.ndarray:
        """Backward-compatible alias for outputs created before version 2."""
        return self.young_endpoint

    @property
    def old_centroid(self) -> np.ndarray:
        """Backward-compatible alias for outputs created before version 2."""
        return self.old_endpoint


@dataclass(frozen=True)
class RadialWinsorizer:
    """Reference-derived radial cap in robustly standardized feature space."""

    center: np.ndarray
    scale: np.ndarray
    feature_scale_floor: float
    n_feature_scales_floored: int
    radius_median: float
    radius_mad: float
    radius_cutoff: float
    mad_cutoff: float


@dataclass(frozen=True)
class PositionWinsorizer:
    """Label-blind symmetric cap for pooled older-adult phenotype positions."""

    center: float
    scale: float
    lower_bound: float
    upper_bound: float
    mad_cutoff: float


def ensure_output_directories() -> None:
    for path in (
        cfg.OUTPUT_ROOT,
        cfg.DATASET_DIR,
        cfg.PHENOTYPE_DIR,
        cfg.INFERENCE_DIR,
        cfg.FINAL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def parse_state_window_key(key: str) -> tuple[int, int]:
    match = re.fullmatch(r"(-?\d+)_(-?\d+)", str(key))
    if not match:
        raise ValueError(f"Unexpected matrix key {key!r}; expected '<state>_<window>'")
    return int(match.group(1)), int(match.group(2))


def feature_metadata() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_index = 0
    for network in cfg.NETWORK_ORDER:
        n_regions = len(cfg.NETWORK_INDICES[network])
        rows.append(
            {
                "feature_index": feature_index,
                "feature_name": f"within__{network}",
                "feature_type": "within",
                "network_1": network,
                "network_2": network,
                "n_edges": n_regions * (n_regions - 1) // 2,
            }
        )
        feature_index += 1
    for network_1, network_2 in combinations(cfg.NETWORK_ORDER, 2):
        rows.append(
            {
                "feature_index": feature_index,
                "feature_name": f"between__{network_1}__{network_2}",
                "feature_type": "between",
                "network_1": network_1,
                "network_2": network_2,
                "n_edges": len(cfg.NETWORK_INDICES[network_1])
                * len(cfg.NETWORK_INDICES[network_2]),
            }
        )
        feature_index += 1
    result = pd.DataFrame(rows)
    if len(result) != 28:
        raise RuntimeError(f"Expected 28 network features; created {len(result)}")
    return result


def detect_matrix_scale(matrix: np.ndarray, source: str) -> tuple[str, str]:
    matrix = np.asarray(matrix, dtype=float)
    upper = np.triu_indices_from(matrix, k=1)
    values = matrix[upper]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{source}: no finite off-diagonal values")
    diagonal = np.diag(matrix)
    diagonal_median = float(np.nanmedian(diagonal))
    if np.max(np.abs(finite)) > 1.0 + 1e-6:
        return "fisher_z", "off-diagonal values exceed the Pearson range"
    if "raw" in Path(source).name.lower() and np.isclose(diagonal_median, 1.0, atol=1e-5):
        return "pearson", "raw filename, unit diagonal, and values within [-1, 1]"
    if np.isclose(diagonal_median, 1.0, atol=1e-5):
        return "pearson", "unit diagonal and values within [-1, 1]"
    raise ValueError(
        f"{source}: matrix scale is ambiguous; set FAPA_MATRIX_SCALE explicitly"
    )


def validate_and_transform_matrix(
    matrix: np.ndarray,
    source: str,
    scale: str,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    expected = cfg.N_ATLAS_REGIONS + cfg.TRIM_TRAILING_REGIONS
    if matrix.ndim != 2 or matrix.shape != (expected, expected):
        raise ValueError(f"{source}: expected {expected}x{expected}; found {matrix.shape}")
    matrix = matrix[: cfg.N_ATLAS_REGIONS, : cfg.N_ATLAS_REGIONS]
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{source}: contains NaN or infinite values")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if asymmetry > 1e-5:
        raise ValueError(f"{source}: matrix is asymmetric (maximum={asymmetry:.3g})")
    upper = np.triu_indices(cfg.N_ATLAS_REGIONS, k=1)
    offdiag = matrix[upper]
    minimum = float(np.min(offdiag))
    maximum = float(np.max(offdiag))
    if scale == "pearson":
        if minimum < -1.0 - 1e-6 or maximum > 1.0 + 1e-6:
            raise ValueError(f"{source}: Pearson values outside [-1, 1]")
        transformed = np.zeros_like(matrix)
        values = np.arctanh(
            np.clip(offdiag, -1 + cfg.FISHER_CLIP_EPSILON, 1 - cfg.FISHER_CLIP_EPSILON)
        )
        transformed[upper] = values
        transformed[(upper[1], upper[0])] = values
    elif scale == "fisher_z":
        transformed = matrix.copy()
        np.fill_diagonal(transformed, 0.0)
    else:
        raise ValueError(f"Unknown matrix scale {scale!r}")
    return transformed, {
        "offdiag_min_input": minimum,
        "offdiag_max_input": maximum,
        "max_asymmetry": asymmetry,
        "analysis_scale": "fisher_z",
    }


def matrix_to_network_features(matrix_fisher_z: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for network in cfg.NETWORK_ORDER:
        indices = np.asarray(cfg.NETWORK_INDICES[network], dtype=int)
        block = matrix_fisher_z[np.ix_(indices, indices)]
        values.append(float(np.mean(block[np.triu_indices_from(block, k=1)])))
    for network_1, network_2 in combinations(cfg.NETWORK_ORDER, 2):
        indices_1 = np.asarray(cfg.NETWORK_INDICES[network_1], dtype=int)
        indices_2 = np.asarray(cfg.NETWORK_INDICES[network_2], dtype=int)
        block = matrix_fisher_z[np.ix_(indices_1, indices_2)]
        values.append(float(np.mean(block)))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (28,) or not np.all(np.isfinite(result)):
        raise ValueError("Network feature vector is not a finite 28-element vector")
    return result


def average_participant_features(dynamic: pd.DataFrame, feature_names: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_names = list(feature_names)
    mode = (
        dynamic.groupby(["participant", "group", "mode"], as_index=False, observed=True)[feature_names]
        .mean()
    )
    counts = mode.groupby(["participant", "group"], observed=True)["mode"].nunique()
    missing = counts[counts != len(cfg.MODES)]
    if not missing.empty:
        raise ValueError(f"Equal-mode averaging requires EC and EO: {missing.index.tolist()}")
    combined = (
        mode.groupby(["participant", "group"], as_index=False, observed=True)[feature_names]
        .mean()
    )
    return mode, combined


def _scale_with_floor(
    values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, float, int]:
    if method == "sd":
        raw_scale = np.std(values, axis=0, ddof=1)
    elif method == "mad":
        center = np.median(values, axis=0)
        raw_scale = cfg.MAD_NORMAL_CONSISTENCY * np.median(
            np.abs(values - center),
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported reference scale method {method!r}")
    positive = raw_scale[np.isfinite(raw_scale) & (raw_scale > 0)]
    quantile_floor = (
        float(np.quantile(positive, cfg.REFERENCE_SCALE_FLOOR_QUANTILE))
        if positive.size
        else cfg.REFERENCE_SCALE_ABSOLUTE_FLOOR
    )
    floor = max(quantile_floor, cfg.REFERENCE_SCALE_ABSOLUTE_FLOOR)
    invalid = ~np.isfinite(raw_scale) | (raw_scale < floor)
    adjusted = raw_scale.copy()
    adjusted[invalid] = floor
    return adjusted, floor, int(np.count_nonzero(invalid))


def fit_radial_winsorizer(reference: np.ndarray) -> RadialWinsorizer:
    """Fit the prespecified 3.5-MAD radial cap from reference participants."""
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or len(reference) == 0:
        raise ValueError("Radial winsorization requires a non-empty 2D reference")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Radial winsorization reference contains non-finite values")

    center = np.median(reference, axis=0)
    scale, feature_floor, n_floored = _scale_with_floor(reference, "mad")
    standardized = (reference - center) / scale
    radii = np.sqrt(np.mean(np.square(standardized), axis=1))
    radius_median = float(np.median(radii))
    radius_mad = float(
        cfg.MAD_NORMAL_CONSISTENCY
        * np.median(np.abs(radii - radius_median))
    )
    radius_mad = max(
        radius_mad,
        cfg.RADIAL_WINSORIZATION_RADIUS_SCALE_FLOOR,
    )
    radius_cutoff = float(
        radius_median + cfg.RADIAL_WINSORIZATION_MAD_CUTOFF * radius_mad
    )
    if not np.isfinite(radius_cutoff) or radius_cutoff <= 0:
        raise ValueError("Radial winsorization produced an invalid radius cutoff")
    return RadialWinsorizer(
        center=center,
        scale=scale,
        feature_scale_floor=feature_floor,
        n_feature_scales_floored=n_floored,
        radius_median=radius_median,
        radius_mad=radius_mad,
        radius_cutoff=radius_cutoff,
        mad_cutoff=cfg.RADIAL_WINSORIZATION_MAD_CUTOFF,
    )


def apply_radial_winsorizer(
    values: np.ndarray,
    winsorizer: RadialWinsorizer,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cap multivariate radius while preserving each vector's direction."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != len(winsorizer.center):
        raise ValueError("Winsorization values do not match the reference dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("Winsorization values contain non-finite entries")

    standardized = (values - winsorizer.center) / winsorizer.scale
    original_radius = np.sqrt(np.mean(np.square(standardized), axis=1))
    shrinkage_factor = np.ones(len(values), dtype=float)
    flagged = original_radius > winsorizer.radius_cutoff
    shrinkage_factor[flagged] = (
        winsorizer.radius_cutoff / original_radius[flagged]
    )
    adjusted_standardized = standardized * shrinkage_factor[:, np.newaxis]
    adjusted = (
        winsorizer.center
        + adjusted_standardized * winsorizer.scale
    )
    adjusted_radius = np.sqrt(
        np.mean(np.square(adjusted_standardized), axis=1)
    )
    return adjusted, {
        "original_radius": original_radius,
        "adjusted_radius": adjusted_radius,
        "shrinkage_factor": shrinkage_factor,
        "is_winsorized": flagged,
    }


def winsorize_axis_data(
    young: np.ndarray,
    old: np.ndarray,
    *targets: np.ndarray,
) -> tuple[RadialWinsorizer, tuple[np.ndarray, ...]]:
    """Fit on pooled YAC+OAC and apply the same cap to every supplied group."""
    winsorizer = fit_radial_winsorizer(np.vstack([young, old]))
    adjusted = tuple(
        apply_radial_winsorizer(values, winsorizer)[0]
        for values in (young, old, *targets)
    )
    return winsorizer, adjusted


def fit_position_winsorizer(older_positions: np.ndarray) -> PositionWinsorizer:
    """Fit a symmetric median/MAD cap without using OAC/TCOA labels."""
    positions = np.asarray(older_positions, dtype=float).reshape(-1)
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("Position winsorization requires finite older-adult scores")
    center = float(np.median(positions))
    scale = float(
        cfg.MAD_NORMAL_CONSISTENCY * np.median(np.abs(positions - center))
    )
    scale = max(scale, cfg.POSITION_WINSORIZATION_SCALE_FLOOR)
    half_width = cfg.POSITION_WINSORIZATION_MAD_CUTOFF * scale
    return PositionWinsorizer(
        center=center,
        scale=scale,
        lower_bound=center - half_width,
        upper_bound=center + half_width,
        mad_cutoff=cfg.POSITION_WINSORIZATION_MAD_CUTOFF,
    )


def apply_position_winsorizer(
    positions: np.ndarray,
    winsorizer: PositionWinsorizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap both tails at common bounds and return adjusted values and flags."""
    values = np.asarray(positions, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Position winsorization values contain non-finite entries")
    adjusted = np.clip(values, winsorizer.lower_bound, winsorizer.upper_bound)
    return adjusted, adjusted != values


def winsorize_older_positions(
    old_positions: np.ndarray,
    tcoa_positions: np.ndarray,
) -> tuple[PositionWinsorizer, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one pooled older-adult cap and apply it to both labeled groups."""
    old = np.asarray(old_positions, dtype=float)
    tcoa = np.asarray(tcoa_positions, dtype=float)
    winsorizer = fit_position_winsorizer(np.concatenate([old, tcoa]))
    adjusted_old, old_flags = apply_position_winsorizer(old, winsorizer)
    adjusted_tcoa, tcoa_flags = apply_position_winsorizer(tcoa, winsorizer)
    return winsorizer, adjusted_old, adjusted_tcoa, old_flags, tcoa_flags


def geometric_median(
    values: np.ndarray,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    distance_epsilon: float | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Return the Euclidean geometric median using modified Weiszfeld updates."""
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Geometric median requires a non-empty 2D array")
    if not np.all(np.isfinite(points)):
        raise ValueError("Geometric median input contains non-finite values")
    tolerance = (
        cfg.GEOMETRIC_MEDIAN_TOLERANCE
        if tolerance is None
        else float(tolerance)
    )
    max_iterations = (
        cfg.GEOMETRIC_MEDIAN_MAX_ITERATIONS
        if max_iterations is None
        else int(max_iterations)
    )
    distance_epsilon = (
        cfg.GEOMETRIC_MEDIAN_DISTANCE_EPSILON
        if distance_epsilon is None
        else float(distance_epsilon)
    )
    if tolerance <= 0 or max_iterations < 1 or distance_epsilon <= 0:
        raise ValueError("Invalid geometric-median convergence settings")
    if len(points) == 1:
        return points[0].copy(), 0, True

    current = np.median(points, axis=0)
    previous_objective = float(
        np.sum(np.linalg.norm(points - current, axis=1))
    )
    for iteration in range(1, max_iterations + 1):
        differences = points - current
        distances = np.linalg.norm(differences, axis=1)
        coincident = distances <= distance_epsilon
        noncoincident = ~coincident
        if not np.any(noncoincident):
            return current, iteration, True

        inverse_distances = 1.0 / distances[noncoincident]
        candidate = np.average(
            points[noncoincident],
            axis=0,
            weights=inverse_distances,
        )
        if np.any(coincident):
            unit_sum = np.sum(
                differences[noncoincident]
                / distances[noncoincident, np.newaxis],
                axis=0,
            )
            residual_norm = float(np.linalg.norm(unit_sum))
            n_coincident = int(np.count_nonzero(coincident))
            if residual_norm <= n_coincident:
                updated = current
            else:
                fraction = max(0.0, 1.0 - n_coincident / residual_norm)
                updated = fraction * candidate + (1.0 - fraction) * current
        else:
            updated = candidate

        movement = float(np.linalg.norm(updated - current))
        threshold = tolerance * max(1.0, float(np.linalg.norm(current)))
        updated_objective = float(
            np.sum(np.linalg.norm(points - updated, axis=1))
        )
        objective_change = abs(previous_objective - updated_objective)
        objective_threshold = tolerance * max(1.0, previous_objective)
        current = updated
        if movement <= threshold or objective_change <= objective_threshold:
            return current, iteration, True
        previous_objective = updated_objective
    return current, max_iterations, False


def fit_axis(
    young: np.ndarray,
    old: np.ndarray,
    estimator: str | None = None,
) -> AxisModel:
    young = np.asarray(young, dtype=float)
    old = np.asarray(old, dtype=float)
    if young.ndim != 2 or old.ndim != 2 or young.shape[1] != old.shape[1]:
        raise ValueError("Young and old reference arrays must share a feature dimension")
    if len(young) == 0 or len(old) == 0:
        raise ValueError("Young and old reference groups must be non-empty")
    if not np.all(np.isfinite(young)) or not np.all(np.isfinite(old)):
        raise ValueError("Reference arrays contain non-finite values")
    estimator = cfg.PRIMARY_AXIS_ESTIMATOR if estimator is None else estimator
    if estimator not in cfg.AXIS_ESTIMATORS:
        raise ValueError(f"Unsupported axis estimator {estimator!r}")

    reference = np.vstack([young, old])
    if estimator == "mean":
        center_method = "pooled reference arithmetic mean"
        scale_method = "pooled reference sample SD"
        endpoint_method = "group arithmetic mean"
        center = np.mean(reference, axis=0)
        scale, floor, n_floored = _scale_with_floor(reference, "sd")
    else:
        center_method = "pooled reference coordinate-wise median"
        scale_method = "pooled reference normal-consistent MAD"
        endpoint_method = (
            "group geometric median in Ledoit-Wolf covariance-whitened space"
            if estimator == "geometric_median"
            else "group coordinate-wise median in Ledoit-Wolf covariance-whitened space"
        )
        center = np.median(reference, axis=0)
        scale, floor, n_floored = _scale_with_floor(reference, "mad")

    young_z = (young - center) / scale
    old_z = (old - center) / scale

    # Estimate covariance from YAC+OAC only. Ledoit-Wolf shrinkage makes the
    # inverse stable when 28 features are estimated from 30 reference people.
    covariance_estimator = LedoitWolf(assume_centered=False).fit(
        np.vstack([young_z, old_z])
    )
    covariance = np.asarray(covariance_estimator.covariance_, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_floor = max(
        float(np.max(eigenvalues)) * cfg.MAHALANOBIS_EIGEN_FLOOR_RELATIVE,
        np.finfo(float).eps,
    )
    stable_eigenvalues = np.maximum(eigenvalues, eigen_floor)
    whitener = (
        eigenvectors * (1.0 / np.sqrt(stable_eigenvalues))
    ) @ eigenvectors.T
    dewhitener = (
        eigenvectors * np.sqrt(stable_eigenvalues)
    ) @ eigenvectors.T
    precision = (eigenvectors * (1.0 / stable_eigenvalues)) @ eigenvectors.T
    young_metric = young_z @ whitener
    old_metric = old_z @ whitener
    if estimator == "geometric_median":
        young_endpoint, young_iterations, young_converged = geometric_median(
            young_metric
        )
        old_endpoint, old_iterations, old_converged = geometric_median(old_metric)
        if not young_converged or not old_converged:
            raise RuntimeError(
                "Geometric-median endpoint failed to converge within "
                f"{cfg.GEOMETRIC_MEDIAN_MAX_ITERATIONS} iterations"
            )
    elif estimator == "coordinate_median":
        young_endpoint = np.median(young_metric, axis=0)
        old_endpoint = np.median(old_metric, axis=0)
        young_iterations = old_iterations = 0
        young_converged = old_converged = True
    else:
        endpoint_method = "group arithmetic mean in Ledoit-Wolf covariance-whitened space"
        young_endpoint = np.mean(young_metric, axis=0)
        old_endpoint = np.mean(old_metric, axis=0)
        young_iterations = old_iterations = 0
        young_converged = old_converged = True

    direction = young_endpoint - old_endpoint
    norm_squared = float(direction @ direction)
    if not np.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise ValueError("The YAC-OAC phenotype direction is degenerate")
    young_endpoint_standardized = young_endpoint @ dewhitener
    old_endpoint_standardized = old_endpoint @ dewhitener
    direction_standardized = (
        young_endpoint_standardized - old_endpoint_standardized
    )
    projection_weight_standardized = (
        whitener @ direction / norm_squared
    )
    return AxisModel(
        center=center,
        scale=scale,
        young_endpoint=young_endpoint,
        old_endpoint=old_endpoint,
        direction=direction,
        young_endpoint_standardized=young_endpoint_standardized,
        old_endpoint_standardized=old_endpoint_standardized,
        direction_standardized=direction_standardized,
        projection_weight_standardized=projection_weight_standardized,
        covariance=covariance,
        precision=precision,
        whitener=whitener,
        dewhitener=dewhitener,
        direction_norm_squared=norm_squared,
        scale_floor=floor,
        n_scaled_features_floored=n_floored,
        estimator=estimator,
        center_method=center_method,
        scale_method=scale_method,
        endpoint_method=endpoint_method,
        distance_metric=cfg.DISTANCE_METRIC,
        young_endpoint_iterations=young_iterations,
        old_endpoint_iterations=old_iterations,
        young_endpoint_converged=young_converged,
        old_endpoint_converged=old_converged,
        ledoit_wolf_shrinkage=float(covariance_estimator.shrinkage_),
        covariance_condition_number=float(
            np.max(stable_eigenvalues) / np.min(stable_eigenvalues)
        ),
        covariance_eigenvalue_floor=eigen_floor,
    )


def _row_correlations(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    template = np.asarray(template, dtype=float)
    centered_values = values - np.mean(values, axis=1, keepdims=True)
    centered_template = template - np.mean(template)
    denominator = np.linalg.norm(centered_values, axis=1) * np.linalg.norm(centered_template)
    correlations = np.divide(
        centered_values @ centered_template,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=denominator > 0,
    )
    return np.clip(correlations, -1 + 1e-7, 1 - 1e-7)


def score_axis(values: np.ndarray, model: AxisModel) -> dict[str, np.ndarray]:
    values = np.atleast_2d(np.asarray(values, dtype=float))
    standardized = (values - model.center) / model.scale
    metric_values = standardized @ model.whitener
    displacement = metric_values - model.old_endpoint
    position = (displacement @ model.direction) / model.direction_norm_squared
    projection = model.old_endpoint + position[:, None] * model.direction
    deviation = np.sqrt(np.mean(np.square(metric_values - projection), axis=1))
    distance_young = np.sqrt(
        np.mean(np.square(metric_values - model.young_endpoint), axis=1)
    )
    distance_old = np.sqrt(
        np.mean(np.square(metric_values - model.old_endpoint), axis=1)
    )
    relative_distance = np.divide(
        distance_old,
        distance_young + distance_old,
        out=np.full_like(distance_old, np.nan),
        where=(distance_young + distance_old) > 0,
    )
    # RSA remains a correlation-pattern robustness metric in the original
    # robust-standardized feature coordinates, not in covariance-whitened space.
    r_young = _row_correlations(standardized, model.young_endpoint_standardized)
    r_old = _row_correlations(standardized, model.old_endpoint_standardized)
    rsa_difference = np.arctanh(r_young) - np.arctanh(r_old)
    standardized_displacement = standardized - model.old_endpoint_standardized
    contributions = (
        standardized_displacement * model.projection_weight_standardized
    )
    return {
        "standardized": standardized,
        "metric_values": metric_values,
        "position": position,
        "deviation": deviation,
        "distance_young": distance_young,
        "distance_old": distance_old,
        "relative_distance_youth": relative_distance,
        "rsa_r_young": r_young,
        "rsa_r_old": r_old,
        "rsa_fisher_difference": rsa_difference,
        "contributions": contributions,
    }


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    pooled = (
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled <= 0:
        return math.nan
    d = (np.mean(a) - np.mean(b)) / math.sqrt(pooled)
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float(correction * d)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted


def hc3_ols(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    leverage = np.einsum("ij,jk,ik->i", design, inverse, design)
    adjusted = residual / np.clip(1 - leverage, 1e-8, None)
    meat = design.T @ ((adjusted ** 2)[:, None] * design)
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0, None))
    t_value = np.divide(beta, standard_error, out=np.full_like(beta, np.nan), where=standard_error > 0)
    return beta, standard_error, t_value, residual


def huber_regression(
    y: np.ndarray,
    design: np.ndarray,
    tuning_constant: float | None = None,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Fit a Huber M-estimator by iteratively reweighted least squares."""
    outcome = np.asarray(y, dtype=float)
    matrix = np.asarray(design, dtype=float)
    if outcome.ndim != 1 or matrix.ndim != 2 or len(outcome) != len(matrix):
        raise ValueError("Huber regression inputs have incompatible shapes")
    if not np.all(np.isfinite(outcome)) or not np.all(np.isfinite(matrix)):
        raise ValueError("Huber regression inputs contain non-finite values")
    tuning = cfg.HUBER_TUNING_CONSTANT if tuning_constant is None else tuning_constant
    tol = cfg.HUBER_TOLERANCE if tolerance is None else tolerance
    limit = cfg.HUBER_MAX_ITERATIONS if max_iterations is None else max_iterations
    beta = np.linalg.pinv(matrix) @ outcome
    weights = np.ones(len(outcome), dtype=float)
    converged = False
    for iteration in range(1, limit + 1):
        residual = outcome - matrix @ beta
        residual_center = np.median(residual)
        scale = cfg.MAD_NORMAL_CONSISTENCY * np.median(
            np.abs(residual - residual_center)
        )
        scale = max(float(scale), 1e-8)
        standardized = residual / scale
        weights = np.ones_like(standardized)
        large = np.abs(standardized) > tuning
        weights[large] = tuning / np.abs(standardized[large])
        weighted_design = matrix * np.sqrt(weights)[:, None]
        weighted_outcome = outcome * np.sqrt(weights)
        updated = np.linalg.pinv(weighted_design) @ weighted_outcome
        if np.linalg.norm(updated - beta) <= tol * max(1.0, np.linalg.norm(beta)):
            beta = updated
            converged = True
            break
        beta = updated
    residual = outcome - matrix @ beta
    return beta, residual, weights, iteration, converged


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Return Cliff's delta: probability(a>b) minus probability(a<b)."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    differences = a[:, None] - b[None, :]
    return float((np.count_nonzero(differences > 0) - np.count_nonzero(differences < 0)) / differences.size)


def percentile_interval(values: np.ndarray, lower: float = 2.5, upper: float = 97.5) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, lower)), float(np.percentile(values, upper))
