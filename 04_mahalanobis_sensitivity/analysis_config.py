"""Configuration for the FC age-phenotype analysis pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Repository root; all default paths are relative to it and can be
# overridden with the FAPA_* environment variables below.
REPO_ROOT = Path(__file__).resolve().parents[1]


ANALYSIS_VERSION = "5.0.0-mahalanobis-ledoit-wolf"

# This is a dedicated sensitivity pipeline. It intentionally retains the
# robust-median preprocessing from Alt-2 and changes only the multivariate
# geometry used after preprocessing.
ANALYSIS_METHOD = "robust_median"
ANALYSIS_METHODS = ("robust_median",)
DISTANCE_METRIC = "Mahalanobis distance with Ledoit-Wolf covariance"

INPUT_ROOT = Path(
    os.environ.get("FAPA_INPUT_ROOT", REPO_ROOT / "data" / "out" / "subjects")
)
OUTPUT_BASE_ROOT = Path(
    os.environ.get(
        "FAPA_OUTPUT_ROOT", REPO_ROOT / "results"
    )
)
OUTPUT_ROOT = OUTPUT_BASE_ROOT / "robust_median" / "mahalanobis_ledoit_wolf"
BEHAVIOR_FILE = Path(
    os.environ.get(
        "FAPA_BEHAVIOR_FILE",
        REPO_ROOT / "data" / "RBANS_miniBEST_GDS.xlsx",
    )
)
MATRIX_FILENAME_TEMPLATE = "{participant}_thresholded_matrices_raw.npz"
MODES = ("EC", "EO")

GROUPS = {
    "YAC": (
        "101", "102", "103", "104", "105", "111", "112", "113",
        "114", "115", "116", "117", "118", "119", "120",
    ),
    "OAC": (
        "202", "205", "206", "207", "208", "209", "210", "211",
        "214", "215", "216", "217", "218", "219", "221",
    ),
    "TCOA": (
        "401", "402", "403", "404", "406", "407", "408", "409",
        "410", "411", "412", "413", "414", "415", "416",
    ),
}

# The authoritative archives contain 100 Schaefer parcels followed by two
# background/medial-wall entries. Only the first 100 enter the analysis.
N_ATLAS_REGIONS = 100
TRIM_TRAILING_REGIONS = 2
FISHER_CLIP_EPSILON = 1e-7
MATRIX_SCALE = os.environ.get("FAPA_MATRIX_SCALE", "auto").lower()

NETWORK_ORDER = (
    "Visual",
    "Somatomotor",
    "DorsalAttention",
    "VentralAttention",
    "Limbic",
    "Frontoparietal",
    "Default",
)

# Canonical indices for Schaefer2018 100-parcel, 7-network order. These cover
# indices 0..99 exactly once and avoid relying on older participant-specific
# assignment files.
NETWORK_INDICES = {
    "Visual": tuple(range(41, 50)) + tuple(range(92, 100)),
    "Somatomotor": tuple(range(35, 41)) + tuple(range(84, 92)),
    "DorsalAttention": tuple(range(17, 25)) + tuple(range(70, 77)),
    "VentralAttention": tuple(range(28, 35)) + tuple(range(79, 84)),
    "Limbic": tuple(range(25, 28)) + tuple(range(77, 79)),
    "Frontoparietal": tuple(range(0, 4)) + tuple(range(50, 59)),
    "Default": tuple(range(4, 17)) + tuple(range(59, 70)),
}

PRIMARY_AXIS_ESTIMATOR = "geometric_median"
AXIS_ESTIMATORS = (
    "geometric_median",
    "coordinate_median",
    "mean",
)

# Robust primary and coordinate-median sensitivity analyses use the pooled
# YAC+OAC median and normal-consistent MAD. The conventional sensitivity
# analysis retains the original pooled mean and sample SD.
MAD_NORMAL_CONSISTENCY = 1.4826
REFERENCE_SCALE_FLOOR_QUANTILE = 0.05
REFERENCE_SCALE_ABSOLUTE_FLOOR = 1e-6

# Multivariate radial winsorization is fitted from the pooled YAC+OAC
# reference participants. A participant is capped when their robust RMS radius
# exceeds the reference median plus this many normal-consistent MADs.
RADIAL_WINSORIZATION_MAD_CUTOFF = float(
    os.environ.get("FAPA_RADIAL_WINSORIZATION_MAD_CUTOFF", "3.5")
)
RADIAL_WINSORIZATION_RADIUS_SCALE_FLOOR = float(
    os.environ.get("FAPA_RADIAL_WINSORIZATION_RADIUS_SCALE_FLOOR", "1e-6")
)

# The simpler branch fits this symmetric cap to phenotype positions pooled
# across all 30 older adults, without using OAC/TCOA labels. Only position is
# capped; the YAC-OAC axis and all other multivariate metrics remain unchanged.
POSITION_WINSORIZATION_MAD_CUTOFF = float(
    os.environ.get("FAPA_POSITION_WINSORIZATION_MAD_CUTOFF", "3.5")
)
POSITION_WINSORIZATION_SCALE_FLOOR = float(
    os.environ.get("FAPA_POSITION_WINSORIZATION_SCALE_FLOOR", "1e-6")
)

HUBER_TUNING_CONSTANT = float(
    os.environ.get("FAPA_HUBER_TUNING_CONSTANT", "1.345")
)
HUBER_TOLERANCE = float(os.environ.get("FAPA_HUBER_TOLERANCE", "1e-8"))
HUBER_MAX_ITERATIONS = int(os.environ.get("FAPA_HUBER_MAX_ITERATIONS", "200"))

# Weiszfeld iteration settings for the geometric median in covariance-whitened
# 28-feature space. Euclidean geometry after whitening is exactly the fitted
# Mahalanobis geometry in the original robust-standardized space.
GEOMETRIC_MEDIAN_TOLERANCE = float(
    os.environ.get("FAPA_GEOMETRIC_MEDIAN_TOLERANCE", "1e-6")
)
GEOMETRIC_MEDIAN_MAX_ITERATIONS = int(
    os.environ.get("FAPA_GEOMETRIC_MEDIAN_MAX_ITERATIONS", "1000")
)
GEOMETRIC_MEDIAN_DISTANCE_EPSILON = float(
    os.environ.get("FAPA_GEOMETRIC_MEDIAN_DISTANCE_EPSILON", "1e-12")
)
MAHALANOBIS_EIGEN_FLOOR_RELATIVE = float(
    os.environ.get("FAPA_MAHALANOBIS_EIGEN_FLOOR_RELATIVE", "1e-12")
)

RANDOM_SEED = int(os.environ.get("FAPA_RANDOM_SEED", "42"))
N_PERMUTATIONS = int(os.environ.get("FAPA_N_PERMUTATIONS", "10000"))
N_BOOTSTRAPS = int(os.environ.get("FAPA_N_BOOTSTRAPS", "5000"))

DATASET_DIR = OUTPUT_ROOT / "01_dataset"
PHENOTYPE_DIR = OUTPUT_ROOT / "02_phenotype"
INFERENCE_DIR = OUTPUT_ROOT / "03_inference"
FINAL_DIR = OUTPUT_ROOT / "04_final_results"


def all_participants() -> tuple[str, ...]:
    return tuple(participant for participants in GROUPS.values() for participant in participants)


def validate_config() -> None:
    if ANALYSIS_METHOD not in ANALYSIS_METHODS:
        raise ValueError(f"FAPA_ALT2_METHOD must be one of {ANALYSIS_METHODS}")
    if MATRIX_SCALE not in {"pearson", "fisher_z", "auto"}:
        raise ValueError("MATRIX_SCALE must be pearson, fisher_z, or auto")
    if N_PERMUTATIONS < 1 or N_BOOTSTRAPS < 1:
        raise ValueError("Permutation and bootstrap counts must be positive")
    if PRIMARY_AXIS_ESTIMATOR not in AXIS_ESTIMATORS:
        raise ValueError("PRIMARY_AXIS_ESTIMATOR must be listed in AXIS_ESTIMATORS")
    if len(AXIS_ESTIMATORS) != 3 or set(AXIS_ESTIMATORS) != {
        "geometric_median",
        "coordinate_median",
        "mean",
    }:
        raise ValueError("AXIS_ESTIMATORS contains an unsupported estimator")
    if MAD_NORMAL_CONSISTENCY <= 0:
        raise ValueError("MAD_NORMAL_CONSISTENCY must be positive")
    if not 0 <= REFERENCE_SCALE_FLOOR_QUANTILE <= 1:
        raise ValueError("REFERENCE_SCALE_FLOOR_QUANTILE must be in [0, 1]")
    if REFERENCE_SCALE_ABSOLUTE_FLOOR <= 0:
        raise ValueError("REFERENCE_SCALE_ABSOLUTE_FLOOR must be positive")
    if RADIAL_WINSORIZATION_MAD_CUTOFF <= 0:
        raise ValueError("RADIAL_WINSORIZATION_MAD_CUTOFF must be positive")
    if RADIAL_WINSORIZATION_RADIUS_SCALE_FLOOR <= 0:
        raise ValueError(
            "RADIAL_WINSORIZATION_RADIUS_SCALE_FLOOR must be positive"
        )
    if POSITION_WINSORIZATION_MAD_CUTOFF <= 0:
        raise ValueError("POSITION_WINSORIZATION_MAD_CUTOFF must be positive")
    if POSITION_WINSORIZATION_SCALE_FLOOR <= 0:
        raise ValueError("POSITION_WINSORIZATION_SCALE_FLOOR must be positive")
    if HUBER_TUNING_CONSTANT <= 0 or HUBER_TOLERANCE <= 0:
        raise ValueError("Huber tuning and tolerance values must be positive")
    if HUBER_MAX_ITERATIONS < 1:
        raise ValueError("HUBER_MAX_ITERATIONS must be positive")
    if GEOMETRIC_MEDIAN_TOLERANCE <= 0:
        raise ValueError("GEOMETRIC_MEDIAN_TOLERANCE must be positive")
    if GEOMETRIC_MEDIAN_MAX_ITERATIONS < 1:
        raise ValueError("GEOMETRIC_MEDIAN_MAX_ITERATIONS must be positive")
    if GEOMETRIC_MEDIAN_DISTANCE_EPSILON <= 0:
        raise ValueError("GEOMETRIC_MEDIAN_DISTANCE_EPSILON must be positive")
    if MAHALANOBIS_EIGEN_FLOOR_RELATIVE <= 0:
        raise ValueError("MAHALANOBIS_EIGEN_FLOOR_RELATIVE must be positive")
    seen = []
    for network in NETWORK_ORDER:
        if network not in NETWORK_INDICES:
            raise ValueError(f"Missing indices for network {network}")
        seen.extend(NETWORK_INDICES[network])
    if sorted(seen) != list(range(N_ATLAS_REGIONS)):
        raise ValueError("NETWORK_INDICES must cover atlas indices 0..99 exactly once")
    participant_ids = list(all_participants())
    if len(participant_ids) != len(set(participant_ids)):
        raise ValueError("Participant IDs must be unique across groups")
