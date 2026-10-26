"""Stage 1: validate thresholded matrices and build participant 28-feature vectors."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_config as cfg
from analysis_utils import (
    average_participant_features,
    detect_matrix_scale,
    ensure_output_directories,
    feature_metadata,
    matrix_to_network_features,
    parse_state_window_key,
    validate_and_transform_matrix,
    write_json,
)


def matrix_path(participant: str, mode: str) -> Path:
    return (
        cfg.INPUT_ROOT
        / participant
        / mode
        / cfg.MATRIX_FILENAME_TEMPLATE.format(participant=participant)
    )


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = feature_metadata()
    feature_names = metadata["feature_name"].tolist()
    dynamic_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    total_files = len(cfg.all_participants()) * len(cfg.MODES)
    file_number = 0

    for group, participants in cfg.GROUPS.items():
        for participant in participants:
            for mode in cfg.MODES:
                file_number += 1
                path = matrix_path(participant, mode)
                print(f"\rLoading {file_number}/{total_files}: {path}", end="", flush=True)
                if not path.exists():
                    validation_rows.append(
                        {
                            "participant": participant,
                            "group": group,
                            "mode": mode,
                            "source_file": str(path),
                            "source_key": "",
                            "status": "missing_file",
                            "message": "Input archive does not exist",
                        }
                    )
                    continue
                try:
                    archive = np.load(path, allow_pickle=False)
                    if not archive.files:
                        raise ValueError("Archive contains no matrices")
                    first = np.asarray(archive[archive.files[0]])
                    if cfg.MATRIX_SCALE == "auto":
                        scale, reason = detect_matrix_scale(first, str(path))
                    else:
                        scale, reason = cfg.MATRIX_SCALE, "explicit configuration"
                    first_offdiag = first[np.triu_indices_from(first, k=1)]
                    scale_rows.append(
                        {
                            "participant": participant,
                            "group": group,
                            "mode": mode,
                            "source_file": str(path),
                            "matrix_scale": scale,
                            "reason": reason,
                            "offdiag_min": float(np.nanmin(first_offdiag)),
                            "offdiag_max": float(np.nanmax(first_offdiag)),
                            "diagonal_median": float(np.nanmedian(np.diag(first))),
                        }
                    )
                    for key in archive.files:
                        try:
                            state, window = parse_state_window_key(key)
                            transformed, diagnostics = validate_and_transform_matrix(
                                archive[key], f"{path}:{key}", scale
                            )
                            values = matrix_to_network_features(transformed)
                            row: dict[str, object] = {
                                "participant": participant,
                                "group": group,
                                "mode": mode,
                                "state": state,
                                "window": window,
                                "source_key": key,
                                "source_file": str(path),
                            }
                            row.update(dict(zip(feature_names, values)))
                            dynamic_rows.append(row)
                            validation_rows.append(
                                {
                                    "participant": participant,
                                    "group": group,
                                    "mode": mode,
                                    "source_file": str(path),
                                    "source_key": key,
                                    "state": state,
                                    "window": window,
                                    "status": "valid",
                                    "message": "",
                                    **diagnostics,
                                }
                            )
                        except Exception as exc:
                            validation_rows.append(
                                {
                                    "participant": participant,
                                    "group": group,
                                    "mode": mode,
                                    "source_file": str(path),
                                    "source_key": key,
                                    "status": "invalid_matrix",
                                    "message": str(exc),
                                }
                            )
                    archive.close()
                except Exception as exc:
                    validation_rows.append(
                        {
                            "participant": participant,
                            "group": group,
                            "mode": mode,
                            "source_file": str(path),
                            "source_key": "",
                            "status": "invalid_file",
                            "message": str(exc),
                        }
                    )
    print()
    dynamic = pd.DataFrame(dynamic_rows)
    validation = pd.DataFrame(validation_rows)
    scale_report = pd.DataFrame(scale_rows)
    if dynamic.empty:
        raise RuntimeError("No valid thresholded matrices were loaded")
    # Always preserve the audit files, including when validation stops the run.
    validation.to_csv(cfg.DATASET_DIR / "matrix_validation.csv", index=False)
    scale_report.to_csv(cfg.DATASET_DIR / "matrix_scale_detection.csv", index=False)
    invalid = validation[validation["status"].ne("valid")]
    if not invalid.empty:
        raise RuntimeError(
            f"Input validation found {len(invalid)} failures. See matrix_validation.csv"
        )

    mode, combined = average_participant_features(dynamic, feature_names)
    expected_participants = len(cfg.all_participants())
    if len(combined) != expected_participants:
        raise RuntimeError(
            f"Expected {expected_participants} participant vectors; created {len(combined)}"
        )
    matrix_counts = (
        dynamic.groupby(["participant", "group", "mode"], observed=True)
        .size()
        .rename("n_matrices")
        .reset_index()
    )

    dynamic.to_csv(
        cfg.DATASET_DIR / "dynamic_network_features.csv.gz",
        index=False,
        compression="gzip",
    )
    mode.to_csv(cfg.DATASET_DIR / "participant_mode_features.csv", index=False)
    combined.to_csv(cfg.DATASET_DIR / "participant_features.csv", index=False)
    matrix_counts.to_csv(cfg.DATASET_DIR / "matrix_counts.csv", index=False)
    metadata.to_csv(cfg.DATASET_DIR / "feature_metadata.csv", index=False)

    fingerprint = hashlib.sha256()
    fingerprint.update(
        combined.sort_values(["group", "participant"])
        .to_csv(index=False)
        .encode("utf-8")
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": cfg.ANALYSIS_VERSION,
        "input_root": str(cfg.INPUT_ROOT),
        "output_root": str(cfg.OUTPUT_ROOT),
        "matrix_filename": cfg.MATRIX_FILENAME_TEMPLATE,
        "matrix_scale": cfg.MATRIX_SCALE,
        "analysis_scale": "Fisher z",
        "n_participants": len(combined),
        "n_modes": len(mode),
        "n_matrices": len(dynamic),
        "n_features": len(feature_names),
        "averaging": "all matrices within mode, then EC and EO equally",
        "state_window_matching": False,
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "groups": {key: list(value) for key, value in cfg.GROUPS.items()},
    }
    write_json(cfg.DATASET_DIR / "dataset_manifest.json", manifest)
    print(
        f"Saved {len(dynamic)} matrix-level vectors and {len(combined)} "
        f"participant vectors to {cfg.DATASET_DIR}"
    )
    return dynamic, mode, combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    cfg.validate_config()
    ensure_output_directories()
    build_features()


if __name__ == "__main__":
    main()
