"""Run the FC age-phenotype analysis pipeline.

By default, all four stages are run in order. Individual stages or an
inclusive stage range can be selected from the command line. Each stage runs
in a fresh Python process so environment overrides are applied consistently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
STATUS_FILE = PIPELINE_DIR / ".pipeline_status.json"

STAGES = {
    1: (
        "Build participant-level connectivity features",
        "01_build_participant_features.py",
    ),
    2: (
        "Calculate phenotype position and deviation",
        "02_calculate_phenotype_scores.py",
    ),
    3: (
        "Run phenotype inference and robustness analyses",
        "03_run_inference_and_behavior.py",
    ),
    4: (
        "Create the final workbook and figures",
        "04_create_results_workbook.py",
    ),
}


def utc_now() -> str:
    """Return a compact UTC timestamp for the run-status record."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_status() -> dict:
    """Read the existing status record, tolerating an absent/old file."""
    if not STATUS_FILE.exists():
        return {"completed_stages": []}
    try:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"completed_stages": []}
    if not isinstance(status, dict):
        return {"completed_stages": []}
    status.setdefault("completed_stages", [])
    return status


def save_status(status: dict) -> None:
    """Write status atomically so an interrupted write cannot corrupt it."""
    temporary_file = STATUS_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_file.replace(STATUS_FILE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all or selected stages of the FC age-phenotype pipeline."
    )
    parser.add_argument(
        "--method",
        choices=("position-winsor", "robust-median", "both"),
        default="both",
        help="Alt-2 analysis branch to run (default: both)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--stages",
        nargs="+",
        type=int,
        choices=STAGES,
        metavar="N",
        help="specific stage number(s) to run, for example: --stages 2 3",
    )
    selection.add_argument(
        "--from-stage",
        type=int,
        choices=STAGES,
        metavar="N",
        help="first stage in an inclusive range (default: 1)",
    )
    parser.add_argument(
        "--to-stage",
        type=int,
        choices=STAGES,
        metavar="N",
        help="last stage in an inclusive range (default: 4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="validation run with 100 permutations and 100 bootstraps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the selected commands without running them",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="list the available stages and exit",
    )
    return parser


def select_stages(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[int]:
    """Resolve explicit stages or an inclusive range into execution order."""
    if args.stages is not None:
        if args.to_stage is not None:
            parser.error("--to-stage cannot be combined with --stages")
        return sorted(set(args.stages))

    first = args.from_stage if args.from_stage is not None else min(STAGES)
    last = args.to_stage if args.to_stage is not None else max(STAGES)
    if first > last:
        parser.error("--from-stage cannot be greater than --to-stage")
    return list(range(first, last + 1))


def validate_stage_files(stage_numbers: list[int]) -> None:
    """Fail before execution when an expected pipeline script is missing."""
    missing = [
        str(PIPELINE_DIR / STAGES[number][1])
        for number in stage_numbers
        if not (PIPELINE_DIR / STAGES[number][1]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required pipeline stage file(s) not found:\n  " + "\n  ".join(missing)
        )


def print_plan(stage_numbers: list[int], quick: bool, method: str) -> None:
    mode = "QUICK VALIDATION (100 permutations, 100 bootstraps)" if quick else "PRODUCTION"
    print("=" * 72, flush=True)
    print("FC AGE-PHENOTYPE ANALYSIS PIPELINE", flush=True)
    print(f"Alt-2 method: {method}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Pipeline folder: {PIPELINE_DIR}", flush=True)
    print("Stages:", flush=True)
    for number in stage_numbers:
        print(f"  {number}. {STAGES[number][0]}", flush=True)
    print("=" * 72, flush=True)


def run_stage(number: int, environment: dict[str, str]) -> int:
    description, filename = STAGES[number]
    script_path = PIPELINE_DIR / filename
    command = [sys.executable, "-u", str(script_path)]

    print(f"\n--- Stage {number}: {description} ---", flush=True)
    print(f"Running: {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=PIPELINE_DIR,
        env=environment,
        check=False,
    )
    return completed.returncode


def main() -> int:
    global STATUS_FILE
    parser = build_parser()
    args = parser.parse_args()

    if args.method == "both":
        forwarded: list[str] = []
        skip_next = False
        for index, argument in enumerate(sys.argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if argument == "--method":
                skip_next = True
                continue
            if argument.startswith("--method="):
                continue
            forwarded.append(argument)
        for method in ("position-winsor", "robust-median"):
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--method",
                method,
                *forwarded,
            ]
            completed = subprocess.run(command, cwd=PIPELINE_DIR, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0

    method_key = args.method.replace("-", "_")
    STATUS_FILE = PIPELINE_DIR / f".pipeline_status_{method_key}.json"

    if args.list_stages:
        for number, (description, filename) in STAGES.items():
            print(f"{number}: {description} ({filename})")
        return 0

    stage_numbers = select_stages(args, parser)
    validate_stage_files(stage_numbers)
    print_plan(stage_numbers, args.quick, args.method)

    environment = os.environ.copy()
    environment["FAPA_ALT2_METHOD"] = method_key
    if args.quick:
        environment["FAPA_N_PERMUTATIONS"] = "100"
        environment["FAPA_N_BOOTSTRAPS"] = "100"

    if args.dry_run:
        for number in stage_numbers:
            print(f"Would run: {sys.executable} -u {PIPELINE_DIR / STAGES[number][1]}")
        return 0

    status = load_status()
    completed_stages = {
        int(number)
        for number in status.get("completed_stages", [])
        if str(number).isdigit() and int(number) in STAGES
    }
    # Rerunning a stage invalidates its prior status and every downstream stage.
    first_stage = min(stage_numbers)
    completed_stages = {number for number in completed_stages if number < first_stage}
    status.update(
        {
            "completed_stages": sorted(completed_stages),
            "last_run_started_utc": utc_now(),
            "last_run_mode": "quick" if args.quick else "production",
            "last_run_requested_stages": stage_numbers,
            "last_run_status": "running",
        }
    )
    save_status(status)

    for number in stage_numbers:
        try:
            return_code = run_stage(number, environment)
        except KeyboardInterrupt:
            status.update(
                {
                    "completed_stages": sorted(completed_stages),
                    "last_run_finished_utc": utc_now(),
                    "last_run_status": "interrupted",
                    "failed_stage": number,
                }
            )
            save_status(status)
            print("\nPipeline interrupted by user.", file=sys.stderr, flush=True)
            return 130

        if return_code != 0:
            status.update(
                {
                    "completed_stages": sorted(completed_stages),
                    "last_run_finished_utc": utc_now(),
                    "last_run_status": "failed",
                    "failed_stage": number,
                    "failed_stage_return_code": return_code,
                }
            )
            save_status(status)
            print(
                f"\nPipeline stopped: stage {number} failed with exit code {return_code}.",
                file=sys.stderr,
                flush=True,
            )
            return return_code

        completed_stages.add(number)
        status["completed_stages"] = sorted(completed_stages)
        status["last_completed_stage"] = number
        save_status(status)
        print(f"Stage {number} completed successfully.", flush=True)

    status.update(
        {
            "completed_stages": sorted(completed_stages),
            "last_run_finished_utc": utc_now(),
            "last_run_status": "completed",
        }
    )
    status.pop("failed_stage", None)
    status.pop("failed_stage_return_code", None)
    save_status(status)
    print("\nPipeline completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
