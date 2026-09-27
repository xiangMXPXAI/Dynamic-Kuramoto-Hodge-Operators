"""Evaluate matched-input feature controls for the Darcy benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import (
    CACHE,
    FEATURE_CONFIGS,
    TASK_TARGET,
    checkpoint_feature_profile,
    feature_profile_details,
    prepare_cache,
)

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
DEFAULT_RUNS = ROOT / "runs" / "baseline" / "main"
DEFAULT_OUTPUT = ROOT / "reports" / "feature_controls"
MODEL_NAMES = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")
TASKS = ("0", "1", "2")


def relative_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    

    error = prediction - target
    return {
        "mse": float(np.mean(np.square(error))),
        "relative_l1": float(
            np.mean(np.abs(error).sum(1) / np.maximum(np.abs(target).sum(1), 1e-12))
        ),
        "relative_l2": float(
            np.mean(
                np.linalg.norm(error, axis=1)
                / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
            )
        ),
    }


def parse_csv(value: str, allowed: tuple[str, ...], option: str) -> tuple[str, ...]:
    requested = allowed if value == "all" else tuple(item for item in value.split(",") if item)
    invalid = tuple(item for item in requested if item not in allowed)
    if not requested or invalid:
        choices = ", ".join(allowed)
        raise ValueError(f"{option} contains unsupported value(s) {invalid}; choices: {choices}")
    return requested


def run_path(root: Path, task: str, profile: str, model: str, seed: int) -> Path:
    return root / f"form_{task}" / profile / model / f"seed_{seed}"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external>/{path.name}"


def load_control(
    root: Path,
    task: str,
    profile: str,
    model: str,
    seed: int,
    test_indices: np.ndarray,
    target_cache: dict[str, np.ndarray],
) -> dict:
    

    run = run_path(root, task, profile, model, seed)
    result_path = run / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    required = (
        run / "best.pt",
        run / "prediction_test_normalized.npy",
        run / "target_test_normalized.npy",
        run / "test_indices.npy",
    )
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{run}: missing required artifact(s): {', '.join(missing)}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("family") != model:
        raise ValueError(f"{run}: family={result.get('family')!r}, expected {model!r}")
    if result.get("task") != f"form_{task}":
        raise ValueError(f"{run}: task={result.get('task')!r}, expected form_{task!s}")
    saved_profile = checkpoint_feature_profile(str(result.get("feature_config", "")))
    if saved_profile != profile:
        raise ValueError(
            f"{run}: feature_config={result.get('feature_config')!r} "
            f"(public={saved_profile!r}), expected {profile!r}"
        )
    saved_indices = np.asarray(np.load(run / "test_indices.npy"), dtype=np.int64)
    if not np.array_equal(saved_indices, test_indices):
        raise ValueError(f"{run}: saved test indices do not match the released Darcy split")

    scale = float(result["normalization"]["y_scale"])
    prediction = np.asarray(np.load(run / "prediction_test_normalized.npy"), dtype=np.float32) * scale
    target = np.asarray(np.load(run / "target_test_normalized.npy"), dtype=np.float32) * scale
    if task not in target_cache:
        target_cache[task] = np.asarray(
            np.load(CACHE / f"{TASK_TARGET[task]}.npy", mmap_mode="r")[test_indices],
            dtype=np.float32,
        )
    expected_target = target_cache[task]
    if target.shape != expected_target.shape or not np.allclose(
        target, expected_target, rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"{run}: saved target is inconsistent with the released Darcy archive")
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError(f"{run}: prediction has invalid shape or non-finite values")

    return {
        "task": f"form_{task}",
        "input_profile": profile,
        "model": model,
        "seed": seed,
        "parameters": int(result["parameters"]),
        "best_epoch": int(result.get("best_epoch", 0)),
        "best_val_relative_l2": float(result["best_val_relative_l2"]),
        "native_input_ranks": ",".join(str(rank) for rank in result["native_input_ranks"]),
        **relative_metrics(prediction, target),
        "run": display_path(run),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument(
        "--profiles",
        default="native,conditioned",
        help="comma-separated input profiles, or 'all' after all profiles have been trained",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS,
        help="baseline run root; relative paths are interpreted from experiments/darcy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="report directory; relative paths are interpreted from experiments/darcy",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless every requested task/profile/model/seed run is present",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    profiles = parse_csv(args.profiles, tuple(FEATURE_CONFIGS), "--profiles")
    tasks = TASKS if args.task == "all" else (args.task,)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    runs_root = args.runs_root if args.runs_root.is_absolute() else ROOT / args.runs_root
    output = args.output if args.output.is_absolute() else ROOT / args.output

    prepare_cache()
    test_indices = np.asarray(np.load(CACHE / "split_test.npy", mmap_mode="r"), dtype=np.int64)
    target_cache: dict[str, np.ndarray] = {}
    rows, absent = [], []
    for task in tasks:
        for profile in profiles:
            for model in models:
                candidate = run_path(runs_root, task, profile, model, args.seed)
                if not (candidate / "result.json").exists():
                    absent.append(display_path(candidate))
                    continue
                rows.append(
                    load_control(
                        runs_root,
                        task,
                        profile,
                        model,
                        args.seed,
                        test_indices,
                        target_cache,
                    )
                )
    if absent and args.require_complete:
        raise FileNotFoundError(
            "Missing requested feature-control runs:\n  - " + "\n  - ".join(absent)
        )
    if not rows:
        raise FileNotFoundError("No requested Darcy feature-control result was found")

    output.mkdir(parents=True, exist_ok=True)
    fields = (
        "task",
        "input_profile",
        "model",
        "seed",
        "parameters",
        "native_input_ranks",
        "best_epoch",
        "best_val_relative_l2",
        "relative_l1",
        "relative_l2",
        "mse",
        "run",
    )
    rows.sort(key=lambda row: (row["task"], row["input_profile"], row["model"]))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "benchmark": "perforated_darcy",
        "purpose": (
            "matched-input control: evaluate whether reference baselines explain the result "
            "when supplied with the same target-safe condition profile as DKHO"
        ),
        "protocol": {
            "fixed_across_profiles": (
                "dataset split, seed, loss, optimizer schedule, baseline family, and native-rank support"
            ),
            "varied_across_profiles": "only documented target-safe condition channels",
            "profile_definitions": {
                name: feature_profile_details(FEATURE_CONFIGS[name]) for name in profiles
            },
        },
        "runs_root": display_path(runs_root),
        "requested": {"tasks": [f"form_{task}" for task in tasks], "profiles": profiles, "models": models},
        "completed_runs": len(rows),
        "missing_runs": absent,
        "metrics": rows,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[feature-control] wrote {len(rows)} validated runs to {output}", flush=True)
    if absent:
        print(f"[feature-control] {len(absent)} requested run(s) are not present", flush=True)


if __name__ == "__main__":
    main()
