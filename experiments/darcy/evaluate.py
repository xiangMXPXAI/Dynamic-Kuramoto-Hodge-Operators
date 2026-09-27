"""Evaluate saved DKHO and baseline predictions for Darcy cochain targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from common import (
    CACHE,
    TASK_TARGET,
    DarcyGeometry,
    checkpoint_feature_profile,
    prepare_cache,
    write_json,
)

ROOT = Path(__file__).resolve().parent
DKHO_ROOTS = (ROOT / "runs" / "dkho" / "large", ROOT / "runs" / "dkho" / "small")
BASE = ROOT / "runs" / "baseline" / "main"
OUT = ROOT / "reports" / "metrics" / "current"


def records(root: Path) -> list[Path]:
    
    
    return sorted(
        set(root.glob("form*/*/*/result.json")) | set(root.glob("form*/*/*/*/result.json"))
    )

def metric(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "mse": float(np.mean(error**2)),
        "relative_l1": float(
            np.mean(np.abs(error).sum(1) / np.maximum(np.abs(truth).sum(1), 1e-12))
        ),
        "relative_l2": float(
            np.mean(
                np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(truth, axis=1), 1e-12)
            )
        ),
    }

def run_group(run: str) -> tuple[str, ...]:
    
    parts = Path(run).parts
    if len(parts) < 3:
        raise ValueError(f"malformed run path: {run}")
    return parts[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    prepare_cache()
    geo = DarcyGeometry()
    args.output.mkdir(parents=True, exist_ok=True)
    test_indices = np.asarray(np.load(CACHE / "split_test.npy", mmap_mode="r"))
    target_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for root in (*DKHO_ROOTS, BASE):
        for path in records(root):
            saved = json.loads(path.read_text(encoding="utf-8"))
            run = path.parent
            task = saved["task"].replace("form_", "").replace("form", "")
            index = np.load(run / "test_indices.npy")
            if not np.array_equal(index, test_indices):
                raise ValueError(f"test index mismatch: {run}")
            scale = float(saved["normalization"]["y_scale"])
            pred = np.load(run / "prediction_test_normalized.npy") * scale
            truth = np.load(run / "target_test_normalized.npy") * scale
            if task not in target_cache:
                target_cache[task] = np.asarray(
                    np.load(CACHE / f"{TASK_TARGET[task]}.npy", mmap_mode="r")[test_indices],
                    dtype=np.float32,
                )
            expected = target_cache[task]
            if truth.shape != expected.shape or not np.allclose(
                truth, expected, rtol=1e-5, atol=1e-6
            ):
                raise ValueError(
                    f"saved target does not match the archive test target: {run}"
                )
            row = {
                key: value
                for key, value in saved.items()
                if key
                not in {
                    "normalization",
                    "input_protocol",
                    "mse",
                    "relative_l1",
                    "relative_l2",
                    "harmonic_relative_l2",
                }
            }
            row.update(metric(pred, truth))
            row["run"] = str(run.relative_to(ROOT))
            row["task"] = f"form_{task}"
            row["input_profile"] = checkpoint_feature_profile(saved["feature_config"])
            row["feature_config"] = row["input_profile"]
            if task == "0":
                interior = geo.node_component == 0
                row["interior_relative_l2"] = float(
                    np.mean(
                        np.linalg.norm((pred - truth)[:, interior], axis=1)
                        / np.maximum(np.linalg.norm(truth[:, interior], axis=1), 1e-12)
                    )
                )
            if task == "1":
                psi = geo.harmonic_basis_1
                gp, gt = pred @ psi, truth @ psi
                row["harmonic_relative_l2"] = float(
                    np.mean(
                        np.linalg.norm(gp - gt, axis=1)
                        / np.maximum(np.linalg.norm(gt, axis=1), 1e-12)
                    )
                )
                curl = (geo.d1 @ pred.T).T
                if "2" not in target_cache:
                    omega_truth = np.asarray(
                        np.load(CACHE / f"{TASK_TARGET['2']}.npy", mmap_mode="r")[test_indices],
                        dtype=np.float32,
                    )
                    target_cache["2"] = omega_truth
                else:
                    omega_truth = target_cache["2"]
                row["curl_to_target_relative_l2"] = float(
                    np.mean(
                        np.linalg.norm(curl - omega_truth, axis=1)
                        / np.maximum(np.linalg.norm(omega_truth, axis=1), 1e-12)
                    )
                )
            rows.append(row)
    
    
    for row in rows:
        if row["task"] != "form_1":
            continue
        
        
        candidate = [
            item
            for item in rows
            if item["family"] == row["family"]
            and item["task"] == "form_2"
            and item["feature_config"] == row["feature_config"]
            and item.get("structure_variant", "full") == row.get("structure_variant", "full")
            and run_group(item["run"]) == run_group(row["run"])
            and Path(item["run"]).name == Path(row["run"]).name
        ]
        if not candidate:
            continue
        form2 = candidate[0]
        p1 = np.load(ROOT / row["run"] / "prediction_test_normalized.npy") * float(
            json.loads((ROOT / row["run"] / "result.json").read_text())["normalization"]["y_scale"]
        )
        p2 = np.load(ROOT / form2["run"] / "prediction_test_normalized.npy") * float(
            json.loads((ROOT / form2["run"] / "result.json").read_text())["normalization"][
                "y_scale"
            ]
        )
        compatible = (geo.d1 @ p1.T).T - p2
        row["predicted_curl_compatibility_relative_l2"] = float(
            np.mean(
                np.linalg.norm(compatible, axis=1) / np.maximum(np.linalg.norm(p2, axis=1), 1e-12)
            )
        )
    fields = sorted({key for row in rows for key in row})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output / "metrics.json",
        {
            "n_runs": len(rows),
            "metrics": rows,
            "notes": {
                "norm": "unweighted Euclidean cochain metrics",
                "curl": "d1 q1 compared with archive omega2",
                "not_reported": "delta q=f physical residual",
            },
        },
    )
    print(f"[evaluate] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
