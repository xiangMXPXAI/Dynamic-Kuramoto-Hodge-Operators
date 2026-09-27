"""Evaluate saved structural ablations for toroidal transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG = "conditioned"
RUN_ROOT = ROOT / "runs" / "dkho" / "small" / "form_0" / CONFIG
FULL = RUN_ROOT / "full" / "seed_42"
ORDER = ("full", "no_dirac", "no_phase", "no_harmonic")
LABEL = {
    "full": "Full TDK-HO",
    "no_dirac": "No Dirac coupling",
    "no_phase": "No phase dynamics",
    "no_harmonic": "No harmonic head",
}
COLOR = {"full": "#173f6c", "no_dirac": "#cc6d4d", "no_phase": "#b58b34", "no_harmonic": "#6b8e63"}


def run_dir(name: str) -> Path:
    return FULL if name == "full" else RUN_ROOT / name / "seed_42"


def load_metrics(strict_paper_budget: bool = False) -> dict:
    rows = {}
    for name in ORDER:
        root = run_dir(name)
        absent = [
            str(root / file)
            for file in ("best.pt", "test_metrics.json")
            if not (root / file).exists()
        ]
        if absent:
            raise FileNotFoundError(f"Incomplete saved structural run for {name}: {absent}")
        row = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
        if row.get("config") != CONFIG:
            raise RuntimeError(f"{name}: unexpected feature configuration {row.get('config')!r}")
        rows[name] = row
    
    if strict_paper_budget and (
        rows["full"]["epochs_completed"] != 300
        or any(
            rows[name]["epochs_completed"] != 200
            for name in ("no_dirac", "no_phase", "no_harmonic")
        )
    ):
        raise RuntimeError(
            "The requested paper-budget check expects full=300 and each ablation=200 epochs"
        )
    if any(
        rows[name]["parameters"] != rows["full"]["parameters"] for name in ("no_dirac", "no_phase")
    ):
        raise RuntimeError("No-Dirac and no-phase must preserve the full model parameter count")
    return rows


def plot(rows: dict, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#fbfcfe",
            "axes.facecolor": "#fbfcfe",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), facecolor="#fbfcfe")
    x = np.arange(len(ORDER))
    labels = ("Full", "No Dirac", "No phase", "No harmonic")
    specs = (
        ("mse", "MSE  ↓", True, "Physical test error"),
        ("relative_l2_mean", "mean relative $L_2$  ↓", False, "Field reconstruction error"),
    )
    for axis, (key, ylabel, log, subtitle) in zip(axes, specs):
        values = np.asarray([rows[name][key] for name in ORDER])
        bars = axis.bar(x, values, color=[COLOR[name] for name in ORDER], width=0.68, zorder=3)
        if log:
            axis.set_yscale("log")
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2e}" if log else f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#263d56",
            )
        axis.set_xticks(x, labels, rotation=16, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle, loc="left", fontsize=11.5, color="#1d334b")
        axis.grid(axis="y", color="#d8e1ea", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.75, bottom=0.18, wspace=0.14)
    fig.suptitle(
        "Toroidal transport C0 · TDK-HO small structural ablation",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.08,
        0.885,
        "Same V + H + L + G + F input condition and validation-best checkpoint selection; only the named structure is removed.",
        fontsize=9.2,
        color="#596d82",
    )
    fig.savefig(
        out / "01_structure_ablation_bars.png", dpi=320, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "metrics" / "current" / "structure_ablation",
    )
    parser.add_argument(
        "--strict-paper-budget",
        action="store_true",
        help="require the released full=300 / ablation=200 epoch schedule",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(args.strict_paper_budget)
    plot(rows, args.output)
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "task": "toroidal_transport C0",
                "condition": CONFIG,
                "seed": 42,
                "variants": list(ORDER),
                "matched_full_reference": str(FULL.relative_to(ROOT)),
                "selection": "saved validation-best test_metrics.json",
                "training_budget": {
                    name: int(row["epochs_completed"]) for name, row in rows.items()
                },
                "note": "No dense predictions were saved for C0 structure variants; table therefore contains the native saved C0 test metrics.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                name: {
                    key: rows[name][key]
                    for key in ("parameters", "mse", "relative_l2_mean", "best_val_epoch")
                }
                for name in ORDER
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
