"""Evaluate saved structural ablations for the magnetostatics benchmark."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

try:  
    from . import evaluate as parent
except ImportError:  
    import evaluate as parent


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs" / "dkho"
REPORT_ROOT = ROOT / "reports" / "metrics" / "current"


FULL = RUN_ROOT / "small" / "form_2" / "conditioned" / "full" / "seed_42"
ABLATION = RUN_ROOT / "small" / "form_2" / "conditioned"
DATA = ROOT / "data" / "cavity_magnetostatics_v1.pkl"
ORDER = ("full", "no_dirac", "no_phase", "no_harmonic")
LABEL = {
    "full": "Full TDK-HO",
    "no_dirac": "No Dirac coupling",
    "no_phase": "No phase dynamics",
    "no_harmonic": "No harmonic head",
}
COLOR = {"full": "#173f6c", "no_dirac": "#cc6d4d", "no_phase": "#b58b34", "no_harmonic": "#6b8e63"}


def run_dir(name: str) -> Path:
    return FULL if name == "full" else ABLATION / name / "seed_42"


def load_runs() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
    predictions, records, target, indices = {}, {}, None, None
    for name in ORDER:
        root = run_dir(name)
        required = [
            root / file
            for file in (
                "prediction_test.npy",
                "target_test.npy",
                "test_indices.npy",
                "result.json",
            )
        ]
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            raise FileNotFoundError(f"{name} has no complete saved test artefacts: {absent}")
        prediction = np.load(root / "prediction_test.npy").astype(np.float32)
        current_target = np.load(root / "target_test.npy").astype(np.float32)
        current_indices = np.load(root / "test_indices.npy")
        if target is None:
            target, indices = current_target, current_indices
        elif not np.array_equal(target, current_target) or not np.array_equal(
            indices, current_indices
        ):
            raise RuntimeError(
                f"{name} test targets or sample IDs differ from full; refusing mixed evaluation"
            )
        predictions[name] = prediction
        records[name] = json.loads((root / "result.json").read_text(encoding="utf-8"))
    return predictions, target, indices, records


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=(1, 2)) / (
        np.linalg.norm(target, axis=(1, 2)) + 1e-12
    )


def create_evaluator(data: dict):
    sys.path.insert(0, str(parent.HSD_ROOT))
    from dataset import VectorFluxMapper
    from spectral_operators import HighOrderSpectralOperators

    host = HighOrderSpectralOperators(data["nodes"], data["elements"], k_list=(64, 64, 64))
    mapper = VectorFluxMapper(data["nodes"], data["elements"], mesh_type="volume")
    return parent.SparseHSDMetrics(host, mapper, data["nodes"], data["elements"])


def save(fig, out: Path, filename: str) -> None:
    fig.savefig(out / filename, dpi=320, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def metric_atlas(rows: dict, out: Path) -> None:
    keys = (
        "divergence_fidelity",
        "curl_mse",
        "vorticity_fidelity",
        "enstrophy_fidelity",
        "gradient_fidelity",
        "spectral_fidelity",
        "energy_fidelity",
        "betti0_score",
        "level_set_iou",
        "vortex_count_accuracy",
    )
    labels = (
        "Div",
        "Curl*",
        "Vort",
        "Enst",
        "Grad",
        "Spec",
        "Energy",
        "$S_{\\beta_0}$",
        "IoU",
        "Vortex",
    )
    matrix = np.asarray(
        [
            [rows[name][key] if key != "curl_mse" else np.exp(-rows[name][key]) for key in keys]
            for name in ORDER
        ]
    )
    fig = plt.figure(figsize=(17.2, 7.0), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.18, 3.1), left=0.075, right=0.98, bottom=0.15, top=0.80, wspace=0.24
    )
    axis = fig.add_subplot(grid[0, 0])
    y = np.arange(len(ORDER))
    mse = np.array([rows[name]["MSE"] for name in ORDER])
    axis.hlines(y, mse.min() * 0.75, mse, color="#d5e0e8", lw=1.5, zorder=1)
    axis.scatter(
        mse,
        y,
        s=100,
        color=[COLOR[name] for name in ORDER],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for yy, value in zip(y, mse):
        axis.annotate(
            f"{value:.2e}",
            (value, yy),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    axis.set_xscale("log")
    axis.set_yticks(y, [LABEL[name] for name in ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("edge-flux MSE  ↓ (log)")
    axis.grid(axis="x", color="#d8e1ea")
    axis.spines[["top", "right", "left"]].set_visible(False)
    heat_axis = fig.add_subplot(grid[0, 1])
    image = heat_axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    heat_axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    heat_axis.set_yticks(range(len(ORDER)), [LABEL[name] for name in ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            heat_axis.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if matrix[i, j] > 0.62 else "#17334d",
            )
    heat_axis.set_title(
        "Physical / topology fidelity  ↑     (* = exp(-Curl MSE))",
        loc="left",
        pad=12,
        fontsize=11,
        color="#1d334b",
    )
    bar = fig.colorbar(image, ax=heat_axis, fraction=0.035, pad=0.02)
    bar.set_label("fidelity score")
    fig.suptitle(
        "Static magnetism · structural ablation on identical 600-test split",
        x=0.075,
        ha="left",
        y=0.96,
        fontsize=16,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.075,
        0.885,
        "All variants use B + L + H + Q/P, the same small capacity and validation-best checkpoint protocol; only the named structure is removed.",
        fontsize=9.3,
        color="#596d82",
    )
    save(fig, out, "01_structure_metric_atlas.png")


def bar_figure(rows: dict, out: Path) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), facecolor="#fbfcfe")
    x = np.arange(len(ORDER))
    labels = ("Full", "No Dirac", "No phase", "No harmonic")
    for axis, key, ylabel, log in (
        (axes[0], "MSE", "edge-flux MSE  ↓", True),
        (axes[1], "Relative_L2", "test relative $L_2$  ↓", False),
    ):
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
        axis.grid(axis="y", color="#d8e1ea", zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(
            "Physical error" if key == "MSE" else "Field reconstruction error",
            loc="left",
            fontsize=11.5,
            color="#1d334b",
        )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.75, bottom=0.18, wspace=0.14)
    fig.suptitle(
        "Static magnetism · TDK-HO small structural ablation",
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
        "Same B + L + H + Q/P condition, same 600 test fields and validation-best selection; only the named structure is removed.",
        fontsize=9.2,
        color="#596d82",
    )
    save(fig, out, "01_structure_ablation_bars.png")


def distribution_figure(predictions: dict, target: np.ndarray, out: Path) -> tuple[int, int, int]:
    per_sample = {name: relative_l2(value, target) for name, value in predictions.items()}
    full_rank = np.argsort(per_sample["full"])
    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": (1.05, 1.35)}, facecolor="#fbfcfe"
    )
    values = [per_sample[name] for name in ORDER]
    violin = axes[0].violinplot(values, showmeans=True, showmedians=True, widths=0.78)
    for body, name in zip(violin["bodies"], ORDER):
        body.set_facecolor(COLOR[name])
        body.set_edgecolor("white")
        body.set_alpha(0.82)
    for part in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        violin[part].set_color("#1d334b")
    axes[0].set_xticks(
        range(1, 5), ["Full", "No Dirac", "No phase", "No harmonic"], rotation=18, ha="right"
    )
    axes[0].set_ylabel("per-sample relative $L_2$  ↓")
    axes[0].set_title("Error distribution and robustness", loc="left", fontsize=12, color="#1d334b")
    axes[0].grid(axis="y", color="#d8e1ea")
    axes[0].spines[["top", "right"]].set_visible(False)
    x = np.arange(len(ORDER))
    quantiles = np.asarray(
        [[np.quantile(per_sample[name], q) for q in (0.1, 0.5, 0.9)] for name in ORDER]
    )
    for name, xx, q in zip(ORDER, x, quantiles):
        axes[1].vlines(xx, q[0], q[2], color=COLOR[name], lw=3.1, alpha=0.9)
        axes[1].scatter(xx, q[1], color=COLOR[name], s=62, edgecolor="white", zorder=3)
        axes[1].annotate(
            f"median={q[1]:.3f}",
            (xx, q[1]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.6,
        )
    axes[1].set_xticks(x, ["Full", "No Dirac", "No phase", "No harmonic"], rotation=18, ha="right")
    axes[1].set_ylabel("relative $L_2$  ↓")
    axes[1].set_title("10–50–90% test-error interval", loc="left", fontsize=12, color="#1d334b")
    axes[1].grid(axis="y", color="#d8e1ea")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Structural ablation · full-test population statistics",
        x=0.075,
        ha="left",
        y=0.99,
        fontsize=15.5,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.075,
        0.915,
        "Sample ranks below are selected only by the full model and then shared by every ablation variant.",
        fontsize=9.2,
        color="#596d82",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save(fig, out, "02_structure_error_distribution.png")
    return int(full_rank[0]), int(full_rank[len(full_rank) // 2]), int(full_rank[-1])


def field_figure(
    points: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    predictions: dict,
    index: int,
    descriptor: str,
    out: Path,
) -> None:
    fields = [
        ("Input source", np.abs(source[index])),
        ("Ground truth |Y|", np.linalg.norm(target[index], axis=-1)),
    ]
    fields += [(LABEL[name], np.linalg.norm(predictions[name][index], axis=-1)) for name in ORDER]
    norm = Normalize(
        *np.quantile(np.concatenate([values for _, values in fields[1:]]), (0.01, 0.99))
    )
    fig = plt.figure(figsize=(16.0, 8.2), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        2, 3, left=0.02, right=0.98, top=0.87, bottom=0.10, wspace=0.02, hspace=0.08
    )
    ids = np.arange(len(points))[::3]
    for number, (label, values) in enumerate(fields):
        axis = fig.add_subplot(grid[number // 3, number % 3], projection="3d")
        axis.scatter(
            points[ids, 0],
            points[ids, 1],
            points[ids, 2],
            c=values[ids],
            s=5.2,
            cmap="RdBu_r" if number == 0 else "magma",
            norm=None if number == 0 else norm,
            alpha=0.94,
            linewidths=0,
            depthshade=False,
        )
        axis.view_init(elev=20, azim=-56)
        axis.set_box_aspect((1, 1, 0.78))
        axis.set_axis_off()
        axis.set_title(label, fontsize=10, color="#1d334b", pad=2)
    bar = fig.add_axes([0.39, 0.04, 0.22, 0.016])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="magma"), cax=bar, orientation="horizontal"
    ).set_label("field magnitude (shared scale)", fontsize=9)
    fig.suptitle(
        f"Structural ablation · 3-D field comparison · {descriptor} full-model sample",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.02,
        0.935,
        "Ground truth and all four variants share one robust magnitude scale. Rendering uses original nodes; no interpolation enters metrics.",
        fontsize=9.2,
        color="#596d82",
    )
    save(fig, out, f"03_structure_field_{descriptor}.png")


def error_figure(
    points: np.ndarray,
    target: np.ndarray,
    predictions: dict,
    index: int,
    descriptor: str,
    out: Path,
) -> None:
    errors = {
        name: np.linalg.norm(predictions[name][index] - target[index], axis=-1) for name in ORDER
    }
    norm = PowerNorm(
        gamma=0.48, vmin=0, vmax=np.quantile(np.concatenate(list(errors.values())), 0.99)
    )
    fig = plt.figure(figsize=(16.0, 4.6), facecolor="#fbfcfe")
    grid = fig.add_gridspec(1, 4, left=0.02, right=0.98, top=0.81, bottom=0.13, wspace=0.02)
    ids = np.arange(len(points))[::3]
    for number, name in enumerate(ORDER):
        axis = fig.add_subplot(grid[0, number], projection="3d")
        axis.scatter(
            points[ids, 0],
            points[ids, 1],
            points[ids, 2],
            c=errors[name][ids],
            s=5.2,
            cmap="inferno",
            norm=norm,
            alpha=0.94,
            linewidths=0,
            depthshade=False,
        )
        axis.view_init(elev=20, azim=-56)
        axis.set_box_aspect((1, 1, 0.78))
        axis.set_axis_off()
        rel = relative_l2(predictions[name][index : index + 1], target[index : index + 1])[0]
        axis.set_title(f"{LABEL[name]}\nrel. $L_2$={rel:.3f}", fontsize=10, color="#1d334b", pad=2)
    bar = fig.add_axes([0.39, 0.055, 0.22, 0.020])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="inferno"), cax=bar, orientation="horizontal"
    ).set_label("node-vector error magnitude (shared scale)", fontsize=9)
    fig.suptitle(
        f"Structural ablation · direct 3-D error atlas · {descriptor} full-model sample",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.02,
        0.915,
        "Every panel is prediction minus the same ground truth. The shared nonlinear scale resolves low-error structure without changing ranking.",
        fontsize=9.2,
        color="#596d82",
    )
    save(fig, out, f"04_structure_error_{descriptor}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT / "structure_ablation",
        help="Directory for derived structural-ablation metrics and figures.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute sparse physical metrics even if metrics.json exists.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parent.configure_style()
    predictions, target, indices, records = load_runs()
    metric_path = args.output / "metrics.json"
    if metric_path.exists() and not args.recompute:
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    else:
        with DATA.open("rb") as handle:
            data = pickle.load(handle)
        evaluator = create_evaluator(data)
        metrics = {}
        for count, name in enumerate(ORDER, 1):
            print(f"[metrics] {count}/{len(ORDER)} {name}", flush=True)
            metrics[name] = evaluator.evaluate(predictions[name], target)
            metrics[name]["parameters"] = int(records[name]["parameters"])
            metrics[name]["Relative_L2"] = float(relative_l2(predictions[name], target).mean())
            metrics[name]["best_val_relative_l2"] = float(records[name]["best_val_relative_l2"])
            metrics[name]["best_epoch"] = int(records[name]["best_epoch"])
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    bar_figure(metrics, args.output)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "condition": "conditioned",
                "test_samples": int(len(target)),
                "variants": list(ORDER),
                "selection": "saved validation-best checkpoint per variant",
                "metric_implementation": "same sparse HSD-compatible evaluator as magnetostatics/evaluate.py",
                "test_indices_sha256_note": "all four test_indices.npy arrays were verified elementwise equal",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        name: {
            key: metrics[name][key] for key in ("parameters", "MSE", "Relative_L2", "best_epoch")
        }
        for name in ORDER
    }
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
