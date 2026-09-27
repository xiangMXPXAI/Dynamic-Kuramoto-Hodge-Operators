"""Evaluate saved DKHO and baseline predictions for magnetostatics."""

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
from matplotlib.lines import Line2D
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import NearestNDInterpolator
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
HSD_ROOT = ROOT / "baselines"
DATA = ROOT / "data" / "cavity_magnetostatics_v1.pkl"
BASELINE_ROOT = ROOT / "runs" / "baseline" / "legacy"
REPORT_ROOT = ROOT / "reports" / "metrics" / "current"
BASELINE_LOG = BASELINE_ROOT / "experiment_log.pkl"
GROUPS = ("small", "large")
GROUP_COLORS = {
    "small": "#367cb9",
    "large": "#173f6c",
}
BASELINE_COLOR = "#d47a55"
FIDELITY_KEYS = (
    "divergence_fidelity",
    "vorticity_fidelity",
    "enstrophy_fidelity",
    "gradient_fidelity",
    "spectral_fidelity",
    "energy_fidelity",
    "betti0_score",
    "level_set_iou",
    "vortex_count_accuracy",
)
CONFIG_ORDER = (
    "native",
    "boundary",
    "boundary_spectral",
    "boundary_diffusion_spectral",
    "conditioned",
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#fbfcfe",
            "axes.facecolor": "#fbfcfe",
            "savefig.facecolor": "#fbfcfe",
            "axes.edgecolor": "#bbc6d2",
            "axes.labelcolor": "#2b4058",
            "xtick.color": "#5a6f85",
            "ytick.color": "#5a6f85",
            "grid.color": "#d9e2eb",
            "grid.alpha": 0.75,
            "axes.titleweight": "semibold",
        }
    )


def group_label(group: str) -> str:
    return {
        "small": "DKHO small",
        "large": "DKHO large",
    }[group]


def config_label(config: str) -> str:
    return {
        "native": "Native",
        "boundary": "B",
        "boundary_spectral": "B + L",
        "boundary_diffusion_spectral": "B + L + H",
        "conditioned": "B + L + H + Q/P",
    }[config]


def clean(axis, grid="x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, zorder=0)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def faces_from_tets(tets: np.ndarray) -> list[tuple[int, int, int]]:
    face_set = set()
    for tet in tets:
        face_set.update(
            (
                tuple(sorted((tet[0], tet[1], tet[2]))),
                tuple(sorted((tet[0], tet[1], tet[3]))),
                tuple(sorted((tet[0], tet[2], tet[3]))),
                tuple(sorted((tet[1], tet[2], tet[3]))),
            )
        )
    return list(face_set)


class SparseHSDMetrics:
    

    def __init__(self, host, mapper, points: np.ndarray, tets: np.ndarray):
        self.host, self.mapper = host, mapper
        self.b1 = host.B1.tocsr().astype(np.float32)
        self.b2 = host.B2.tocsr().astype(np.float32)
        self.abs_b1 = abs(mapper.B1).tocsr().astype(np.float32)
        self.edge_vectors = mapper.edge_vectors.astype(np.float32)
        self.edge_dirs = mapper.edge_dirs.astype(np.float32)
        self.degree = mapper.node_degree[:, 0].astype(np.float32)
        self.faces = faces_from_tets(tets)
        self.face_array = np.asarray(self.faces, dtype=np.int64)
        self.face_area, self.node_area = self._geometry(points, tets)
        self.face_adj = self._face_adjacency()
        self.eigen = self._eigenvalues()

    def _geometry(self, points, tets):
        face_area = np.empty(len(self.faces), np.float32)
        for index, face in enumerate(self.faces):
            a, b, c = points[list(face)]
            face_area[index] = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        node_area = np.zeros(len(points), np.float32)
        for tet in tets:
            a, b, c, d = points[tet]
            volume = abs(np.dot(b - a, np.cross(c - a, d - a))) / 6.0
            node_area[tet] += volume / 4.0
        return face_area, node_area

    def _face_adjacency(self) -> csr_matrix:
        owners: dict[tuple[int, int], list[int]] = {}
        for index, (a, b, c) in enumerate(self.faces):
            for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                owners.setdefault(edge, []).append(index)
        rows, cols = [], []
        for indices in owners.values():
            if len(indices) == 2:
                rows += indices
                cols += indices[::-1]
        return csr_matrix(
            (np.ones(len(rows), np.uint8), (rows, cols)), shape=(len(self.faces), len(self.faces))
        )

    def _eigenvalues(self) -> np.ndarray:
        phi = self.host.Phi0.astype(np.float64)
        lap = self.host.L0.astype(np.float64)
        return np.array(
            [value @ (lap @ value) / max(value @ value, 1e-12) for value in phi.T], np.float32
        )

    def vectors_to_flux(self, vectors: np.ndarray) -> np.ndarray:
        result = np.zeros((len(vectors), self.abs_b1.shape[0]), np.float32)
        for component in range(3):
            average = 0.5 * self.abs_b1.dot(vectors[:, :, component].T).T
            result += average * self.edge_vectors[None, :, component]
        return result

    def velocity_magnitude(self, flux: np.ndarray) -> np.ndarray:
        vector = np.empty((len(flux), self.abs_b1.shape[1], 3), np.float32)
        for component in range(3):
            vector[:, :, component] = (
                self.abs_b1.T.dot((flux * self.edge_dirs[:, component]).T).T / self.degree[None]
            )
        return np.linalg.norm(vector, axis=-1)

    def vorticity(self, flux: np.ndarray) -> np.ndarray:
        return (self.b2.dot(flux.T).T / np.maximum(self.face_area[None], 1e-10)).astype(np.float32)

    def _components(self, values: np.ndarray, threshold: float) -> int:
        active = np.flatnonzero(values >= threshold)
        if len(active) == 0:
            return 0
        return int(connected_components(self.face_adj[active, :][:, active], directed=False)[0])

    def _vortex_counts(self, absolute_vorticity: np.ndarray) -> np.ndarray:
        indptr, indices = self.face_adj.indptr, self.face_adj.indices
        if np.any(np.diff(indptr) == 0):
            return np.zeros(len(absolute_vorticity), np.int32)
        out = np.empty(len(absolute_vorticity), np.int32)
        for start in range(0, len(absolute_vorticity), 12):
            block = absolute_vorticity[start : start + 12]
            neighbor_max = np.maximum.reduceat(block[:, indices], indptr[:-1], axis=1)
            out[start : start + len(block)] = (
                (block >= block.max(1, keepdims=True) * 0.3) & (block >= neighbor_max)
            ).sum(1)
        return out

    def evaluate(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        fp, ft = self.vectors_to_flux(prediction), self.vectors_to_flux(target)
        vp, vt = self.vorticity(fp), self.vorticity(ft)
        face_weights = self.face_area / self.face_area.sum()
        div_error = np.mean((self.b1.T.dot(fp.T).T - self.b1.T.dot(ft.T).T) ** 2)
        mse = np.mean((fp - ft) ** 2)
        curl_mse = np.mean(np.sum((vp - vt) ** 2 * face_weights[None], axis=1))
        dot = np.sum(face_weights[None] * vp * vt, axis=1)
        vort_fid = np.mean(
            (
                1
                + dot
                / (
                    np.sqrt(
                        np.sum(face_weights[None] * vp**2, axis=1)
                        * np.sum(face_weights[None] * vt**2, axis=1)
                    )
                    + 1e-10
                )
            )
            / 2
        )
        ep, et = np.sum(face_weights[None] * vp**2, axis=1), np.sum(
            face_weights[None] * vt**2, axis=1
        )
        enst = np.mean(1 / (1 + np.abs(np.log((ep + 1e-10) / (et + 1e-10)))))
        energy_p, energy_t = np.sum(fp**2, axis=1), np.sum(ft**2, axis=1)
        energy = np.mean(1 / (1 + np.abs(np.log((energy_p + 1e-10) / (energy_t + 1e-10)))))
        magp, magt = self.velocity_magnitude(fp), self.velocity_magnitude(ft)
        gradp, gradt = self.b1.dot(magp.T).T, self.b1.dot(magt.T).T
        grad = np.mean(
            (
                1
                + np.sum(gradp * gradt, axis=1)
                / (np.linalg.norm(gradp, axis=1) * np.linalg.norm(gradt, axis=1) + 1e-10)
            )
            / 2
        )
        phi = self.host.Phi0[:, :20].astype(np.float32) * self.node_area[:, None]
        cp, ct = magp @ phi, magt @ phi
        weights = 1 / (self.eigen[:20] + 0.1)
        weights /= weights.sum()
        spec = np.mean(
            np.exp(
                -2
                * np.sqrt(np.sum(weights[None] * (cp - ct) ** 2, axis=1))
                / (np.sqrt(np.sum(weights[None] * ct**2, axis=1)) + 1e-10)
            )
        )
        ap, at = np.abs(vp), np.abs(vt)
        b0 = []
        for level in (0.2, 0.5, 0.8):
            for pred_row, target_row in zip(ap, at):
                threshold = target_row.min() + level * (target_row.max() - target_row.min())
                a, b = self._components(pred_row, threshold), self._components(
                    target_row, threshold
                )
                b0.append(np.exp(-1.5 * abs(a - b) / max(a, b, 1)))
        iou = []
        for pred_row, target_row in zip(ap, at):
            levels = np.linspace(target_row.min(), target_row.max(), 12)[1:-1]
            local = []
            for level in levels:
                pm, tm = pred_row >= level, target_row >= level
                local.append(
                    np.sum(face_weights * (pm & tm)) / max(np.sum(face_weights * (pm | tm)), 1e-12)
                )
            iou.append(np.mean(local))
        nc_p, nc_t = self._vortex_counts(ap), self._vortex_counts(at)
        vortex_score = np.zeros_like(nc_p, dtype=np.float64)
        both_zero = (nc_p == 0) & (nc_t == 0)
        both_nonzero = (nc_p != 0) & (nc_t != 0)
        vortex_score[both_zero] = 1.0
        vortex_score[both_nonzero] = 1.0 - np.abs(
            nc_p[both_nonzero] - nc_t[both_nonzero]
        ) / np.maximum(nc_p[both_nonzero], nc_t[both_nonzero])
        vortex = np.mean(vortex_score)
        return {
            "n_samples": int(len(prediction)),
            "MSE": float(mse),
            "divergence_fidelity": float(np.exp(-10 * div_error)),
            "curl_mse": float(curl_mse),
            "vorticity_fidelity": float(vort_fid),
            "enstrophy_fidelity": float(enst),
            "gradient_fidelity": float(grad),
            "spectral_fidelity": float(spec),
            "energy_fidelity": float(energy),
            "betti0_score": float(np.mean(b0)),
            "level_set_iou": float(np.sqrt(np.mean(iou))),
            "vortex_count_accuracy": float(vortex),
        }


def load_groups() -> tuple[dict, dict, dict, np.ndarray]:
    metrics, predictions, manifests = {}, {}, {}
    reference = None
    for group in GROUPS:
        root = ROOT / "runs" / "dkho" / group
        manifests[group] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        run_root = root / "form_2"
        for run in sorted(run_root.glob("*/*/seed_*")):
            if not (run / "result.json").is_file():
                continue
            prediction, target, indices = (
                np.load(run / "prediction_test.npy"),
                np.load(run / "target_test.npy"),
                np.load(run / "test_indices.npy"),
            )
            if reference is None:
                reference = (target, indices)
            else:
                assert np.array_equal(target, reference[0]) and np.array_equal(
                    indices, reference[1]
                ), f"test mismatch: {run}"
            row = json.loads((run / "result.json").read_text(encoding="utf-8"))
            key = f"TDK-{group}/{run.parent.parent.name}"
            metrics[key] = {"parameters": row["parameters"]}
            predictions[key] = prediction.astype(np.float32)
    return metrics, predictions, manifests, reference[0]


def align_archived_baseline_predictions(
    baseline: dict, sorted_test_indices: np.ndarray
) -> dict[str, np.ndarray]:
    
    all_indices = np.arange(len(pickle.load(DATA.open("rb"))["Y_data"]))
    _, archived_order = train_test_split(all_indices, test_size=0.2, random_state=42)
    position = {int(sample): i for i, sample in enumerate(archived_order)}
    permutation = np.asarray(
        [position[int(sample)] for sample in sorted_test_indices], dtype=np.int64
    )
    scale = float(baseline["scaling"]["y_scale"])
    return {
        name: values[permutation].astype(np.float32) * scale
        for name, values in baseline["vector_predictions"].items()
        if name in ("HSD", "GNO", "FNO", "MGN", "DeepONet", "GeoFNO")
    }


def metric_atlas(metrics: dict, out: Path) -> None:
    
    
    
    names = sorted(metrics, key=lambda name: metrics[name]["MSE"])
    labels = [
        (
            f"{group_label(name.split('/')[0][4:])} · {config_label(name.rsplit('/', 1)[1])}"
            if name.startswith("TDK-")
            else name
        )
        for name in names
    ]
    colors = [
        GROUP_COLORS[name.split("/")[0][4:]] if name.startswith("TDK-") else BASELINE_COLOR
        for name in names
    ]
    heat_keys = ("divergence_fidelity", "curl_mse", *FIDELITY_KEYS[1:])
    heat = np.asarray(
        [
            [
                metrics[name][key] if key != "curl_mse" else np.exp(-metrics[name][key])
                for key in heat_keys
            ]
            for name in names
        ]
    )
    columns = (
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
    fig = plt.figure(figsize=(22, 11), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.28, 3.72), left=0.075, right=0.93, top=0.84, bottom=0.10, wspace=0.10
    )
    y = np.arange(len(names))
    ax = fig.add_subplot(grid[0, 0])
    values = np.asarray([metrics[name]["MSE"] for name in names])
    ax.hlines(y, values.min() * 0.73, values, color="#d6e0e9", lw=1.3, zorder=1)
    ax.scatter(values, y, s=95, c=colors, edgecolors="white", linewidths=1.25, zorder=3)
    for yy, value in zip(y, values):
        ax.annotate(
            f"{value:.2e}",
            (value, yy),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.1,
            color="#425972",
        )
    ax.set(
        xscale="log",
        xlim=(values.min() * 0.62, values.max() * 3.1),
        yticks=y,
        yticklabels=labels,
        xlabel="edge-flux MSE  ·  lower is better",
    )
    ax.invert_yaxis()
    clean(ax, "x")
    ax.set_title("Accuracy", loc="left", pad=16, fontsize=13, color="#15283f")
    for boundary in (
        np.flatnonzero(np.asarray([name.startswith("TDK-") for name in names]))[-1] + 0.5,
    ):
        ax.axhline(boundary, color="#aab9c8", lw=1.1, ls=(0, (3, 3)), zorder=2)
    hm = fig.add_subplot(grid[0, 1])
    image = hm.imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    hm.set(
        xticks=np.arange(len(columns)), xticklabels=columns, yticks=y, yticklabels=[""] * len(names)
    )
    hm.xaxis.tick_top()
    hm.tick_params(axis="x", pad=9, labelsize=9)
    for yy in range(len(names)):
        for xx in range(len(columns)):
            value = heat[yy, xx]
            hm.text(
                xx,
                yy,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.62 else "#163047",
            )
    hm.axhline(boundary, color="#f7fafc", lw=2.0)
    for spine in hm.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(image, ax=hm, shrink=0.74, pad=0.025)
    cb.set_label("fidelity  ·  higher is better  (Curl* = exp(−curl MSE))", fontsize=9)
    fig.suptitle(
        "Magnetostatics · unified physical metric atlas",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.075,
        0.925,
        "Same fixed 600-sample test split, physical units, and sparse HSD-compatible evaluator.  Blue: TDK-HO; orange: archived baselines.",
        fontsize=10,
        color="#596d82",
    )
    save(fig, out, "01_metric_atlas.png")


def pareto(metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.6), constrained_layout=True)
    for name, row in metrics.items():
        if name.startswith("TDK-"):
            group = name.split("/")[0].removeprefix("TDK-")
            ax.scatter(
                row["parameters"],
                row["MSE"],
                color=GROUP_COLORS[group],
                s=58,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
        else:
            ax.scatter(
                row["parameters"],
                row["MSE"],
                marker="s",
                color=BASELINE_COLOR,
                s=78,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
    points = sorted(metrics.items(), key=lambda item: item[1]["parameters"])
    frontier = []
    best = np.inf
    for name, row in points:
        if row["MSE"] < best:
            frontier.append((name, row))
            best = row["MSE"]
    ax.plot(
        [r["parameters"] for _, r in frontier],
        [r["MSE"] for _, r in frontier],
        "--",
        color="#172e48",
        linewidth=2.1,
        alpha=0.8,
        label="empirical frontier",
    )
    candidates = [
        min((n for n in metrics if n.startswith(f"TDK-{g}/")), key=lambda n: metrics[n]["MSE"])
        for g in GROUPS
    ] + ["HSD", "FNO", "DeepONet"]
    for name in candidates:
        row = metrics[name]
        label = config_label(name.rsplit("/", 1)[1]) if name.startswith("TDK-") else name
        ax.annotate(
            label,
            (row["parameters"], row["MSE"]),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=8.8,
            color="#2d425a",
            bbox={"boxstyle": "round,pad=.22", "fc": "#fbfcfe", "ec": "none", "alpha": 0.84},
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="trainable parameters",
        ylabel="edge-flux MSE",
        title="Accuracy–capacity frontier",
    )
    clean(ax, "both")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=GROUP_COLORS[g],
            label=group_label(g),
            markersize=7,
        )
        for g in GROUPS
    ] + [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=BASELINE_COLOR,
            label="archived baseline",
            markersize=7,
        )
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8.5,
        loc="upper right",
        title="model family",
        title_fontsize=8.5,
    )
    ax.text(
        0.015,
        0.025,
        "Lower-left is preferable. Dashed line: non-dominated models.",
        transform=ax.transAxes,
        fontsize=9,
        color="#596d82",
    )
    save(fig, out, "02_accuracy_parameter_tradeoff.png")


def population_diagnostics(
    predictions: dict, target: np.ndarray, source: np.ndarray, out: Path
) -> None:
    
    order = list(predictions)
    colors = {
        "TDK-HO small": "#173f6c",
        "TDK-HO large": "#367cb9",
        "HSD": "#d47a55",
        "GNO": "#b55a43",
        "FNO": "#c88e39",
        "MGN": "#9a7652",
        "DeepONet": "#8a5a93",
        "GeoFNO": "#5b8c85",
    }
    relative = {
        name: np.linalg.norm(value - target, axis=(1, 2))
        / (np.linalg.norm(target, axis=(1, 2)) + 1e-12)
        for name, value in predictions.items()
    }
    source_strength = np.sqrt(np.mean(source**2, axis=1))
    fig = plt.figure(figsize=(18.5, 10.2), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        2, 2, left=0.07, right=0.97, bottom=0.10, top=0.86, hspace=0.34, wspace=0.22
    )
    ax = fig.add_subplot(grid[0, 0])
    for name in order:
        values = np.sort(relative[name])
        q = np.linspace(0, 1, len(values))
        ax.plot(values, q, color=colors.get(name, "#596d82"), lw=2.2, label=name, alpha=0.95)
    ax.set(
        xscale="log",
        xlabel="per-sample relative $L_2$",
        ylabel="empirical CDF",
        title="Error distribution over the complete test population",
    )
    clean(ax, "both")
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[0, 1])
    logs = [np.log10(np.maximum(relative[name], 1e-9)) for name in order]
    violin = ax.violinplot(logs, showmeans=False, showmedians=True, showextrema=False)
    for body, name in zip(violin["bodies"], order):
        body.set_facecolor(colors.get(name, "#596d82"))
        body.set_edgecolor("white")
        body.set_alpha(0.82)
    violin["cmedians"].set_color("#172e48")
    violin["cmedians"].set_linewidth(1.6)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=25, ha="right")
    ax.set(ylabel=r"$\log_{10}$ relative $L_2$", title="Median and spread; lower is better")
    clean(ax, "y")
    ax = fig.add_subplot(grid[1, :])
    quantiles = np.quantile(source_strength, np.linspace(0, 1, 9))
    centers = 0.5 * (quantiles[:-1] + quantiles[1:])
    for name in order:
        values = []
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            mask = (source_strength >= lo) & (source_strength <= hi)
            values.append(np.median(relative[name][mask]))
        ax.plot(
            centers, values, marker="o", ms=4, lw=2.1, color=colors.get(name, "#596d82"), label=name
        )
    ax.set(
        xlabel="source-field RMS amplitude (equal-count bins)",
        ylabel="median relative $L_2$",
        title="Robustness across source-field strength",
    )
    clean(ax, "both")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Magnetostatics: population-level generalisation diagnostics",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.07,
        0.915,
        "Each curve and violin uses all 600 held-out fields; no sample selection is involved.",
        fontsize=9.5,
        color="#596d82",
    )
    save(fig, out, "07_population_error_diagnostics.png")


def audit_deeponet_parameters(metrics: dict, out: Path) -> dict:
    

    def count(path: Path) -> tuple[int, list[int]]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return sum(value.numel() for value in state.values()), list(
            state["branch_net.0.weight"].shape
        )

    records = {}
    base = BASELINE_ROOT / "model_deeponet.pt"
    n, shape = count(base)
    records["original_base"] = {
        "checkpoint": str(base.relative_to(ROOT)),
        "parameters": n,
        "first_branch_weight": shape,
    }
    if int(metrics["DeepONet"]["parameters"]) != n:
        raise RuntimeError("DeepONet parameter count disagrees with its saved baseline checkpoint")
    control_root = (
        ROOT.parents[1]
        / "baselines"
        / "hsd"
        / "feature_ablation_outputs"
        / "magnetostatics"
        / "DeepONet"
    )
    for checkpoint in sorted(control_root.glob("*__normal/seed_42/best_val.pt")):
        n, shape = count(checkpoint)
        records[checkpoint.parents[1].name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT.parents[1])),
            "parameters": n,
            "first_branch_weight": shape,
        }
    (out / "deeponet_parameter_audit.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def ablation(metrics: dict, out: Path) -> None:
    configs = [
        config
        for config in CONFIG_ORDER
        if all(f"TDK-{group}/{config}" in metrics for group in GROUPS)
    ]
    if not configs:
        return
    fig, axes = plt.subplots(
        1, len(GROUPS), figsize=(6.2 * len(GROUPS), 5.4), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(configs))
    for ax, group in zip(axes, GROUPS):
        values = np.array([metrics[f"TDK-{group}/{c}"]["MSE"] for c in configs])
        best = int(values.argmin())
        colors = [GROUP_COLORS[group]] * len(values)
        colors[best] = "#e5a043"
        ax.bar(x, values, color=colors, zorder=3)
        ax.plot(x, values, color="#264a69", marker="o", markersize=3.5, zorder=4)
        ax.scatter(
            best, values[best], marker="*", s=150, color="#ffe08a", edgecolor="#925922", zorder=5
        )
        ax.set_yscale("log")
        ax.set_xticks(x, [config_label(c) for c in configs], rotation=25, ha="right")
        ax.set_title(group_label(group), loc="left", fontsize=12, color="#1d334b")
        clean(ax)
        ax.text(
            0.03,
            0.04,
            f"best: {config_label(configs[best])}",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=.28", "fc": "#f3f6fa", "ec": "#d5e0e9"},
        )
    axes[0].set_ylabel("edge-flux MSE (log)")
    save(fig, out, "03_tdk_ablation.png")


def sample_panel(points, source, target, predictions, index, title, out: Path, name: str) -> None:
    z = points[:, 2]
    keep = np.abs(z - np.median(z)) <= np.quantile(np.abs(z - np.median(z)), 0.16)
    xy = points[keep, :2]
    ids = np.flatnonzero(keep)[:: max(1, keep.sum() // 350)]
    fields = [
        ("Source", source[index]),
        ("Target |B|", np.linalg.norm(target[index], axis=1)),
        *[(label, np.linalg.norm(value[index], axis=1)) for label, value in predictions.items()],
    ]
    fields = fields[:6]
    magpool = np.concatenate([value[keep] for _, value in fields[1:]])
    norm = Normalize(*np.quantile(magpool, [0.01, 0.99]))
    fig, axes = plt.subplots(2, 6, figsize=(18.5, 6), constrained_layout=True)
    for col, (label, value) in enumerate(fields):
        ax = axes[0, col]
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=value[keep],
            s=8,
            cmap="viridis" if col == 0 else "magma",
            norm=norm if col else None,
            linewidths=0,
        )
        ax.set_title(label, fontsize=10, color="#1c324a")
        ax.set_axis_off()
        ax.set_aspect("equal")
        if col > 0:
            vec = predictions.get(label, target)[index]
            ax.quiver(
                points[ids, 0],
                points[ids, 1],
                vec[ids, 0],
                vec[ids, 1],
                color="white",
                alpha=0.65,
                scale=6,
                width=0.003,
            )
    for col, (label, value) in enumerate(fields):
        ax = axes[1, col]
        error = (
            np.zeros_like(value)
            if col < 2
            else np.linalg.norm(predictions[label][index] - target[index], axis=1)
            / (np.sqrt(np.mean(target[index] ** 2)) + 1e-10)
        )
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=error[keep],
            s=8,
            cmap="magma",
            norm=Normalize(0, 1.3),
            linewidths=0,
        )
        ax.set_title("error / RMS" if col > 1 else "", fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.suptitle(title, x=0.02, ha="left", fontsize=16, fontweight="bold", color="#15283f")
    save(fig, out, name)


def sample_panel_3d(
    points, source, target, predictions, sample_metrics, index, title, out: Path, name: str
) -> None:
    

    def magnitude(value):
        return np.abs(value) if value.ndim == 1 else np.linalg.norm(value, axis=-1)

    panels = [("Source", magnitude(source[index])), ("Ground truth |B|", magnitude(target[index]))]
    panels += [(label, magnitude(value[index])) for label, value in predictions.items()]
    
    
    pool = np.concatenate([value for _, value in panels[1:]])
    norm = Normalize(*np.quantile(pool, [0.01, 0.99]))
    ids = np.arange(len(points))[::3]
    fig = plt.figure(figsize=(20, 8.1), facecolor="#fbfcfe")
    
    
    grid = fig.add_gridspec(
        2, 5, left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.01, hspace=0.06
    )
    for panel_index, (label, value) in enumerate(panels):
        axis = fig.add_subplot(grid[panel_index // 5, panel_index % 5], projection="3d")
        axis.scatter(
            points[ids, 0],
            points[ids, 1],
            points[ids, 2],
            c=value[ids],
            s=5.5,
            cmap="magma",
            norm=norm,
            linewidths=0,
            alpha=0.94,
            depthshade=False,
        )
        axis.view_init(elev=20, azim=-56)
        axis.set_box_aspect((1, 1, 0.78))
        axis.set_axis_off()
        suffix = (
            ""
            if label not in sample_metrics
            else f"\nrel. L2={sample_metrics[label][0][index]:.3f}  MSE={sample_metrics[label][1][index]:.2e}"
        )
        axis.set_title(label + suffix, fontsize=9.2, color="#1c324a", pad=2)
    bar = fig.add_axes([0.38, 0.035, 0.24, 0.016])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="magma"), cax=bar, orientation="horizontal"
    ).set_label("field magnitude", fontsize=9, labelpad=2)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=16, fontweight="bold", color="#15283f")
    fig.text(
        0.02,
        0.945,
        "Same 3-D nodes and shared robust magnitude scale; both selected TDK-HO models are shown explicitly.",
        fontsize=9.5,
        color="#596d82",
    )
    save(fig, out, name)


def sample_error_panel_3d(
    points, source, target, predictions, sample_metrics, index, title, out: Path, name: str
) -> None:
    
    items = [("Source", np.abs(source[index])), ("Ground truth", np.zeros(len(points), np.float32))]
    items += [
        (label, np.linalg.norm(value[index] - target[index], axis=-1))
        for label, value in predictions.items()
    ]
    pool = np.concatenate([value for _, value in items[2:]])
    norm = Normalize(0, np.quantile(pool, 0.99))
    ids = np.arange(len(points))[::3]
    fig = plt.figure(figsize=(20, 8.1), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        2, 5, left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.01, hspace=0.06
    )
    for panel_index, (label, value) in enumerate(items):
        axis = fig.add_subplot(grid[panel_index // 5, panel_index % 5], projection="3d")
        axis.scatter(
            points[ids, 0],
            points[ids, 1],
            points[ids, 2],
            c=value[ids],
            s=5.5,
            cmap="magma",
            norm=norm,
            linewidths=0,
            alpha=0.94,
            depthshade=False,
        )
        axis.view_init(elev=20, azim=-56)
        axis.set_box_aspect((1, 1, 0.78))
        axis.set_axis_off()
        suffix = (
            ""
            if label not in sample_metrics
            else f"\nrel. L2={sample_metrics[label][0][index]:.3f}  MSE={sample_metrics[label][1][index]:.2e}"
        )
        axis.set_title(label + suffix, fontsize=9.2, color="#1c324a", pad=2)
    bar = fig.add_axes([0.38, 0.035, 0.24, 0.016])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="magma"), cax=bar, orientation="horizontal"
    ).set_label("pointwise vector error magnitude (shared scale)", fontsize=9, labelpad=2)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=16, fontweight="bold", color="#15283f")
    fig.text(
        0.02,
        0.945,
        "Prediction panels show absolute node-vector error against the same ground-truth field; source and truth are zero-error references.",
        fontsize=9.5,
        color="#596d82",
    )
    save(fig, out, name)


def hsd_style_slice_atlas(
    points: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    index: int,
    sample_label: str,
    out: Path,
) -> None:
    
    center = points.mean(axis=0)
    z0 = float(np.median(points[:, 2]))
    extent = float(np.max(np.abs(points[:, :2] - center[:2]))) * 1.03
    grid_size = 180
    axis = np.linspace(center[0] - extent, center[0] + extent, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    query = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z0)]
    radius = np.linalg.norm(points - center, axis=1)
    radial_query = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (z0 - center[2]) ** 2)
    inner, outer = np.quantile(radius, [0.015, 0.995])
    domain_mask = (radial_query < inner * 0.96) | (radial_query > outer * 1.01)

    def interp(values: np.ndarray) -> np.ndarray:
        result = NearestNDInterpolator(points, values)(query).reshape(grid_size, grid_size)
        result[domain_mask] = np.nan
        return result

    names = list(predictions)
    magnitudes = {"Ground truth |B|": np.linalg.norm(target[index], axis=1)}
    magnitudes.update(
        {name: np.linalg.norm(value[index], axis=1) for name, value in predictions.items()}
    )
    all_magnitude = np.concatenate([value for value in magnitudes.values()])
    field_norm = Normalize(0, np.quantile(all_magnitude, 0.995))
    error = {
        name: np.linalg.norm(value[index] - target[index], axis=1)
        for name, value in predictions.items()
    }
    error_norm = PowerNorm(
        gamma=0.48, vmin=0, vmax=np.quantile(np.concatenate(list(error.values())), 0.995)
    )
    
    panels = [
        (
            "Source $\\rho$",
            source[index],
            "RdBu_r",
            Normalize(*np.quantile(source[index], [0.01, 0.99])),
        ),
        *[(name, value, "magma", field_norm) for name, value in magnitudes.items()],
    ]
    fig, axes = plt.subplots(2, 5, figsize=(19, 8.0), facecolor="#fbfcfe")
    for panel, ax in zip(panels, axes.ravel()):
        title, values, cmap, norm = panel
        image = ax.pcolormesh(
            xx, yy, interp(values), shading="auto", cmap=cmap, norm=norm, rasterized=True
        )
        ax.set(title=title, aspect="equal")
        ax.set_axis_off()
        if title != "Source $\\rho$":
            vectors = target[index] if title == "Ground truth |B|" else predictions[title][index]
            step = 15
            vx = interp(vectors[:, 0])[::step, ::step]
            vy = interp(vectors[:, 1])[::step, ::step]
            ax.quiver(
                xx[::step, ::step],
                yy[::step, ::step],
                vx,
                vy,
                color="white",
                alpha=0.58,
                width=0.0025,
                scale=7.5,
                headwidth=3.2,
            )
    source_bar = fig.add_axes([0.18, 0.055, 0.18, 0.015])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=panels[0][3], cmap="RdBu_r"),
        cax=source_bar,
        orientation="horizontal",
    ).set_label("source $\\rho$", fontsize=8)
    field_bar = fig.add_axes([0.64, 0.055, 0.18, 0.015])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=field_norm, cmap="magma"),
        cax=field_bar,
        orientation="horizontal",
    ).set_label("$|B|$ (shared scale)", fontsize=8)
    fig.suptitle(
        f"Magnetostatics ·  z={z0:.3f} field slice · {sample_label}",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.03,
        0.942,
        "Input, truth and all eight predictions use the same planar cut; white arrows show the in-plane vector direction.",
        fontsize=9.3,
        color="#596d82",
    )
    
    
    
    fig.subplots_adjust(left=0.015, right=0.985, top=0.87, bottom=0.13, wspace=0.035, hspace=0.20)
    save(fig, out, f"08_slice_field_{sample_label}.png")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.0), facecolor="#fbfcfe")
    for ax, name in zip(axes.ravel(), names):
        image = ax.pcolormesh(
            xx,
            yy,
            interp(error[name]),
            shading="auto",
            cmap="inferno",
            norm=error_norm,
            rasterized=True,
        )
        ax.set(title=name, aspect="equal")
        ax.set_axis_off()
        rel = np.linalg.norm(predictions[name][index] - target[index]) / (
            np.linalg.norm(target[index]) + 1e-12
        )
        ax.text(
            0.03,
            0.04,
            f"rel. $L_2$={rel:.3f}",
            transform=ax.transAxes,
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=.25", "fc": "#101820", "ec": "none", "alpha": 0.72},
        )
    bar = fig.add_axes([0.39, 0.055, 0.22, 0.016])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=error_norm, cmap="inferno"), cax=bar, orientation="horizontal"
    ).set_label("node-vector error magnitude (shared scale)", fontsize=8)
    fig.suptitle(
        f"Magnetostatics ·  z={z0:.3f} error slice · {sample_label}",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.03,
        0.942,
        "Every panel is prediction minus the same ground truth; the nonlinear colour scale resolves low-error structure without changing rankings.",
        fontsize=9.3,
        color="#596d82",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.13, wspace=0.045, hspace=0.20)
    save(fig, out, f"09_slice_error_{sample_label}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_ROOT,
        help="Directory for derived metrics and diagnostic figures.",
    )
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    configure_style()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HSD_ROOT))
    from dataset import VectorFluxMapper
    from spectral_operators import HighOrderSpectralOperators

    with DATA.open("rb") as handle:
        data = pickle.load(handle)
    metrics, predictions, _, target = load_groups()
    baseline = pickle.load(BASELINE_LOG.open("rb"))
    cache = out / "tdk_metrics.json"
    if cache.exists() and not args.recompute:
        computed = json.loads(cache.read_text(encoding="utf-8"))
    else:
        host = HighOrderSpectralOperators(data["nodes"], data["elements"], k_list=(64, 64, 64))
        evaluator = SparseHSDMetrics(
            host,
            VectorFluxMapper(data["nodes"], data["elements"], mesh_type="volume"),
            data["nodes"],
            data["elements"],
        )
        computed = {}
        for index, (name, prediction) in enumerate(predictions.items(), 1):
            print(f"[metrics] {index}/{len(predictions)} {name}", flush=True)
            computed[name] = evaluator.evaluate(prediction, target)
            computed[name]["parameters"] = metrics[name]["parameters"]
        cache.write_text(json.dumps(computed, indent=2), encoding="utf-8")
    metrics.update(computed)

    sorted_test_indices = np.load(
        ROOT / "runs" / "dkho" / GROUPS[0] / "form_2" / "conditioned" / "full" / "seed_42" / "test_indices.npy"
    )
    baseline_predictions = align_archived_baseline_predictions(baseline, sorted_test_indices)
    
    
    
    
    baseline_cache = out / "baseline_metrics_aligned_physical_v2.json"
    if baseline_cache.exists() and not args.recompute:
        baseline_metrics = json.loads(baseline_cache.read_text(encoding="utf-8"))
    else:
        host = HighOrderSpectralOperators(data["nodes"], data["elements"], k_list=(64, 64, 64))
        evaluator = SparseHSDMetrics(
            host,
            VectorFluxMapper(data["nodes"], data["elements"], mesh_type="volume"),
            data["nodes"],
            data["elements"],
        )
        baseline_metrics = {}
        for index, (name, prediction) in enumerate(baseline_predictions.items(), 1):
            print(f"[baseline metrics] {index}/{len(baseline_predictions)} {name}", flush=True)
            baseline_metrics[name] = evaluator.evaluate(prediction, target)
            baseline_metrics[name]["parameters"] = int(baseline["params"][name])
        baseline_cache.write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")
    metrics.update(baseline_metrics)
    for name, prediction in {**predictions, **baseline_predictions}.items():
        rel = np.linalg.norm(prediction - target, axis=(1, 2)) / (
            np.linalg.norm(target, axis=(1, 2)) + 1e-12
        )
        metrics[name]["Relative_L2"] = float(rel.mean())
    small = min(
        (name for name in metrics if name.startswith("TDK-small/")),
        key=lambda name: metrics[name]["MSE"],
    )
    large = min(
        (name for name in metrics if name.startswith("TDK-large/")),
        key=lambda name: metrics[name]["MSE"],
    )
    metric_atlas(metrics, out)
    pareto(metrics, out)
    visual = {
        "TDK-HO small": predictions[small],
        "TDK-HO large": predictions[large],
        **baseline_predictions,
    }
    visual_sample_metrics = {
        label: (
            np.linalg.norm(value - target, axis=(1, 2))
            / (np.linalg.norm(target, axis=(1, 2)) + 1e-12),
            np.mean((value - target) ** 2, axis=(1, 2)),
        )
        for label, value in visual.items()
    }
    rel = np.linalg.norm(predictions[small] - target, axis=(1, 2)) / (
        np.linalg.norm(target, axis=(1, 2)) + 1e-12
    )
    rank = np.argsort(rel)
    source = np.asarray(data["X_data"], np.float32)[sorted_test_indices]
    for file, index, title in [
        ("04_sample_best.png", rank[0], "Magnetostatics vector field · low-error sample #1 / 600"),
        (
            "04_sample_rank05.png",
            rank[4],
            "Magnetostatics vector field · low-error sample #5 / 600",
        ),
        (
            "04_sample_median.png",
            rank[len(rank) // 2],
            "Magnetostatics vector field · median-error sample",
        ),
        (
            "04_sample_worst.png",
            rank[-1],
            "Magnetostatics vector field · high-error sample #600 / 600",
        ),
    ]:
        sample_panel(data["nodes"], source, target, visual, int(index), title, out, file)
    for file, index, title in [
        ("05_3d_sample_best.png", rank[0], "Magnetostatics · 3-D low-error sample #1 / 600"),
        (
            "05_3d_sample_median.png",
            rank[len(rank) // 2],
            "Magnetostatics · 3-D median-error sample",
        ),
        ("05_3d_sample_worst.png", rank[-1], "Magnetostatics · 3-D high-error sample #600 / 600"),
    ]:
        sample_panel_3d(
            data["nodes"],
            source,
            target,
            visual,
            visual_sample_metrics,
            int(index),
            title,
            out,
            file,
        )
    for file, index, title in [
        (
            "06_3d_error_best.png",
            rank[0],
            "Magnetostatics · 3-D error atlas · low-error sample #1 / 600",
        ),
        (
            "06_3d_error_median.png",
            rank[len(rank) // 2],
            "Magnetostatics · 3-D error atlas · median-error sample",
        ),
        (
            "06_3d_error_worst.png",
            rank[-1],
            "Magnetostatics · 3-D error atlas · high-error sample #600 / 600",
        ),
    ]:
        sample_error_panel_3d(
            data["nodes"],
            source,
            target,
            visual,
            visual_sample_metrics,
            int(index),
            title,
            out,
            file,
        )
    
    
    for sample_label, sample_index in (
        ("best", int(rank[0])),
        ("median", int(rank[len(rank) // 2])),
        ("worst", int(rank[-1])),
    ):
        hsd_style_slice_atlas(
            data["nodes"], source, target, visual, sample_index, sample_label, out
        )
    
    
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "tdk_groups": list(GROUPS),
                "baseline_condition": "base",
                "test_samples": int(len(target)),
                "provenance": "saved validation-best TDK predictions and aligned archived base baseline predictions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"small_best": small, "large_best": large, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
