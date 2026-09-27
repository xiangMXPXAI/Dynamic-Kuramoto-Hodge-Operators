"""Evaluate saved C1 and C2 predictions for toroidal transport."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
try:
    from .generate_cochain_data import oriented_complex
except ImportError:  
    sys.path.insert(0, str(HERE))
    from generate_cochain_data import oriented_complex

SOURCE = HERE.parent / "data" / "torus_transport_v1.pkl"

TDK_CONFIG = "conditioned"
BASELINE_MAIN_CONFIG = "native"
FEATURE_FULL_CONFIG = "conditioned"
BASELINE_NAMES = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")
COLORS = {
    "TDK-HO small": "#2d75a8",
    "TDK-HO large": "#12375a",
    "GNO": "#c3684b",
    "FNO": "#bf8a31",
    "MGN": "#807152",
    "DeepONet": "#84578e",
    "GeoFNO": "#4f887f",
    "HSD": "#ba544a",
}


@dataclass
class Run:
    name: str
    task: str
    path: Path
    parameters: int
    prediction: np.ndarray
    target: np.ndarray
    result: dict


@dataclass
class Geometry:
    points: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    face_edges: np.ndarray
    face_signs: np.ndarray
    d0: sparse.csr_matrix
    d1: sparse.csr_matrix
    node_area: np.ndarray
    face_area: np.ndarray
    node_adj: sparse.csr_matrix
    face_adj: sparse.csr_matrix
    harmonic: tuple[np.ndarray, np.ndarray, np.ndarray]
    lpe: tuple[np.ndarray, np.ndarray, np.ndarray]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "figure.facecolor": "#fbfcfe",
            "axes.facecolor": "#fbfcfe",
            "savefig.facecolor": "#fbfcfe",
            "axes.edgecolor": "#b7c4d0",
            "axes.labelcolor": "#243b53",
            "xtick.color": "#52677c",
            "ytick.color": "#52677c",
            "grid.color": "#dce5ed",
            "grid.alpha": 0.82,
            "axes.titleweight": "semibold",
        }
    )


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, zorder=0)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def build_geometry(cache: Path) -> Geometry:
    cache.mkdir(parents=True, exist_ok=True)
    static = cache / "torus_geometry.npz"
    if static.exists():
        z = np.load(static)
        points, faces, edges, fe, fs = (
            z["points"],
            z["faces"],
            z["edges"],
            z["face_edges"],
            z["face_signs"],
        )
        d0 = sparse.csr_matrix(
            (
                np.tile(np.array([-1.0, 1.0]), len(edges)),
                (np.repeat(np.arange(len(edges)), 2), edges.ravel()),
            ),
            shape=(len(edges), len(points)),
        )
        d1 = sparse.csr_matrix(
            (fs.ravel(), (np.repeat(np.arange(len(faces)), 3), fe.ravel())),
            shape=(len(faces), len(edges)),
        )
        node_area, area = z["node_area"], z["face_area"]
    else:
        with SOURCE.open("rb") as handle:
            raw = pickle.load(handle)
        points, faces = np.asarray(raw["points"], np.float64), np.asarray(raw["faces"], np.int64)
        edges, fe, fs, d0, d1 = oriented_complex(faces, len(points))
        cross = np.cross(
            points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
        )
        area = 0.5 * np.linalg.norm(cross, axis=1)
        node_area = np.zeros(len(points))
        [np.add.at(node_area, faces[:, i], area / 3) for i in range(3)]
        np.savez_compressed(
            static,
            points=points,
            faces=faces,
            edges=edges,
            face_edges=fe,
            face_signs=fs,
            node_area=node_area,
            face_area=area,
        )
    if (d1 @ d0).nnz and np.max(np.abs((d1 @ d0).data)) > 1e-7:
        raise RuntimeError("invalid torus orientation: d1 d0 != 0")
    node_adj = (abs(d0).T @ abs(d0)).tocsr()
    node_adj.setdiag(0)
    node_adj.eliminate_zeros()
    face_adj = (abs(d1) @ abs(d1).T).tocsr()
    face_adj.setdiag(0)
    face_adj.eliminate_zeros()

    def eig(lap, count, zero):
        n = lap.shape[0]
        k = min(max(count + 5, 8), n - 2)
        if zero:
            values, vectors = eigsh(
                lap.astype(np.float64), k=k, sigma=-1e-7, which="LM", tol=1e-8, maxiter=60000
            )
        else:
            values, vectors = eigsh(
                lap.astype(np.float64), k=k, which="SM", tol=1e-7, maxiter=60000
            )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        take = (
            np.flatnonzero(values <= 1e-5)[:count]
            if zero
            else np.flatnonzero(values > 1e-5)[:count]
        )
        if len(take) != count:
            raise RuntimeError(
                f"spectral audit: expected {count} {'harmonic' if zero else 'positive'} modes, got {len(take)}"
            )
        return vectors[:, take]

    l0, l1, l2 = d0.T @ d0, d0 @ d0.T + d1.T @ d1, d1 @ d1.T
    return Geometry(
        points,
        faces,
        edges,
        fe,
        fs,
        d0,
        d1,
        node_area,
        area,
        node_adj,
        face_adj,
        (eig(l0, 1, True), eig(l1, 2, True), eig(l2, 1, True)),
        (eig(l0, 16, False), eig(l1, 16, False), eig(l2, 16, False)),
    )


def read_run(
    name: str,
    root: Path,
    task: str,
    baseline: bool = False,
    variant: str = "full",
    config: str | None = None,
) -> Run:
    config = config or (BASELINE_MAIN_CONFIG if baseline else TDK_CONFIG)
    path = root / f"form_{task}" / config
    path = path / (name if baseline else variant) / "seed_42"
    required = (
        path / "best.pt",
        path / "result.json",
        path / "prediction_test_normalized.npy",
        path / "target_test_normalized.npy",
    )
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("missing saved evaluation artefacts: " + ", ".join(missing))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    scale = float(result["normalization"]["y_scale"])
    p = np.asarray(np.load(path / "prediction_test_normalized.npy"), np.float64) * scale
    y = np.asarray(np.load(path / "target_test_normalized.npy"), np.float64) * scale
    if p.shape != y.shape:
        raise RuntimeError(f"shape mismatch at {path}: {p.shape} versus {y.shape}")
    return Run(name, task, path, int(result["parameters"]), p, y, result)


def rel(pred, target):
    error = pred - target
    return {
        "MSE": float(np.mean(error**2)),
        "MAE": float(np.mean(np.abs(error))),
        "Rel L1": float(np.mean(np.abs(error).sum(1) / np.maximum(np.abs(target).sum(1), 1e-12))),
        "Rel L2": float(
            np.mean(
                np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
            )
        ),
    }


def cosine(pred, target):
    return float(
        np.mean(
            (
                1
                + np.einsum("bi,bi->b", pred, target)
                / (np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1) + 1e-12)
            )
            / 2
        )
    )


def energy(pred, target):
    a, b = np.sum(pred**2, axis=1), np.sum(target**2, axis=1)
    return float(np.mean(1 / (1 + np.abs(np.log((a + 1e-12) / (b + 1e-12))))))


def topology(pred, target, adjacency, weights):
    sb = []
    si = []
    for p, y in zip(pred, target):
        beta = []
        iou = []
        for alpha in (0.2, 0.5, 0.8):
            th = y.min() + alpha * (y.max() - y.min())
            pm, tm = p >= th, y >= th
            pi, ti = np.flatnonzero(pm), np.flatnonzero(tm)
            bp = (
                0
                if not len(pi)
                else connected_components(adjacency[pi, :][:, pi], directed=False)[0]
            )
            bt = (
                0
                if not len(ti)
                else connected_components(adjacency[ti, :][:, ti], directed=False)[0]
            )
            beta.append(np.exp(-1.5 * abs(bp - bt) / max(bp, bt, 1)))
            iou.append(np.sum(weights * (pm & tm)) / max(np.sum(weights * (pm | tm)), 1e-12))
        sb.append(np.mean(beta))
        si.append(np.mean(iou))
    return float(np.mean(sb)), float(np.sqrt(np.mean(si)))


def metrics(run: Run, geo: Geometry) -> dict:
    p, y, rank = run.prediction, run.target, int(run.task)
    out = rel(p, y)
    lpe = geo.lpe[rank]
    harmonic = geo.harmonic[rank]
    spec = lambda a: a @ lpe
    spectral = float(
        np.mean(
            np.exp(
                -np.linalg.norm(spec(p) - spec(y), axis=1)
                / np.maximum(np.linalg.norm(spec(y), axis=1), 1e-12)
            )
        )
    )
    harmonic = float(
        np.mean(
            np.linalg.norm((p - y) @ harmonic, axis=1)
            / np.maximum(np.linalg.norm(y @ harmonic, axis=1), 1e-12)
        )
    )
    if rank == 1:
        codp, cody = (geo.d0.T @ p.T).T, (geo.d0.T @ y.T).T
        curlp, curly = (geo.d1 @ p.T).T, (geo.d1 @ y.T).T
        out.update(
            {
                "Co-diff Fid": cosine(codp, cody),
                "Curl MSE": float(np.mean((curlp - curly) ** 2)),
                "Curl Fid": cosine(curlp, curly),
                "Enstrophy Fid": energy(curlp, curly),
                "Energy Fid": energy(p, y),
                "Spectral Fid": spectral,
                "Harmonic Rel L2": harmonic,
                "Curl Target Rel L2": float(
                    np.mean(
                        np.linalg.norm(curlp - curly, axis=1)
                        / np.maximum(np.linalg.norm(curly, axis=1), 1e-12)
                    )
                ),
            }
        )
    else:
        cobp, coby = (geo.d1.T @ p.T).T, (geo.d1.T @ y.T).T
        beta, iou = topology(p, y, geo.face_adj, geo.face_area / geo.face_area.sum())
        out.update(
            {
                "Co-boundary Fid": cosine(cobp, coby),
                "Co-boundary MSE": float(np.mean((cobp - coby) ** 2)),
                "Coexact Energy Fid": energy(cobp, coby),
                "Energy Fid": energy(p, y),
                "Spectral Fid": spectral,
                "Harmonic Rel L2": harmonic,
                "S_beta0": beta,
                "IoU": iou,
            }
        )
    return out


def columns(task: str):
    return (
        [
            "MSE",
            "MAE",
            "Rel L1",
            "Rel L2",
            "Co-diff Fid",
            "Curl MSE",
            "Curl Fid",
            "Enstrophy Fid",
            "Energy Fid",
            "Spectral Fid",
            "Harmonic Rel L2",
            "Curl Target Rel L2",
        ]
        if task == "1"
        else [
            "MSE",
            "MAE",
            "Rel L1",
            "Rel L2",
            "Co-boundary Fid",
            "Co-boundary MSE",
            "Coexact Energy Fid",
            "Energy Fid",
            "Spectral Fid",
            "Harmonic Rel L2",
            "S_beta0",
            "IoU",
        ]
    )


def atlas(rows, task, out):
    cols = columns(task)
    low = {
        "MSE",
        "MAE",
        "Rel L1",
        "Rel L2",
        "Curl MSE",
        "Co-boundary MSE",
        "Harmonic Rel L2",
        "Curl Target Rel L2",
    }
    values = np.empty((len(rows), len(cols)))
    for j, c in enumerate(cols):
        x = np.array([r[c] for r in rows])
        values[:, j] = np.exp(-np.log1p(x / max(np.median(x[x > 0]), 1e-12))) if c in low else x
    fig = plt.figure(figsize=(22, 7.3))
    gs = fig.add_gridspec(
        1, 2, width_ratios=(1.25, 4.2), left=0.07, right=0.95, top=0.80, bottom=0.14, wspace=0.08
    )
    ax = fig.add_subplot(gs[0])
    yy = np.arange(len(rows))
    mse = np.array([r["MSE"] for r in rows])
    ax.hlines(yy, mse.min() * 0.55, mse, color="#d7e2eb", lw=1.5)
    for y, r in zip(yy, rows):
        ax.scatter(r["MSE"], y, s=68, color=COLORS[r["Model"]], edgecolor="white", lw=0.9, zorder=3)
        ax.annotate(
            f"{r['MSE']:.2e}",
            (r["MSE"], y),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    ax.set(
        xscale="log",
        ylim=(len(rows) - 0.45, -0.55),
        yticks=yy,
        yticklabels=[r["Model"].replace("TDK-HO ", "TDK-") for r in rows],
        xlabel="physical MSE  (lower is better)",
    )
    clean(ax, "x")
    ax.axhline(1.5, color="#91a6ba", lw=1.2, ls="--")
    heat = fig.add_subplot(gs[1])
    im = heat.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    heat.set(
        xticks=np.arange(len(cols)),
        xticklabels=[c.replace(" ", "\n") for c in cols],
        yticks=yy,
        yticklabels=[""] * len(rows),
    )
    heat.xaxis.tick_top()
    heat.tick_params(axis="x", labelsize=7.4, pad=8)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            heat.text(
                j,
                i,
                f"{r[c]:.2e}" if r[c] < 0.001 else f"{r[c]:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if values[i, j] > 0.56 else "#163047",
            )
    heat.axhline(1.5, lw=2, color="white")
    [s.set_visible(False) for s in heat.spines.values()]
    fig.colorbar(im, ax=heat, shrink=0.72, pad=0.022, label="direction-aware display score")
    title = "C1 terminal covariant flux" if task == "1" else "C2 terminal face-integrated mass"
    fig.suptitle(
        f"Toroidal transport — {title}: 12-metric atlas",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.07,
        0.905,
        "Same held-out saved predictions; raw physical metrics are printed in cells. Colour only assists cross-metric scanning.",
        fontsize=9,
        color="#596d82",
    )
    save(fig, out / f"01_c{task}_metric_atlas.png")


def tradeoff(rows, task, out):
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.1), constrained_layout=True)
    for ax, key, label in zip(
        axes, ("MSE", "Rel L1", "Rel L2"), ("MSE", "relative L1", "relative L2")
    ):
        for r in rows:
            ax.scatter(
                r["Parameters"],
                r[key],
                color=COLORS[r["Model"]],
                marker="o" if r["Model"].startswith("TDK") else "s",
                s=75,
                edgecolor="white",
                lw=1,
                zorder=3,
            )
            ax.annotate(
                r["Model"].replace("TDK-HO ", "TDK-"),
                (r["Parameters"], r[key]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="trainable parameters",
            ylabel=label,
            title=f"{label} vs capacity",
        )
        clean(ax, "both")
    save(fig, out / f"02_c{task}_accuracy_parameter_tradeoff.png")


def population(runs, task, out):
    per = {
        r.name: np.linalg.norm(r.prediction - r.target, axis=1)
        / np.maximum(np.linalg.norm(r.target, axis=1), 1e-12)
        for r in runs
    }
    names = [r.name for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.4), constrained_layout=True)
    for n in names:
        x = np.sort(per[n])
        axes[0].plot(
            x, np.linspace(0, 1, len(x)), lw=2, color=COLORS[n], label=n.replace("TDK-HO ", "TDK-")
        )
    axes[0].set(
        xscale="log",
        xlabel="per-sample relative L2",
        ylabel="empirical CDF",
        title="All saved held-out samples",
    )
    clean(axes[0], "both")
    axes[0].legend(ncol=2, frameon=False, fontsize=7.5, loc="lower right")
    vio = axes[1].violinplot(
        [np.log10(np.maximum(per[n], 1e-12)) for n in names], showmedians=True, showextrema=False
    )
    for b, n in zip(vio["bodies"], names):
        b.set(facecolor=COLORS[n], edgecolor="white", alpha=0.84)
    vio["cmedians"].set(color="#142b42", lw=1.5)
    axes[1].set(
        xticks=np.arange(1, len(names) + 1),
        xticklabels=[n.replace("TDK-HO ", "TDK-") for n in names],
        ylabel="log10 relative L2",
        title="Error dispersion and median",
    )
    axes[1].tick_params(axis="x", rotation=23)
    clean(axes[1])
    fig.suptitle(
        f"Toroidal transport C{task}: full test-set error distribution",
        x=0.03,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#15283f",
    )
    save(fig, out / f"03_c{task}_population_error.png")


def cochain_plot(ax, geo, values, task, norm, cmap):
    if getattr(ax, "name", "") == "3d":
        if task == "1":
            coll = Line3DCollection(
                geo.points[geo.edges], cmap=cmap, norm=norm, linewidths=0.43, antialiased=True
            )
            coll.set_array(values)
            ax.add_collection3d(coll)
        else:
            coll = Poly3DCollection(
                geo.points[geo.faces], cmap=cmap, norm=norm, linewidths=0, antialiased=True
            )
            coll.set_array(values)
            ax.add_collection3d(coll)
        lo, hi = geo.points.min(0), geo.points.max(0)
        ax.set(
            xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]), box_aspect=(1, 1, 0.55)
        )
        ax.view_init(elev=24, azim=-52)
        ax.set_axis_off()
        return coll
    if task == "1":
        coll = LineCollection(
            geo.points[geo.edges][:, :, :2], cmap=cmap, norm=norm, linewidths=0.38
        )
        coll.set_array(values)
        ax.add_collection(coll)
        ax.set(
            xlim=(geo.points[:, 0].min(), geo.points[:, 0].max()),
            ylim=(geo.points[:, 1].min(), geo.points[:, 1].max()),
            aspect="equal",
        )
    else:
        poly = PolyCollection(
            geo.points[geo.faces][:, :, :2], array=values, cmap=cmap, norm=norm, linewidths=0
        )
        ax.add_collection(poly)
        ax.set(
            xlim=(geo.points[:, 0].min(), geo.points[:, 0].max()),
            ylim=(geo.points[:, 1].min(), geo.points[:, 1].max()),
            aspect="equal",
        )
    ax.set_axis_off()
    return coll if task == "1" else poly


def samples(runs, geo, task, out):
    anchor = next(x for x in runs if x.name == "TDK-HO large")
    error = np.linalg.norm(anchor.prediction - anchor.target, axis=1) / np.maximum(
        np.linalg.norm(anchor.target, axis=1), 1e-12
    )
    idxs = {
        "low": int(np.argsort(error)[max(0, int(0.1 * len(error)) - 1)]),
        "median": int(np.argsort(error)[int(0.5 * len(error))]),
        "high": int(np.argsort(error)[min(len(error) - 1, int(0.9 * len(error)))]),
    }
    for tag, idx in idxs.items():
        truth = anchor.target[idx]
        vmax = np.quantile(np.abs(truth), 0.995)
        norm = Normalize(-vmax, vmax)
        errmax = max(np.quantile(np.abs(r.prediction[idx] - truth), 0.995) for r in runs)
        enorm = Normalize(0, errmax)
        for group, label in ((runs[:2], "tdk"), (runs[2:], "baseline")):
            if label == "baseline":
                fig, grid = plt.subplots(
                    2,
                    4,
                    figsize=(17.4, 8.1),
                    subplot_kw={"projection": "3d"},
                    constrained_layout=True,
                )
                axes = list(grid.flat)
                plot_axes = [axes[0], *axes[1:4], *axes[4:7]]
                axes[7].set_axis_off()
            else:
                fig, grid = plt.subplots(
                    1,
                    len(group) + 1,
                    figsize=(4.25 * (len(group) + 1), 4.5),
                    subplot_kw={"projection": "3d"},
                    constrained_layout=True,
                )
                axes = list(np.atleast_1d(grid))
                plot_axes = axes
            artist = cochain_plot(plot_axes[0], geo, truth, task, norm, "coolwarm")
            plot_axes[0].set_title("ground truth", loc="left", color="#253e57")
            for ax, r in zip(plot_axes[1:], group):
                cochain_plot(ax, geo, r.prediction[idx], task, norm, "coolwarm")
                ax.set_title(r.name.replace("TDK-HO ", "TDK-"), loc="left", color="#253e57")
            fig.colorbar(artist, ax=plot_axes, shrink=0.7, pad=0.015, label="native cochain value")
            fig.suptitle(
                f"C{task} {tag}-error sample — prediction",
                x=0.02,
                ha="left",
                fontsize=13,
                fontweight="bold",
            )
            save(fig, out / f"04_c{task}_{tag}_{label}_prediction.png")
            shape = (2, 3) if label == "baseline" else (1, len(group))
            fig, grid = plt.subplots(
                *shape,
                figsize=(12.8, 8.0) if label == "baseline" else (4.25 * len(group), 4.15),
                subplot_kw={"projection": "3d"},
                constrained_layout=True,
            )
            axes = list(np.atleast_1d(grid).flat)
            artist = None
            for ax, r in zip(axes, group):
                artist = cochain_plot(
                    ax, geo, np.abs(r.prediction[idx] - truth), task, enorm, "magma"
                )
                ax.set_title(
                    f"{r.name.replace('TDK-HO ','TDK-')} |error|", loc="left", color="#253e57"
                )
            fig.colorbar(
                artist, ax=axes, shrink=0.7, pad=0.015, label="absolute native-cochain error"
            )
            fig.suptitle(
                f"C{task} {tag}-error sample — direct errors",
                x=0.02,
                ha="left",
                fontsize=13,
                fontweight="bold",
            )
            save(fig, out / f"05_c{task}_{tag}_{label}_error.png")


def controls(task, roots, geo, out):
    entries = []
    for name in BASELINE_NAMES:
        try:
            
            
            def config_read(config):
                p = roots["base"] / f"form_{task}" / config / name / "seed_42"
                result = json.loads((p / "result.json").read_text())
                s = float(result["normalization"]["y_scale"])
                return Run(
                    name,
                    task,
                    p,
                    int(result["parameters"]),
                    np.load(p / "prediction_test_normalized.npy") * s,
                    np.load(p / "target_test_normalized.npy") * s,
                    result,
                )

            native, conditioned = config_read("native"), config_read(FEATURE_FULL_CONFIG)
            a, b = metrics(native, geo), metrics(conditioned, geo)
            entries.append(
                {
                    "Model": name,
                    "Core Rel L2": a["Rel L2"],
                    "Full Rel L2": b["Rel L2"],
                    "Core MSE": a["MSE"],
                    "Full MSE": b["MSE"],
                    "Delta Rel L2 (%)": 100 * (b["Rel L2"] - a["Rel L2"]) / max(a["Rel L2"], 1e-12),
                }
            )
        except FileNotFoundError:
            pass
    if not entries:
        return entries
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), constrained_layout=True)
    y = np.arange(len(entries))
    for ax, l, r, label in (
        (axes[0], "Core Rel L2", "Full Rel L2", "relative L2"),
        (axes[1], "Core MSE", "Full MSE", "physical MSE"),
    ):
        for yy, row in zip(y, entries):
            ax.plot([row[l], row[r]], [yy, yy], color="#c6d3df", lw=2)
            ax.scatter(row[l], yy, color="#9aa9b7", s=55, edgecolor="white", zorder=3)
            ax.scatter(
                row[r],
                yy,
                color=COLORS[row["Model"]],
                marker="s",
                s=60,
                edgecolor="white",
                zorder=3,
            )
        ax.set(
            xscale="log",
            yticks=y,
            yticklabels=[r["Model"] for r in entries],
            xlabel=label,
            title="baseline feature response: native → conditioned",
        )
        ax.invert_yaxis()
        clean(ax, "x")
    fig.suptitle(
        f"Toroidal transport C{task}: six-baseline feature control",
        x=0.03,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#15283f",
    )
    save(fig, out / f"06_c{task}_baseline_feature_control.png")
    return entries


def structural(task, roots, geo, out):
    rows = []
    for variant in ("full", "no_dirac", "no_phase", "no_harmonic"):
        try:
            r = read_run(
                "TDK-HO small",
                roots["small"],
                task,
                False,
                variant,
            )
            rows.append({"Variant": variant, "Parameters": r.parameters, **metrics(r, geo)})
        except FileNotFoundError:
            pass
    if not rows:
        return rows
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    x = np.arange(len(rows))
    labels = [r["Variant"].replace("_", " ") for r in rows]
    for ax, key, label in ((axes[0], "Rel L2", "relative L2"), (axes[1], "MSE", "physical MSE")):
        bars = ax.bar(
            x,
            [r[key] for r in rows],
            color=["#12375a" if r["Variant"] == "full" else "#9baaba" for r in rows],
            width=0.62,
        )
        ax.set(
            xticks=x,
            xticklabels=labels,
            ylabel=label,
            title=f"TDK-HO small structural ablation — {label}",
        )
        clean(ax)
        [ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)]
    fig.suptitle(
        f"Toroidal transport C{task}: full feature condition, structural controls",
        x=0.03,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#15283f",
    )
    save(fig, out / f"07_c{task}_tdk_structure_control.png")
    return rows


def write_table(rows, cols):
    lines = [
        "| Model | Parameters | " + " | ".join(cols) + " |",
        "|---|---:" + "|".join(["---:" for _ in cols]) + "|",
    ]
    for r in rows:
        lines.append(
            "| "
            + r["Model"]
            + f" | {r['Parameters']:,} | "
            + " | ".join(f"{r[c]:.3e}" if r[c] < 0.001 else f"{r[c]:.4f}" for c in cols)
            + " |"
        )
    return lines


def report(task, rows, feature_rows, structural_rows, out):
    title = (
        "C1: terminal covariant transport flux"
        if task == "1"
        else "C2: terminal face-integrated mass"
    )
    metrics_note = (
        "Co-diff, Curl MSE/Fid, Enstrophy, Energy, spectral and harmonic errors"
        if task == "1"
        else "Co-boundary, coexact energy, spectral, harmonic, S_beta0 and IoU"
    )
    lines = [
        f"# Toroidal transport {title}",
        "",
        "## Evaluation protocol",
        "",
        "All numbers are recomputed in physical units from saved predictions written after each validation-best checkpoint was reloaded. No checkpoint is retrained, tuned or overwritten. The eight primary runs use the same saved test-array ordering for that task. The operator is unweighted: all derivatives in the derived metrics use the fixed oriented incidences $$d_0$$ and $$d_1$$.",
        "",
        f"The 12 columns combine base errors with native cochain diagnostics: {metrics_note}. Lower is better for error columns; fidelity, $$S_{{\\beta_0}}$$ and IoU are higher-is-better.",
        "",
        "## Eight-model primary comparison",
        "",
        *write_table(rows, columns(task)),
        "",
        f"![metric atlas](01_c{task}_metric_atlas.png)",
        "",
        f"![capacity tradeoff](02_c{task}_accuracy_parameter_tradeoff.png)",
        "",
        "## Full test-set distribution",
        "",
        f"![population error](03_c{task}_population_error.png)",
        "",
        "## Quantile-selected visual comparisons",
        "",
        "Samples are selected at the 10th, 50th and 90th percentile of TDK-HO large's per-sample relative L2. Predictions and direct cochain errors are intentionally split into TDK-HO and baseline panels to keep each panel legible.",
        "",
    ]
    for tag in ("low", "median", "high"):
        lines += [
            f"### {tag} error sample",
            "",
            f"![TDK prediction](04_c{task}_{tag}_tdk_prediction.png)",
            "",
            f"![baseline prediction](04_c{task}_{tag}_baseline_prediction.png)",
            "",
            f"![TDK error](05_c{task}_{tag}_tdk_error.png)",
            "",
            f"![baseline error](05_c{task}_{tag}_baseline_error.png)",
            "",
        ]
    lines += [
        "## Feature-control experiment",
        "",
        "Only models for which both native and conditioned checkpoints exist are shown. This isolates the effect of the optional condition features for each baseline without inventing a missing DKHO run.",
        "",
    ]
    if feature_rows:
        lines += (
            [
                "| baseline | native Rel L2 | conditioned Rel L2 | native MSE | conditioned MSE | Delta Rel L2 (%) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            + [
                f"| {r['Model']} | {r['Core Rel L2']:.4f} | {r['Full Rel L2']:.4f} | {r['Core MSE']:.3e} | {r['Full MSE']:.3e} | {r['Delta Rel L2 (%)']:+.1f}% |"
                for r in feature_rows
            ]
            + ["", f"![feature control](06_c{task}_baseline_feature_control.png)", ""]
        )
    lines += [
        "## TDK-HO structural controls",
        "",
        "The full condition is contrasted with actual saved `no_dirac`, `no_phase`, and `no_harmonic` small-model checkpoints. Their 200-epoch budget differs from the full model's 300 epochs; the plot is structural evidence, not a budget-matched leaderboard.",
        "",
    ]
    if structural_rows:
        lines += (
            ["| Variant | Parameters | Rel L2 | MSE |", "|---|---:|---:|---:|"]
            + [
                f"| {r['Variant']} | {r['Parameters']:,} | {r['Rel L2']:.4f} | {r['MSE']:.3e} |"
                for r in structural_rows
            ]
            + ["", f"![structural control](07_c{task}_tdk_structure_control.png)", ""]
        )
    (out / f"REPORT_C{task}.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global TDK_CONFIG, BASELINE_MAIN_CONFIG, FEATURE_FULL_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="1,2")
    p.add_argument(
        "--output", type=Path, default=HERE.parent / "reports" / "cochain_evaluation"
    )
    p.add_argument("--tdk-feature-config", default=TDK_CONFIG)
    p.add_argument("--baseline-main-config", default=BASELINE_MAIN_CONFIG)
    p.add_argument("--feature-control-full-config", default=FEATURE_FULL_CONFIG)
    args = p.parse_args()
    TDK_CONFIG = args.tdk_feature_config
    BASELINE_MAIN_CONFIG = args.baseline_main_config
    FEATURE_FULL_CONFIG = args.feature_control_full_config
    style()
    output = args.output.resolve()
    geo = build_geometry(output / "cache")
    roots = {
        "small": HERE.parent / "runs" / "dkho" / "small",
        "large": HERE.parent / "runs" / "dkho" / "large",
        "base": HERE.parent / "runs" / "baseline" / "main",
    }
    all_summary = {}
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        if task not in ("1", "2"):
            p.error("this evaluator handles C1/C2; scalar C0 is already reported in ../report.md")
        out = output / f"form_{task}"
        out.mkdir(parents=True, exist_ok=True)
        runs = [
            read_run("TDK-HO small", roots["small"], task),
            read_run("TDK-HO large", roots["large"], task),
        ] + [read_run(n, roots["base"], task, True) for n in BASELINE_NAMES]
        if any(r.prediction.shape != runs[0].prediction.shape for r in runs):
            raise RuntimeError("primary runs do not share a saved test-array shape")
        rows = [
            {
                "Model": r.name,
                "Parameters": r.parameters,
                **metrics(r, geo),
                "Checkpoint": str(r.path.relative_to(HERE.parent)),
            }
            for r in runs
        ]
        atlas(rows, task, out)
        tradeoff(rows, task, out)
        population(runs, task, out)
        samples(runs, geo, task, out)
        feature = controls(task, roots, geo, out)
        structure = structural(task, roots, geo, out)
        with (out / "main_metrics.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "main_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (out / "feature_control.json").write_text(json.dumps(feature, indent=2), encoding="utf-8")
        (out / "structure_control.json").write_text(
            json.dumps(structure, indent=2), encoding="utf-8"
        )
        report(task, rows, feature, structure, out)
        all_summary[f"form_{task}"] = rows
        print(f"[torus cochain evaluation] form_{task}: wrote {out}", flush=True)
    manifest_path = output / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    tasks = sorted(set(prior.get("tasks", [])) | set(all_summary))
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": tasks,
                "geometry": {
                    "n0": len(geo.points),
                    "n1": len(geo.edges),
                    "n2": len(geo.faces),
                    "betti": [1, 2, 1],
                },
                "selection": {
                    "tdk": TDK_CONFIG,
                    "baseline_main": BASELINE_MAIN_CONFIG,
                    "baseline_feature_control": ["native", FEATURE_FULL_CONFIG],
                },
                "provenance": "saved validation-best predictions only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
