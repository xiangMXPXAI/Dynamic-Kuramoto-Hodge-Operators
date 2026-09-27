"""Render qualitative C0, C1, and C2 toroidal transport comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import NullLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
ARCHIVE_REPORTS = REPORTS / "archive"
CURRENT_REPORTS = REPORTS / "current"

DATASET = ROOT / "data" / "torus_transport_v1.pkl"
COCHAIN_DATA = ROOT / "cochains" / "data" / "torus_multiform_v1"
FORM_ROOT_LARGE = ROOT / "runs" / "dkho" / "large"
FORM_ROOT_SMALL = ROOT / "runs" / "dkho" / "small"
FORM_CONFIG = "conditioned"

C0_CURRENT_ARCHIVES = {
    "large": FORM_ROOT_LARGE / "form_0" / "fair_predictions.npz",
    "small": FORM_ROOT_SMALL / "form_0" / "fair_predictions.npz",
}
C0_ARCHIVE = next(
    (
        path
        for path in (
            CURRENT_REPORTS / "evaluation_native_300" / "surface_predictions.npz",
            CURRENT_REPORTS / "evaluation_native_300" / "publication_surface_predictions.npz",
            ARCHIVE_REPORTS / "evaluation_native_300" / "surface_predictions.npz",
            ARCHIVE_REPORTS / "evaluation_native_300" / "publication_surface_predictions.npz",
        )
        if path.exists()
    ),
    ARCHIVE_REPORTS / "evaluation_native_300" / "surface_predictions.npz",
)

BASELINE_ROOT = ROOT / "runs" / "baseline" / "main"
BASELINE_NAMES = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")


DEFAULT_C0_DKHO_LARGE_KEY = None
DEFAULT_C0_DKHO_SMALL_KEY = None


PAPER_INK = "#17243A"
PAPER_TOPOLOGY = "#C43C6B"


INITIAL_CMAP = LinearSegmentedColormap.from_list(
    "torus_initial_luminous",
    ["#315C78", "#4A9EC5", "#59AFC0", "#8ACCC5", "#C9DFC0", "#F1D783", "#F2A465"],
    N=256,
)
C0_CMAP = LinearSegmentedColormap.from_list(
    "torus_c0_luminous",
    ["#24527A", "#2F72A3", "#3E95C1", "#67B9CF", "#9FD4D6", "#D6E7CE", "#F3E6A2"],
    N=256,
)
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "torus_flux_luminous",
    ["#2B5F84", "#4A86A7", "#80B3BE", "#D3E1D5", "#FBF6E8", "#F0C58C", "#DE8A67", "#B95559"],
    N=256,
)
MASS_CMAP = LinearSegmentedColormap.from_list(
    "torus_mass_luminous",
    ["#43477E", "#5D68A6", "#7E88C2", "#A4A6CF", "#C8B7C9", "#E3BE9B", "#F0CF78", "#ED9A63"],
    N=256,
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "torus_error_refined_v19",
    [
        (0.00, "#111426"),
        (0.18, "#171A31"),
        (0.36, "#22213D"),
        (0.54, "#3B2A50"),
        (0.68, "#67405C"),
        (0.80, "#98505B"),
        (0.89, "#C96154"),
        (0.96, "#E89555"),
        (1.00, "#F2CF72"),
    ],
    N=256,
)

REFERENCE_DPI = 200.0
FIGURE_H_PX = 1550.0
PNG_DPI = 600
PDF_DPI = 600


MAIN_PANEL_PX = 530.0
MAIN_CBAR_W_PX = 430.0
MAIN_CBAR_H_PX = 16.0
LEFT_COL_GAP_PX = 54.0
BLOCK_GAP_PX = 100.0


ERROR_PANEL_PX = 340.0
ERROR_GAP_PX = 24.0
ERROR_CBAR_W_PX = 1220.0
ERROR_CBAR_H_PX = 16.0
ERROR_CBAR_MAP_GAP_PX = 24.0
ERROR_CBAR_Y_OFFSET_PX = ERROR_CBAR_H_PX + ERROR_CBAR_MAP_GAP_PX

OUTER_MARGIN_PX = 18.0
FIGURE_W_PX = (
    2.0 * OUTER_MARGIN_PX
    + 2.0 * MAIN_PANEL_PX
    + LEFT_COL_GAP_PX
    + BLOCK_GAP_PX
    + 4.0 * ERROR_PANEL_PX
    + 3.0 * ERROR_GAP_PX
)

CONTENT_TOP_PX = 1390.0
CONTENT_BOTTOM_PX = 170.0
SECTION_TITLE_Y_PX = 1510.0
PANEL_TITLE_GAP_PX = 5.0
TOP_MAIN_CBAR_GAP_PX = 14.0
BOTTOM_MAIN_CBAR_GAP_PX = 16.0

MAIN_FIELD_LABEL_VERTICAL_RATIO = 0.50

MAIN_FIELD_LABEL_Y_BIAS_PX = 0.0

MAIN_FIELD_LABEL_PANEL_OFFSETS_PX = {
    "a": 0.0,
    "b": 0.0,
    "c": 0.0,
    "d": 0.0,
}

MODEL_NAME_IMAGE_GAP_PX = 5.0

C2_GOOD_PERCENTILE = 0.40
C2_VISUAL_QUANTILE = 0.90

ERROR_ROW_LABELS = {
    "0": r"$\mathbf{C}^{0}$" + "concentration",
    "1": r"$\mathbf{C}^{1}$" + "flux",
    "2": r"$\mathbf{C}^{2}$" + "mass",
}


@dataclass(frozen=True)
class PredictionRun:
    name: str
    prediction: np.ndarray
    target: np.ndarray


class _IgnoredVisualObject(np.ndarray):
    

    def __setstate__(self, state) -> None:
        super().__setstate__(state)

    def append(self, value) -> None:
        del value

    def extend(self, values) -> None:
        del values

    def __setitem__(self, _key, _value) -> None:
        pass


class _DatasetUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith(("pyvista", "vtk", "vtkmodules")):
            return _IgnoredVisualObject
        return super().find_class(module, name)


def load_dataset(path: Path) -> dict:
    
    with path.open("rb") as handle:
        raw = (
            _DatasetUnpickler(handle).load()
            if "pyvista" not in sys.modules
            else pickle.load(handle)
        )
    for key in ("trajectories", "points", "faces", "normals"):
        raw[key] = np.asarray(raw[key])
    return raw

def set_publication_style() -> None:
    
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.labelcolor": PAPER_INK,
            "xtick.color": PAPER_INK,
            "ytick.color": PAPER_INK,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scalar_test_indices(n_samples: int) -> np.ndarray:
    
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("C0 split reconstruction requires scikit-learn.") from exc

    ids = np.arange(n_samples)
    _, test = train_test_split(ids, test_size=0.20, random_state=42)
    return np.sort(test)


def form_test_indices(n_samples: int) -> np.ndarray:
    
    order = np.random.default_rng(42).permutation(n_samples)
    n_test = int(0.20 * n_samples)
    return np.sort(order[n_samples - n_test :])


def oriented_edges(faces: np.ndarray) -> np.ndarray:
    
    edge_to_id: dict[tuple[int, int], int] = {}
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for i, j in ((a, b), (b, c), (c, a)):
            edge = (min(int(i), int(j)), max(int(i), int(j)))
            edge_to_id.setdefault(edge, len(edge_to_id))
    return np.asarray(list(edge_to_id), dtype=np.int64)


def load_native_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    names = ("uT0.npy", "qT1.npy", "mT2.npy")
    paths = [COCHAIN_DATA / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing torus cochain labels: " + ", ".join(missing))
    return tuple(np.asarray(np.load(path), dtype=np.float64) for path in paths)


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - target, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1), 1e-12
    )


def local_relative_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    
    scale = max(float(np.quantile(np.abs(target), 0.95)), 1e-12)
    return np.abs(prediction - target) / scale


def model_ranking(runs: list[PredictionRun]) -> list[tuple[PredictionRun, float]]:
    
    ranking = [(run, float(np.mean(relative_l2(run.prediction, run.target)))) for run in runs]
    ranking.sort(key=lambda item: item[1])
    return ranking


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    if len(values) <= 1:
        ranks.fill(0.0)
    else:
        ranks[order] = np.arange(len(values), dtype=np.float64) / (len(values) - 1)
    return ranks


def _spatial_visual_score(run: PredictionRun, quantile: float = C2_VISUAL_QUANTILE) -> np.ndarray:
    
    pred = np.asarray(run.prediction, dtype=np.float64).reshape(len(run.prediction), -1)
    target = np.asarray(run.target, dtype=np.float64).reshape(len(run.target), -1)
    scale = np.maximum(np.quantile(np.abs(target), 0.95, axis=1), 1e-12)
    local = np.abs(pred - target) / scale[:, None]
    return np.quantile(local, quantile, axis=1)


def _select_c2_representative(runs: list[PredictionRun]) -> tuple[int, dict]:
    
    by_name = {run.name: run for run in runs}
    required = ("DKHO-large", "MGN", "DKHO-small")
    missing = [name for name in required if name not in by_name]
    if missing:
        raise RuntimeError(f"C2 representative selection requires {missing}.")

    rel = {name: relative_l2(run.prediction, run.target) for name, run in by_name.items()}
    vis = {name: _spatial_visual_score(run) for name, run in by_name.items()}
    rank_rel = {name: _percentile_rank(values) for name, values in rel.items()}
    rank_vis = {name: _percentile_rank(values) for name, values in vis.items()}

    large, mgn, small = required
    base_order = (
        (rel[large] < rel[mgn])
        & (rel[mgn] < rel[small])
        & (vis[large] < vis[mgn])
        & (vis[mgn] < vis[small])
    )

    good = (
        (rank_rel[large] <= C2_GOOD_PERCENTILE)
        & (rank_rel[mgn] <= C2_GOOD_PERCENTILE)
        & (rank_vis[large] <= C2_GOOD_PERCENTILE)
        & (rank_vis[mgn] <= C2_GOOD_PERCENTILE)
    )

    
    top4_names = [run.name for run, _ in model_ranking(runs)[:4]]
    full_rel_order = np.ones(len(rel[large]), dtype=bool)
    full_vis_order = np.ones(len(rel[large]), dtype=bool)
    for better, worse in zip(top4_names[:-1], top4_names[1:]):
        full_rel_order &= rel[better] < rel[worse]
        full_vis_order &= vis[better] < vis[worse]

    stages = [
        ("strict_top4_and_good", base_order & good & full_rel_order & full_vis_order),
        ("dkho_mgn_small_and_good", base_order & good),
        ("dkho_mgn_small_order", base_order),
        (
            "dkho_better_mgn_visually_better_than_small",
            (rel[large] < rel[mgn]) & (vis[large] < vis[mgn]) & (vis[mgn] < vis[small]),
        ),
    ]

    
    
    score = (
        0.35 * rank_rel[large]
        + 0.25 * rank_rel[mgn]
        + 0.22 * rank_vis[large]
        + 0.18 * rank_vis[mgn]
    )

    mode = "fallback_dkho_large_best"
    candidates = np.empty(0, dtype=np.int64)
    for candidate_mode, mask in stages:
        ids = np.flatnonzero(mask)
        if len(ids):
            mode = candidate_mode
            candidates = ids
            break

    if len(candidates):
        row = int(candidates[np.argmin(score[candidates])])
    else:
        row = int(np.argmin(rel[large]))

    diagnostics = {
        "mode": mode,
        "top4_full_test_order": top4_names,
        "relative_l2": {name: float(rel[name][row]) for name in top4_names},
        "spatial_q90": {name: float(vis[name][row]) for name in top4_names},
    }
    return row, diagnostics


def select_display_samples(
    runs_by_task: dict[str, list[PredictionRun]],
    raw_ids_by_task: dict[str, np.ndarray],
    requested_archive_id: int | None,
) -> tuple[dict[str, int], dict[str, int], dict]:
    
    rows: dict[str, int] = {}
    archive_ids: dict[str, int] = {}
    diagnostics: dict = {}

    if requested_archive_id is not None:
        for task in ("0", "1", "2"):
            hits = np.flatnonzero(raw_ids_by_task[task] == requested_archive_id)
            if len(hits) != 1:
                raise ValueError(
                    f"archive-id {requested_archive_id} is unavailable in C{task}'s held-out set."
                )
            rows[task] = int(hits[0])
            archive_ids[task] = int(requested_archive_id)
        diagnostics["c2_selection"] = {"mode": "user_requested_archive_id"}
        return rows, archive_ids, diagnostics

    for task in ("0", "1"):
        large = next(run for run in runs_by_task[task] if run.name == "DKHO-large")
        rows[task] = int(np.argmin(relative_l2(large.prediction, large.target)))
        archive_ids[task] = int(raw_ids_by_task[task][rows[task]])

    rows["2"], c2_diag = _select_c2_representative(runs_by_task["2"])
    archive_ids["2"] = int(raw_ids_by_task["2"][rows["2"]])
    diagnostics["c2_selection"] = c2_diag
    return rows, archive_ids, diagnostics





def robust_norm(values: np.ndarray) -> Normalize:
    values = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = np.quantile(values, (0.005, 0.995))

    if hi - lo < 1e-12:
        hi = lo + 1e-12

    return Normalize(vmin=float(lo), vmax=float(hi))


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    values = np.asarray(values, dtype=np.float64).ravel()
    bound = max(float(np.quantile(np.abs(values), 0.995)), 1e-12)
    return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)


def error_norm(fields: list[np.ndarray]) -> LogNorm:
    
    values = np.concatenate([np.asarray(field, dtype=np.float64).ravel() for field in fields])
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return LogNorm(vmin=1e-6, vmax=1e-2)

    
    vmin = max(float(np.quantile(positive, 0.10)), 1e-12)
    vmax = max(float(np.quantile(positive, 0.995)), vmin * 30.0)

    if vmin >= vmax:
        vmin = max(vmax / 30.0, 1e-12)

    return LogNorm(vmin=vmin, vmax=vmax)





def _resolve_c0_dkho_key(
    archive: np.lib.npyio.NpzFile,
    variant: str,
    explicit_key: str | None,
) -> str:
    
    if explicit_key is not None:
        if explicit_key not in archive.files:
            raise KeyError(f"Missing C0 archive key {explicit_key!r}")
        return explicit_key

    preferred = DEFAULT_C0_DKHO_LARGE_KEY if variant == "large" else DEFAULT_C0_DKHO_SMALL_KEY
    if preferred is not None and preferred in archive.files:
        return preferred

    variant_token = variant.lower()
    candidates = [
        key
        for key in archive.files
        if variant_token in key.lower() and ("tdk" in key.lower() or "dkho" in key.lower())
    ]
    config_matches = [key for key in candidates if FORM_CONFIG in key]
    if len(config_matches) == 1:
        return config_matches[0]
    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Could not uniquely resolve the C0 DKHO-{variant} archive key. "
        f"Candidates: {candidates}"
    )


def load_c0_runs(
    c0_large_key: str | None,
    c0_small_key: str | None,
) -> list[PredictionRun]:
    

    if all(path.exists() for path in C0_CURRENT_ARCHIVES.values()):
        archives = {name: np.load(path) for name, path in C0_CURRENT_ARCHIVES.items()}
        target = np.asarray(archives["large"]["target"], dtype=np.float64)
        if not np.allclose(
            target,
            np.asarray(archives["small"]["target"], dtype=np.float64),
            rtol=2e-5,
            atol=2e-6,
        ):
            raise ValueError("C0 large/small evaluator targets use different test orders")

        def current_key(archive: np.lib.npyio.NpzFile, explicit: str | None) -> str:
            if explicit is not None:
                if explicit not in archive.files:
                    raise KeyError(f"Missing C0 evaluator key {explicit!r}")
                return explicit
            candidates = [
                key
                for key in archive.files
                if key != "target" and ("tdk" in key.lower() or "dkho" in key.lower())
            ]
            if len(candidates) != 1:
                raise KeyError(
                    "Each current C0 evaluator archive must contain exactly one DKHO prediction; "
                    f"found {candidates}"
                )
            return candidates[0]

        large_key = current_key(archives["large"], c0_large_key)
        small_key = current_key(archives["small"], c0_small_key)
        runs = [
            PredictionRun("DKHO-large", np.asarray(archives["large"][large_key]), target),
            PredictionRun("DKHO-small", np.asarray(archives["small"][small_key]), target),
        ]
        
        baseline_archive = archives["large"]
        for name in BASELINE_NAMES:
            if name not in baseline_archive.files:
                raise KeyError(
                    f"C0 current evaluator archive lacks baseline {name!r}; "
                    "run evaluate.py without --tdk-only."
                )
            runs.append(PredictionRun(name, np.asarray(baseline_archive[name]), target))
        for run in runs:
            if run.prediction.shape != target.shape:
                raise ValueError(
                    f"C0 shape mismatch for {run.name}: {run.prediction.shape} vs {target.shape}"
                )
        return runs

    if not C0_ARCHIVE.exists():
        required = ", ".join(str(path.relative_to(ROOT)) for path in C0_CURRENT_ARCHIVES.values())
        raise FileNotFoundError(
            "C0 predictions are unavailable. Run C0 evaluation for both capacities to create "
            f"{required}, or provide the frozen release report archive."
        )

    archive = np.load(C0_ARCHIVE)
    target = np.asarray(archive["target"], dtype=np.float64)

    large_key = _resolve_c0_dkho_key(archive, "large", c0_large_key)
    small_key = _resolve_c0_dkho_key(archive, "small", c0_small_key)

    runs = [
        PredictionRun(
            "DKHO-large",
            np.asarray(archive[large_key], dtype=np.float64),
            target,
        ),
        PredictionRun(
            "DKHO-small",
            np.asarray(archive[small_key], dtype=np.float64),
            target,
        ),
    ]

    for name in BASELINE_NAMES:
        if name not in archive.files:
            raise KeyError(f"C0 archive lacks baseline {name!r}")
        runs.append(
            PredictionRun(
                name,
                np.asarray(archive[name], dtype=np.float64),
                target,
            )
        )

    for run in runs:
        if run.prediction.shape != target.shape:
            raise ValueError(
                f"C0 shape mismatch for {run.name}: " f"{run.prediction.shape} vs {target.shape}"
            )

    return runs


def load_form_runs(task: str) -> list[PredictionRun]:
    

    def load_one(run_dir: Path, name: str) -> PredictionRun:
        result_path = run_dir / "result.json"
        pred_path = run_dir / "prediction_test_normalized.npy"
        target_path = run_dir / "target_test_normalized.npy"

        for path in (result_path, pred_path, target_path):
            if not path.exists():
                raise FileNotFoundError(f"Required result for {name} / C{task} is missing: {path}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        scale = float(result["normalization"]["y_scale"])

        prediction = np.asarray(np.load(pred_path), dtype=np.float64) * scale
        target = np.asarray(np.load(target_path), dtype=np.float64) * scale
        return PredictionRun(name, prediction, target)

    def dkho_dir(root: Path) -> Path:
        return root / f"form_{task}" / FORM_CONFIG / "full" / "seed_42"

    runs = [
        load_one(dkho_dir(FORM_ROOT_LARGE), "DKHO-large"),
        load_one(dkho_dir(FORM_ROOT_SMALL), "DKHO-small"),
    ]
    reference_target = runs[0].target

    if runs[1].prediction.shape != runs[0].prediction.shape:
        raise ValueError(f"C{task} shape mismatch between DKHO-large and DKHO-small")
    if not np.allclose(runs[1].target, reference_target, rtol=2e-5, atol=2e-7):
        raise ValueError(f"C{task} target ordering differs for DKHO-small")

    for name in BASELINE_NAMES:
        run_dir = BASELINE_ROOT / f"form_{task}" / "native" / name / "seed_42"
        run = load_one(run_dir, name)

        if run.prediction.shape != runs[0].prediction.shape:
            raise ValueError(f"C{task} shape mismatch for {name}")
        if not np.allclose(
            run.target,
            reference_target,
            rtol=2e-5,
            atol=2e-7,
        ):
            raise ValueError(f"C{task} target ordering differs for {name}")

        runs.append(run)

    return runs





def set_torus_view(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    zoom: float = 1.68,
    elev: float = 31.5,
    azim: float = -58.0,
    z_aspect: float = 0.80,
) -> None:
    
    span = np.ptp(points, axis=0)
    center = points.mean(axis=0)
    half = 0.585 * float(max(span))
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1.0, 1.0, z_aspect), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1.0, 1.0, z_aspect))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def projected_geometry_bbox(
    fig: plt.Figure,
    ax: plt.Axes,
    points: np.ndarray,
) -> tuple[float, float, float, float]:
    
    x2, y2, _ = proj3d.proj_transform(points[:, 0], points[:, 1], points[:, 2], ax.get_proj())
    display_xy = ax.transData.transform(np.column_stack((x2, y2)))
    figure_xy = fig.transFigure.inverted().transform(display_xy)
    left, bottom = np.min(figure_xy, axis=0)
    right, top = np.max(figure_xy, axis=0)
    return float(left), float(bottom), float(right), float(top)


def draw_surface(
    ax: plt.Axes,
    points: np.ndarray,
    faces: np.ndarray,
    *,
    facecolors,
    edgecolors="none",
    linewidth: float = 0.0,
    alpha: float = 1.0,
) -> Poly3DCollection:
    artist = Poly3DCollection(
        points[faces],
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidth,
        antialiased=False,
        alpha=alpha,
    )
    artist.set_rasterized(True)
    ax.add_collection3d(artist)
    return artist


def node_to_face(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[faces].mean(axis=1)


def stratified_nodes(points: np.ndarray, count: int) -> np.ndarray:
    
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    radial = np.linalg.norm(points[:, :2], axis=1)
    major_r = float(radial.mean())
    phi = np.mod(np.arctan2(points[:, 2], radial - major_r), 2 * np.pi)
    n_theta = max(5, int(round(math.sqrt(count * 2.0))))
    n_phi = max(4, int(math.ceil(count / n_theta)))
    selected: list[int] = []
    for i in range(n_theta):
        for j in range(n_phi):
            target_theta = 2 * np.pi * (i + 0.5) / n_theta
            target_phi = 2 * np.pi * (j + 0.5) / n_phi
            dtheta = np.angle(np.exp(1j * (theta - target_theta)))
            dphi = np.angle(np.exp(1j * (phi - target_phi)))
            selected.append(int(np.argmin(dtheta**2 + 0.55 * dphi**2)))
    return np.unique(selected)[:count]


def torus_chart(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    xy_center = points[:, :2].mean(axis=0)
    z_center = float(points[:, 2].mean())
    xy = points[:, :2] - xy_center
    rho = np.linalg.norm(xy, axis=1)
    major_radius = float(np.mean(rho))
    theta = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * np.pi)
    phi = np.mod(np.arctan2(points[:, 2] - z_center, rho - major_radius), 2.0 * np.pi)
    minor_radius = float(np.median(np.hypot(rho - major_radius, points[:, 2] - z_center)))
    center = np.array([xy_center[0], xy_center[1], z_center])
    return theta, phi, major_radius, minor_radius, center


def chart_to_torus(
    theta: np.ndarray,
    phi: np.ndarray,
    major_radius: float,
    minor_radius: float,
    center: np.ndarray,
) -> np.ndarray:
    radial = major_radius + minor_radius * np.cos(phi)
    return np.column_stack(
        (
            center[0] + radial * np.cos(theta),
            center[1] + radial * np.sin(theta),
            center[2] + minor_radius * np.sin(phi),
        )
    )


def torus_generator_cycle(
    points: np.ndarray,
    *,
    generator: str,
    fixed_angle: float,
    samples: int = 220,
    lift: float = 0.018,
) -> np.ndarray:
    _, _, major_r, minor_r, center = torus_chart(points)
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
    if generator == "alpha":
        theta, phi = t, np.full_like(t, fixed_angle)
    elif generator == "beta":
        theta, phi = np.full_like(t, fixed_angle), t
    else:
        raise ValueError(generator)
    curve = chart_to_torus(theta, phi, major_r, minor_r, center)
    normal = np.column_stack(
        (np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi))
    )
    return curve + lift * normal


def draw_topology_cycles(ax: plt.Axes, points: np.ndarray) -> None:
    alpha = torus_generator_cycle(points, generator="alpha", fixed_angle=0.18 * np.pi, lift=0.022)
    beta = torus_generator_cycle(points, generator="beta", fixed_angle=-0.30 * np.pi, lift=0.022)
    ax.plot(
        alpha[:, 0],
        alpha[:, 1],
        alpha[:, 2],
        color=PAPER_TOPOLOGY,
        linewidth=1.45,
        alpha=0.96,
        solid_capstyle="round",
        zorder=8,
    )
    ax.plot(
        beta[:, 0],
        beta[:, 1],
        beta[:, 2],
        color=PAPER_TOPOLOGY,
        linewidth=1.45,
        alpha=0.96,
        solid_capstyle="round",
        zorder=8,
    )
    ia = int(0.10 * (len(alpha) - 1))
    ib = int(0.36 * (len(beta) - 1))
    ax.text(
        alpha[ia, 0],
        alpha[ia, 1],
        alpha[ia, 2],
        r"$\alpha$",
        color=PAPER_TOPOLOGY,
        fontsize=13.5,
        fontweight="bold",
        zorder=10,
    )
    ax.text(
        beta[ib, 0],
        beta[ib, 1],
        beta[ib, 2],
        r"$\beta$",
        color=PAPER_TOPOLOGY,
        fontsize=13.5,
        fontweight="bold",
        zorder=10,
    )





def draw_initial_panel(
    ax: plt.Axes,
    points: np.ndarray,
    faces: np.ndarray,
    initial: np.ndarray,
    norm: Normalize,
) -> None:
    draw_surface(
        ax,
        points,
        faces,
        facecolors=INITIAL_CMAP(norm(node_to_face(initial, faces))),
        edgecolors=(0.88, 0.93, 0.95, 0.34),
        linewidth=0.070,
        alpha=1.0,
    )
    draw_topology_cycles(ax, points)
    set_torus_view(ax, points)


def draw_c0_panel(
    ax: plt.Axes,
    points: np.ndarray,
    values: np.ndarray,
    norm: Normalize,
) -> None:
    count = min(360, len(points))
    ids = stratified_nodes(points, count)
    p = points[ids]
    ratio = np.clip(norm(values[ids]), 0.0, 1.0)

    radial = np.linalg.norm(points[:, :2], axis=1)
    major_r = float(radial.mean())
    n = np.column_stack(
        (
            (radial - major_r) * points[:, 0] / np.maximum(radial, 1e-12),
            (radial - major_r) * points[:, 1] / np.maximum(radial, 1e-12),
            points[:, 2] - float(points[:, 2].mean()),
        )
    )
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    elev = math.radians(31.5)
    azim = math.radians(-58.0)
    camera_dir = np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ]
    )
    facing = np.clip(0.5 + 0.5 * (n[ids] @ camera_dir), 0.0, 1.0)

    colors = C0_CMAP(ratio)
    colors[:, 3] = 0.72 + 0.28 * facing
    sizes = 13.0 + 6.0 * facing
    ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        s=sizes,
        c=colors,
        edgecolors="none",
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )
    set_torus_view(ax, points)


def draw_c1_panel(
    ax: plt.Axes,
    points: np.ndarray,
    edges: np.ndarray,
    values: np.ndarray,
    signed_edge_norm: TwoSlopeNorm,
) -> tuple[cm.ScalarMappable, TwoSlopeNorm]:
    values = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values)
    bound = max(abs(float(signed_edge_norm.vmin)), abs(float(signed_edge_norm.vmax)))
    mag = np.clip(abs_values / max(bound, 1e-12), 0.0, 1.0)
    colors = SIGNED_CMAP(signed_edge_norm(values))
    colors[:, 3] = 0.52 + 0.46 * mag**0.68
    widths = 0.42 + 0.54 * mag**0.78
    wire = Line3DCollection(
        points[edges],
        colors=colors,
        linewidths=widths,
        antialiased=True,
        capstyle="round",
        joinstyle="round",
    )
    wire.set_rasterized(True)
    ax.add_collection3d(wire)
    set_torus_view(ax, points)
    mapper = cm.ScalarMappable(norm=signed_edge_norm, cmap=SIGNED_CMAP)
    mapper.set_array([])
    return mapper, signed_edge_norm


def draw_c2_panel(
    ax: plt.Axes,
    points: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    norm: Normalize,
) -> None:
    draw_surface(
        ax,
        points,
        faces,
        facecolors=MASS_CMAP(norm(values)),
        edgecolors=(0.86, 0.92, 0.95, 0.52),
        linewidth=0.080,
        alpha=1.0,
    )
    set_torus_view(ax, points)


def draw_error_thumbnail(
    ax: plt.Axes,
    task: str,
    error: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    norm: LogNorm,
) -> None:
    
    safe = np.maximum(np.asarray(error, dtype=np.float64), norm.vmin)
    ratio = np.clip(norm(safe), 0.0, 1.0)

    if task == "0":
        
        colors = ERROR_CMAP(ratio)
        colors[:, 3] = 0.94
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=10.0,
            c=colors,
            edgecolors="none",
            linewidths=0.0,
            depthshade=False,
            rasterized=True,
        )

    elif task == "1":
        
        colors = ERROR_CMAP(ratio)
        colors[:, 3] = 0.94
        wire = Line3DCollection(
            points[edges],
            colors=colors,
            linewidths=0.62,
            antialiased=True,
            capstyle="round",
            joinstyle="round",
        )
        wire.set_rasterized(True)
        ax.add_collection3d(wire)

    elif task == "2":
        
        draw_surface(
            ax,
            points,
            faces,
            facecolors=ERROR_CMAP(ratio),
            edgecolors=(0.86, 0.90, 0.94, 0.24),
            linewidth=0.045,
            alpha=1.0,
        )

    else:
        raise ValueError(task)

    set_torus_view(ax, points)


def error_row(
    fig: plt.Figure,
    image_rect: tuple[float, float, float, float],
    cbar_rect: tuple[float, float, float, float],
    task: str,
    models: list[PredictionRun],
    row: int,
    points: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
) -> list[tuple[plt.Axes, str]]:
    
    if len(models) != 4:
        raise ValueError(f"error_row expects exactly four models, got {len(models)}")

    left, bottom, width, height = image_rect
    errors = [local_relative_error(run.prediction[row], run.target[row]) for run in models]
    norm = error_norm(errors)

    fig_w, fig_h = fig.get_size_inches()
    gap = ERROR_GAP_PX / FIGURE_W_PX
    image_w = min(ERROR_PANEL_PX / FIGURE_W_PX, (width - 3.0 * gap) / 4.0)
    image_h = image_w * (fig_w / fig_h)
    if image_h > height:
        image_h = height
        image_w = image_h * (fig_h / fig_w)

    used_w = 4.0 * image_w + 3.0 * gap
    x0 = left + 0.5 * (width - used_w)
    y0 = bottom + 0.5 * (height - image_h)

    label_specs: list[tuple[plt.Axes, str]] = []
    for idx, (run, error) in enumerate(zip(models, errors)):
        x = x0 + idx * (image_w + gap)
        axis = fig.add_axes([x, y0, image_w, image_h], projection="3d")
        draw_error_thumbnail(axis, task, error, points, faces, edges, norm)
        label_specs.append((axis, run.name))

    row_label = ERROR_ROW_LABELS.get(task, rf"$\mathbf{{C}}^{{{task}}}$")

    fig.text(
        left - 32.0 / FIGURE_W_PX,  
        y0 + 0.5 * image_h,
        row_label,
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=PAPER_INK,
        rotation=90,  
        linespacing=1.15,
    )

    mapper = cm.ScalarMappable(norm=norm, cmap=ERROR_CMAP)
    mapper.set_array([])
    cax = fig.add_axes(cbar_rect)
    cbar = fig.colorbar(mapper, cax=cax, orientation="horizontal")
    ticks, ticklabels = compact_colorbar_ticks(norm, logarithmic=True)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=8.2, pad=1.4, length=2.4, width=0.60)
    cbar.ax.xaxis.set_minor_locator(NullLocator())
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    cbar.outline.set_linewidth(0.62)
    cbar.outline.set_edgecolor("#5C6878")

    cb_left, cb_bottom, cb_width, cb_height = cbar_rect
    fig.text(
        cb_left + 0.5 * cb_width,
        0.5 * (y0 + cb_bottom + cb_height),
        "Relative error",
        ha="center",
        va="center",
        fontsize=8.7,
        color=PAPER_INK,
    )
    return label_specs


def format_scientific_tick(value: float) -> str:
    
    if not np.isfinite(value) or value <= 0:
        return "0"

    if 1e-2 <= value < 100.0:
        return f"{value:.2g}"
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)
    if abs(mantissa - 1.0) < 0.05:
        return rf"$10^{{{exponent}}}$"
    return f"{mantissa:.1g}e{exponent}"


def compact_colorbar_ticks(
    norm: Normalize, *, logarithmic: bool = False
) -> tuple[np.ndarray, list[str]]:
    
    fractions = np.asarray((0.10, 0.50, 0.90))
    if logarithmic:
        ticks = norm.vmin * (norm.vmax / norm.vmin) ** fractions
    else:
        ticks = norm.vmin + (norm.vmax - norm.vmin) * fractions
    return ticks, [format_scientific_tick(float(value)) for value in ticks]


def main_colorbar(
    fig: plt.Figure,
    artist,
    rect: tuple[float, float, float, float],
    *,
    ticks=None,
    ticklabels=None,
) -> None:
    
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=8.4, pad=1.4, length=2.6, width=0.60)
    cbar.ax.xaxis.set_minor_locator(NullLocator())
    cbar.ax.xaxis.get_offset_text().set_visible(False)
    cbar.outline.set_linewidth(0.66)
    cbar.outline.set_edgecolor("#7C8996")


def place_main_field_label(
    fig: plt.Figure,
    ax: plt.Axes,
    points: np.ndarray,
    cbar_rect: tuple[float, float, float, float],
    text: str,
    *,
    panel_offset_px: float = 0.0,
) -> None:
    
    _, geometry_bottom, _, _ = projected_geometry_bbox(fig, ax, points)
    left, bottom, width, height = cbar_rect

    bar_top_px = (bottom + height) * FIGURE_H_PX
    geometry_bottom_px = geometry_bottom * FIGURE_H_PX

    ratio = float(MAIN_FIELD_LABEL_VERTICAL_RATIO)
    label_center_px = (
        bar_top_px
        + ratio * (geometry_bottom_px - bar_top_px)
        + MAIN_FIELD_LABEL_Y_BIAS_PX
        + panel_offset_px
    )

    fig.text(
        left + 0.5 * width,
        label_center_px / FIGURE_H_PX,
        text,
        ha="center",
        va="center",
        fontsize=9.1,
        color=PAPER_INK,
    )


def place_model_name(
    fig: plt.Figure,
    ax: plt.Axes,
    points: np.ndarray,
    name: str,
) -> None:
    
    left, _, right, geometry_top = projected_geometry_bbox(fig, ax, points)
    fig.text(
        0.5 * (left + right),
        geometry_top + MODEL_NAME_IMAGE_GAP_PX / FIGURE_H_PX,
        name,
        ha="center",
        va="bottom",
        fontsize=11.4,
        fontweight="bold",
        color=PAPER_INK,
    )





def render_figure(
    output_dir: Path,
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    trajectories: np.ndarray,
    edges: np.ndarray,
    runs_by_task: dict[str, list[PredictionRun]],
    raw_ids_by_task: dict[str, np.ndarray],
    requested_archive_id: int | None,
) -> Path:
    required = {"0", "1", "2"}
    if set(runs_by_task) != required:
        raise ValueError("runs_by_task must contain exactly '0', '1', '2'.")

    dkho_large = {
        task: next(run for run in runs if run.name == "DKHO-large")
        for task, runs in runs_by_task.items()
    }
    rows, raw_ids, selection_diagnostics = select_display_samples(
        runs_by_task, raw_ids_by_task, requested_archive_id
    )

    rankings = {task: model_ranking(runs) for task, runs in runs_by_task.items()}
    top4 = {task: [run for run, _score in rankings[task][:4]] for task in ("0", "1", "2")}

    
    initial = trajectories[raw_ids["0"], 0]
    c0_pred = dkho_large["0"].prediction[rows["0"]]
    c0_true = dkho_large["0"].target[rows["0"]]
    c1_pred = dkho_large["1"].prediction[rows["1"]]
    c2_pred = dkho_large["2"].prediction[rows["2"]]
    c2_true = dkho_large["2"].target[rows["2"]]

    initial_norm = robust_norm(initial)
    c0_norm = robust_norm(np.concatenate((c0_pred, c0_true)))
    c2_norm = robust_norm(np.concatenate((c2_pred, c2_true)))
    c1_signed_norm = symmetric_norm(c1_pred)


    fig = plt.figure(
        figsize=(FIGURE_W_PX / REFERENCE_DPI, FIGURE_H_PX / REFERENCE_DPI),
        facecolor="#FFFFFF",
    )

    def px_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / FIGURE_W_PX, y / FIGURE_H_PX, w / FIGURE_W_PX, h / FIGURE_H_PX)

    LEFT_BLOCK_W = 2.0 * MAIN_PANEL_PX + LEFT_COL_GAP_PX
    ERROR_TOTAL_W = 4.0 * ERROR_PANEL_PX + 3.0 * ERROR_GAP_PX
    TOTAL_CONTENT_W = LEFT_BLOCK_W + BLOCK_GAP_PX + ERROR_TOTAL_W

    LEFT_X0 = 0.5 * (FIGURE_W_PX - TOTAL_CONTENT_W)
    LEFT_X1 = LEFT_X0 + MAIN_PANEL_PX + LEFT_COL_GAP_PX
    LEFT_BLOCK_RIGHT = LEFT_X0 + LEFT_BLOCK_W
    RIGHT_X0 = LEFT_BLOCK_RIGHT + BLOCK_GAP_PX
    RIGHT_X1 = RIGHT_X0 + ERROR_TOTAL_W

    
    BOTTOM_MAIN_Y = CONTENT_BOTTOM_PX
    TOP_MAIN_Y = CONTENT_TOP_PX - MAIN_PANEL_PX

    A_MAIN = px_rect(LEFT_X0, TOP_MAIN_Y, MAIN_PANEL_PX, MAIN_PANEL_PX)
    B_MAIN = px_rect(LEFT_X1, TOP_MAIN_Y, MAIN_PANEL_PX, MAIN_PANEL_PX)
    C_MAIN = px_rect(LEFT_X0, BOTTOM_MAIN_Y, MAIN_PANEL_PX, MAIN_PANEL_PX)
    D_MAIN = px_rect(LEFT_X1, BOTTOM_MAIN_Y, MAIN_PANEL_PX, MAIN_PANEL_PX)

    cbar_x_offset = 0.5 * (MAIN_PANEL_PX - MAIN_CBAR_W_PX)
    TOP_CBAR_Y = TOP_MAIN_Y - MAIN_CBAR_H_PX - TOP_MAIN_CBAR_GAP_PX
    BOTTOM_CBAR_Y = BOTTOM_MAIN_Y - MAIN_CBAR_H_PX - BOTTOM_MAIN_CBAR_GAP_PX
    A_CBAR = px_rect(LEFT_X0 + cbar_x_offset, TOP_CBAR_Y, MAIN_CBAR_W_PX, MAIN_CBAR_H_PX)
    B_CBAR = px_rect(LEFT_X1 + cbar_x_offset, TOP_CBAR_Y, MAIN_CBAR_W_PX, MAIN_CBAR_H_PX)
    C_CBAR = px_rect(LEFT_X0 + cbar_x_offset, BOTTOM_CBAR_Y, MAIN_CBAR_W_PX, MAIN_CBAR_H_PX)
    D_CBAR = px_rect(LEFT_X1 + cbar_x_offset, BOTTOM_CBAR_Y, MAIN_CBAR_W_PX, MAIN_CBAR_H_PX)

    ax_a = fig.add_axes(A_MAIN, projection="3d")
    ax_b = fig.add_axes(B_MAIN, projection="3d")
    ax_c = fig.add_axes(C_MAIN, projection="3d")
    ax_d = fig.add_axes(D_MAIN, projection="3d")

    
    ERROR_CBAR_X = RIGHT_X0 + 0.5 * (ERROR_TOTAL_W - ERROR_CBAR_W_PX)

    
    
    
    ERROR_ROW_GAP = ((CONTENT_TOP_PX - CONTENT_BOTTOM_PX) - 3.0 * ERROR_PANEL_PX) / 2.0
    if ERROR_ROW_GAP <= 0:
        raise RuntimeError("Error rows do not fit inside the shared content envelope.")

    ERROR_Y = {
        "2": CONTENT_BOTTOM_PX,
        "1": CONTENT_BOTTOM_PX + ERROR_PANEL_PX + ERROR_ROW_GAP,
        "0": CONTENT_TOP_PX - ERROR_PANEL_PX,
    }
    ERROR_CBAR_Y = {task: ERROR_Y[task] - ERROR_CBAR_Y_OFFSET_PX for task in ("0", "1", "2")}

    
    
    LEFT_BLOCK_CENTER_X = 0.5 * (LEFT_X0 + LEFT_BLOCK_RIGHT)
    RIGHT_BLOCK_CENTER_X = 0.5 * (RIGHT_X0 + RIGHT_X1)

    fig.text(
        LEFT_BLOCK_CENTER_X / FIGURE_W_PX,
        SECTION_TITLE_Y_PX / FIGURE_H_PX,
        "Geometric Domain & Physical Fields",
        ha="center",
        va="center",
        fontsize=16.4,
        fontweight="bold",
        color=PAPER_INK,
    )
    fig.text(
        RIGHT_BLOCK_CENTER_X / FIGURE_W_PX,
        SECTION_TITLE_Y_PX / FIGURE_H_PX,
        "Top-4 Relative Spatial Errors",
        ha="center",
        va="center",
        fontsize=16.4,
        fontweight="bold",
        color=PAPER_INK,
    )

    
    title_specs = [
        (A_MAIN, r"(a) Geometry and Initial Condition"),
        (B_MAIN, r"(b) $\mathbf{C}^{0}$ Terminal Concentration"),
        (C_MAIN, r"(c) $\mathbf{C}^{1}$ Oriented Flux"),
        (D_MAIN, r"(d) $\mathbf{C}^{2}$ Face Mass"),
    ]
    for rect, title in title_specs:
        left, bottom, width, height = rect
        fig.text(
            left + 0.5 * width,
            bottom + height + PANEL_TITLE_GAP_PX / FIGURE_H_PX,
            title,
            ha="center",
            va="bottom",
            fontsize=12.4,
            fontweight="bold",
            color=PAPER_INK,
        )

    
    draw_initial_panel(ax_a, points, faces, initial, initial_norm)
    initial_mapper = cm.ScalarMappable(norm=initial_norm, cmap=INITIAL_CMAP)
    initial_mapper.set_array([])
    initial_ticks, initial_labels = compact_colorbar_ticks(initial_norm)
    main_colorbar(
        fig,
        initial_mapper,
        A_CBAR,
        ticks=initial_ticks,
        ticklabels=initial_labels,
    )

    
    draw_c0_panel(ax_b, points, c0_pred, c0_norm)
    c0_mapper = cm.ScalarMappable(norm=c0_norm, cmap=C0_CMAP)
    c0_mapper.set_array([])
    c0_ticks, c0_labels = compact_colorbar_ticks(c0_norm)
    main_colorbar(
        fig,
        c0_mapper,
        B_CBAR,
        ticks=c0_ticks,
        ticklabels=c0_labels,
    )

    
    flux_mapper, flux_norm = draw_c1_panel(ax_c, points, edges, c1_pred, c1_signed_norm)
    flux_ticks = np.asarray([flux_norm.vmin, 0.0, flux_norm.vmax])
    flux_labels = [
        (
            f"-{format_scientific_tick(abs(float(flux_norm.vmin))).replace('$', '')}"
            if flux_norm.vmin < 0
            else format_scientific_tick(float(flux_norm.vmin)).replace("$", "")
        ),
        "0",
        format_scientific_tick(float(flux_norm.vmax)).replace("$", ""),
    ]
    main_colorbar(
        fig,
        flux_mapper,
        C_CBAR,
        ticks=flux_ticks,
        ticklabels=flux_labels,
    )

    
    draw_c2_panel(ax_d, points, faces, c2_pred, c2_norm)
    c2_mapper = cm.ScalarMappable(norm=c2_norm, cmap=MASS_CMAP)
    c2_mapper.set_array([])
    c2_ticks, c2_labels = compact_colorbar_ticks(c2_norm)
    main_colorbar(
        fig,
        c2_mapper,
        D_CBAR,
        ticks=c2_ticks,
        ticklabels=c2_labels,
    )

    model_label_specs: list[tuple[plt.Axes, str]] = []
    for task in ("0", "1", "2"):
        model_label_specs.extend(
            error_row(
                fig,
                px_rect(RIGHT_X0, ERROR_Y[task], ERROR_TOTAL_W, ERROR_PANEL_PX),
                px_rect(ERROR_CBAR_X, ERROR_CBAR_Y[task], ERROR_CBAR_W_PX, ERROR_CBAR_H_PX),
                task,
                top4[task],
                rows[task],
                points,
                faces,
                edges,
            )
        )

    
    
    fig.canvas.draw()
    for panel_key, ax, cbar_rect, text in (
        ("a", ax_a, A_CBAR, r"Initial concentration  $u^0$"),
        ("b", ax_b, B_CBAR, r"Terminal concentration  $u^T$"),
        ("c", ax_c, C_CBAR, r"Oriented edge flux  $q_e$"),
        ("d", ax_d, D_CBAR, r"Face mass  $m^2$"),
    ):
        place_main_field_label(
            fig,
            ax,
            points,
            cbar_rect,
            text,
            panel_offset_px=MAIN_FIELD_LABEL_PANEL_OFFSETS_PX[panel_key],
        )

    for axis, name in model_label_specs:
        place_model_name(fig, axis, points, name)

    
    
    
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "torus_cochain_comparison.png"
    pdf = png.with_suffix(".pdf")

    fig.savefig(
        png,
        dpi=PNG_DPI,
        facecolor="#FFFFFF",
        bbox_inches=None,
        pad_inches=0.0,
    )
    fig.savefig(
        pdf,
        dpi=PDF_DPI,
        facecolor="#FFFFFF",
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(fig)

    metadata = {
        "figure": png.name,
        "palette": "luminous",
        "palette_note": "fixed luminous publication palette",
        "canvas_reference_px": [int(FIGURE_W_PX), int(FIGURE_H_PX)],
        "png_dpi": PNG_DPI,
        "png_output_px": [
            int(FIGURE_W_PX * PNG_DPI / REFERENCE_DPI),
            int(FIGURE_H_PX * PNG_DPI / REFERENCE_DPI),
        ],
        "sample_selection": (
            "C0/C1: best DKHO-large; C2: strong ranking-consistent DKHO-large/MGN sample"
            if requested_archive_id is None
            else "user-requested raw archive id"
        ),
        "selection_diagnostics": selection_diagnostics,
        "test_rows": rows,
        "archive_ids": raw_ids,
        "top4_error_models": {task: [run.name for run in top4[task]] for task in ("0", "1", "2")},
        "ranking_metric": "full-test mean relative-L2; lower is better",
        "layout": (
            f"{int(FIGURE_W_PX)}x{int(FIGURE_H_PX)} compact reference canvas; "
            f"{int(OUTER_MARGIN_PX)}px outer horizontal margins; centered two-block composition with "
            f"{int(BLOCK_GAP_PX)}px middle gap; left 2x2 530px main panels; right 3x4 matrix "
            "of 340px Top-4 relative spatial error maps on native supports; equal error-row gaps; "
            "labels anchored to projected torus bounds rather than 3D axes boxes; semantic labels "
            "remain separated from numeric ticks."
        ),
        "error_field": (
            "|prediction-target| / Q95(|target|); shared per-form robust LogNorm; "
            "C0 stays on vertices, C1 stays on native edges, C2 stays on faces; "
            "no vertex/edge-to-face display projection."
        ),
    }

    (output_dir / "torus_cochain_comparison.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[figure] {png}", flush=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return png





def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "qualitative_comparison",
        help="Output directory.",
    )
    parser.add_argument(
        "--archive-id",
        type=int,
        default=None,
        help=(
            "Optional raw trajectory id. Default: C0/C1 use DKHO-large best samples; "
            "C2 uses a strong ranking-consistent DKHO-large/MGN sample."
        ),
    )
    parser.add_argument(
        "--c0-key",
        default=None,
        help="Override the DKHO-large key in the C0 prediction archive.",
    )
    parser.add_argument(
        "--c0-small-key",
        default=None,
        help="Override the DKHO-small key in the C0 prediction archive.",
    )

    args = parser.parse_args()
    set_publication_style()

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    raw = load_dataset(DATASET)
    trajectories = np.asarray(raw["trajectories"], dtype=np.float64)
    points = np.asarray(raw["points"], dtype=np.float64)
    faces = np.asarray(raw["faces"], dtype=np.int64)
    normals = np.asarray(raw["normals"], dtype=np.float64)
    edges = oriented_edges(faces)

    c0_runs = load_c0_runs(args.c0_key, args.c0_small_key)
    c1_runs = load_form_runs("1")
    c2_runs = load_form_runs("2")
    runs_by_task = {"0": c0_runs, "1": c1_runs, "2": c2_runs}

    scalar_ids = scalar_test_indices(len(trajectories))
    form_ids = form_test_indices(len(trajectories))
    raw_ids_by_task = {"0": scalar_ids, "1": form_ids, "2": form_ids}

    dkho0 = next(run for run in c0_runs if run.name == "DKHO-large")
    dkho1 = next(run for run in c1_runs if run.name == "DKHO-large")
    dkho2 = next(run for run in c2_runs if run.name == "DKHO-large")

    if len(dkho0.target) != len(scalar_ids):
        raise RuntimeError("C0 saved predictions do not match the deterministic test split.")
    if len(dkho1.target) != len(form_ids) or len(dkho2.target) != len(form_ids):
        raise RuntimeError("C1/C2 saved predictions do not match the deterministic test split.")

    c0_target = trajectories[:, -1]
    _, raw_c1, raw_c2 = load_native_targets()

    if not np.allclose(dkho0.target, c0_target[scalar_ids], rtol=2e-5, atol=2e-6):
        raise RuntimeError("C0 frozen targets do not match the archived trajectory split.")
    if not np.allclose(dkho1.target, raw_c1[form_ids], rtol=2e-5, atol=2e-6):
        raise RuntimeError("C1 frozen targets do not match the reconstructed native target.")
    if not np.allclose(dkho2.target, raw_c2[form_ids], rtol=2e-5, atol=2e-7):
        raise RuntimeError("C2 frozen targets do not match the reconstructed native target.")

    render_figure(
        args.output.resolve(),
        points,
        faces,
        normals,
        trajectories,
        edges,
        runs_by_task,
        raw_ids_by_task,
        args.archive_id,
    )


if __name__ == "__main__":
    main()
