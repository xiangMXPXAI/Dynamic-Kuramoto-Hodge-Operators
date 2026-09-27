"""Render qualitative magnetostatics predictions and error diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent

DEFAULT_DATA = ROOT / "data" / "cavity_magnetostatics_v1.pkl"
def default_dkho_run(scale: str) -> Path:
    return ROOT / "runs" / "dkho" / scale / "form_2" / "conditioned" / "full" / "seed_42"


DEFAULT_DKHO_LARGE_RUN = default_dkho_run("large")
DEFAULT_DKHO_SMALL_RUN = default_dkho_run("small")
DEFAULT_BASELINE_LOG = ROOT / "runs" / "baseline" / "legacy" / "experiment_log.pkl"
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "comparison"

EPS = 1.0e-12

@dataclass(frozen=True)
class FigurePalette:
    
    ink: str = "#000000"
    secondary: str = "#000000"
    hairline: str = "#5A5A5A"

    shell: str = "#D6E7ED"
    shell_edge: str = "#89AABA"
    cavity: str = "#FFF7E8"
    cavity_edge: str = "#536C82"

    source_cmap: str = "mag_source_topjournal_v6"
    field_cmap: str = "mag_field_topjournal_v2"
    flux_cmap: str = "mag_flux_topjournal_v2"
    error_cmap: str = "hot"
    outer_geometry_cmap: str = "mag_geometry_outer_v6"
    inner_geometry_cmap: str = "mag_geometry_inner_v6"


PALETTE = FigurePalette()


@dataclass
class ExperimentData:
    nodes: np.ndarray
    elements: np.ndarray
    inner_boundary: np.ndarray
    outer_boundary: np.ndarray
    source: np.ndarray
    target: np.ndarray
    predictions: dict[str, np.ndarray]
    test_indices: np.ndarray
    geometry: dict
    population_relative_l2: dict[str, float]

@dataclass(frozen=True)
class BoundarySurfaces:
    inner_faces: np.ndarray
    outer_faces: np.ndarray
    inner_radius: float
    outer_radius: float


def register_colormaps() -> None:
    
    maps = {
        
        PALETTE.source_cmap: [
            "#23458D",  
            "#4483B0",
            "#72B4C8",
            "#ACDDD8",
            "#F3EFE4",  
            "#E5C37C",
            "#D99562",
            "#CA3A27",
            "#CD3C5E",  
        ],
        
        PALETTE.field_cmap: [
            "#293B73",
            "#315D8E",
            "#2F7FA2",
            "#2D9CA7",
            "#45B6A3",
            "#7FCC98",
            "#B8D878",
            "#E3DB64",
            "#F3C85A",
        ],
        
        PALETTE.flux_cmap: [
            "#315B9A",
            "#5D97C0",
            "#A9D0DE",
            "#F7F2E8",
            "#F5D17E",
            "#ED9A69",
            "#D96863",
            "#B64B60",
        ],
        
        PALETTE.outer_geometry_cmap: [
            "#F0F5F6",
            "#DDEBED",
            "#C5DDE1",
            "#A5C8CF",
            "#7FABB7",
            "#5E8799",
        ],
        PALETTE.inner_geometry_cmap: [
            "#F7F6F1",
            "#E9ECE7",
            "#D5DFDA",
            "#B9CDC6",
            "#93B2AA",
            "#6F938C",
        ],
    }

    for name, colors in maps.items():
        if name not in matplotlib.colormaps:
            matplotlib.colormaps.register(LinearSegmentedColormap.from_list(name, colors, N=256))

def configure_style() -> None:
    register_colormaps()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12.0,
            "text.color": PALETTE.ink,
            "axes.titlecolor": PALETTE.ink,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelcolor": PALETTE.ink,
            "xtick.color": PALETTE.ink,
            "ytick.color": PALETTE.ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

def _require(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, prediction.ndim))
    numerator = np.linalg.norm(prediction - target, axis=axes)
    denominator = np.linalg.norm(target, axis=axes)
    return numerator / np.maximum(denominator, EPS)


def _align_baselines(
    archive: Mapping,
    test_indices: np.ndarray,
    n_total_samples: int,
) -> dict[str, np.ndarray]:
    
    all_indices = np.arange(n_total_samples)
    _, archived_test_order = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
    )

    position = {int(sample_id): row for row, sample_id in enumerate(archived_test_order)}

    try:
        permutation = np.asarray([position[int(sample_id)] for sample_id in test_indices])
    except KeyError as exc:
        raise ValueError("DKHO and baseline test splits are incompatible") from exc

    scale = float(archive["scaling"]["y_scale"])
    allowed = ("HSD", "GNO", "FNO", "MGN", "DeepONet", "GeoFNO")

    return {
        name: np.asarray(
            archive["vector_predictions"][name][permutation],
            dtype=np.float32,
        )
        * scale
        for name in allowed
        if name in archive["vector_predictions"]
    }


def _load_dkho_run(
    run_path: Path,
    model_name: str,
    raw_target_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    _require(run_path / "prediction_test.npy", f"{model_name} prediction array")
    _require(run_path / "target_test.npy", f"{model_name} target array")
    _require(run_path / "test_indices.npy", f"{model_name} test-index array")

    prediction = np.load(run_path / "prediction_test.npy").astype(np.float32)
    target = np.load(run_path / "target_test.npy").astype(np.float32)
    test_indices = np.load(run_path / "test_indices.npy").astype(np.int64)

    if prediction.shape != target.shape:
        raise ValueError(
            f"{model_name} prediction/target shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )
    if len(test_indices) != len(prediction):
        raise ValueError(
            f"{model_name} test-index length {len(test_indices)} does not match "
            f"prediction rows {len(prediction)}"
        )

    expected_target = np.asarray(raw_target_all)[test_indices]
    if not np.array_equal(target, expected_target):
        raise ValueError(f"Saved {model_name} targets do not match dataset test rows")

    return prediction, target, test_indices


def _align_prediction_to_reference(
    prediction: np.ndarray,
    source_indices: np.ndarray,
    reference_indices: np.ndarray,
    *,
    model_name: str,
) -> np.ndarray:
    
    if np.array_equal(source_indices, reference_indices):
        return prediction

    position = {int(sample_id): row for row, sample_id in enumerate(source_indices)}
    missing = [int(sample_id) for sample_id in reference_indices if int(sample_id) not in position]
    if missing:
        raise ValueError(
            f"{model_name} test split is incompatible with DKHO-large; "
            f"missing {len(missing)} canonical test samples (e.g. {missing[:5]})"
        )

    permutation = np.asarray(
        [position[int(sample_id)] for sample_id in reference_indices],
        dtype=np.int64,
    )
    return prediction[permutation]


def load_experiment(
    dataset_path: Path,
    dkho_large_run: Path,
    dkho_small_run: Path,
    baseline_log: Path,
) -> ExperimentData:
    
    _require(dataset_path, "magnetostatics dataset")
    _require(baseline_log, "baseline experiment archive")

    with dataset_path.open("rb") as handle:
        raw = pickle.load(handle)

    raw_target_all = np.asarray(raw["Y_data"])

    large_pred, large_target, large_indices = _load_dkho_run(
        dkho_large_run,
        "DKHO-large",
        raw_target_all,
    )
    small_pred_raw, small_target_raw, small_indices = _load_dkho_run(
        dkho_small_run,
        "DKHO-small",
        raw_target_all,
    )

    
    
    small_pred = _align_prediction_to_reference(
        small_pred_raw,
        small_indices,
        large_indices,
        model_name="DKHO-small",
    )
    small_target = _align_prediction_to_reference(
        small_target_raw,
        small_indices,
        large_indices,
        model_name="DKHO-small target",
    )
    if not np.array_equal(small_target, large_target):
        raise ValueError("Aligned DKHO-small targets do not match DKHO-large targets")

    with baseline_log.open("rb") as handle:
        archive = pickle.load(handle)
    baselines = _align_baselines(
        archive,
        large_indices,
        len(raw_target_all),
    )
    del archive
    gc.collect()

    predictions = {
        "DKHO-large": large_pred,
        "DKHO-small": small_pred,
        **baselines,
    }
    population = {
        name: float(np.mean(_relative_l2(values, large_target)))
        for name, values in predictions.items()
    }

    geometry = dict(raw.get("config", {}).get("geometry", {}))
    geometry.setdefault("sphere_center", (0.0, 0.0, 0.0))
    geometry.setdefault("sphere_radius", 0.4)
    geometry.setdefault("domain_radius", 2.0)

    return ExperimentData(
        nodes=np.asarray(raw["nodes"], dtype=np.float64),
        elements=np.asarray(raw["elements"], dtype=np.int64),
        inner_boundary=np.asarray(raw["inner_boundary"], dtype=np.int64),
        outer_boundary=np.asarray(raw["outer_boundary"], dtype=np.int64),
        source=np.asarray(raw["X_data"], dtype=np.float32)[large_indices],
        target=large_target,
        predictions=predictions,
        test_indices=large_indices,
        geometry=geometry,
        population_relative_l2=population,
    )


def extract_boundary_surfaces(
    data: ExperimentData,
) -> BoundarySurfaces:
    
    tets = data.elements
    faces = np.concatenate(
        (
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ),
        axis=0,
    )
    faces = np.sort(faces, axis=1)

    unique_faces, counts = np.unique(
        faces,
        axis=0,
        return_counts=True,
    )
    boundary = unique_faces[counts == 1]
    if len(boundary) == 0:
        raise ValueError("No tetrahedral boundary faces were found")

    center = np.asarray(
        data.geometry["sphere_center"],
        dtype=float,
    )
    radius_nodes = np.linalg.norm(
        data.nodes - center,
        axis=1,
    )

    inner_radius = float(np.median(radius_nodes[data.inner_boundary]))
    outer_radius = float(np.median(radius_nodes[data.outer_boundary]))

    face_radius = np.linalg.norm(
        data.nodes[boundary].mean(axis=1) - center,
        axis=1,
    )
    is_inner = np.abs(face_radius - inner_radius) < np.abs(face_radius - outer_radius)

    inner_faces = boundary[is_inner]
    outer_faces = boundary[~is_inner]

    if len(inner_faces) == 0 or len(outer_faces) == 0:
        raise ValueError("Could not separate inner and outer spherical boundaries")

    return BoundarySurfaces(
        inner_faces=inner_faces,
        outer_faces=outer_faces,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
    )


def choose_sample(
    data: ExperimentData,
    mode: str,
    explicit_index: int | None,
) -> int:
    
    rel_large = _relative_l2(
        data.predictions["DKHO-large"],
        data.target,
    )

    if explicit_index is not None:
        if not 0 <= explicit_index < len(rel_large):
            raise IndexError(f"sample-index must be in [0, {len(rel_large) - 1}]")
        return int(explicit_index)

    if mode == "representative":
        
        
        
        ranked_all = sorted(
            data.predictions.keys(),
            key=lambda name: data.population_relative_l2[name],
        )
        required = [name for name in ("DKHO-large", "DKHO-small") if name in data.predictions]
        remaining = [name for name in ranked_all if name not in required]
        models = required + remaining[: max(0, 6 - len(required))]
        models = sorted(
            models[:6],
            key=lambda name: data.population_relative_l2[name],
        )

        error_matrix = np.stack(
            [_relative_l2(data.predictions[name], data.target) for name in models],
            axis=0,
        )
        log_error = np.log(np.maximum(error_matrix, EPS))
        center = np.median(log_error, axis=1, keepdims=True)
        q25 = np.quantile(log_error, 0.25, axis=1, keepdims=True)
        q75 = np.quantile(log_error, 0.75, axis=1, keepdims=True)
        spread = np.maximum(q75 - q25, 0.15)
        robust_distance = np.mean(
            np.abs(log_error - center) / spread,
            axis=0,
        )
        return int(np.argmin(robust_distance))

    order = np.argsort(rel_large)
    if mode == "best":
        return int(order[0])
    if mode == "median":
        return int(order[len(order) // 2])
    if mode == "worst":
        return int(order[-1])

    raise ValueError(mode)





def _robust_symmetric_norm(
    values: Iterable[np.ndarray],
    quantile: float = 0.99,
) -> TwoSlopeNorm:
    array = np.concatenate([np.ravel(value) for value in values])
    finite = array[np.isfinite(array)]
    limit = float(np.quantile(np.abs(finite), quantile))
    limit = max(limit, 1.0e-8)

    return TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )


def _robust_positive_norm(
    values: Iterable[np.ndarray],
    quantile: float = 0.992,
) -> Normalize:
    
    array = np.concatenate([np.ravel(value) for value in values])
    finite = array[np.isfinite(array)]
    vmax = float(np.quantile(finite, quantile))
    return PowerNorm(
        gamma=0.72,
        vmin=0.0,
        vmax=max(vmax, 1.0e-8),
        clip=True,
    )


def _set_3d_axis(
    ax,
    radius: float,
    *,
    elev: float = 21.0,
    azim: float = -54.0,
    zoom: float = 1.08,
    box_aspect: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)

    try:
        ax.set_box_aspect(
            box_aspect,
            zoom=zoom,
        )
    except TypeError:
        ax.set_box_aspect(box_aspect)

    ax.set_axis_off()
    ax.patch.set_alpha(0.0)


def _add_poly_surface(
    ax,
    nodes: np.ndarray,
    faces: np.ndarray,
    *,
    facecolor,
    edgecolor,
    alpha: float,
    linewidth: float,
    zorder: float = 1.0,
) -> Poly3DCollection:
    poly = Poly3DCollection(
        nodes[faces],
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        antialiased=True,
        zorder=zorder,
    )
    ax.add_collection3d(poly)
    return poly


def _cutaway_faces(
    nodes: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    
    center = nodes[faces].mean(axis=1)
    wedge = (center[:, 0] > 0.0) & (center[:, 1] < 0.15)
    return faces[~wedge]


def _build_scalar_interpolator(
    nodes: np.ndarray,
    values: np.ndarray,
):
    linear = LinearNDInterpolator(
        nodes,
        values,
        fill_value=np.nan,
    )
    nearest = NearestNDInterpolator(
        nodes,
        values,
    )

    def evaluate(
        points: np.ndarray,
    ) -> np.ndarray:
        result = np.asarray(
            linear(points),
            dtype=float,
        )
        missing = ~np.isfinite(result)
        if np.any(missing):
            result[missing] = nearest(points[missing])
        return result

    return evaluate


def _plot_scalar_plane(
    ax,
    evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    plane: str,
    radius_inner: float,
    radius_outer: float,
    norm: Normalize,
    cmap: str,
    alpha: float,
    grid_size: int = 88,
    zorder: float = 2.0,
) -> None:
    
    span = np.linspace(
        -radius_outer,
        radius_outer,
        grid_size,
    )
    first, second = np.meshgrid(
        span,
        span,
    )
    zero = np.zeros_like(first)

    if plane == "x":
        x, y, z = zero, first, second
    elif plane == "y":
        x, y, z = first, zero, second
    elif plane == "z":
        x, y, z = first, second, zero
    else:
        raise ValueError(plane)

    query = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    values = evaluator(query).reshape(x.shape)

    radius = np.sqrt(x * x + y * y + z * z)
    visible = (radius >= radius_inner) & (radius <= radius_outer)

    rgba = matplotlib.colormaps[cmap](norm(values))
    rgba[..., 3] = np.where(
        visible,
        alpha,
        0.0,
    )

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=rgba,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        zorder=zorder,
    )


def _draw_shell_context(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    *,
    outer_alpha: float,
    inner_alpha: float = 0.98,
) -> None:
    
    outer = _cutaway_faces(
        data.nodes,
        surfaces.outer_faces,
    )

    _add_poly_surface(
        ax,
        data.nodes,
        outer,
        facecolor=PALETTE.shell,
        edgecolor=PALETTE.shell_edge,
        alpha=outer_alpha,
        linewidth=0.055,
        zorder=1,
    )

    _add_poly_surface(
        ax,
        data.nodes,
        surfaces.inner_faces,
        facecolor=PALETTE.cavity,
        edgecolor=PALETTE.cavity_edge,
        alpha=inner_alpha,
        linewidth=0.085,
        zorder=5,
    )


def _select_glyph_nodes(
    nodes: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    *,
    per_shell: int = 18,
) -> np.ndarray:
    
    directions = _fibonacci_directions(per_shell)
    radii = np.asarray(
        [
            max(inner_radius * 1.38, inner_radius + 0.08),
            0.56 * outer_radius,
            0.79 * outer_radius,
        ],
        dtype=float,
    )

    queries = np.concatenate(
        [directions * radius for radius in radii],
        axis=0,
    )

    tree = cKDTree(nodes)
    _, ids = tree.query(queries, k=1)
    return np.unique(np.asarray(ids, dtype=np.int64))


def _plot_vector_glyphs(
    ax,
    nodes: np.ndarray,
    vector: np.ndarray,
    ids: np.ndarray,
    norm: Normalize,
    *,
    length_min: float,
    length_max: float,
    linewidth: float,
    alpha: float = 0.97,
) -> None:
    
    values = vector[ids]
    magnitude = np.linalg.norm(values, axis=1)
    direction = values / np.maximum(magnitude[:, None], EPS)

    ratio = np.clip(norm(magnitude), 0.0, 1.0)
    length = length_min + (length_max - length_min) * ratio**0.80
    arrows = direction * length[:, None]

    colors = matplotlib.colormaps[PALETTE.field_cmap](0.03 + 0.97 * ratio)

    ax.quiver(
        nodes[ids, 0],
        nodes[ids, 1],
        nodes[ids, 2],
        arrows[:, 0],
        arrows[:, 1],
        arrows[:, 2],
        color=colors,
        linewidth=linewidth,
        arrow_length_ratio=0.30,
        normalize=False,
        pivot="middle",
        alpha=alpha,
    )


def _plot_source_cloud(
    ax,
    nodes: np.ndarray,
    source: np.ndarray,
    source_norm: Normalize,
    *,
    max_points: int = 360,
) -> None:
    
    finite = np.isfinite(source)
    strength = np.abs(source)
    threshold = float(np.quantile(strength[finite], 0.60))
    ids = np.flatnonzero(finite & (strength >= threshold))

    if len(ids) > max_points:
        order = np.argsort(strength[ids])[::-1]
        
        
        pick = np.linspace(0, len(order) - 1, max_points).astype(int)
        ids = ids[order[pick]]

    ratio = np.clip(source_norm(source[ids]), 0.0, 1.0)
    sizes = 3.0 + 10.0 * np.clip(
        strength[ids] / max(float(np.quantile(strength[finite], 0.98)), EPS),
        0.0,
        1.0,
    )

    ax.scatter(
        nodes[ids, 0],
        nodes[ids, 1],
        nodes[ids, 2],
        s=sizes,
        c=matplotlib.colormaps[PALETTE.source_cmap](ratio),
        edgecolors="none",
        depthshade=False,
        alpha=0.24,
        rasterized=True,
        zorder=4,
    )


def _orientation_facecolors(
    nodes: np.ndarray,
    faces: np.ndarray,
    cmap_name: str,
    *,
    center: np.ndarray,
    alpha: float,
    light_direction: np.ndarray = np.array([0.55, -0.35, 0.76]),
) -> np.ndarray:
    
    triangles = nodes[faces]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    normal = np.cross(e1, e2)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), EPS)

    centroid = triangles.mean(axis=1) - center[None, :]
    
    flip = np.sum(normal * centroid, axis=1) < 0.0
    normal[flip] *= -1.0

    light = np.asarray(light_direction, dtype=float)
    light /= max(float(np.linalg.norm(light)), EPS)

    
    value = 0.15 + 0.85 * (0.5 + 0.5 * (normal @ light))
    rgba = matplotlib.colormaps[cmap_name](np.clip(value, 0.0, 1.0))
    rgba[:, 3] = alpha
    return rgba


def _plot_spherical_guide(
    ax,
    center: np.ndarray,
    radius: float,
    *,
    plane: str,
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
    zorder: float = 8.0,
) -> None:
    
    t = np.linspace(0.0, 2.0 * np.pi, 360)
    c, s = np.cos(t), np.sin(t)

    if plane == "xy":
        xyz = np.column_stack([radius * c, radius * s, np.zeros_like(t)])
    elif plane == "xz":
        xyz = np.column_stack([radius * c, np.zeros_like(t), radius * s])
    elif plane == "yz":
        xyz = np.column_stack([np.zeros_like(t), radius * c, radius * s])
    else:
        raise ValueError(plane)

    xyz = xyz + center[None, :]
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
    )


def _draw_geometry_topology_shell(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
) -> None:
    
    center = np.asarray(data.geometry["sphere_center"], dtype=float)

    
    outer = _cutaway_faces(data.nodes, surfaces.outer_faces)
    outer_rgba = _orientation_facecolors(
        data.nodes,
        outer,
        PALETTE.outer_geometry_cmap,
        center=center,
        alpha=0.19,
        light_direction=np.array([0.40, -0.28, 0.87]),
    )
    outer_poly = Poly3DCollection(
        data.nodes[outer],
        facecolors=outer_rgba,
        edgecolors=(0.30, 0.49, 0.58, 0.16),
        linewidths=0.055,
        antialiased=True,
        zorder=1,
    )
    outer_poly.set_rasterized(True)
    ax.add_collection3d(outer_poly)

    
    inner_rgba = _orientation_facecolors(
        data.nodes,
        surfaces.inner_faces,
        PALETTE.inner_geometry_cmap,
        center=center,
        alpha=0.96,
        light_direction=np.array([-0.32, 0.46, 0.83]),
    )
    inner_poly = Poly3DCollection(
        data.nodes[surfaces.inner_faces],
        facecolors=inner_rgba,
        edgecolors=(0.20, 0.39, 0.42, 0.30),
        linewidths=0.075,
        antialiased=True,
        zorder=7,
    )
    inner_poly.set_rasterized(True)
    ax.add_collection3d(inner_poly)

    
    
    for plane in ("xy", "xz"):
        _plot_spherical_guide(
            ax,
            center,
            surfaces.outer_radius * 1.002,
            plane=plane,
            color="#527D8C",
            linewidth=0.68,
            alpha=0.38,
            zorder=4,
        )


def _draw_cavity_topology_marker(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
) -> None:
    
    center = np.asarray(data.geometry["sphere_center"], dtype=float)
    radius = surfaces.inner_radius * 1.045

    
    
    accent = "#A0139E"
    accent_soft = "#A2249E"
    text_dark = "#AE3FAE"

    
    
    _plot_spherical_guide(
        ax,
        center,
        radius,
        plane="xy",
        color=accent,
        linewidth=1.42,
        alpha=0.98,
        zorder=12,
    )
    _plot_spherical_guide(
        ax,
        center,
        radius,
        plane="xz",
        color=accent_soft,
        linewidth=0.96,
        alpha=0.78,
        linestyle="--",
        zorder=11,
    )

    
    p = center + np.array([0.86 * radius, -0.22 * radius, 0.20 * radius])
    n_in = -p / max(float(np.linalg.norm(p)), EPS)
    ax.quiver(
        p[0],
        p[1],
        p[2],
        n_in[0],
        n_in[1],
        n_in[2],
        length=0.22 * surfaces.outer_radius,
        normalize=True,
        arrow_length_ratio=0.30,
        linewidth=1.35,
        color=accent,
        alpha=0.98,
        pivot="tail",
    )

    
    
    ax.annotate(
        r"$\Gamma_{\rm in}\simeq S^2$",
        xy=(0.49, 0.42),
        xytext=(0.19, 0.29),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=11.0,
        fontweight="semibold",
        color=accent,
        arrowprops={
            "arrowstyle": "-",
            "color": accent,
            "linewidth": 0.92,
            "shrinkA": 0.0,
            "shrinkB": 1.5,
        },
        zorder=30,
    )

    ax.text2D(
        0.565,
        0.445,
        r"$\mathbf{n}_{\rm in}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color=accent,
        zorder=31,
    )





def plot_geometry_and_source(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    source: np.ndarray,
    source_norm: Normalize,
) -> None:
    
    _draw_geometry_topology_shell(ax, data, surfaces)

    evaluator = _build_scalar_interpolator(data.nodes, source)

    
    _plot_scalar_plane(
        ax,
        evaluator,
        plane="y",
        radius_inner=surfaces.inner_radius * 1.015,
        radius_outer=surfaces.outer_radius * 0.965,
        norm=source_norm,
        cmap=PALETTE.source_cmap,
        alpha=0.62,
        grid_size=118,
        zorder=3,
    )

    
    _plot_scalar_plane(
        ax,
        evaluator,
        plane="z",
        radius_inner=surfaces.inner_radius * 1.03,
        radius_outer=surfaces.outer_radius * 0.90,
        norm=source_norm,
        cmap=PALETTE.source_cmap,
        alpha=0.045,
        grid_size=78,
        zorder=2,
    )

    _plot_source_cloud(
        ax,
        data.nodes,
        source,
        source_norm,
        max_points=190,
    )

    _draw_cavity_topology_marker(ax, data, surfaces)

    _set_3d_axis(
        ax,
        surfaces.outer_radius * 1.01,
        elev=20.0,
        azim=-49.0,
        zoom=1.31,
    )

    
    
    outer_color = "#B03FA6"
    ax.annotate(
        r"$\Gamma_{\rm out}$",
        xy=(0.78, 0.66),
        xytext=(0.77, 0.76),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="semibold",
        color=outer_color,
        arrowprops={
            "arrowstyle": "-",
            "color": outer_color,
            "linewidth": 0.86,
            "shrinkA": 0.0,
            "shrinkB": 1.5,
        },
        zorder=30,
    )





def _build_vector_interpolator(
    nodes: np.ndarray,
    vector: np.ndarray,
):
    linear = LinearNDInterpolator(
        nodes,
        vector,
        fill_value=np.nan,
    )

    def evaluate(
        point: np.ndarray,
    ) -> np.ndarray | None:
        value = np.asarray(linear(np.asarray(point)[None, :]))[0]
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return None
        return value

    return evaluate


def _unit_field(
    evaluator,
    point: np.ndarray,
    sign: float,
) -> np.ndarray | None:
    value = evaluator(point)
    if value is None:
        return None

    magnitude = float(np.linalg.norm(value))
    if magnitude < 1.0e-9:
        return None

    return sign * value / magnitude


def _rk4_step(
    evaluator,
    point: np.ndarray,
    step: float,
    sign: float,
) -> np.ndarray | None:
    k1 = _unit_field(
        evaluator,
        point,
        sign,
    )
    if k1 is None:
        return None

    k2 = _unit_field(
        evaluator,
        point + 0.5 * step * k1,
        sign,
    )
    if k2 is None:
        return None

    k3 = _unit_field(
        evaluator,
        point + 0.5 * step * k2,
        sign,
    )
    if k3 is None:
        return None

    k4 = _unit_field(
        evaluator,
        point + step * k3,
        sign,
    )
    if k4 is None:
        return None

    return point + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _integrate_half_line(
    evaluator,
    seed: np.ndarray,
    sign: float,
    inner_radius: float,
    outer_radius: float,
    *,
    step: float = 0.032,
    max_steps: int = 220,
) -> np.ndarray:
    points = [np.asarray(seed, dtype=float)]

    for _ in range(max_steps):
        candidate = _rk4_step(
            evaluator,
            points[-1],
            step,
            sign,
        )
        if candidate is None:
            break

        radius = float(np.linalg.norm(candidate))

        if radius <= inner_radius * 1.004 or radius >= outer_radius * 1.004:
            break

        if np.linalg.norm(candidate - points[-1]) < 1.0e-7:
            break

        points.append(candidate)

    return np.asarray(points)


def _fibonacci_directions(
    n: int,
) -> np.ndarray:
    index = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n
    radius = np.sqrt(
        np.maximum(
            1.0 - z * z,
            0.0,
        )
    )
    phi = index * np.pi * (3.0 - np.sqrt(5.0))

    return np.column_stack(
        (
            radius * np.cos(phi),
            radius * np.sin(phi),
            z,
        )
    )


def trace_field_lines(
    nodes: np.ndarray,
    vector: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    n_lines: int,
) -> list[np.ndarray]:
    
    evaluator = _build_vector_interpolator(
        nodes,
        vector,
    )

    directions = _fibonacci_directions(n_lines)

    
    
    seed_bands = np.asarray(
        [
            1.16 * inner_radius,
            0.52 * outer_radius,
            0.72 * outer_radius,
            0.86 * outer_radius,
        ]
    )
    seed_radii = seed_bands[np.arange(n_lines) % len(seed_bands)]
    seeds = directions * seed_radii[:, None]

    lines: list[np.ndarray] = []

    for seed in seeds:
        backward = _integrate_half_line(
            evaluator,
            seed,
            -1.0,
            inner_radius,
            outer_radius,
        )
        forward = _integrate_half_line(
            evaluator,
            seed,
            +1.0,
            inner_radius,
            outer_radius,
        )

        line = np.concatenate(
            (
                backward[::-1],
                forward[1:],
            ),
            axis=0,
        )

        if len(line) >= 6:
            lines.append(line)

    return lines


def _line_segment_magnitudes(
    lines: list[np.ndarray],
    nodes: np.ndarray,
    magnitude: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    evaluator = _build_scalar_interpolator(
        nodes,
        magnitude,
    )

    all_segments: list[np.ndarray] = []
    all_values: list[np.ndarray] = []

    for line in lines:
        segments = np.stack(
            (
                line[:-1],
                line[1:],
            ),
            axis=1,
        )
        midpoint = segments.mean(axis=1)

        all_segments.extend(segments)
        all_values.append(evaluator(midpoint))

    if not all_segments:
        return [], np.empty(0)

    return (
        all_segments,
        np.concatenate(all_values),
    )


def plot_flux_field(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    vector: np.ndarray,
    field_norm: Normalize,
    *,
    n_lines: int,
) -> None:
    
    _draw_shell_context(
        ax,
        data,
        surfaces,
        outer_alpha=0.105,
        inner_alpha=0.995,
    )

    magnitude = np.linalg.norm(vector, axis=1)
    mag_eval = _build_scalar_interpolator(
        data.nodes,
        magnitude,
    )

    _plot_scalar_plane(
        ax,
        mag_eval,
        plane="z",
        radius_inner=surfaces.inner_radius * 1.01,
        radius_outer=surfaces.outer_radius * 0.985,
        norm=field_norm,
        cmap=PALETTE.field_cmap,
        alpha=0.23,
        grid_size=100,
        zorder=2,
    )

    
    
    lines = trace_field_lines(
        data.nodes,
        vector,
        surfaces.inner_radius,
        surfaces.outer_radius,
        min(max(12, n_lines // 2), 18),
    )

    segments, values = _line_segment_magnitudes(
        lines,
        data.nodes,
        magnitude,
    )

    if segments:
        scaled = np.clip(field_norm(values), 0.0, 1.0)
        widths = 0.58 + 1.10 * scaled**0.80

        halo = Line3DCollection(
            segments,
            colors=(1.0, 1.0, 1.0, 0.60),
            linewidths=widths + 0.74,
            antialiased=True,
            zorder=5,
        )
        halo.set_rasterized(True)
        ax.add_collection3d(halo)

        collection = Line3DCollection(
            segments,
            cmap=PALETTE.field_cmap,
            norm=field_norm,
            linewidths=widths,
            alpha=0.92,
            zorder=6,
        )
        collection.set_array(values)
        collection.set_rasterized(True)
        ax.add_collection3d(collection)

    glyph_ids = _select_glyph_nodes(
        data.nodes,
        surfaces.inner_radius,
        surfaces.outer_radius,
        per_shell=18,
    )
    _plot_vector_glyphs(
        ax,
        data.nodes,
        vector,
        glyph_ids,
        field_norm,
        length_min=0.11,
        length_max=0.31,
        linewidth=0.76,
        alpha=0.96,
    )

    _set_3d_axis(
        ax,
        surfaces.outer_radius * 1.01,
        elev=20.0,
        azim=-49.0,
        zoom=1.30,
    )





def select_comparison_models(
    data: ExperimentData,
    count: int = 6,
) -> list[str]:
    
    ranked_all = sorted(
        data.predictions.keys(),
        key=lambda name: data.population_relative_l2[name],
    )
    required = [name for name in ("DKHO-large", "DKHO-small") if name in data.predictions]
    remaining = [name for name in ranked_all if name not in required]
    selected = required + remaining[: max(0, count - len(required))]
    selected = sorted(
        selected[:count],
        key=lambda name: data.population_relative_l2[name],
    )
    return selected


def _select_comparison_glyph_nodes(
    nodes: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    inner_radius: float,
    outer_radius: float,
    *,
    n_focus: int = 34,
    context_per_shell: int = 9,
) -> np.ndarray:
    
    context_ids = _select_glyph_nodes(
        nodes,
        inner_radius,
        outer_radius,
        per_shell=context_per_shell,
    )

    model_stack = np.stack(
        [np.asarray(predictions[name], dtype=float) for name in predictions],
        axis=0,
    )
    ensemble_mean = np.mean(model_stack, axis=0)
    disagreement = np.sqrt(
        np.mean(
            np.sum((model_stack - ensemble_mean[None, :, :]) ** 2, axis=-1),
            axis=0,
        )
    )

    radius = np.linalg.norm(nodes, axis=1)
    valid = (radius > inner_radius * 1.04) & (radius < outer_radius * 0.985)
    candidate = np.flatnonzero(valid)
    if len(candidate) == 0:
        return np.asarray(context_ids, dtype=np.int64)

    
    
    order = candidate[np.argsort(disagreement[candidate])[::-1]]
    min_sep = 0.16 * outer_radius
    focus: list[int] = []
    for idx in order:
        if not np.isfinite(disagreement[idx]):
            continue
        if not focus:
            focus.append(int(idx))
        else:
            d = np.linalg.norm(nodes[np.asarray(focus)] - nodes[idx], axis=1)
            if np.all(d >= min_sep):
                focus.append(int(idx))
        if len(focus) >= n_focus:
            break

    
    
    if len(focus) < n_focus:
        existing = set(focus)
        for idx in order:
            idx = int(idx)
            if idx not in existing:
                focus.append(idx)
                existing.add(idx)
            if len(focus) >= n_focus:
                break

    return np.unique(
        np.concatenate(
            [
                np.asarray(context_ids, dtype=np.int64),
                np.asarray(focus, dtype=np.int64),
            ]
        )
    )


def _prediction_pairwise_diagnostics(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, float]:
    
    names = list(predictions)
    result: dict[str, float] = {}
    for i, name_i in enumerate(names):
        a = np.asarray(predictions[name_i], dtype=float)
        for name_j in names[i + 1 :]:
            b = np.asarray(predictions[name_j], dtype=float)
            denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), EPS)
            result[f"{name_i} vs {name_j}"] = float(np.linalg.norm(a - b) / denom)
    return result


def _build_slice_grid(
    surfaces: BoundarySurfaces,
    *,
    grid_size: int = 132,
    inner_scale: float = 1.02,
    outer_scale: float = 0.995,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    span = np.linspace(-surfaces.outer_radius, surfaces.outer_radius, grid_size)
    x, y = np.meshgrid(span, span)
    query = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    radius = np.hypot(x, y)
    visible = (radius >= surfaces.inner_radius * inner_scale) & (
        radius <= surfaces.outer_radius * outer_scale
    )
    return x, y, query, visible


def _interpolate_slice(
    nodes: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    
    linear = LinearNDInterpolator(nodes, values, fill_value=np.nan)
    nearest = NearestNDInterpolator(nodes, values)

    result = np.asarray(linear(query), dtype=float)
    missing = ~np.isfinite(result)
    if np.any(missing):
        result[missing] = nearest(query[missing])

    return result.reshape(shape)


def _draw_annular_slice_frame(
    ax,
    surfaces: BoundarySurfaces,
    *,
    highlight: bool = False,
) -> None:
    
    outer = plt.Circle(
        (0.0, 0.0),
        surfaces.outer_radius,
        fill=False,
        edgecolor="#8FA6B5",
        linewidth=0.48,
        alpha=0.88,
    )
    inner = plt.Circle(
        (0.0, 0.0),
        surfaces.inner_radius,
        facecolor="white",
        edgecolor="#6B7B89",
        linewidth=0.62,
        alpha=1.0,
        zorder=8,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    lim = surfaces.outer_radius * 1.055
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_prediction_glyph_3d(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    vector: np.ndarray,
    truth_vector: np.ndarray,
    magnitude_bias: np.ndarray,
    bias_norm: TwoSlopeNorm,
    glyph_ids: np.ndarray,
    field_scale: float,
    *,
    model_name: str,
) -> None:
    
    _draw_shell_context(
        ax,
        data,
        surfaces,
        outer_alpha=0.018,
        inner_alpha=0.64,
    )

    pred = np.asarray(vector[glyph_ids], dtype=float)
    truth = np.asarray(truth_vector[glyph_ids], dtype=float)

    pred_mag = np.linalg.norm(pred, axis=1)
    true_mag = np.linalg.norm(truth, axis=1)
    pred_dir = pred / np.maximum(pred_mag[:, None], EPS)
    true_dir = truth / np.maximum(true_mag[:, None], EPS)

    
    
    pred_ratio = np.clip(pred_mag / max(field_scale, EPS), 0.0, 1.18)
    true_ratio = np.clip(true_mag / max(field_scale, EPS), 0.0, 1.18)
    pred_length = 0.50 * pred_ratio
    true_length = 0.50 * true_ratio

    
    
    
    gt_arrows = true_dir * true_length[:, None]
    ax.quiver(
        data.nodes[glyph_ids, 0],
        data.nodes[glyph_ids, 1],
        data.nodes[glyph_ids, 2],
        gt_arrows[:, 0],
        gt_arrows[:, 1],
        gt_arrows[:, 2],
        color="#66717D",
        linewidth=1.10,
        arrow_length_ratio=0.28,
        normalize=False,
        pivot="middle",
        alpha=0.30,
        zorder=3,
    )

    bias = np.asarray(magnitude_bias[glyph_ids], dtype=float)
    bias_ratio = np.clip(bias_norm(bias), 0.0, 1.0)
    colors = matplotlib.colormaps[PALETTE.source_cmap](bias_ratio)
    colors[:, 3] = 0.94

    pred_arrows = pred_dir * pred_length[:, None]
    ax.quiver(
        data.nodes[glyph_ids, 0],
        data.nodes[glyph_ids, 1],
        data.nodes[glyph_ids, 2],
        pred_arrows[:, 0],
        pred_arrows[:, 1],
        pred_arrows[:, 2],
        color=colors,
        linewidth=0.92,
        arrow_length_ratio=0.30,
        normalize=False,
        pivot="middle",
        alpha=0.98,
        zorder=5,
    )

    
    
    bias_strength = np.abs(bias) / max(
        abs(float(bias_norm.vmin)),
        abs(float(bias_norm.vmax)),
        EPS,
    )
    ax.scatter(
        data.nodes[glyph_ids, 0],
        data.nodes[glyph_ids, 1],
        data.nodes[glyph_ids, 2],
        s=3.2 + 8.0 * np.clip(bias_strength, 0.0, 1.0) ** 0.75,
        c=colors,
        edgecolors="none",
        alpha=0.72,
        depthshade=False,
        rasterized=True,
        zorder=6,
    )

    _set_3d_axis(
        ax,
        surfaces.outer_radius * 1.005,
        elev=20.0,
        azim=-49.0,
        zoom=1.38,
    )

    ax.set_title(
        model_name,
        fontsize=11.4,
        fontweight="bold",
        color=PALETTE.ink,
        pad=0.8,
    )


def plot_residual_glyph_3d(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    residual_vector: np.ndarray,
    error_nodes: np.ndarray,
    error_norm: PowerNorm,
    glyph_ids: np.ndarray,
    *,
    model_name: str,
) -> None:
    
    _draw_shell_context(
        ax,
        data,
        surfaces,
        outer_alpha=0.016,
        inner_alpha=0.58,
    )

    
    

    residual = np.asarray(residual_vector[glyph_ids], dtype=float)
    magnitude = np.linalg.norm(residual, axis=1)
    direction = residual / np.maximum(magnitude[:, None], EPS)

    normalized_error = np.asarray(error_nodes[glyph_ids], dtype=float)
    ratio = np.clip(error_norm(normalized_error), 0.0, 1.0)

    
    
    
    colors = matplotlib.colormaps[PALETTE.error_cmap](0.03 + 0.97 * ratio)
    colors[:, 3] = 0.42 + 0.56 * ratio

    ax.scatter(
        data.nodes[glyph_ids, 0],
        data.nodes[glyph_ids, 1],
        data.nodes[glyph_ids, 2],
        s=4.5 + 17.0 * ratio**0.80,
        c=colors,
        edgecolors="none",
        depthshade=False,
        rasterized=True,
        zorder=5,
    )

    
    
    
    arrow_length = 0.045 + 0.54 * ratio**0.78
    arrows = direction * arrow_length[:, None]

    ax.quiver(
        data.nodes[glyph_ids, 0],
        data.nodes[glyph_ids, 1],
        data.nodes[glyph_ids, 2],
        arrows[:, 0],
        arrows[:, 1],
        arrows[:, 2],
        color=colors,
        linewidth=0.72,
        arrow_length_ratio=0.31,
        normalize=False,
        pivot="middle",
        alpha=0.98,
    )

    _set_3d_axis(
        ax,
        surfaces.outer_radius * 1.005,
        elev=20.0,
        azim=-49.0,
        zoom=1.38,
    )


def _radial_component(
    nodes: np.ndarray,
    vector: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    
    radial = nodes - center[None, :]
    radius = np.linalg.norm(radial, axis=1, keepdims=True)
    direction = radial / np.maximum(radius, EPS)
    return np.sum(vector * direction, axis=1)


def plot_radial_component_slice(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    radial_nodes: np.ndarray,
    radial_norm: TwoSlopeNorm,
) -> None:
    
    x, y, query, visible = _build_slice_grid(surfaces, grid_size=132)
    radial_slice = _interpolate_slice(
        data.nodes,
        np.asarray(radial_nodes, dtype=float),
        query,
        x.shape,
    )
    masked = np.ma.array(radial_slice, mask=~visible)

    ax.pcolormesh(
        x,
        y,
        masked,
        cmap=PALETTE.flux_cmap,
        norm=radial_norm,
        shading="gouraud",
        rasterized=True,
        zorder=1,
    )

    
    
    try:
        ax.contour(
            x,
            y,
            masked,
            levels=[0.0],
            colors="#FFFFFF",
            linewidths=0.42,
            alpha=0.62,
            zorder=5,
        )
    except ValueError:
        pass

    _draw_annular_slice_frame(ax, surfaces)


def _local_vector_error(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    
    denominator = max(
        float(
            np.quantile(
                np.linalg.norm(truth, axis=1),
                0.95,
            )
        ),
        EPS,
    )
    return np.linalg.norm(prediction - truth, axis=1) / denominator


def _shared_error_norm(
    fields: list[np.ndarray],
) -> PowerNorm:
    
    pooled = np.concatenate([np.asarray(field, dtype=float).ravel() for field in fields])
    finite = pooled[np.isfinite(pooled)]

    if len(finite) == 0:
        return PowerNorm(gamma=0.40, vmin=0.0, vmax=1.0)

    vmax = float(np.quantile(finite, 0.995))
    vmax = max(vmax, 1.0e-8)

    return PowerNorm(
        gamma=0.40,
        vmin=0.0,
        vmax=vmax,
        clip=True,
    )


def plot_error_square_slice(
    ax,
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    error_nodes: np.ndarray,
    error_norm: PowerNorm,
) -> None:
    
    x, y, query, valid = _build_slice_grid(
        surfaces,
        grid_size=132,
        inner_scale=1.015,
        outer_scale=0.995,
    )

    error_slice = _interpolate_slice(
        data.nodes,
        np.asarray(error_nodes, dtype=float),
        query,
        x.shape,
    )

    masked = np.ma.array(
        error_slice,
        mask=~valid,
    )

    
    
    ax.set_facecolor("#F5F6F6")

    ax.pcolormesh(
        x,
        y,
        masked,
        cmap=PALETTE.error_cmap,
        norm=error_norm,
        shading="gouraud",
        rasterized=True,
        zorder=1,
    )

    
    
    contour_levels = np.linspace(
        0.22 * error_norm.vmax,
        0.82 * error_norm.vmax,
        3,
    )
    try:
        ax.contour(
            x,
            y,
            masked,
            levels=contour_levels,
            colors="#FFFFFF",
            linewidths=0.28,
            alpha=0.18,
            zorder=4,
        )
    except ValueError:
        pass

    
    outer = plt.Circle(
        (0.0, 0.0),
        surfaces.outer_radius,
        fill=False,
        edgecolor="#748B9A",
        linewidth=0.58,
        alpha=0.72,
        zorder=7,
    )
    inner = plt.Circle(
        (0.0, 0.0),
        surfaces.inner_radius,
        facecolor="#F5F6F6",
        edgecolor="#526877",
        linewidth=0.70,
        alpha=1.0,
        zorder=8,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    
    
    r = surfaces.outer_radius
    frame = plt.Rectangle(
        (-r, -r),
        2.0 * r,
        2.0 * r,
        fill=False,
        edgecolor="#AAB6BE",
        linewidth=0.48,
        alpha=0.78,
        zorder=9,
    )
    ax.add_patch(frame)

    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _error_colorbar(
    fig,
    rect,
    *,
    norm: PowerNorm,
    cmap: str,
    label: str,
) -> None:
    
    cax = fig.add_axes(rect)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    mapper.set_array([])

    cbar = fig.colorbar(
        mapper,
        cax=cax,
        orientation="horizontal",
    )
    cbar.outline.set_edgecolor(PALETTE.hairline)
    cbar.outline.set_linewidth(0.82)

    
    
    ticks = np.linspace(0.0, float(norm.vmax), 6, dtype=float)

    superscript = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

    def _fmt(value: float) -> str:
        
        
        if value == 0.0:
            return "0"
        if abs(value) < 1.0e-2:
            exponent = int(np.floor(np.log10(abs(value))))
            mantissa = value / (10.0**exponent)
            return f"{mantissa:.1f}×10{str(exponent).translate(superscript)}"
        if abs(value) < 0.1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")

    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([_fmt(v) for v in ticks])
    cbar.ax.tick_params(
        axis="x",
        labelsize=11.0,
        colors=PALETTE.ink,
        length=3.0,
        width=0.78,
        pad=1.7,
        direction="out",
    )
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_color(PALETTE.ink)
        tick_label.set_fontweight("bold")

    cbar.set_label(
        label,
        fontsize=13.0,
        fontweight="bold",
        color=PALETTE.ink,
        labelpad=2.4,
    )
    cbar.ax.xaxis.label.set_color(PALETTE.ink)
    cbar.ax.xaxis.label.set_fontweight("bold")








def _compact_horizontal_colorbar(
    fig,
    rect,
    *,
    norm: Normalize,
    cmap: str,
    label: str,
) -> None:
    
    cax = fig.add_axes(rect)

    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    scalar.set_array([])

    cbar = fig.colorbar(
        scalar,
        cax=cax,
        orientation="horizontal",
    )

    cbar.outline.set_edgecolor(PALETTE.hairline)
    cbar.outline.set_linewidth(0.82)

    cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    cbar.ax.tick_params(
        axis="x",
        labelsize=11.0,
        colors=PALETTE.ink,
        length=3.0,
        width=0.78,
        pad=1.8,
        direction="out",
    )
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_color(PALETTE.ink)
        tick_label.set_fontweight("bold")

    cbar.set_label(
        label,
        fontsize=13.0,
        fontweight="bold",
        color=PALETTE.ink,
        labelpad=2.6,
    )
    cbar.ax.xaxis.label.set_color(PALETTE.ink)
    cbar.ax.xaxis.label.set_fontweight("bold")





def create_paper_figure(
    data: ExperimentData,
    surfaces: BoundarySurfaces,
    sample_row: int,
    n_streamlines: int,
) -> tuple[plt.Figure, dict]:
    
    source = data.source[sample_row]
    truth = data.target[sample_row]

    
    
    
    
    finite_source = np.asarray(source)[np.isfinite(source)]
    if finite_source.size:
        print(
            "[Viz] Displayed source-density range: "
            f"[{finite_source.min():.6g}, {finite_source.max():.6g}] "
            f"(test row={sample_row})"
        )

    models = select_comparison_models(data, count=6)
    if not models:
        raise ValueError("No predictions are available for the six-model comparison")

    predictions = {name: data.predictions[name][sample_row] for name in models}

    source_norm = _robust_symmetric_norm([source], quantile=0.975)

    truth_magnitude = np.linalg.norm(truth, axis=1)
    prediction_magnitudes = {name: np.linalg.norm(predictions[name], axis=1) for name in models}
    all_magnitudes = [truth_magnitude] + [prediction_magnitudes[name] for name in models]
    field_norm = _robust_positive_norm(all_magnitudes, quantile=0.992)

    
    
    
    truth_q95 = max(float(np.quantile(truth_magnitude, 0.95)), EPS)
    magnitude_bias = {
        name: (prediction_magnitudes[name] - truth_magnitude) / truth_q95 for name in models
    }
    bias_norm = _robust_symmetric_norm(
        list(magnitude_bias.values()),
        quantile=0.985,
    )
    field_scale = max(
        float(np.quantile(np.concatenate(all_magnitudes), 0.97)),
        EPS,
    )

    
    
    
    comparison_glyph_ids = _select_comparison_glyph_nodes(
        data.nodes,
        predictions,
        surfaces.inner_radius,
        surfaces.outer_radius,
        n_focus=34,
        context_per_shell=9,
    )

    pairwise_prediction_difference = _prediction_pairwise_diagnostics(predictions)
    if pairwise_prediction_difference:
        closest_pair, closest_difference = min(
            pairwise_prediction_difference.items(),
            key=lambda item: item[1],
        )
        print(
            f"[Viz] Closest prediction pair: {closest_pair}; "
            f"relative difference={closest_difference:.3e}"
        )
    nearly_identical_pairs = {
        pair: value for pair, value in pairwise_prediction_difference.items() if value < 1.0e-7
    }
    if nearly_identical_pairs:
        print(
            "[Viz warning] Near-identical prediction arrays detected: "
            + ", ".join(f"{pair}={value:.3e}" for pair, value in nearly_identical_pairs.items())
        )

    residual_vectors = {name: predictions[name] - truth for name in models}
    error_fields = {name: _local_vector_error(predictions[name], truth) for name in models}
    error_norm = _shared_error_norm(list(error_fields.values()))

    
    
    sample_errors = {
        name: float(np.linalg.norm(residual_vectors[name]) / max(np.linalg.norm(truth), EPS))
        for name in models
    }

    
    
    
    
    
    W, H = 2200.0, 980.0

    
    OUTER_X = 30.0
    TOP_TITLE_Y = 952.0
    LEFT_W = 440.0
    SECTION_GAP = 18.0
    RIGHT_X = OUTER_X + LEFT_W + SECTION_GAP
    RIGHT_W = W - RIGHT_X - OUTER_X

    
    LEFT_PANEL = 330.0
    LEFT_PANEL_X = OUTER_X + 0.5 * (LEFT_W - LEFT_PANEL)
    LEFT_TOP_Y = 570.0
    LEFT_BOTTOM_Y = 120.0
    LEFT_CBAR_GAP = 17.0
    LEFT_CBAR_W = 320.0
    LEFT_CBAR_H = 11.0

    
    RIGHT_COL_GAP = 6.0
    RIGHT_MINI_W = (RIGHT_W - 5.0 * RIGHT_COL_GAP) / 6.0
    RIGHT_MINI_H = 210.0

    RIGHT_ROW1_Y = 690.0  
    RIGHT_ROW2_Y = 380.0  
    RIGHT_ROW3_Y = 80.0  

    RIGHT_CBAR_W = 0.80 * RIGHT_W
    RIGHT_CBAR_H = 11.0
    RIGHT_ROW1_CBAR_Y = 660.0
    RIGHT_ROW2_CBAR_Y = 350.0
    RIGHT_ROW3_CBAR_Y = 43.0

    RIGHT_ROW_LABEL_X = RIGHT_X - 8.0

    fig = plt.figure(figsize=(W / 100.0, H / 100.0), facecolor="white")

    
    
    
    ax_a = fig.add_axes(
        (LEFT_PANEL_X / W, LEFT_TOP_Y / H, LEFT_PANEL / W, LEFT_PANEL / H),
        projection="3d",
    )
    ax_b = fig.add_axes(
        (LEFT_PANEL_X / W, LEFT_BOTTOM_Y / H, LEFT_PANEL / W, LEFT_PANEL / H),
        projection="3d",
    )

    plot_geometry_and_source(ax_a, data, surfaces, source, source_norm)
    plot_flux_field(
        ax_b,
        data,
        surfaces,
        truth,
        field_norm,
        n_lines=max(24, n_streamlines),
    )

    
    
    
    
    axes_pred: list = []
    axes_residual: list = []
    axes_error: list = []

    for col in range(6):
        x = RIGHT_X + col * (RIGHT_MINI_W + RIGHT_COL_GAP)

        axes_pred.append(
            fig.add_axes(
                (x / W, RIGHT_ROW1_Y / H, RIGHT_MINI_W / W, RIGHT_MINI_H / H),
                projection="3d",
            )
        )
        axes_residual.append(
            fig.add_axes(
                (x / W, RIGHT_ROW2_Y / H, RIGHT_MINI_W / W, RIGHT_MINI_H / H),
                projection="3d",
            )
        )
        axes_error.append(
            fig.add_axes((x / W, RIGHT_ROW3_Y / H, RIGHT_MINI_W / W, RIGHT_MINI_H / H))
        )

    for ax, name in zip(axes_pred, models):
        plot_prediction_glyph_3d(
            ax,
            data,
            surfaces,
            predictions[name],
            truth,
            magnitude_bias[name],
            bias_norm,
            comparison_glyph_ids,
            field_scale,
            model_name=name,
        )

    for ax, name in zip(axes_residual, models):
        plot_residual_glyph_3d(
            ax,
            data,
            surfaces,
            residual_vectors[name],
            error_fields[name],
            error_norm,
            comparison_glyph_ids,
            model_name=name,
        )

    for ax, name in zip(axes_error, models):
        plot_error_square_slice(
            ax,
            data,
            surfaces,
            error_fields[name],
            error_norm,
        )

    for group in (axes_pred, axes_residual, axes_error):
        for ax in group[len(models) :]:
            ax.set_axis_off()

    
    
    
    fig.text(
        (OUTER_X + 0.5 * LEFT_W) / W,
        TOP_TITLE_Y / H,
        "Geometric Domain & Physical Fields",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color=PALETTE.ink,
    )
    fig.text(
        (RIGHT_X + 0.5 * RIGHT_W) / W,
        TOP_TITLE_Y / H,
        "Top-6 Model Predictions and Error Diagnostics",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color=PALETTE.ink,
    )

    row_labels = (
        (RIGHT_ROW1_Y, RIGHT_MINI_H, "Prediction"),
        (RIGHT_ROW2_Y, RIGHT_MINI_H, "3-D Residual"),
        (RIGHT_ROW3_Y, RIGHT_MINI_H, "Local Error"),
    )
    for y, height, label in row_labels:
        fig.text(
            RIGHT_ROW_LABEL_X / W,
            (y + 0.5 * height) / H,
            label,
            ha="right",
            va="center",
            rotation=90,
            fontsize=11.6,
            fontweight="bold",
            color=PALETTE.ink,
        )

    
    
    
    
    left_bar_x = OUTER_X + 0.5 * (LEFT_W - LEFT_CBAR_W)
    left_top_bar_y = LEFT_TOP_Y - LEFT_CBAR_GAP - LEFT_CBAR_H
    left_bottom_bar_y = LEFT_BOTTOM_Y - LEFT_CBAR_GAP - LEFT_CBAR_H
    right_bar_x = RIGHT_X + 0.5 * (RIGHT_W - RIGHT_CBAR_W)

    _compact_horizontal_colorbar(
        fig,
        (left_bar_x / W, left_top_bar_y / H, LEFT_CBAR_W / W, LEFT_CBAR_H / H),
        norm=source_norm,
        cmap=PALETTE.source_cmap,
        label=r"Source density $\rho$",
    )
    _compact_horizontal_colorbar(
        fig,
        (left_bar_x / W, left_bottom_bar_y / H, LEFT_CBAR_W / W, LEFT_CBAR_H / H),
        norm=field_norm,
        cmap=PALETTE.field_cmap,
        label=r"Reference flux magnitude $\|\mathbf{B}\|_2$",
    )
    _compact_horizontal_colorbar(
        fig,
        (right_bar_x / W, RIGHT_ROW1_CBAR_Y / H, RIGHT_CBAR_W / W, RIGHT_CBAR_H / H),
        norm=bias_norm,
        cmap=PALETTE.source_cmap,
        label=(
            r"Signed magnitude bias  "
            r"$(\|\widehat{\mathbf{B}}\|_2-\|\mathbf{B}\|_2)/"
            r"Q_{0.95}(\|\mathbf{B}\|_2)$"
        ),
    )
    _error_colorbar(
        fig,
        (right_bar_x / W, RIGHT_ROW2_CBAR_Y / H, RIGHT_CBAR_W / W, RIGHT_CBAR_H / H),
        norm=error_norm,
        cmap=PALETTE.error_cmap,
        label=(r"3-D residual magnitude  $\|\Delta\mathbf{B}\|_2/" r"Q_{0.95}(\|\mathbf{B}\|_2)$"),
    )
    _error_colorbar(
        fig,
        (right_bar_x / W, RIGHT_ROW3_CBAR_Y / H, RIGHT_CBAR_W / W, RIGHT_CBAR_H / H),
        norm=error_norm,
        cmap=PALETTE.error_cmap,
        label=(
            r"Local vector error  $\|\widehat{\mathbf{B}}-\mathbf{B}\|_2/"
            r"Q_{0.95}(\|\mathbf{B}\|_2)$"
        ),
    )

    metadata = {
        "sample_selection": "representative multi-model sample unless explicitly overridden",
        "test_row": int(sample_row),
        "dataset_index": int(data.test_indices[sample_row]),
        "dkho_variants_loaded": [
            name for name in ("DKHO-large", "DKHO-small") if name in data.predictions
        ],
        "comparison_models": models,
        "selection_rule": (
            "DKHO-large + DKHO-small guaranteed, remaining slots filled by the "
            "best baselines by full-test mean relative-L2; final columns sorted by score"
        ),
        "population_relative_l2": {
            name: float(data.population_relative_l2[name]) for name in models
        },
        "displayed_sample_relative_l2": sample_errors,
        "source_density_range": (
            [float(finite_source.min()), float(finite_source.max())] if finite_source.size else None
        ),
        "pairwise_prediction_relative_difference": pairwise_prediction_difference,
        "comparison_glyph_count": int(len(comparison_glyph_ids)),
        "rows": [
            "3-D predicted vectors over thin GT references; arrow length uses a shared physical scale and color shows signed magnitude bias",
            "3-D residual vectors ΔB = B_pred - B_true; direction by glyph orientation, magnitude by shared color/length/size",
            "normalized local vector error on z=0",
        ],
        "error_definition": (
            "nodewise ||B_pred-B_true||_2 / Q95(||B_true||_2), " "shared PowerNorm gamma=0.40"
        ),
        "figure_layout": (
            "2200x980 compact canvas; left 2x1 physical panels; right three compact "
            "six-model rows (prediction+bias / 3-D residual / z=0 local error); "
            "five shared horizontal colorbars"
        ),
    }

    return fig, metadata





def save_outputs(
    fig: plt.Figure,
    metadata: dict,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved: list[Path] = []

    for suffix in (
        "png",
        "pdf",
        "svg",
    ):
        path = output_dir / (f"{stem}.{suffix}")

        kwargs = {"dpi": dpi} if suffix == "png" else {}

        
        
        fig.savefig(
            path,
            facecolor="white",
            **kwargs,
        )
        saved.append(path)

    manifest = output_dir / (f"{stem}.json")
    manifest.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
    saved.append(manifest)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Render the compact DKHO-vs-baseline spherical-cavity magnetostatics figure.")
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATA,
        help="Canonical cavity-magnetostatics dataset.",
    )
    parser.add_argument(
        "--dkho-large-run",
        type=Path,
        default=DEFAULT_DKHO_LARGE_RUN,
        help="Validation-best DKHO-large run directory.",
    )
    parser.add_argument(
        "--dkho-small-run",
        type=Path,
        default=DEFAULT_DKHO_SMALL_RUN,
        help="Validation-best DKHO-small run directory.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=DEFAULT_BASELINE_LOG,
        help="Frozen six-baseline prediction log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the rendered qualitative figure and manifest.",
    )

    parser.add_argument(
        "--sample",
        choices=(
            "representative",
            "best",
            "median",
            "worst",
        ),
        default="representative",
        help=(
            "Qualitative sample selection. 'representative' is the default and "
            "chooses a sample jointly close to all displayed models' median "
            "test errors; best/median/worst retain DKHO-large rank behavior."
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=("Explicit row in the aligned test split."),
    )

    parser.add_argument(
        "--streamlines",
        type=int,
        default=34,
        help=("Number of deterministic 3-D field-line seeds."),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster resolution for PNG output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_style()

    data = load_experiment(
        args.dataset,
        args.dkho_large_run,
        args.dkho_small_run,
        args.baseline_log,
    )

    surfaces = extract_boundary_surfaces(data)

    sample_row = choose_sample(
        data,
        args.sample,
        args.sample_index,
    )

    fig, metadata = create_paper_figure(
        data,
        surfaces,
        sample_row,
        n_streamlines=max(
            20,
            args.streamlines,
        ),
    )

    stem = f"magnetostatics_field_comparison_{args.sample}"

    if args.sample_index is not None:
        stem = "magnetostatics_field_comparison_" f"testrow_{sample_row:03d}"

    saved = save_outputs(
        fig,
        metadata,
        args.output_dir,
        stem,
        args.dpi,
    )
    plt.close(fig)

    print("Compact DKHO comparison figure generated:")
    for path in saved:
        print(f"  {path}")

    sample_rel = metadata["displayed_sample_relative_l2"]
    summary = []
    for name in ("DKHO-large", "DKHO-small"):
        if name in sample_rel:
            summary.append(f"{name} {100.0 * sample_rel[name]:.3f}%")
    suffix = " / " + " / ".join(summary) if summary else ""
    print(
        f"sample: test row {metadata['test_row']} / "
        f"dataset index {metadata['dataset_index']}" + suffix
    )


if __name__ == "__main__":
    main()
