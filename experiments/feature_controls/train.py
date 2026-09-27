"""Train matched-input feature controls for baseline models."""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
MODEL_NAMES = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")
TASK_FEATURES = {
    "darcy": (),
    "magnetostatics": ("boundary", "lpe", "heat", "qp"),
    "toroidal_transport": ("velocity", "heat", "lpe", "global", "torus_fourier"),
}
TASK_RANK = {"magnetostatics": "form_2", "toroidal_transport": "form_0"}


def delegate_darcy_control(args: argparse.Namespace) -> None:
    

    if args.features:
        raise ValueError(
            "Darcy profiles are named protocols; use --profile rather than --features"
        )
    if args.feature_mode != "normal":
        raise ValueError("--feature-mode is only defined for magnetostatics and toroidal_transport")
    if args.preflight:
        raise ValueError("Darcy has no node-only preflight; run darcy/baselines/train.py directly")
    output_root = None
    if args.output_root:
        candidate = Path(args.output_root)
        output_root = candidate if candidate.is_absolute() else EXPERIMENTS / "darcy" / candidate
    if args.evaluate_only:
        command = [
            sys.executable,
            str(EXPERIMENTS / "darcy" / "evaluate_feature_controls.py"),
            "--task",
            args.rank,
            "--model",
            args.model,
            "--profiles",
            args.profile,
            "--seed",
            str(args.seed),
        ]
        if output_root is not None:
            command.extend(("--runs-root", str(output_root)))
    else:
        command = [
            sys.executable,
            str(EXPERIMENTS / "darcy" / "baselines" / "train.py"),
            "--model",
            args.model,
            "--task",
            args.rank,
            "--feature-configs",
            args.profile,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--lpe-dim",
            str(args.lpe_dim),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--device",
            args.device,
        ]
        if output_root is not None:
            command.extend(("--output-root", str(output_root)))
    print("[feature-controls] delegating Darcy command:\n  " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def feature_profile(task: str, enabled: set[str]) -> str:
    

    profiles = {
        "magnetostatics": {
            frozenset(): "native",
            frozenset({"boundary"}): "boundary",
            frozenset({"boundary", "lpe"}): "boundary_spectral",
            frozenset({"boundary", "lpe", "heat"}): "boundary_diffusion_spectral",
            frozenset({"boundary", "lpe", "heat", "qp"}): "conditioned",
        },
        "toroidal_transport": {
            frozenset(): "native",
            frozenset({"velocity"}): "flow",
            frozenset({"velocity", "heat"}): "flow_diffusion",
            frozenset({"velocity", "heat", "lpe"}): "flow_diffusion_spectral",
            frozenset({"velocity", "heat", "lpe", "global"}): "flow_global_spectral",
            frozenset({"velocity", "heat", "lpe", "global", "torus_fourier"}): "conditioned",
        },
    }
    return profiles[task].get(frozenset(enabled), "custom")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_of(requested: str) -> torch.device:
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA unavailable; using CPU.", flush=True)
    return torch.device("cpu")


def split_indices(n: int) -> dict[str, np.ndarray]:
    indices = np.arange(n)
    train_val, test = train_test_split(indices, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, random_state=42)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def edges_from_simplices(simplices: np.ndarray) -> np.ndarray:
    width = simplices.shape[1]
    edges = {
        tuple(sorted((int(s[i]), int(s[j]))))
        for s in simplices
        for i in range(width)
        for j in range(i + 1, width)
    }
    return np.asarray(sorted(edges), dtype=np.int64)


def d0_from_edges(edges: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    ids = np.arange(len(edges))
    return sparse.csr_matrix(
        (
            np.r_[-np.ones(len(ids)), np.ones(len(ids))],
            (np.r_[ids, ids], np.r_[edges[:, 0], edges[:, 1]]),
        ),
        shape=(len(edges), n_nodes),
        dtype=np.float32,
    )


def positive_lpe(lap: sparse.spmatrix, count: int, cache: Path) -> np.ndarray:
    if cache.exists():
        value = np.load(cache)["lpe0"]
        if value.shape == (lap.shape[0], count):
            return value.astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = lap.shape[0]
    request = min(n - 1, max(count + 8, 2 * count + 4))
    while True:
        values, vectors = eigsh(
            lap.astype(np.float64), k=request, which="SM", tol=1e-4, maxiter=20000
        )
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        kept = np.flatnonzero(values > 1e-7)[:count]
        if len(kept) == count:
            out = vectors[:, kept]
            signs = np.sign(out[np.abs(out).argmax(0), np.arange(count)])
            out *= np.where(signs == 0, 1.0, signs)[None]
            np.savez_compressed(cache, lpe0=out.astype(np.float32))
            return out.astype(np.float32)
        if request >= n - 1:
            raise RuntimeError(f"LPE needs {count} positive modes but mesh has only {len(kept)}")
        request = min(n - 1, max(request + 8, 2 * request))


def node_area(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (points[faces[:, i]] for i in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    result = np.zeros(len(points), np.float32)
    for column in range(3):
        np.add.at(result, faces[:, column], area / 3)
    return result / max(float(result.sum()), 1e-12)


def static_node_weights(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    tetra = points[tetrahedra]
    volume = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                tetra[:, 1] - tetra[:, 0],
                np.cross(tetra[:, 2] - tetra[:, 0], tetra[:, 3] - tetra[:, 0]),
            )
        )
        / 6
    )
    result = np.zeros(len(points), np.float32)
    for column in range(4):
        np.add.at(result, tetrahedra[:, column], volume.astype(np.float32) / 4)
    return result / max(float(result.sum()), 1e-12)


def normalize_added(features: np.ndarray, train_val: np.ndarray) -> np.ndarray:
    
    if features.shape[-1] <= 1:
        return features.astype(np.float32)
    mean = features[train_val, :, 1:].mean(axis=(0, 1), keepdims=True)
    std = features[train_val, :, 1:].std(axis=(0, 1), keepdims=True) + 1e-6
    result = features.copy()
    result[:, :, 1:] = (result[:, :, 1:] - mean) / std
    return result.astype(np.float32)


def validate_features(features: np.ndarray, n_samples: int, n_nodes: int, task: str) -> None:
    
    if features.ndim != 3 or features.shape[:2] != (n_samples, n_nodes):
        raise ValueError(
            f"{task}: expected feature shape ({n_samples}, {n_nodes}, C), got {features.shape}"
        )
    if features.shape[-1] < 1:
        raise ValueError(f"{task}: feature tensor must retain the raw-source channel")
    if not np.isfinite(features).all():
        bad = int(np.size(features) - np.isfinite(features).sum())
        raise ValueError(f"{task}: feature construction produced {bad} non-finite values")


def make_static_features(
    x: np.ndarray,
    points: np.ndarray,
    elements: np.ndarray,
    data: dict,
    enabled: set[str],
    cache: Path,
    lpe_dim: int,
) -> np.ndarray:
    n_samples, n_nodes = x.shape
    edges = edges_from_simplices(elements)
    d0 = d0_from_edges(edges, n_nodes)
    pieces = [x[:, :, None]]
    if "boundary" in enabled:
        radius = np.linalg.norm(points, axis=1)
        inner = np.asarray(
            data.get("inner_boundary", np.flatnonzero(radius < radius.max() * 0.35)), dtype=np.int64
        )
        outer = np.asarray(
            data.get("outer_boundary", np.flatnonzero(radius > radius.max() * 0.80)), dtype=np.int64
        )
        mi = np.zeros(n_nodes, np.float32)
        mo = mi.copy()
        mi[inner] = 1
        mo[outer] = 1
        scale = max(float(np.linalg.norm(points.max(0) - points.min(0))), 1e-12)
        di = cKDTree(points[inner]).query(points)[0] / scale
        do = cKDTree(points[outer]).query(points)[0] / scale
        radial = points / np.maximum(radius[:, None], 1e-12)
        fixed = np.c_[mi, mo, di, do, radial * (mo - mi)[:, None]].astype(np.float32)
        pieces.append(np.broadcast_to(fixed, (n_samples, n_nodes, fixed.shape[1])))
    if "lpe" in enabled:
        lpe = positive_lpe(d0.T @ d0, lpe_dim, cache / f"static_lpe0_k{lpe_dim}.npz")
        pieces.append(np.broadcast_to(lpe, (n_samples, n_nodes, lpe_dim)))
    if "heat" in enabled:
        lap = d0.T @ d0
        current = x.copy()
        heat = []
        for _ in range(4):
            current = current - 0.025 * (lap @ current.T).T
            heat.append(current[:, :, None])
        pieces.extend(heat)
    if "qp" in enabled:
        weights = static_node_weights(points, elements)
        q = (x * weights[None]).sum(1, keepdims=True)
        p = np.einsum("bn,n,nc->bc", x, weights, points)
        values = np.concatenate([q, p], axis=1).astype(np.float32)
        pieces.append(np.broadcast_to(values[:, None], (n_samples, n_nodes, 4)))
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def make_torus_features(
    x: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    enabled: set[str],
    cache: Path,
    lpe_dim: int,
) -> np.ndarray:
    n_samples, n_nodes = x.shape
    edges = edges_from_simplices(faces)
    d0 = d0_from_edges(edges, n_nodes)
    pieces = [x[:, :, None]]
    area_weights = node_area(points, faces)
    if "velocity" in enabled:
        raw = np.c_[-points[:, 1], points[:, 0], np.zeros(n_nodes)]
        velocity = raw - (raw * normals).sum(1, keepdims=True) * normals
        edge_vector = points[edges[:, 1]] - points[edges[:, 0]]
        v_edge = (0.5 * (velocity[edges[:, 0]] + velocity[edges[:, 1]]) * edge_vector).sum(1)
        div_v = d0.T @ v_edge
        u_edge = 0.5 * (x[:, edges[:, 0]] + x[:, edges[:, 1]])
        q_edge = u_edge * v_edge[None]
        div_q = (d0.T @ q_edge.T).T
        speed = np.linalg.norm(velocity, axis=1)
        
        
        face_curl = np.zeros((n_samples, len(faces)), np.float32)
        edge_map = {tuple(edge): index for index, edge in enumerate(edges)}
        for fi, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted((int(a), int(b))))
                sign = 1.0 if (int(a), int(b)) == key else -1.0
                face_curl[:, fi] += sign * q_edge[:, edge_map[key]]
        average = np.zeros((n_samples, n_nodes), np.float32)
        degree = np.zeros(n_nodes, np.float32)
        for col in range(3):
            np.add.at(degree, faces[:, col], 1.0)
            for sample in range(n_samples):
                np.add.at(average[sample], faces[:, col], face_curl[sample])
        average /= np.maximum(degree, 1.0)[None]
        dynamic = np.stack(
            [
                np.broadcast_to(div_v, (n_samples, n_nodes)),
                div_q,
                np.broadcast_to(speed, (n_samples, n_nodes)),
                average,
            ],
            axis=-1,
        )
        pieces.append(dynamic.astype(np.float32))
    if "heat" in enabled:
        lap = d0.T @ d0
        h1 = x - 0.03 * (lap @ x.T).T
        h2 = h1 - 0.03 * (lap @ h1.T).T
        pieces.append(np.stack([h1, h2], axis=-1).astype(np.float32))
    if "lpe" in enabled:
        lpe = positive_lpe(d0.T @ d0, lpe_dim, cache / f"torus_lpe0_k{lpe_dim}.npz")
        pieces.append(np.broadcast_to(lpe, (n_samples, n_nodes, lpe_dim)))
    mean = (x * area_weights[None]).sum(1)
    centered = x - mean[:, None]
    energy = np.sqrt((np.square(x) * area_weights[None]).sum(1))
    scale = np.sqrt((np.square(centered) * area_weights[None]).sum(1))
    gradient = x[:, edges[:, 1]] - x[:, edges[:, 0]]
    grad_scale = np.sqrt(np.mean(np.square(gradient), axis=1))
    if "global" in enabled:
        stats = np.stack([mean, energy, scale, grad_scale], axis=1).astype(np.float32)
        pieces.append(np.broadcast_to(stats[:, None], (n_samples, n_nodes, 4)))
    if "torus_fourier" in enabled:
        radial_xy = np.linalg.norm(points[:, :2], axis=1)
        major = float(radial_xy.mean())
        theta = np.arctan2(points[:, 1], points[:, 0])
        phi = np.arctan2(points[:, 2], radial_xy - major)
        basis = np.stack(
            [
                entry
                for m, n in ((1, 0), (0, 1), (1, 1), (1, -1))
                for entry in (np.cos(m * theta + n * phi), np.sin(m * theta + n * phi))
            ],
            axis=1,
        )
        coefficients = np.einsum("bn,n,nk->bk", x, area_weights, basis).astype(np.float32)
        pieces.append(np.broadcast_to(coefficients[:, None], (n_samples, n_nodes, 8)))
    return np.concatenate(pieces, axis=-1).astype(np.float32)


class BatchAdapter:
    def __init__(self, manager, points: np.ndarray, model_name: str, task: str, host=None):
        self.manager, self.points, self.name, self.task, self.host = (
            manager,
            points,
            model_name,
            task,
            host,
        )

    def __call__(self, model, features: torch.Tensor) -> torch.Tensor:
        if self.name == "GNO":
            value, _ = self.manager.prepare_gno_batch(features)
            return model(value, self.manager.pts).permute(0, 2, 1)
        if self.name == "FNO":
            value, _ = self.manager.prepare_fno_batch(features)
            return self.manager.decode_fno_output(model(value))
        if self.name == "GeoFNO":
            coords, value = self.manager.prepare_geofno_batch(features)
            return model(coords, value)
        if self.name == "DeepONet":
            return model(features.reshape(features.shape[0], -1), self.manager.pts)
        if self.name == "MGN":
            nodes = torch.cat([features, self.manager.pts[None].expand(len(features), -1, -1)], -1)
            return torch.stack(
                [
                    model(nodes[i], self.manager.edge_index, self.manager.edge_attr)
                    for i in range(len(features))
                ]
            )
        
        source = features[:, :, 0]
        phi0 = torch.as_tensor(self.host.Phi0, device=source.device, dtype=source.dtype)
        phi1 = torch.as_tensor(self.host.Phi1, device=source.device, dtype=source.dtype)
        c0 = source @ phi0
        c1 = (
            source
            @ torch.as_tensor(self.host.B1.T.toarray(), device=source.device, dtype=source.dtype)
        ) @ phi1
        c2 = torch.zeros(
            len(source), self.host.Phi2.shape[1], device=source.device, dtype=source.dtype
        )
        
        
        
        return model(c0, c1, c2, features)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(item.numel() for item in model.parameters() if item.requires_grad)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    axes = tuple(range(1, prediction.ndim))
    numerator = prediction.sub(target).square().sum(axes).sqrt()
    denominator = target.square().sum(axes).sqrt().clamp_min(1e-6)
    return (numerator / denominator).mean()


def hsd_loss(
    task: str, total: torch.Tensor, base: torch.Tensor, residual: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    
    main = relative_l2(total, target)
    sparse = residual.abs().mean()
    if task == "toroidal_transport":
        
        return main + 1e-2 * sparse, main
    
    
    base_loss = relative_l2(base, target)
    smooth = (
        (total[:, 1:] - total[:, :-1]).square().mean()
        if total.shape[1] > 1
        else total.new_zeros(())
    )
    return main + 0.3 * base_loss + 1e-2 * sparse + 0.05 * smooth, main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASK_FEATURES)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
        help="One process trains one model; use shell loops for serial runs.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated requested feature switches; empty is the raw-source baseline.",
    )
    parser.add_argument(
        "--profile",
        default="conditioned",
        help=(
            "Darcy only: named input profile (native, spectral, diffusion_spectral, "
            "conditioned, or all). Ignored by the other tasks."
        ),
    )
    parser.add_argument(
        "--rank",
        choices=("0", "1", "2", "all"),
        default="all",
        help="Darcy only: supervised cochain rank. Ignored by the other tasks.",
    )
    parser.add_argument(
        "--feature-mode", choices=("normal", "zero_added", "shuffle_added"), default="normal"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lpe-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Optional output root. Darcy paths are interpreted relative to experiments/darcy; "
            "the other tasks default to experiments/{task}/runs/controls."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Build features and run one no-gradient forward pass, then exit.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load the saved best_val.pt for this exact feature run and save physical-scale test predictions; never trains or overwrites weights.",
    )
    args = parser.parse_args()
    if args.task == "darcy":
        if args.preflight:
            parser.error("Darcy preflight is provided by experiments/darcy/baselines/train.py")
        delegate_darcy_control(args)
        return
    enabled = {item for item in args.features.split(",") if item}
    illegal = enabled - set(TASK_FEATURES[args.task])
    if illegal:
        raise ValueError(f"Unsupported for {args.task}: {sorted(illegal)}")
    seed_all(args.seed)
    device = device_of(args.device)
    task_root = EXPERIMENTS / args.task
    baseline_dir = task_root / "baselines"
    sys.path.insert(0, str(task_root))
    sys.path.insert(0, str(baseline_dir))
    models = importlib.import_module("models")
    dataset = importlib.import_module("dataset")
    cfg = importlib.import_module("config").Config
    with open(
        task_root
        / (
            "data/cavity_magnetostatics_v1.pkl"
            if args.task == "magnetostatics"
            else "data/torus_transport_v1.pkl"
        ),
        "rb",
    ) as file:
        data = pickle.load(file)
    if args.task == "magnetostatics":
        
        
        raw_x = np.asarray(data["X_data"], dtype=np.float32)
        raw_y = np.asarray(data["Y_data"], dtype=np.float32)
        points = np.asarray(data.get("points", data.get("nodes")), dtype=np.float32)
        simplices = np.asarray(data.get("elements", data.get("faces")), dtype=np.int64)
    else:
        trajectory = np.asarray(data["trajectories"], dtype=np.float32)
        if trajectory.ndim != 3:
            raise ValueError(
                f"toroidal_transport: trajectories must be (samples,time,nodes), got {trajectory.shape}"
            )
        raw_x, raw_y = trajectory[:, 0], trajectory[:, -1]
        points = np.asarray(data["points"], dtype=np.float32)
        simplices = np.asarray(data["faces"], dtype=np.int64)
    split = split_indices(len(raw_x))
    train_val = np.r_[split["train"], split["val"]]
    x_scale = float(np.abs(raw_x[train_val]).max() + 1e-9)
    y_scale = float(np.abs(raw_y[train_val]).max() + 1e-9)
    x, y = raw_x / x_scale, raw_y / y_scale
    cache = task_root / "reports" / "cache" / "feature_controls"
    if args.task == "magnetostatics":
        features = make_static_features(x, points, simplices, data, enabled, cache, args.lpe_dim)
        manager = dataset.DataManager(
            len(points),
            points,
            device,
            simplices=simplices,
            grid_res=cfg.FNO_GRID_RES,
            mesh_type="volume",
        )
        out_dim = 3
    else:
        normals = np.asarray(data["normals"], dtype=np.float32)
        features = make_torus_features(x, points, simplices, normals, enabled, cache, args.lpe_dim)
        manager = dataset.DataManager(
            len(points), points, device, faces=simplices, grid_res=cfg.FNO_GRID_RES
        )
        out_dim = 1
        y = y[:, :, None]
    validate_features(features, len(raw_x), len(points), args.task)
    features = normalize_added(features, train_val)
    if args.feature_mode == "zero_added" and features.shape[-1] > 1:
        features[:, :, 1:] = 0
    if args.feature_mode == "shuffle_added" and features.shape[-1] > 1:
        features[:, :, 1:] = features[
            np.random.default_rng(args.seed + 137).permutation(len(features)), :, 1:
        ]
    input_dim = features.shape[-1]
    host = None
    if args.model == "GNO":
        model = models.GNO(
            input_dim + 3,
            out_dim,
            cfg.GNO_HIDDEN_CHANNELS,
            cfg.GNO_PROJECTION_CHANNELS,
            cfg.GNO_N_LAYERS,
            cfg.GNO_RADIUS,
        )
    elif args.model == "FNO":
        model = models.FNO3d(
            input_dim + 1, out_dim, cfg.FNO_HIDDEN_CHANNELS, cfg.FNO_MODES, cfg.FNO_N_LAYERS
        )
    elif args.model == "MGN":
        model = models.LightweightMGN(
            input_dim + 3, 3, out_dim, cfg.MGN_HIDDEN_DIM, cfg.MGN_NUM_LAYERS
        )
    elif args.model == "DeepONet":
        model = models.DeepONet(
            len(points) * input_dim,
            3,
            out_dim,
            cfg.DEEPONET_BRANCH_LAYERS,
            cfg.DEEPONET_TRUNK_LAYERS,
            cfg.DEEPONET_BASIS_DIM,
        )
    elif args.model == "GeoFNO":
        model = models.GeoFNO(
            cfg.GEOFNO_MODES,
            cfg.GEOFNO_WIDTH,
            cfg.GEOFNO_LAYERS,
            cfg.GEOFNO_GRID_RES,
            out_dim,
            in_features=input_dim,
        )
    else:
        spectral = importlib.import_module("spectral_operators")
        host = spectral.HighOrderSpectralOperators(points, simplices, k_list=(cfg.K_EIGENS,) * 3)
        phi0 = torch.from_numpy(host.Phi0.astype(np.float32)).to(device)
        base = models.SpectralVectorOperator(
            host.Md0, host.Md1, cfg.K_EIGENS, cfg.K_EIGENS, cfg.K_EIGENS, cfg.SPECTRAL_HIDDEN_DIMS
        )
        cls = (
            models.HSDSpectralVectorFNO
            if hasattr(models, "HSDSpectralVectorFNO")
            else models.HSDSpectralFNO
        )
        model = cls(
            base,
            manager,
            phi0,
            cfg.HSD_FNO_MODES,
            cfg.HSD_FNO_HIDDEN,
            cfg.HSD_FNO_LAYERS,
            fno_in_channels=input_dim + 1,
        )
    model = model.to(device)
    adapter = BatchAdapter(manager, points, args.model, args.task, host)
    x_t, y_t = torch.from_numpy(features).to(device), torch.from_numpy(y.astype(np.float32)).to(
        device
    )
    if args.preflight:
        count = min(max(1, args.batch_size), len(x_t), 2)
        model.eval()
        with torch.no_grad():
            output = adapter(model, x_t[:count])
            prediction = output[0] if args.model == "HSD" else output
        if prediction.shape != y_t[:count].shape:
            raise RuntimeError(
                f"{args.model}: prediction shape {tuple(prediction.shape)} does not match target {tuple(y_t[:count].shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"{args.model}: non-finite prediction in preflight")
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "task": args.task,
                    "model": args.model,
                    "features": sorted(enabled),
                    "feature_channels": input_dim,
                    "input_shape": list(x_t[:count].shape),
                    "output_shape": list(prediction.shape),
                    "parameters": parameter_count(model),
                    "device": str(device),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    root = Path(args.output_root) if args.output_root else task_root / "runs" / "controls"
    profile = feature_profile(args.task, enabled)
    run = root / args.model.lower() / TASK_RANK[args.task] / profile / args.feature_mode / f"seed_{args.seed}"
    if args.evaluate_only:
        checkpoint = run / "best_val.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Saved feature-ablation checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        prediction_parts = []
        with torch.no_grad():
            for start in range(0, len(split["test"]), args.batch_size):
                batch = x_t[split["test"]][start : start + args.batch_size]
                output = adapter(model, batch)
                prediction_parts.append(
                    (output[0] if args.model == "HSD" else output).cpu().numpy()
                )
        prediction = np.concatenate(prediction_parts, axis=0)
        target = y_t[split["test"]].cpu().numpy()
        
        
        
        
        np.savez_compressed(
            run / "test_predictions_physical.npz",
            prediction=(prediction * y_scale).astype(np.float32),
            target=(target * y_scale).astype(np.float32),
            test_indices=split["test"],
        )
        print(
            json.dumps(
                {
                    "evaluation_only": "saved",
                    "run": str(run),
                    "samples": int(len(prediction)),
                    "prediction_shape": list(prediction.shape),
                    "device": str(device),
                },
                indent=2,
            ),
            flush=True,
        )
        return
    train_loader = DataLoader(
        TensorDataset(x_t[split["train"]], y_t[split["train"]]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run.mkdir(parents=True, exist_ok=True)
    best, best_state, history, start = float("inf"), None, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = adapter(model, xb)
            if args.model == "HSD":
                prediction, base_prediction, residual_prediction = output
                loss, main_loss = hsd_loss(
                    args.task, prediction, base_prediction, residual_prediction, yb
                )
            else:
                prediction = output
                main_loss = loss = relative_l2(prediction, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(main_loss.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            output = adapter(model, x_t[split["val"]])
            validation = float(
                relative_l2(output[0] if args.model == "HSD" else output, y_t[split["val"]])
            )
        train_loss = total / max(len(train_loader), 1)
        history.append(
            {"epoch": epoch, "train_relative_l2": train_loss, "val_relative_l2": validation}
        )
        if validation < best:
            best, best_state = validation, {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[{args.model}] epoch={epoch:03d} train_rel_l2={train_loss:.6e} val_rel_l2={validation:.6e} elapsed={time.time()-start:.1f}s",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = adapter(model, x_t[split["test"]])
        prediction = output[0] if args.model == "HSD" else output
        target = y_t[split["test"]]
    mse = float(torch.mean((prediction - target).square()))
    rel = float(
        torch.sqrt(
            (
                (prediction - target).square().sum((1, 2))
                / target.square().sum((1, 2)).clamp_min(1e-12)
            ).mean()
        )
    )
    torch.save(model.state_dict(), run / "best_val.pt")
    report = {
        "task": args.task,
        "model": args.model,
        "features": sorted(enabled),
        "feature_mode": args.feature_mode,
        "feature_profile": profile,
        "input_channels": input_dim,
        "parameters": parameter_count(model),
        "epochs_completed": args.epochs,
        "best_val_relative_l2": best,
        "test_mse_normalized": mse,
        "test_relative_l2": rel,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "split": {key: int(len(value)) for key, value in split.items()},
        "training_protocol": "controlled: relative-L2 supervision for all baselines; task-native HSD regularizers; full requested epochs; best relative-L2 validation checkpoint",
        "base_input_contract": "raw source only when features=[]; task-native coordinates, occupancy mask, MGN edge geometry and HSD spectral lifting are unchanged",
        "elapsed_seconds": time.time() - start,
    }
    (run / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
