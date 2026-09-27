"""Provide shared data, geometry, feature, model, and runtime utilities for Darcy."""

from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from torch import Tensor, nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "perforated_darcy_v1.npz"
CACHE = ROOT / "cache"


def portable_path(path: str | Path) -> str:
    

    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"path must be inside the repository: {candidate}") from error
TASK_TARGET = {"0": "p0", "1": "q1", "2": "omega2"}
_RESIDENT_LOADERS: dict[tuple, "ResidentBatchLoader"] = {}


@dataclass(frozen=True)
class FeatureConfig:
    

    name: str
    lpe: bool = False
    heat: bool = False
    global_broadcast: bool = False


FEATURE_CONFIGS = {
    "native": FeatureConfig("native"),
    "spectral": FeatureConfig("spectral", lpe=True),
    "diffusion_spectral": FeatureConfig("diffusion_spectral", lpe=True, heat=True),
    "conditioned": FeatureConfig("conditioned", lpe=True, heat=True, global_broadcast=True),
}

CHECKPOINT_PROFILE_ALIASES = {"core": "native"}


def checkpoint_feature_profile(name: str) -> str:
    

    normalized = CHECKPOINT_PROFILE_ALIASES.get(name, name)
    if normalized not in FEATURE_CONFIGS:
        raise KeyError(f"unknown Darcy checkpoint input profile: {name!r}")
    return normalized

FEATURE_PROFILE_DETAILS = {
    "native": {
        "base_channels": (
            "source field, prescribed hole values, geometric coordinates, "
            "boundary support, and known permeability descriptors"
        ),
        "added_channels": (),
    },
    "spectral": {
        "base_channels": (
            "source field, prescribed hole values, geometric coordinates, "
            "boundary support, and known permeability descriptors"
        ),
        "added_channels": ("fixed-complex Laplacian positional encoding",),
    },
    "diffusion_spectral": {
        "base_channels": (
            "source field, prescribed hole values, geometric coordinates, "
            "boundary support, and known permeability descriptors"
        ),
        "added_channels": (
            "fixed-complex Laplacian positional encoding",
            "three source-derived diffusion probes",
        ),
    },
    "conditioned": {
        "base_channels": (
            "source field, prescribed hole values, geometric coordinates, "
            "boundary support, and known permeability descriptors"
        ),
        "added_channels": (
            "fixed-complex Laplacian positional encoding",
            "three source-derived diffusion probes",
            "six broadcast source/boundary summary statistics",
        ),
    },
}


def feature_profile_details(config: FeatureConfig) -> dict[str, object]:
    

    if config.name not in FEATURE_PROFILE_DETAILS:
        raise KeyError(f"unknown Darcy input profile: {config.name}")
    details = FEATURE_PROFILE_DETAILS[config.name]
    return {
        "profile": config.name,
        "base_channels": details["base_channels"],
        "added_channels": details["added_channels"],
        "target_safe": True,
        "construction": "prescribed conditions and fixed-complex operators only",
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA unavailable; using CPU.", flush=True)
    return torch.device("cpu")


def configure_runtime(device: torch.device, tf32: bool) -> None:
    
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if tf32:
        torch.set_float32_matmul_precision("high")


def autocast_context(device: torch.device, precision: str):
    
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "amp-bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "amp-fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, precision: str):
    
    enabled = device.type == "cuda" and precision == "amp-fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def maybe_compile(model: nn.Module, device: torch.device, mode: str) -> nn.Module:
    
    if device.type != "cuda" or mode == "off" or not hasattr(torch, "compile"):
        return model
    selected = "reduce-overhead" if mode == "auto" else mode
    try:
        torch._dynamo.config.suppress_errors = True
        return torch.compile(model, mode=selected, dynamic=True)
    except (
        Exception
    ) as error:  
        print(f"[runtime] torch.compile unavailable ({error}); using eager mode", flush=True)
        return model


def parameter_count(model: nn.Module) -> int:
    return sum(item.numel() for item in model.parameters() if item.requires_grad)


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    dims = tuple(range(1, pred.ndim))
    return torch.sqrt(
        (pred.sub(target).square().sum(dims) / target.square().sum(dims).clamp_min(1e-12))
    ).mean()


def relative_l1(pred: Tensor, target: Tensor) -> Tensor:
    dims = tuple(range(1, pred.ndim))
    return (pred.sub(target).abs().sum(dims) / target.abs().sum(dims).clamp_min(1e-12)).mean()


def prepare_cache(force: bool = False) -> Path:
    
    required = [
        "f_samples",
        "g1",
        "g2",
        "p0",
        "q1",
        "omega2",
        "split_train",
        "split_val",
        "split_test",
    ]
    stamp = CACHE / "cache_manifest.json"
    if not force and stamp.exists() and all((CACHE / f"{key}.npy").exists() for key in required):
        return CACHE
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Darcy archive not found: {ARCHIVE}")
    CACHE.mkdir(parents=True, exist_ok=True)
    print("[cache] extracting compressed Darcy archive to .npy arrays (one-time)", flush=True)
    with np.load(ARCHIVE) as data:
        for key in required:
            
            
            value = np.asarray(data[key])
            np.save(CACHE / f"{key}.npy", value)
            del value
    stamp.write_text(
        json.dumps(
            {
                "archive": portable_path(ARCHIVE),
                "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                "format": "npy-v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CACHE


class DarcyGeometry:
    

    def __init__(self, lpe_dim: int = 16) -> None:
        if not ARCHIVE.exists():
            raise FileNotFoundError(ARCHIVE)
        with np.load(ARCHIVE) as data:
            self.points = np.asarray(data["points"], dtype=np.float32)
            self.faces = np.asarray(data["triangles"], dtype=np.int64)
            self.edges = np.asarray(data["edges"], dtype=np.int64)
            self.areas = np.asarray(data["areas"], dtype=np.float32)
            self.node_component = np.asarray(data["node_component"], dtype=np.int64)
            self.edge_component = np.asarray(data["edge_component"], dtype=np.int64)
            self.c_edge = np.asarray(data["c_e"], dtype=np.float32)[:, None]
            self.c_face = np.asarray(data["c_f"], dtype=np.float32)[:, None]

            self.kappa_face = np.asarray(data["kappa_face"], dtype=np.float32).reshape(
                len(data["triangles"]), 4
            )
            self.harmonic_basis_1 = np.asarray(data["harmonic_basis_1"], dtype=np.float32)
            self.beta1 = int(data["beta1"])
            d0 = sparse.csr_matrix(
                (data["d0_data"], (data["d0_row"], data["d0_col"])), shape=tuple(data["d0_shape"])
            )
            d1 = sparse.csr_matrix(
                (data["d1_data"], (data["d1_row"], data["d1_col"])), shape=tuple(data["d1_shape"])
            )
        self.d0, self.d1 = d0.astype(np.float32), d1.astype(np.float32)
        self.n0, self.n1, self.n2 = len(self.points), len(self.edges), len(self.faces)
        assert self.d0.shape == (self.n1, self.n0)
        assert self.d1.shape == (self.n2, self.n1)
        assert np.max(np.abs((self.d1 @ self.d0).data), initial=0.0) < 1e-6
        self.tail, self.head = self.edges[:, 0], self.edges[:, 1]
        self.face_edges = self.d1.indices.reshape(self.n2, 3).astype(np.int64)
        self.face_signs = self.d1.data.reshape(self.n2, 3).astype(np.float32)
        self.edge_mid = ((self.points[self.tail] + self.points[self.head]) * 0.5).astype(np.float32)
        self.edge_vec = (self.points[self.head] - self.points[self.tail]).astype(np.float32)
        self.edge_length = np.linalg.norm(self.edge_vec, axis=1, keepdims=True).astype(np.float32)
        self.edge_dir = self.edge_vec / np.maximum(self.edge_length, 1e-8)
        self.face_mid = self.points[self.faces].mean(axis=1).astype(np.float32)
        self.node_area = np.zeros((self.n0, 1), dtype=np.float32)
        for local in range(3):
            np.add.at(self.node_area[:, 0], self.faces[:, local], self.areas / 3.0)
        self.node_pos = self._normalise(self.points)
        self.edge_pos = self._normalise(self.edge_mid)
        self.face_pos = self._normalise(self.face_mid)
        self.node_boundary_onehot = self._onehot(self.node_component, 4)
        self.edge_boundary_onehot = self._onehot(self.edge_component, 4)
        self.face_boundary = self.node_boundary_onehot[self.faces].mean(axis=1).astype(np.float32)
        self.c_edge_to_node = self._edge_to_node(self.c_edge)
        self.c_face_to_node = self._face_to_node(self.c_face)
        self.kappa_edge = self._face_to_edge(self.kappa_face)
        self.kappa_node = self._face_to_node(self.kappa_face)
        edge_normal = np.stack([-self.edge_dir[:, 1], self.edge_dir[:, 0]], axis=1)
        edge_tensor = self.kappa_edge.reshape(self.n1, 2, 2)
        self.kappa_edge_invariants = np.stack(
            [
                np.einsum("ei,eij,ej->e", self.edge_dir, edge_tensor, self.edge_dir),
                np.einsum("ei,eij,ej->e", edge_normal, edge_tensor, edge_normal),
                np.einsum("ei,eij,ej->e", self.edge_dir, edge_tensor, edge_normal),
            ],
            axis=1,
        ).astype(np.float32)
        self.lpe_dim = lpe_dim
        self.lpe: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _normalise(value: np.ndarray) -> np.ndarray:
        return (
            (value - value.mean(0, keepdims=True)) / (value.std(0, keepdims=True) + 1e-6)
        ).astype(np.float32)

    @staticmethod
    def _onehot(labels: np.ndarray, nclass: int) -> np.ndarray:
        out = np.zeros((len(labels), nclass), dtype=np.float32)
        valid = (labels >= 0) & (labels < nclass)
        out[np.arange(len(labels))[valid], labels[valid]] = 1.0
        return out

    def _edge_to_node(self, edge_value: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n0, edge_value.shape[1]), dtype=np.float32)
        count = np.zeros((self.n0, 1), dtype=np.float32)
        np.add.at(out, self.tail, edge_value)
        np.add.at(out, self.head, edge_value)
        np.add.at(count, self.tail, 1.0)
        np.add.at(count, self.head, 1.0)
        return out / np.maximum(count, 1.0)

    def _face_to_node(self, face_value: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n0, face_value.shape[1]), dtype=np.float32)
        count = np.zeros((self.n0, 1), dtype=np.float32)
        for local in range(3):
            np.add.at(out, self.faces[:, local], face_value)
            np.add.at(count, self.faces[:, local], 1.0)
        return out / np.maximum(count, 1.0)

    def _face_to_edge(self, face_value: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n1, face_value.shape[1]), dtype=np.float32)
        count = np.zeros((self.n1, 1), dtype=np.float32)
        for local in range(3):
            np.add.at(out, self.face_edges[:, local], face_value)
            np.add.at(count, self.face_edges[:, local], 1.0)
        return out / np.maximum(count, 1.0)

    def ensure_lpe(self) -> None:
        if self.lpe is not None:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"darcy_lpe_unweighted_k{self.lpe_dim}.npz"
        if path.exists():
            saved = np.load(path)
            candidate = (saved["lpe0"], saved["lpe1"], saved["lpe2"])
            if (
                candidate[0].shape == (self.n0, self.lpe_dim)
                and candidate[1].shape == (self.n1, self.lpe_dim)
                and candidate[2].shape == (self.n2, self.lpe_dim)
            ):
                self.lpe = tuple(np.asarray(item, dtype=np.float32) for item in candidate)
                return
        print(f"[preprocess] calculating unweighted LPE k={self.lpe_dim}", flush=True)

        def positive_basis(lap: sparse.spmatrix) -> np.ndarray:
            n = lap.shape[0]
            request = min(max(self.lpe_dim + 8, 2 * self.lpe_dim + 4), n - 1)
            while True:
                value, vector = eigsh(
                    lap.astype(np.float64), k=request, which="SM", tol=1e-4, maxiter=30000
                )
                order = np.argsort(value)
                value, vector = value[order], vector[:, order]
                keep = np.flatnonzero(value > 1e-7)[: self.lpe_dim]
                if len(keep) == self.lpe_dim:
                    result = vector[:, keep]
                    signs = np.sign(
                        result[np.abs(result).argmax(axis=0), np.arange(result.shape[1])]
                    )
                    return (result * np.where(signs == 0, 1.0, signs)[None, :]).astype(np.float32)
                if request >= n - 1:
                    raise RuntimeError(
                        f"insufficient positive LPE modes for n={n}, requested={self.lpe_dim}, found={len(keep)}"
                    )
                request = min(n - 1, max(request + 8, request * 2))

        self.lpe = (
            positive_basis(self.d0.T @ self.d0),
            positive_basis(self.d0 @ self.d0.T + self.d1.T @ self.d1),
            positive_basis(self.d1 @ self.d1.T),
        )
        np.savez_compressed(path, lpe0=self.lpe[0], lpe1=self.lpe[1], lpe2=self.lpe[2])


class IncidenceOps(nn.Module):
    

    def __init__(self, geo: DarcyGeometry) -> None:
        super().__init__()
        self.n0, self.n1, self.n2 = geo.n0, geo.n1, geo.n2
        self.register_buffer("tail", torch.from_numpy(geo.tail))
        self.register_buffer("head", torch.from_numpy(geo.head))
        self.register_buffer("face_edges", torch.from_numpy(geo.face_edges))
        self.register_buffer("face_signs", torch.from_numpy(geo.face_signs))
        self.register_buffer("faces", torch.from_numpy(geo.faces))

    def d0(self, value: Tensor) -> Tensor:
        return value[:, self.head] - value[:, self.tail]

    def delta1(self, value: Tensor) -> Tensor:
        out = torch.zeros(
            value.shape[0], self.n0, value.shape[-1], device=value.device, dtype=value.dtype
        )
        out.index_add_(1, self.tail, -value)
        out.index_add_(1, self.head, value)
        return out

    def d1(self, value: Tensor) -> Tensor:

        signs = self.face_signs.to(dtype=value.dtype)
        return (value[:, self.face_edges] * signs[None, :, :, None]).sum(2)

    def delta2(self, value: Tensor) -> Tensor:
        out = torch.zeros(
            value.shape[0], self.n1, value.shape[-1], device=value.device, dtype=value.dtype
        )
        signs = self.face_signs.to(dtype=value.dtype)
        for local in range(3):
            out.index_add_(1, self.face_edges[:, local], value * signs[None, :, local, None])
        return out


class DarcyMemmapDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        task: Literal["0", "1", "2"],
        split: str,
        stats: dict[str, float],
        cache: Path | None = None,
    ) -> None:
        cache = prepare_cache() if cache is None else cache
        self.indices = np.load(cache / f"split_{split}.npy", mmap_mode="r")
        self.f = np.load(cache / "f_samples.npy", mmap_mode="r")
        self.g1 = np.load(cache / "g1.npy", mmap_mode="r")
        self.g2 = np.load(cache / "g2.npy", mmap_mode="r")
        self.y = np.load(cache / f"{TASK_TARGET[task]}.npy", mmap_mode="r")
        self.stats = stats

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int) -> tuple[Tensor, Tensor, Tensor]:
        index = int(self.indices[row])
        f = (np.asarray(self.f[index], dtype=np.float32) - self.stats["f_mean"]) / self.stats[
            "f_std"
        ]
        g = np.asarray([self.g1[index], self.g2[index]], dtype=np.float32)
        g = (g - self.stats["g_mean"]) / self.stats["g_std"]
        y = np.asarray(self.y[index], dtype=np.float32) / self.stats["y_scale"]
        return torch.from_numpy(f), torch.from_numpy(g), torch.from_numpy(y)


class ResidentBatchLoader:

    def __init__(
        self, dataset: DarcyMemmapDataset, batch_size: int, shuffle: bool, device: torch.device
    ) -> None:
        index = np.asarray(dataset.indices, dtype=np.int64)
        f = (
            np.asarray(dataset.f[index], dtype=np.float32) - dataset.stats["f_mean"]
        ) / dataset.stats["f_std"]
        g = np.stack(
            [
                np.asarray(dataset.g1[index], dtype=np.float32),
                np.asarray(dataset.g2[index], dtype=np.float32),
            ],
            axis=-1,
        )
        g = (g - dataset.stats["g_mean"]) / dataset.stats["g_std"]
        y = np.asarray(dataset.y[index], dtype=np.float32) / dataset.stats["y_scale"]
        
        
        self.f = torch.from_numpy(f.copy()).to(device, non_blocking=True)
        self.g = torch.from_numpy(g.copy()).to(device, non_blocking=True)
        self.y = torch.from_numpy(y.copy()).to(device, non_blocking=True)
        self.batch_size, self.shuffle = batch_size, shuffle

    def __len__(self) -> int:
        return math.ceil(self.f.shape[0] / self.batch_size)

    def __iter__(self):
        n = self.f.shape[0]
        order = (
            torch.randperm(n, device=self.f.device)
            if self.shuffle
            else torch.arange(n, device=self.f.device)
        )
        for begin in range(0, n, self.batch_size):
            row = order[begin : begin + self.batch_size]
            yield self.f.index_select(0, row), self.g.index_select(0, row), self.y.index_select(
                0, row
            )

def _resident_bytes(dataset: DarcyMemmapDataset) -> int:
    return len(dataset) * (dataset.f.shape[1] + 2 + dataset.y.shape[1]) * 4


def make_loader(
    task: Literal["0", "1", "2"],
    split: str,
    stats: dict[str, float],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    resident: str,
    workers: int,
) -> DataLoader | ResidentBatchLoader:
    
    dataset = DarcyMemmapDataset(task, split, stats)
    use_resident = resident == "gpu"
    if resident == "auto" and device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        use_resident = _resident_bytes(dataset) <= free // 4
    if use_resident:
        if device.type != "cuda":
            print(
                "[data] GPU-resident data requested without CUDA; using memmap loader", flush=True
            )
        else:
            key = (
                task,
                split,
                batch_size,
                shuffle,
                str(device),
                stats["f_mean"],
                stats["f_std"],
                stats["g_mean"],
                stats["g_std"],
                stats["y_scale"],
            )
            cached = _RESIDENT_LOADERS.get(key)
            if cached is not None:
                return cached
            print(
                f"[data] {split}: resident GPU tensors ({_resident_bytes(dataset) / 2**20:.1f} MiB)",
                flush=True,
            )
            loader = ResidentBatchLoader(dataset, batch_size, shuffle, device)
            _RESIDENT_LOADERS[key] = loader
            return loader
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **options)


def task_stats(task: str, cache: Path | None = None) -> dict[str, float]:
    cache = prepare_cache() if cache is None else cache
    train = np.load(cache / "split_train.npy", mmap_mode="r")
    f = np.load(cache / "f_samples.npy", mmap_mode="r")
    g1, g2 = np.load(cache / "g1.npy", mmap_mode="r"), np.load(cache / "g2.npy", mmap_mode="r")
    y = np.load(cache / f"{TASK_TARGET[task]}.npy", mmap_mode="r")
    
    
    f_train = np.asarray(f[train], dtype=np.float32)
    y_train = np.asarray(y[train], dtype=np.float32)
    g_train = np.concatenate([np.asarray(g1[train]), np.asarray(g2[train])])
    result = {
        "f_mean": float(f_train.mean()),
        "f_std": float(f_train.std() + 1e-6),
        "g_mean": float(g_train.mean()),
        "g_std": float(g_train.std() + 1e-6),
        "y_scale": float(np.max(np.abs(y_train)) + 1e-8),
    }
    del f_train, y_train, g_train
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def config_dict(config: FeatureConfig) -> dict:
    return asdict(config)
