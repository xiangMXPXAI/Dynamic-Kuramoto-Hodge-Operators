"""Shared archive, geometry, features and runtime utilities for torus C0/C1/C2 tasks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from torch import Tensor, nn
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    velocity: bool = True
    lpe: bool = False
    heat: bool = False
    global_broadcast: bool = False
    torus_fourier: bool = False


FEATURE_CONFIGS = {
    "native": FeatureConfig("native", velocity=True),
    "spectral": FeatureConfig("spectral", velocity=True, lpe=True),
    "diffusion_spectral": FeatureConfig("diffusion_spectral", velocity=True, lpe=True, heat=True),
    "flow_global_spectral": FeatureConfig(
        "flow_global_spectral", velocity=True, lpe=True, heat=True, global_broadcast=True
    ),
    "conditioned": FeatureConfig(
        "conditioned",
        velocity=True,
        lpe=True,
        heat=True,
        global_broadcast=True,
        torus_fourier=True,
    ),
}


CHECKPOINT_PROFILE_ALIASES = {
    "velocity_012_heat_lpe_global_fourier": "conditioned",
}

def checkpoint_feature_config(name: str) -> FeatureConfig:
    

    return FEATURE_CONFIGS[CHECKPOINT_PROFILE_ALIASES.get(name, name)]


def resolve_device(value: str) -> torch.device:
    if value in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if value == "cuda":
        print("[device] CUDA unavailable; using CPU", flush=True)
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def splits(count: int, seed: int = 42) -> dict[str, np.ndarray]:
    if count < 10:
        raise ValueError("Need at least ten samples for a train/val/test split")
    rng = np.random.default_rng(seed)
    index = rng.permutation(count)
    ntest = int(0.20 * count)
    nval = int(0.15 * (count - ntest))
    return {
        "train": np.sort(index[: count - ntest - nval]),
        "val": np.sort(index[count - ntest - nval : count - ntest]),
        "test": np.sort(index[count - ntest :]),
    }


class TorusGeometry:
    

    def __init__(self, root: Path, lpe_dim: int = 8) -> None:
        self.root = Path(root)
        meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.meta, self.lpe_dim = meta, lpe_dim
        for name in (
            "points",
            "faces",
            "normals",
            "edges",
            "face_edges",
            "face_signs",
            "edge_vec",
            "edge_mid",
            "edge_length",
            "face_area",
            "face_normal",
            "face_mid",
            "node_area",
            "velocity0",
            "velocity1",
        ):
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        self.n0, self.n1, self.n2 = len(self.points), len(self.edges), len(self.faces)
        self.tail, self.head = np.asarray(self.edges[:, 0]), np.asarray(self.edges[:, 1])

        self.edge_dir = np.array(self.edge_vec, dtype=np.float32, copy=True)
        self.edge_length = np.empty((self.n1, 0), dtype=np.float32)
        self.node_pos, self.edge_pos, self.face_pos = (
            self._norm(np.asarray(x)) for x in (self.points, self.edge_mid, self.face_mid)
        )
        self.d0, self.d1 = self._incidence()
        self.lpe: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.harmonic: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        radius = np.linalg.norm(np.asarray(self.points[:, :2]), axis=1)
        R = float(radius.mean())
        theta = np.arctan2(self.points[:, 1], self.points[:, 0])
        phi = np.arctan2(self.points[:, 2], radius - R)
        self.torus_fourier = np.stack(
            [
                item
                for a, b in ((1, 0), (0, 1), (1, 1), (1, -1))
                for item in (np.cos(a * theta + b * phi), np.sin(a * theta + b * phi))
            ],
            1,
        ).astype(np.float32)

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return ((x - x.mean(0)) / (x.std(0) + 1e-6)).astype(np.float32)

    def _incidence(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        d0 = sparse.csr_matrix(
            (
                np.tile(np.asarray([-1.0, 1.0], np.float32), self.n1),
                (np.repeat(np.arange(self.n1), 2), np.asarray(self.edges).reshape(-1)),
            ),
            shape=(self.n1, self.n0),
        )
        rows = np.repeat(np.arange(self.n2), 3)
        d1 = sparse.csr_matrix(
            (
                np.asarray(self.face_signs).reshape(-1),
                (rows, np.asarray(self.face_edges).reshape(-1)),
            ),
            shape=(self.n2, self.n1),
        )
        product = d1 @ d0
        if product.nnz and np.max(np.abs(product.data)) > 1e-6:
            raise RuntimeError("d1 d0 != 0")
        return d0, d1

    def _basis(self, lap: sparse.spmatrix, needed: int, positive: bool) -> np.ndarray:
        n = lap.shape[0]
        if positive:
            k = min(max(needed + 4, 6), n - 2)
            value, vector = eigsh(lap.astype(np.float64), k=k, which="SM", tol=1e-7, maxiter=40000)
        else:
            
            
            
            k = min(max(needed + 2, 4), n - 2)
            value, vector = eigsh(
                lap.astype(np.float64), k=k, sigma=-1e-6, which="LM", tol=1e-9, maxiter=40000
            )
        order = np.argsort(value)
        value, vector = value[order], vector[:, order]

        zero_tol = 1e-5
        keep = (
            np.flatnonzero(value > zero_tol)[:needed]
            if positive
            else np.flatnonzero(value <= zero_tol)[:needed]
        )
        if len(keep) != needed:
            raise RuntimeError(
                f"Insufficient {'positive' if positive else 'harmonic'} modes: wanted {needed}, found {len(keep)}"
            )
        out = vector[:, keep]
        out *= np.sign(out[np.abs(out).argmax(0), np.arange(out.shape[1])])[None]
        return out.astype(np.float32)

    def ensure_lpe(self) -> None:
        if self.lpe is not None:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"lpe_k{self.lpe_dim}.npz"
        if path.exists():
            saved = np.load(path)
            candidate = (saved["lpe0"], saved["lpe1"], saved["lpe2"])
            if all(x.shape[1] == self.lpe_dim for x in candidate):
                self.lpe = tuple(x.astype(np.float32) for x in candidate)
                return
        l0, l1, l2 = (
            self.d0.T @ self.d0,
            self.d0 @ self.d0.T + self.d1.T @ self.d1,
            self.d1 @ self.d1.T,
        )
        self.lpe = (
            self._basis(l0, self.lpe_dim, True),
            self._basis(l1, self.lpe_dim, True),
            self._basis(l2, self.lpe_dim, True),
        )
        np.savez_compressed(path, lpe0=self.lpe[0], lpe1=self.lpe[1], lpe2=self.lpe[2])

    def ensure_harmonic(self) -> None:
        if self.harmonic is not None:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / "harmonic_unweighted_v2_shiftinvert.npz"
        if path.exists():
            saved = np.load(path)
            got = (saved["h0"], saved["h1"], saved["h2"])
            if tuple(x.shape[1] for x in got) == (1, 2, 1):
                self.harmonic = tuple(x.astype(np.float32) for x in got)
                return
        l0, l1, l2 = (
            self.d0.T @ self.d0,
            self.d0 @ self.d0.T + self.d1.T @ self.d1,
            self.d1 @ self.d1.T,
        )
        self.harmonic = (
            self._basis(l0, 1, False),
            self._basis(l1, 2, False),
            self._basis(l2, 1, False),
        )
        np.savez_compressed(path, h0=self.harmonic[0], h1=self.harmonic[1], h2=self.harmonic[2])

DarcyGeometry = TorusGeometry

class IncidenceOps(nn.Module):
    def __init__(self, geo: TorusGeometry) -> None:
        super().__init__()
        self.n0, self.n1 = geo.n0, geo.n1
        
        
        
        self.register_buffer(
            "tail", torch.from_numpy(np.array(geo.tail, dtype=np.int64, copy=True))
        )
        self.register_buffer(
            "head", torch.from_numpy(np.array(geo.head, dtype=np.int64, copy=True))
        )
        self.register_buffer(
            "face_edges", torch.from_numpy(np.array(geo.face_edges, dtype=np.int64, copy=True))
        )
        self.register_buffer(
            "face_signs", torch.from_numpy(np.array(geo.face_signs, dtype=np.float32, copy=True))
        )

    def d0(self, x: Tensor) -> Tensor:
        return x[:, self.head] - x[:, self.tail]

    def delta1(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], self.n0, x.shape[-1], device=x.device, dtype=x.dtype)
        out.index_add_(1, self.tail, -x)
        out.index_add_(1, self.head, x)
        return out

    def d1(self, x: Tensor) -> Tensor:
        return (x[:, self.face_edges] * self.face_signs[None, :, :, None]).sum(2)

    def delta2(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], self.n1, x.shape[-1], device=x.device, dtype=x.dtype)
        for local in range(3):
            out.index_add_(1, self.face_edges[:, local], x * self.face_signs[None, :, local, None])
        return out


class FormDataset(Dataset):
    def __init__(
        self,
        root: Path,
        task: str,
        split: str,
        stats: dict[str, float],
        indices: np.ndarray | None = None,
    ) -> None:
        self.root, self.task, self.stats = Path(root), task, stats
        self.source = np.load(self.root / "u0.npy", mmap_mode="r")
        self.target = np.load(
            self.root / {"0": "uT0.npy", "1": "qT1.npy", "2": "mT2.npy"}[task], mmap_mode="r"
        )
        self.indices = splits(len(self.source))[split] if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        idx = self.indices[i]
        source = np.array(self.source[idx], dtype=np.float32, copy=True) / self.stats["x_scale"]
        target = np.array(self.target[idx], dtype=np.float32, copy=True) / self.stats["y_scale"]
        return torch.from_numpy(source), torch.from_numpy(target)


def task_stats(root: Path, task: str) -> dict[str, float]:
    source = np.load(Path(root) / "u0.npy", mmap_mode="r")
    target = np.load(
        Path(root) / {"0": "uT0.npy", "1": "qT1.npy", "2": "mT2.npy"}[task], mmap_mode="r"
    )
    idx = splits(len(source))["train"]
    return {
        "x_scale": float(np.max(np.abs(source[idx])) + 1e-8),
        "y_scale": float(np.max(np.abs(target[idx])) + 1e-8),
    }


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    return (
        torch.linalg.vector_norm(pred - target, dim=1)
        / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-10)
    ).mean()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
