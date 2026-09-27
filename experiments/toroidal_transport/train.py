"""Train the toroidal transport model and structural ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs" / "dkho"
DATA_PATH = ROOT / "data" / "torus_transport_v1.pkl"
REPOSITORY_ROOT = ROOT.parents[1]

def portable_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"path must be inside the repository: {candidate}") from error

def portable_arguments(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = portable_path(value)
        elif isinstance(value, str) and Path(value).is_absolute():
            values[key] = portable_path(value)
        else:
            values[key] = value
    return values

CACHE_ROOT = ROOT / "cache"


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    velocity: bool
    face: bool
    heat: bool
    lpe: bool
    global_broadcast: bool = False
    torus_fourier: bool = False

FEATURE_CONFIGS = {

    "native": FeatureConfig("native", False, False, False, False),
    "flow": FeatureConfig("flow", True, False, False, False),
    "flow_diffusion": FeatureConfig("flow_diffusion", True, True, True, False),
    "flow_diffusion_spectral": FeatureConfig("flow_diffusion_spectral", True, True, True, True),
    "flow_global_spectral": FeatureConfig(
        "flow_global_spectral", True, True, True, True, global_broadcast=True
    ),
    "conditioned": FeatureConfig(
        "conditioned", True, True, True, True, global_broadcast=True, torus_fourier=True
    ),
}
STRUCTURE_VARIANTS = ("full", "no_dirac", "no_phase", "no_harmonic")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA requested but unavailable; falling back to CPU.", flush=True)
    return torch.device("cpu")


def oriented_complex(
    faces: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    
    edge_map: dict[tuple[int, int], int] = {}
    face_entries: list[tuple[int, int, int]] = []
    for face in faces:
        row: list[tuple[int, int]] = []
        for a, b in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (min(a, b), max(a, b))
            if key not in edge_map:
                edge_map[key] = len(edge_map)
            row.append((edge_map[key], 1 if (a, b) == key else -1))
        face_entries.extend(
            (fi, ei, sign) for fi, (ei, sign) in [(len(face_entries) // 3, item) for item in row]
        )
    edges = np.asarray(list(edge_map.keys()), dtype=np.int64)
    ne = len(edges)
    d0 = sparse.lil_matrix((ne, n_nodes), dtype=np.float32)
    for e, (a, b) in enumerate(edges):
        d0[e, a] = -1.0
        d0[e, b] = 1.0
    rr, cc, vv = zip(*face_entries)
    d1 = sparse.csr_matrix(
        (np.asarray(vv, np.float32), (np.asarray(rr), np.asarray(cc))), shape=(len(faces), ne)
    )
    return edges, d0.tocsr(), d1


def tri_geometry(
    points: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normals = np.cross(b - a, c - a)
    doubled = np.linalg.norm(normals, axis=1, keepdims=True)
    area = 0.5 * doubled[:, 0]
    normals = normals / np.maximum(doubled, 1e-12)
    return area.astype(np.float32), normals.astype(np.float32), ((a + b + c) / 3).astype(np.float32)


class Geometry:
    def __init__(
        self, points: np.ndarray, faces: np.ndarray, normals: np.ndarray, lpe_dim: int = 8
    ):
        self.points = points.astype(np.float32)
        self.faces = faces.astype(np.int64)
        self.normals = normals.astype(np.float32)
        self.n0 = len(points)
        self.edges, d0, d1 = oriented_complex(self.faces, self.n0)
        self.n1, self.n2 = d0.shape[0], d1.shape[0]
        assert np.max(np.abs((d1 @ d0).data), initial=0.0) < 1e-6
        self.d0_np, self.d1_np = d0, d1
        self.face_edges = d1.indices.reshape(self.n2, 3).astype(np.int64)
        self.face_signs = d1.data.reshape(self.n2, 3).astype(np.float32)
        self.edge_mid = (
            (self.points[self.edges[:, 0]] + self.points[self.edges[:, 1]]) / 2
        ).astype(np.float32)
        self.edge_vec = (self.points[self.edges[:, 1]] - self.points[self.edges[:, 0]]).astype(
            np.float32
        )
        self.edge_len = np.linalg.norm(self.edge_vec, axis=1, keepdims=True).astype(np.float32)
        self.face_area, self.face_normal, self.face_mid = tri_geometry(self.points, self.faces)
        self.node_area = np.zeros(self.n0, dtype=np.float32)
        for column in range(3):
            np.add.at(self.node_area, self.faces[:, column], self.face_area / 3.0)
        self.node_pos = self._norm_coordinates(self.points)
        self.edge_pos = self._norm_coordinates(self.edge_mid)
        self.face_pos = self._norm_coordinates(self.face_mid)
        self.velocity_node = self._background_velocity()
        self.velocity_edge = (
            (
                0.5
                * (self.velocity_node[self.edges[:, 0]] + self.velocity_node[self.edges[:, 1]])
                * self.edge_vec
            )
            .sum(1, keepdims=True)
            .astype(np.float32)
        )
        self.speed_node = np.linalg.norm(self.velocity_node, axis=1, keepdims=True).astype(
            np.float32
        )
        self.div_velocity = (d0.T @ self.velocity_edge).astype(np.float32)
        self.curl_velocity = (d1 @ self.velocity_edge).astype(np.float32)
        
        
        radial_xy = np.linalg.norm(self.points[:, :2], axis=1)
        major_radius = float(radial_xy.mean())
        theta = np.arctan2(self.points[:, 1], self.points[:, 0])
        phi = np.arctan2(self.points[:, 2], radial_xy - major_radius)
        modes = ((1, 0), (0, 1), (1, 1), (1, -1))
        self.torus_fourier_basis = np.stack(
            [
                item
                for m, n in modes
                for item in (np.cos(m * theta + n * phi), np.sin(m * theta + n * phi))
            ],
            axis=1,
        ).astype(np.float32)
        self.lpe_dim = lpe_dim
        self.lpe: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _norm_coordinates(x: np.ndarray) -> np.ndarray:
        return ((x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)).astype(
            np.float32
        )

    def _background_velocity(self) -> np.ndarray:
        
        raw = np.stack([-self.points[:, 1], self.points[:, 0], np.zeros(self.n0)], axis=1)
        return (raw - (raw * self.normals).sum(1, keepdims=True) * self.normals).astype(np.float32)

    def _lpe(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        def basis(lap: sparse.spmatrix, count: int) -> np.ndarray:
            n = lap.shape[0]
            
            
            vals, vecs = eigsh(
                lap.astype(np.float64), k=min(count + 3, n - 2), which="SM", tol=1e-4
            )
            order = np.argsort(vals)
            vals, vecs = vals[order], vecs[:, order]
            keep = np.flatnonzero(vals > 1e-7)[:count]
            if len(keep) != count:
                raise RuntimeError(
                    f"Insufficient positive eigenvectors: wanted {count}, found {len(keep)}"
                )
            out = vecs[:, keep]
            
            
            out *= np.sign(out[np.abs(out).argmax(axis=0), np.arange(out.shape[1])])[None, :]
            return out.astype(np.float32)

        l0 = self.d0_np.T @ self.d0_np
        l1 = self.d0_np @ self.d0_np.T + self.d1_np.T @ self.d1_np
        l2 = self.d1_np @ self.d1_np.T
        return basis(l0, k), basis(l1, k), basis(l2, k)

    def ensure_lpe(self) -> None:
        if self.lpe is None:
            
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            cache = CACHE_ROOT / f"geometry_lpe_k{self.lpe_dim}.npz"
            if cache.exists():
                saved = np.load(cache)
                candidate = (saved["lpe0"], saved["lpe1"], saved["lpe2"])
                if all(array.ndim == 2 and array.shape[1] == self.lpe_dim for array in candidate):
                    self.lpe = candidate
                else:
                    self.lpe = self._lpe(self.lpe_dim)
                    np.savez_compressed(cache, lpe0=self.lpe[0], lpe1=self.lpe[1], lpe2=self.lpe[2])
            else:
                self.lpe = self._lpe(self.lpe_dim)
                np.savez_compressed(cache, lpe0=self.lpe[0], lpe1=self.lpe[1], lpe2=self.lpe[2])

    def tensors(self, device: torch.device) -> dict[str, Tensor]:
        arr = {
            "p0": torch.from_numpy(self.node_pos),
            "p1": torch.from_numpy(self.edge_pos),
            "p2": torch.from_numpy(self.face_pos),
            "v1": torch.from_numpy(self.velocity_edge),
            "speed0": torch.from_numpy(self.speed_node),
            "divv0": torch.from_numpy(self.div_velocity),
            "curlv2": torch.from_numpy(self.curl_velocity),
            "face_area": torch.from_numpy(self.face_area[:, None]),
            "face_normal": torch.from_numpy(self.face_normal),
            "node_area": torch.from_numpy(self.node_area[:, None]),
            "torus_fourier": torch.from_numpy(self.torus_fourier_basis),
        }
        if self.lpe is not None:
            arr.update(
                {
                    "lpe0": torch.from_numpy(self.lpe[0]),
                    "lpe1": torch.from_numpy(self.lpe[1]),
                    "lpe2": torch.from_numpy(self.lpe[2]),
                }
            )
        return {key: value.to(device) for key, value in arr.items()}


class IncidenceOps(nn.Module):
    

    def __init__(self, geo: Geometry, device: torch.device):
        super().__init__()
        self.n0, self.n1 = geo.n0, geo.n1
        self.register_buffer("tail", torch.from_numpy(geo.edges[:, 0]).to(device))
        self.register_buffer("head", torch.from_numpy(geo.edges[:, 1]).to(device))
        self.register_buffer("face_edges", torch.from_numpy(geo.face_edges).to(device))
        self.register_buffer("face_signs", torch.from_numpy(geo.face_signs).to(device))

    def d0(self, x: Tensor) -> Tensor:
        return x[:, self.head] - x[:, self.tail]

    def delta1(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], self.n0, x.shape[2], dtype=x.dtype, device=x.device)
        out.index_add_(1, self.tail, -x)
        out.index_add_(1, self.head, x)
        return out

    def d1(self, x: Tensor) -> Tensor:
        return (x[:, self.face_edges] * self.face_signs[None, :, :, None]).sum(2)

    def delta2(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], self.n1, x.shape[2], dtype=x.dtype, device=x.device)
        for local in range(3):
            out.index_add_(1, self.face_edges[:, local], x * self.face_signs[None, :, local, None])
        return out


class FeatureBuilder(nn.Module):
    def __init__(self, geo: Geometry, cfg: FeatureConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.geo = geo
        if cfg.lpe:
            geo.ensure_lpe()
        for name, value in geo.tensors(device).items():
            self.register_buffer(name, value)
        self.ops = IncidenceOps(geo, device)
        self.dims = self._dims()

    def _dims(self) -> tuple[int, int, int]:
        d0, d1, d2 = 4, 4, 1  
        if self.cfg.velocity:
            d0 += 3  
            d1 += 2  
        if self.cfg.face:
            d2 += 9  
        if self.cfg.heat:
            d0 += 2
        if self.cfg.lpe:
            assert self.geo.lpe is not None
            d0 += self.geo.lpe[0].shape[1]
            d1 += self.geo.lpe[1].shape[1]
            d2 += self.geo.lpe[2].shape[1]
        if self.cfg.global_broadcast:
            
            d0 += 2
            d1 += 2
            d2 += 2
        if self.cfg.torus_fourier:
            
            d0 += 8
            d1 += 8
            d2 += 8
        return d0, d1, d2

    @staticmethod
    def _expand(x: Tensor, b: int) -> Tensor:
        return x.unsqueeze(0).expand(b, -1, -1)

    def _global_conditions(self, u_scalar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        
        w = self.node_area[:, 0] / self.node_area[:, 0].sum().clamp_min(1e-12)
        mean = (u_scalar * w).sum(1)
        centered = u_scalar - mean[:, None]
        energy = torch.sqrt((u_scalar.square() * w).sum(1).clamp_min(1e-12))
        scale = torch.sqrt((centered.square() * w).sum(1).clamp_min(1e-12))
        grad = self.ops.d0(u_scalar.unsqueeze(-1)).squeeze(-1)
        grad_scale = torch.sqrt(grad.square().mean(1).clamp_min(1e-12))
        stats = torch.stack([mean, energy, scale, grad_scale], dim=-1)
        fourier = (u_scalar.unsqueeze(-1) * w[None, :, None] * self.torus_fourier[None]).sum(1)
        topo = torch.cat([stats, fourier], dim=-1) if self.cfg.torus_fourier else stats
        return stats, fourier, topo

    def forward(self, u: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        b = u.shape[0]
        stats, fourier, topology_condition = self._global_conditions(u)
        u = u.unsqueeze(-1)
        grad = self.ops.d0(u)
        f0 = [u, self._expand(self.p0, b)]
        f1 = [grad, self._expand(self.p1, b)]
        
        
        
        f2 = [torch.zeros(b, self.geo.n2, 1, device=u.device)]
        if self.cfg.velocity:
            ue = 0.5 * (u[:, self.geo.edges[:, 0]] + u[:, self.geo.edges[:, 1]])
            q = ue * self._expand(self.v1, b)
            divq = self.ops.delta1(q)
            f0 += [self._expand(self.divv0, b), divq, self._expand(self.speed0, b)]
            f1 += [self._expand(self.v1, b), q]
            if self.cfg.face:
                curlq = self.ops.d1(q)
                f2 += [
                    self._expand(self.curlv2, b),
                    curlq,
                    self._expand(self.p2, b),
                    self._expand(self.face_area, b),
                    self._expand(self.face_normal, b),
                ]
        if self.cfg.heat:
            lap = self.ops.delta1(grad)
            heat1 = u - 0.03 * lap
            heat2 = heat1 - 0.03 * self.ops.delta1(self.ops.d0(heat1))
            f0 += [heat1, heat2]
        if self.cfg.lpe:
            f0 += [self._expand(self.lpe0, b)]
            f1 += [self._expand(self.lpe1, b)]
            f2 += [self._expand(self.lpe2, b)]
        if self.cfg.global_broadcast:
            
            
            local_global = stats[:, None, 1:3]
            f0 += [local_global.expand(-1, self.geo.n0, -1)]
            f1 += [local_global.expand(-1, self.geo.n1, -1)]
            f2 += [local_global.expand(-1, self.geo.n2, -1)]
        if self.cfg.torus_fourier:
            f0 += [fourier[:, None].expand(-1, self.geo.n0, -1)]
            f1 += [fourier[:, None].expand(-1, self.geo.n1, -1)]
            f2 += [fourier[:, None].expand(-1, self.geo.n2, -1)]
        return (torch.cat(f0, -1), torch.cat(f1, -1), torch.cat(f2, -1)), topology_condition


def point_mlp(fin: int, hidden: int, fout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(fin, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, fout)
    )


class DiracKuramotoLayer(nn.Module):
    def __init__(self, hidden: int, channels: int, microsteps: int, slow_initial_skip: bool = True):
        super().__init__()
        self.microsteps, self.channels = microsteps, channels
        self.slow_initial_skip = slow_initial_skip
        self.omega = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.gate = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.lag = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.log_sigma = nn.Parameter(torch.zeros(3))
        self.dt_logit = nn.Parameter(torch.tensor(-1.35))
        self.phase = nn.ModuleList([nn.Linear(channels * 2, hidden) for _ in range(3)])
        
        
        
        
        self.update = nn.ModuleList([point_mlp(hidden * 2, hidden, hidden) for _ in range(3)])
        
        
        
        if slow_initial_skip:
            self.initial_skip = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(3)]
            )
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])

    def forward(
        self, z: list[Tensor], z_initial: list[Tensor], theta: list[Tensor], ops: IncidenceOps
    ) -> tuple[list[Tensor], list[Tensor]]:
        dt = 0.02 + 0.10 * torch.sigmoid(self.dt_logit)
        sigma = F.softplus(self.log_sigma) + 1e-4
        for _ in range(self.microsteps):
            z0, z1, z2 = z
            t0, t1, t2 = theta
            g01 = ops.d0(t0) - torch.tanh(self.gate[1](z1)) * t1
            a10 = ops.delta1(t1) + torch.tanh(self.gate[0](z0)) * t0
            g12 = ops.d1(t1) - torch.tanh(self.gate[2](z2)) * t2
            a21 = ops.delta2(t2) + torch.tanh(self.gate[1](z1)) * t1
            lag01, lag10, lag12, lag21 = (
                0.5 * torch.tanh(self.lag[1](z1)),
                0.5 * torch.tanh(self.lag[0](z0)),
                0.5 * torch.tanh(self.lag[2](z2)),
                0.5 * torch.tanh(self.lag[1](z1)),
            )
            dot0 = -self.omega[0](z0) - sigma[0] * ops.delta1(torch.sin(g01 - lag01))
            dot1 = -self.omega[1](z1) - sigma[1] * (
                ops.d0(torch.sin(a10 - lag10)) + ops.delta2(torch.sin(g12 - lag12))
            )
            dot2 = -self.omega[2](z2) - sigma[2] * ops.d1(torch.sin(a21 - lag21))
            theta = [t0 + dt * dot0, t1 + dt * dot1, t2 + dt * dot2]
        t0, t1, t2 = theta
        out = []
        for r, (zr, z0r, tr) in enumerate(zip(z, z_initial, theta)):
            phase = self.phase[r](torch.cat([torch.cos(tr), torch.sin(tr)], -1))
            value = zr + self.update[r](torch.cat([zr, phase], -1))
            if self.slow_initial_skip:
                value = value + self.initial_skip[r](z0r)
            out.append(self.norm[r](value))
        return out, theta


class NoDiracKuramotoLayer(DiracKuramotoLayer):
    

    def forward(
        self, z: list[Tensor], z_initial: list[Tensor], theta: list[Tensor], ops: IncidenceOps
    ) -> tuple[list[Tensor], list[Tensor]]:
        del ops
        dt = 0.02 + 0.10 * torch.sigmoid(self.dt_logit)
        sigma = F.softplus(self.log_sigma) + 1e-4
        for _ in range(self.microsteps):
            theta = [
                theta[r]
                + dt
                * (
                    -self.omega[r](z[r])
                    - sigma[r]
                    * torch.sin(
                        torch.tanh(self.gate[r](z[r])) * theta[r]
                        - 0.5 * torch.tanh(self.lag[r](z[r]))
                    )
                )
                for r in range(3)
            ]
        out = []
        for r, (zr, z0r, tr) in enumerate(zip(z, z_initial, theta)):
            value = zr + self.update[r](
                torch.cat([zr, self.phase[r](torch.cat([torch.cos(tr), torch.sin(tr)], -1))], -1)
            )
            if self.slow_initial_skip:
                value = value + self.initial_skip[r](z0r)
            out.append(self.norm[r](value))
        return out, theta


class FeatureDiracLayer(nn.Module):
    

    def __init__(self, hidden: int, channels: int, slow_initial_skip: bool = True):
        super().__init__()
        self.channels, self.slow_initial_skip = channels, slow_initial_skip
        
        
        self.omega = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.gate = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.lag = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.log_sigma = nn.Parameter(torch.zeros(3))
        self.dt_logit = nn.Parameter(torch.tensor(-1.35))
        self.channel_readout = nn.ModuleList([nn.Linear(2 * channels, hidden) for _ in range(3)])
        self.update = nn.ModuleList([point_mlp(2 * hidden, hidden, hidden) for _ in range(3)])
        if slow_initial_skip:
            self.initial_skip = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False) for _ in range(3)]
            )
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])

    def carrier(self, z: list[Tensor]) -> list[Tensor]:
        return [
            torch.sigmoid(self.gate[r](z[r])) * torch.tanh(self.omega[r](z[r]))
            + torch.tanh(self.lag[r](z[r]))
            for r in range(3)
        ]

    def decoder_channels(self, value: Tensor, rank: int) -> Tensor:
        carrier = torch.sigmoid(self.gate[rank](value)) * torch.tanh(
            self.omega[rank](value)
        ) + torch.tanh(self.lag[rank](value))
        return torch.cat([carrier, torch.tanh(self.gate[rank](value))], -1)

    def forward(self, z: list[Tensor], z_initial: list[Tensor], ops: IncidenceOps) -> list[Tensor]:
        x0, x1, x2 = self.carrier(z)
        sigma = F.softplus(self.log_sigma) + 1e-4
        messages = [
            sigma[0] * ops.delta1(x1),
            sigma[1] * (ops.d0(x0) + ops.delta2(x2)),
            sigma[2] * ops.d1(x1),
        ]
        dt = 0.02 + 0.10 * torch.sigmoid(self.dt_logit)
        out = []
        for r, (zr, z0r, xr, message) in enumerate(zip(z, z_initial, (x0, x1, x2), messages)):
            value = zr + dt * self.update[r](
                torch.cat([zr, self.channel_readout[r](torch.cat([message, xr], -1))], -1)
            )
            if self.slow_initial_skip:
                value = value + self.initial_skip[r](z0r)
            out.append(self.norm[r](value))
        return out


class TDKHO(nn.Module):
    def __init__(
        self,
        geo: Geometry,
        cfg: FeatureConfig,
        hidden: int,
        layers: int,
        channels: int,
        microsteps: int,
        device: torch.device,
        phase_init: str = "zero",
        harmonic_active: bool = False,
        slow_initial_skip: bool = True,
        variant: str = "full",
    ):
        super().__init__()
        if phase_init not in {"zero", "learnable"}:
            raise ValueError(f"Unknown phase initialization: {phase_init}")
        if variant not in STRUCTURE_VARIANTS:
            raise ValueError(f"Unknown structural variant: {variant}")
        self.phase_init_mode = phase_init
        self.variant = variant
        self.uses_phase = variant in {"full", "no_dirac", "no_harmonic"}
        self.harmonic_active = harmonic_active and variant != "no_harmonic"
        self.features = FeatureBuilder(geo, cfg, device)
        self.enc = nn.ModuleList([point_mlp(d, hidden, hidden) for d in self.features.dims])
        self.slow_initial_skip = slow_initial_skip
        layer = (
            DiracKuramotoLayer
            if variant in {"full", "no_harmonic"}
            else NoDiracKuramotoLayer if variant == "no_dirac" else FeatureDiracLayer
        )
        self.layers = nn.ModuleList(
            [
                (
                    layer(hidden, channels, microsteps, slow_initial_skip)
                    if layer is not FeatureDiracLayer
                    else layer(hidden, channels, slow_initial_skip)
                )
                for _ in range(layers)
            ]
        )
        if self.uses_phase and phase_init == "learnable":
            
            
            self.theta_init = nn.ModuleList([nn.Linear(hidden, channels) for _ in range(3)])
        self.decoder = point_mlp(hidden + 2 * channels, hidden, 1)
        if self.harmonic_active:
            topology_dim = 12 if cfg.torus_fourier else 4
            self.topology_head = point_mlp(topology_dim, hidden, 1)

    def forward(self, u: Tensor) -> Tensor:
        fs, topology_condition = self.features(u)
        z = [encoder(feat) for encoder, feat in zip(self.enc, fs)]
        z_initial = list(z)
        if self.uses_phase:
            if self.phase_init_mode == "zero":
                theta = [
                    torch.zeros(
                        value.shape[0],
                        value.shape[1],
                        self.layers[0].channels,
                        device=value.device,
                        dtype=value.dtype,
                    )
                    for value in z
                ]
            else:
                theta = [
                    math.pi * torch.tanh(head(value)) for head, value in zip(self.theta_init, z)
                ]
            for layer in self.layers:
                z, theta = layer(z, z_initial, theta, self.features.ops)
            decoded = [z[0], torch.cos(theta[0]), torch.sin(theta[0])]
        else:
            for layer in self.layers:
                z = layer(z, z_initial, self.features.ops)
            decoded = [z[0], self.layers[-1].decoder_channels(z[0], 0)]
        local = self.decoder(torch.cat(decoded, -1)).squeeze(-1)
        if not self.harmonic_active:
            return local
        
        
        local = local - local.mean(1, keepdim=True)
        return local + self.topology_head(topology_condition) / math.sqrt(self.features.geo.n0)


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(((pred - target) ** 2).sum(1) / (target.square().sum(1) + 1e-10)).mean()


def train_loss(
    model: TDKHO, x: Tensor, y: Tensor, gradient_weight: float = 0.0
) -> tuple[Tensor, Tensor]:
    pred = model(x)
    data = relative_l2(pred, y)
    gp = model.features.ops.d0(pred.unsqueeze(-1))
    gy = model.features.ops.d0(y.unsqueeze(-1))
    grad = relative_l2(gp.squeeze(-1), gy.squeeze(-1))
    return data + gradient_weight * grad, pred


@torch.no_grad()
def predict(model: TDKHO, x: Tensor, batch: int = 16) -> Tensor:
    model.eval()
    return torch.cat([model(x[i : i + batch]) for i in range(0, len(x), batch)], 0)


@torch.no_grad()
def metrics(model: TDKHO, x: Tensor, y: Tensor, y_scale: float) -> tuple[dict, Tensor]:
    p = predict(model, x)
    p0, y0 = p * y_scale, y * y_scale
    diff = p0 - y0
    rel = torch.sqrt(diff.square().sum(1) / (y0.square().sum(1) + 1e-10))
    gp = model.features.ops.d0(p0.unsqueeze(-1)).squeeze(-1)
    gy = model.features.ops.d0(y0.unsqueeze(-1)).squeeze(-1)
    grad_rel = torch.sqrt((gp - gy).square().sum(1) / (gy.square().sum(1) + 1e-10))
    dot = (gp * gy).sum(1)
    gf = ((1 + dot / (gp.square().sum(1).sqrt() * gy.square().sum(1).sqrt() + 1e-10)) / 2).mean()
    
    
    area = torch.from_numpy(model.features.geo.node_area).to(p0.device)
    mass_delta = ((p0 - y0) * area).sum(1).abs()
    l1_scale = (y0.abs() * area).sum(1).clamp_min(1e-8)
    ep, ey = gp.square().sum(1), gy.square().sum(1)
    energy = (ep - ey).abs() / (ey.abs() + 1e-8)
    thresholds = 0.5 * (y0.amin(1) + y0.amax(1))
    pm, ym = p0 >= thresholds[:, None], y0 >= thresholds[:, None]
    iou = (pm & ym).sum(1).float() / ((pm | ym).sum(1).float() + 1e-8)
    ss_res, ss_tot = diff.square().sum(), ((y0 - y0.mean()) ** 2).sum()
    result = {
        "mse": float(diff.square().mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(diff.square().mean().sqrt()),
        "relative_l2_mean": float(rel.mean()),
        "relative_l2_median": float(rel.median()),
        "relative_l2_std": float(rel.std(unbiased=False)),
        "r2": float(1 - ss_res / (ss_tot + 1e-10)),
        "gradient_relative_l2": float(grad_rel.mean()),
        "gradient_fidelity": float(gf),
        "area_mass_absolute_error": float(mass_delta.mean()),
        "area_mass_l1_normalized_error": float((mass_delta / l1_scale).mean()),
        "mean_gradient_energy_relative_error": float(energy.mean()),
        "levelset_iou_50": float(iou.mean()),
    }
    return result, p0.cpu()


def split_indices(n: int) -> dict[str, np.ndarray]:
    idx = np.arange(n)
    trval, te = train_test_split(idx, test_size=0.2, random_state=42)
    tr, va = train_test_split(trval, test_size=0.15, random_state=42)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def harmonic_diagnosis(x_all: np.ndarray, y_all: np.ndarray, train: np.ndarray) -> dict:
    
    n0 = y_all.shape[1]
    ax = x_all[train].mean(1) * math.sqrt(n0)
    ay = y_all[train].mean(1) * math.sqrt(n0)
    target_std = float(ay.std())
    target_rms = float(np.sqrt(np.mean(y_all[train] ** 2)))
    variation = target_std / max(target_rms, 1e-12)
    corr = (
        float(np.corrcoef(ax, ay)[0, 1]) if ax.std() > 1e-12 and ay.std() > 1e-12 else float("nan")
    )
    active = variation >= 1e-5
    return {
        "basis": "H0=span{1/sqrt(n0)}",
        "input_h0_std": float(ax.std()),
        "target_h0_std": target_std,
        "target_field_rms": target_rms,
        "relative_target_variation": variation,
        "input_target_h0_correlation": corr,
        "threshold": 1e-5,
        "constant_mode": not active,
        "harmonic_head_active": active,
    }


def run_one(
    cfg: FeatureConfig,
    seed: int,
    args: argparse.Namespace,
    geo: Geometry,
    data: dict,
    device: torch.device,
    output_root: Path,
    variant: str = "full",
) -> dict:
    run_dir = output_root / cfg.name / variant / f"seed_{seed}"
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists() and not args.rerun:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    x_all, y_all = trajectories[:, 0], trajectories[:, -1]
    splits = split_indices(len(x_all))
    harmonic = harmonic_diagnosis(x_all, y_all, splits["train"])
    x_scale = float(np.abs(x_all[splits["train"]]).max() + 1e-8)
    y_scale = float(np.abs(y_all[splits["train"]]).max() + 1e-8)
    x = torch.from_numpy(x_all / x_scale).to(device)
    y = torch.from_numpy(y_all / y_scale).to(device)
    loaders = DataLoader(
        TensorDataset(x[splits["train"]], y[splits["train"]]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = TDKHO(
        geo,
        cfg,
        args.hidden,
        args.layers,
        args.channels,
        args.microsteps,
        device,
        args.phase_init,
        harmonic["harmonic_head_active"],
        variant=variant,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    best, best_epoch, best_state = float("inf"), 0, None
    hist = {"train": [], "val": []}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loaders:
            optim.zero_grad(set_to_none=True)
            loss, _ = train_loss(model, xb, yb, args.gradient_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
        train_value = total / len(loaders)
        model.eval()
        with torch.no_grad():
            val_loss, _ = train_loss(model, x[splits["val"]], y[splits["val"]], args.gradient_loss)
        val_value = float(val_loss)
        hist["train"].append(train_value)
        hist["val"].append(val_value)
        if val_value < best:
            best, best_epoch = val_value, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"[{cfg.name}/{variant} seed={seed}] epoch {epoch:03d}/{args.epochs} train={train_value:.6f} val={val_value:.6f} elapsed={time.time()-start:.1f}s",
                flush=True,
            )
    model.load_state_dict(best_state)
    result, p = metrics(model, x[splits["test"]], y[splits["test"]], y_scale)
    architecture = {
        "version": "dkho_torus_v2_initial_skip",
        "structure_variant": variant,
        "uses_phase": model.uses_phase,
        "slow_initial_skip": True,
        "slow_update": (
            "LN(z_r + U_r([z_r, cos(theta_r), sin(theta_r)]) + S_r(z_r_initial))"
            if model.uses_phase
            else "LN(z_r + dt*U_r([z_r, direct_Dirac_message_r]) + S_r(z_r_initial))"
        ),
    }
    result.update(
        {
            "family": "DKHO",
            "task": "form_0",
            "feature_config": cfg.name,
            "config": cfg.name,
            "structure_variant": variant,
            "uses_phase": model.uses_phase,
            "seed": seed,
            "epochs_completed": args.epochs,
            "best_val_epoch": best_epoch,
            "best_val_loss": best,
            "wall_seconds": time.time() - start,
            "parameters": sum(q.numel() for q in model.parameters()),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "hodge_star": False,
            "boundary_features": False,
            "velocity_is_fixed_dataset_condition": True,
            "phase_initialization": args.phase_init if model.uses_phase else None,
            "architecture": architecture,
            **harmonic,
            "harmonic_head_active": model.harmonic_active,
        }
    )
    
    
    
    
    np.save(run_dir / "prediction_test_normalized.npy", (p / y_scale).numpy().astype(np.float32))
    np.save(
        run_dir / "target_test_normalized.npy",
        y[splits["test"]].detach().cpu().numpy().astype(np.float32),
    )
    np.save(run_dir / "test_indices.npy", splits["test"].astype(np.int64))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "args": portable_arguments(args),
            "harmonic_diagnosis": harmonic,
            "structure_variant": variant,
            "architecture": architecture,
        },
        run_dir / "best.pt",
    )
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(results: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for result in results:
        label = (
            result["config"]
            if result.get("structure_variant", "full") == "full"
            else f"{result['config']}/structure_{result['structure_variant']}"
        )
        by.setdefault(label, []).append(result)
    summary: dict[str, dict] = {}
    for name, rows in by.items():
        item = {"n_seeds": len(rows)}
        for key, value in rows[0].items():
            if isinstance(value, (int, float)) and key not in {
                "seed",
                "best_val_epoch",
                "epochs_completed",
            }:
                vals = np.asarray([r[key] for r in rows], dtype=float)
                item[key] = float(vals.mean())
                item[f"{key}_std"] = float(vals.std())
        item["seeds"] = [r["seed"] for r in rows]
        summary[name] = item
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(FEATURE_CONFIGS))
    parser.add_argument(
        "--variants", default="full", help="comma-separated: full,no_dirac,no_phase,no_harmonic"
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=100)
    
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--microsteps", type=int, default=2)
    parser.add_argument(
        "--lpe-dim",
        type=int,
        default=8,
        help="Number of nonzero Laplacian eigenvectors appended per form when LPE is enabled.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gradient-loss", type=float, default=0.0)
    parser.add_argument("--phase-init", choices=("zero", "learnable"), default="zero")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--capacity", choices=("small", "large"), default="small")
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional explicit root; default: runs/dkho/{capacity}/form_0.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="worker mode: leave aggregation/reporting to a final collector",
    )
    args = parser.parse_args()
    if args.lpe_dim <= 0:
        parser.error("--lpe-dim must be a positive integer")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants or any(item not in STRUCTURE_VARIANTS for item in variants):
        parser.error(f"--variants must be drawn from {STRUCTURE_VARIANTS}")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else RUNS_ROOT / args.capacity / "form_0"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    geo = Geometry(data["points"], data["faces"], data["normals"], lpe_dim=args.lpe_dim)
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    harmonic = harmonic_diagnosis(
        trajectories[:, 0], trajectories[:, -1], split_indices(len(trajectories))["train"]
    )
    print(f"[harmonic diagnosis] {json.dumps(harmonic)}", flush=True)
    if "no_harmonic" in variants and not harmonic["harmonic_head_active"]:
        parser.error("no_harmonic is undefined: the diagnosed full model has no explicit H0 head.")
    selected = [FEATURE_CONFIGS[name] for name in args.configs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    manifest = {
        "dataset": portable_path(DATA_PATH),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "device": str(device),
        "args": portable_arguments(args),
        "harmonic": harmonic,
        "architecture": {
            "version": "dkho_torus_v2_initial_skip",
            "slow_initial_skip": True,
            "slow_update": "LN(z_r + U_r([z_r, cos(theta_r), sin(theta_r)]) + S_r(z_r_initial))",
        },
        "n_nodes": geo.n0,
        "n_edges": geo.n1,
        "n_faces": geo.n2,
        "betti_expected": [1, 2, 1],
        "weighted_hodge": False,
        "boundary_features": False,
        "split_sizes": {
            key: len(value) for key, value in split_indices(len(data["trajectories"])).items()
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_results = [
        run_one(cfg, seed, args, geo, data, device, output_root, variant)
        for cfg in selected
        for variant in variants
        for seed in seeds
    ]
    if not args.no_summary:
        summary = aggregate(all_results)
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
