"""Implement native-support baseline models for Darcy cochain targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from torch import Tensor, nn
from torch.nn import functional as F

from common import CACHE, DarcyGeometry


@dataclass(frozen=True)
class ReferenceSpec:
    gno_hidden: int
    gno_projection: int
    gno_layers: int
    gno_radius: float
    fno_width: int
    fno_layers: int
    fno_modes: int
    grid: int
    mgn_hidden: int
    mgn_layers: int
    branch_layers: tuple[int, ...]
    trunk_layers: tuple[int, ...]
    basis: int
    geo_modes: int
    geo_width: int
    geo_layers: int
    geo_grid: int
    hsd_k: int
    hsd_hidden: tuple[int, ...]
    hsd_fno_modes: int
    hsd_fno_width: int
    hsd_fno_layers: int


SPECS = {
    
    "toroidal": ReferenceSpec(
        120,
        68,
        3,
        0.15,
        21,
        3,
        9,
        16,
        72,
        8,
        (96, 96, 64),
        (68, 68, 64),
        64,
        9,
        18,
        4,
        16,
        64,
        (32, 32),
        9,
        12,
        6,
    ),
    
    
    "magnetostatics": ReferenceSpec(
        84,
        96,
        5,
        0.20,
        25,
        2,
        9,
        16,
        58,
        10,
        (64, 64, 64),
        (64, 64, 64),
        74,
        9,
        25,
        2,
        16,
        64,
        (32, 32),
        9,
        15,
        4,
    ),
}


def pad3(value: np.ndarray) -> np.ndarray:
    return np.concatenate([value.astype(np.float32), np.zeros((len(value), 1), np.float32)], axis=1)


def entity_coordinates(geo: DarcyGeometry, task: str) -> np.ndarray:
    return pad3({"0": geo.points, "1": geo.edge_mid, "2": geo.face_mid}[task])


def entity_latent(nodes: Tensor, task: str, tail: Tensor, head: Tensor, faces: Tensor) -> Tensor:
    
    if task == "0":
        return nodes
    if task == "1":
        return 0.5 * (nodes[:, tail] + nodes[:, head])
    return nodes[:, faces].mean(2)


def native_input_ranks(name: str) -> tuple[int, ...]:
    
    if name == "HSD":
        return (0, 1, 2)
    if name in ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO"):
        return (0,)
    raise KeyError(name)


class OriginalMLP(nn.Module):
    
    def __init__(self, dims: tuple[int, ...], out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = dims[0]
        for width in dims[1:]:
            layers += [nn.Linear(current, width), nn.GELU()]
            current = width
        layers += [nn.Linear(current, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SparseArchivedGraphConv(nn.Module):

    def __init__(self, hidden: int, geo: DarcyGeometry, radius: float) -> None:
        super().__init__()
        pairs = cKDTree(geo.points).query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            raise RuntimeError(f"GNO radius {radius} produced no Darcy neighbours")
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        self.register_buffer("row", torch.from_numpy(rows.astype(np.int64)))
        self.register_buffer("col", torch.from_numpy(cols.astype(np.int64)))
        self.register_buffer("pos", torch.from_numpy(pad3(geo.points)))
        
        
        degree = np.bincount(rows, minlength=geo.n0).astype(np.float32)
        position_sum = np.zeros((geo.n0, 3), dtype=np.float32)
        np.add.at(position_sum, rows, pad3(geo.points)[cols])
        self.register_buffer("inv_degree", torch.from_numpy(1.0 / np.maximum(degree, 1.0)))
        self.register_buffer(
            "relative_mean_position",
            torch.from_numpy(position_sum / np.maximum(degree[:, None], 1.0) - pad3(geo.points)),
        )
        aggregation = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([rows, cols]).astype(np.int64)),
            torch.from_numpy((1.0 / np.maximum(degree[rows], 1.0)).astype(np.float32)),
            size=(geo.n0, geo.n0),
        ).coalesce()
        self.register_buffer("aggregation", aggregation, persistent=False)
        
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        flat = x.permute(1, 0, 2).reshape(n, -1)
        aggregate = torch.sparse.mm(self.aggregation, flat)
        aggregate = aggregate.reshape(n, b, -1).permute(1, 0, 2)
        rel = self.relative_mean_position.to(dtype=x.dtype)[None].expand(b, -1, -1)
        message = self.message_net(torch.cat([x, aggregate, rel], -1))
        return self.update_net(torch.cat([x, message], -1))


class ArchivedGNO(nn.Module):
    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.geo, self.task = geo, task
        h = spec.gno_hidden
        
        self.input_adapter = nn.Linear(dims[0], 4)
        self.register_buffer("tail", torch.from_numpy(geo.tail.astype(np.int64)))
        self.register_buffer("head", torch.from_numpy(geo.head.astype(np.int64)))
        self.register_buffer("faces", torch.from_numpy(geo.faces.astype(np.int64)))
        self.lifting = nn.Sequential(nn.Conv1d(4, h, 1), nn.GELU(), nn.Conv1d(h, h, 1))
        self.graph_layers = nn.ModuleList(
            SparseArchivedGraphConv(h, geo, spec.gno_radius) for _ in range(spec.gno_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(spec.gno_layers))
        self.projection = nn.Sequential(
            nn.Conv1d(h, spec.gno_projection, 1), nn.GELU(), nn.Conv1d(spec.gno_projection, 1, 1)
        )

    def forward(self, forms: list[Tensor]) -> Tensor:
        h = self.lifting(self.input_adapter(forms[0]).transpose(1, 2)).transpose(1, 2)
        for layer, norm in zip(self.graph_layers, self.norms):
            h = h + norm(layer(h))
        h = entity_latent(h, self.task, self.tail, self.head, self.faces)
        return self.projection(h.transpose(1, 2)).squeeze(1)


class ArchivedMGN(nn.Module):
    

    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.geo, self.task = geo, task
        h = spec.mgn_hidden
        self.input_adapter = nn.Linear(dims[0], 4)
        self.register_buffer("tail", torch.from_numpy(geo.tail.astype(np.int64)))
        self.register_buffer("head", torch.from_numpy(geo.head.astype(np.int64)))
        self.register_buffer("faces", torch.from_numpy(geo.faces.astype(np.int64)))
        self.node_encoder = nn.Linear(4, h)
        self.edge_encoder = nn.Linear(3, h)
        self.register_buffer(
            "src", torch.from_numpy(np.concatenate([geo.tail, geo.head]).astype(np.int64))
        )
        self.register_buffer(
            "dst", torch.from_numpy(np.concatenate([geo.head, geo.tail]).astype(np.int64))
        )
        
        
        
        edge_attr = np.concatenate([geo.edge_dir, geo.edge_length], 1)
        reverse_edge_attr = np.concatenate([-geo.edge_dir, geo.edge_length], 1)
        self.register_buffer(
            "edge_attr",
            torch.from_numpy(np.concatenate([edge_attr, reverse_edge_attr], 0).astype(np.float32)),
        )
        self.edge_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h)
            )
            for _ in range(spec.mgn_layers)
        )
        self.node_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h)
            )
            for _ in range(spec.mgn_layers)
        )
        self.decoder = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, forms: list[Tensor]) -> Tensor:
        x = self.node_encoder(self.input_adapter(forms[0]))
        edge = self.edge_encoder(self.edge_attr)[None].expand(x.shape[0], -1, -1)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            edge = edge_mlp(torch.cat([x[:, self.src], x[:, self.dst], edge], -1))
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(1, self.dst, edge.to(dtype=x.dtype))
            update = node_mlp(torch.cat([x, aggregate], -1)).to(dtype=x.dtype)
            x = x + update
        x = entity_latent(x, self.task, self.tail, self.head, self.faces)
        return self.decoder(x).squeeze(-1)


class SpectralConv2dArchived(nn.Module):
    

    def __init__(self, cin: int, cout: int, modes: int) -> None:
        super().__init__()
        scale = 1 / (cin * cout)
        self.modes = modes
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        
        input_dtype = x.dtype
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x.float())
        m1, m2 = min(self.modes, h), min(self.modes, w // 2 + 1)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2]
        )
        out[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out, s=(h, w)).to(dtype=input_dtype)


class ArchivedFNO2d(nn.Module):
    

    def __init__(self, width: int, layers: int, modes: int) -> None:
        super().__init__()
        self.lifting = nn.Conv2d(2, width, 1)
        self.spectral_layers = nn.ModuleList(
            SpectralConv2dArchived(width, width, modes) for _ in range(layers)
        )
        self.local_layers = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(layers))
        self.projection = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1)
        )
        self.layers = layers

    def latent(self, x: Tensor) -> Tensor:
        h = self.lifting(x)
        for index, (spectral, local, norm) in enumerate(
            zip(self.spectral_layers, self.local_layers, self.norms)
        ):
            h = norm(spectral(h) + local(h))
            if index < self.layers - 1:
                h = F.gelu(h)
        return h

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(self.latent(x))


class Raster2d(nn.Module):
    def __init__(self, geo: DarcyGeometry, grid: int) -> None:
        super().__init__()
        self.grid = grid

        def index(pos: np.ndarray) -> Tensor:
            ix = np.clip(np.rint((pos[:, 0] + 1) * 0.5 * (grid - 1)), 0, grid - 1).astype(np.int64)
            iy = np.clip(np.rint((pos[:, 1] + 1) * 0.5 * (grid - 1)), 0, grid - 1).astype(np.int64)
            return torch.from_numpy(iy * grid + ix)

        self.register_buffer("node", index(geo.points))
        self.register_buffer("edge", index(geo.edge_mid))
        self.register_buffer("face", index(geo.face_mid))
        count = np.bincount(self.node.cpu().numpy(), minlength=grid * grid).astype(np.float32)
        self.register_buffer("node_inverse_count", torch.from_numpy(1.0 / np.maximum(count, 1.0)))

    def splat(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        flat = torch.zeros(b, c, self.grid * self.grid, device=x.device, dtype=x.dtype)
        flat.index_add_(2, self.node, x.transpose(1, 2))
        return (flat * self.node_inverse_count.to(dtype=x.dtype)[None, None]).reshape(
            b, c, self.grid, self.grid
        )

    def query(self, x: Tensor, task: str) -> Tensor:
        idx = {"0": self.node, "1": self.edge, "2": self.face}[task]
        return x.flatten(2)[:, :, idx].transpose(1, 2)


class ArchivedFNO(nn.Module):
    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.task = task
        self.adapter = nn.Linear(dims[0], 2)
        self.raster = Raster2d(geo, spec.grid)
        self.core = ArchivedFNO2d(spec.fno_width, spec.fno_layers, spec.fno_modes)

    def forward(self, forms: list[Tensor]) -> Tensor:
        return self.raster.query(
            self.core(self.raster.splat(self.adapter(forms[0]))), self.task
        ).squeeze(-1)


class ArchivedDeepONet(nn.Module):
    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.task = task
        self.adapter = nn.Linear(dims[0], 1)
        self.register_buffer("coords", torch.from_numpy(entity_coordinates(geo, task)))
        
        self.branch = OriginalMLP((geo.n0, *spec.branch_layers), spec.basis)
        self.trunk = OriginalMLP((3, *spec.trunk_layers), spec.basis)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, forms: list[Tensor]) -> Tensor:
        b = self.branch(self.adapter(forms[0]).squeeze(-1))
        t = self.trunk(self.coords[None])
        return (t * b[:, None]).sum(-1) + self.bias


class ArchivedGeoFNO(nn.Module):
    

    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.task = task
        self.grid = spec.geo_grid
        self.adapter = nn.Linear(dims[0], 1)
        self.register_buffer("source_coords", torch.from_numpy(pad3(geo.points)))
        self.register_buffer("query_coords", torch.from_numpy(entity_coordinates(geo, task)))
        self.s_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )
        self.fc0 = nn.Linear(4, spec.geo_width)
        self.convs = nn.ModuleList(
            SpectralConv2dArchived(spec.geo_width, spec.geo_width, spec.geo_modes)
            for _ in range(spec.geo_layers)
        )
        self.ws = nn.ModuleList(
            nn.Conv2d(spec.geo_width, spec.geo_width, 1) for _ in range(spec.geo_layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm2d(spec.geo_width) for _ in range(spec.geo_layers))
        self.fc1 = nn.Linear(spec.geo_width, 64)
        self.fc2 = nn.Linear(64, 1)
        self.layers = spec.geo_layers

    def _splat(self, feature: Tensor, eta: Tensor) -> Tensor:
        b, n, c = feature.shape
        g = self.grid
        xy = eta[..., :2]
        ix = torch.clamp(torch.round(xy[..., 0] * (g - 1)).long(), 0, g - 1)
        iy = torch.clamp(torch.round(xy[..., 1] * (g - 1)).long(), 0, g - 1)
        idx = iy * g + ix
        flat = torch.zeros(b, c, g * g, device=feature.device, dtype=feature.dtype)
        flat.scatter_add_(2, idx[:, None].expand(-1, c, -1), feature.transpose(1, 2))
        count = torch.zeros(b, 1, g * g, device=feature.device, dtype=feature.dtype)
        count.scatter_add_(
            2, idx[:, None], torch.ones(b, 1, n, device=feature.device, dtype=feature.dtype)
        )
        return (flat / count.clamp_min(1)).reshape(b, c, g, g)

    def forward(self, forms: list[Tensor]) -> Tensor:
        b = forms[0].shape[0]
        src = self.source_coords[None].expand(b, -1, -1)
        eta = self.s_net(src.reshape(-1, 3)).reshape_as(src)
        x = self.fc0(torch.cat([src, self.adapter(forms[0])], -1))
        h = self._splat(x, eta)
        for i, (conv, w, norm) in enumerate(zip(self.convs, self.ws, self.norms)):
            h = norm(conv(h) + w(h))
            if i < self.layers - 1:
                h = F.gelu(h)
        q = self.query_coords[None].expand(b, -1, -1)
        eq = self.s_net(q.reshape(-1, 3)).reshape_as(q)[..., :2]
        grid = (eq * 2 - 1).view(b, -1, 1, 2)
        sample = (
            F.grid_sample(h, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            .squeeze(-1)
            .transpose(1, 2)
        )
        return self.fc2(F.gelu(self.fc1(sample))).squeeze(-1)


def basis(geo: DarcyGeometry, task: str, k: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"darcy_reference_hsd_form{task}_k{k}.npy"
    n = {"0": geo.n0, "1": geo.n1, "2": geo.n2}[task]
    if path.exists():
        x = np.load(path)
        if x.shape == (n, k):
            return x.astype(np.float32)
    lap = {
        "0": geo.d0.T @ geo.d0,
        "1": geo.d0 @ geo.d0.T + geo.d1.T @ geo.d1,
        "2": geo.d1 @ geo.d1.T,
    }[task]
    _, v = eigsh(lap.astype(float), k=k, which="SM", tol=1e-4, maxiter=30000)
    np.save(path, v.astype(np.float32))
    return v.astype(np.float32)


class PhysicsEncoder(nn.Module):
    

    def __init__(self, m01: Tensor, m12: Tensor) -> None:
        super().__init__()
        self.register_buffer("m01", m01)
        self.register_buffer("m12", m12)

    def forward(self, c0: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
        delta1_c1 = c1 @ self.m01
        d0_c0 = c0 @ self.m01.T
        delta2_c2 = c2 @ self.m12
        d1_c1 = c1 @ self.m12.T
        return torch.cat([c0, delta1_c1, c1, d0_c0, delta2_c2, c2, d1_c1], -1)


class SpectralAmp(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.arange(1, k + 1).float())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gain[None]


class PhysicsGMLP(nn.Module):
    def __init__(self, fin: int, fout: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(fin)
        self.w1 = nn.Linear(fin, fout)
        self.w2 = nn.Linear(fin, fout)
        self.wo = nn.Linear(fout, fout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.wo(self.w1(self.norm(x)) * F.silu(self.w2(self.norm(x))))
        return x + y if x.shape[-1] == y.shape[-1] else y


class CouplingErrorMLP(nn.Module):
    

    def __init__(self, k_total: int, hidden: int = 64) -> None:
        super().__init__()
        self.global_proj = nn.Linear(k_total, hidden // 2)
        self.local_proj = nn.Linear(3, hidden // 2)
        self.body = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, coords: Tensor, coeffs: Tensor) -> Tensor:
        b, n = coeffs.shape[0], coords.shape[0]
        local = self.local_proj(coords)[None].expand(b, -1, -1)
        global_feature = self.global_proj(coeffs)[:, None].expand(-1, n, -1)
        return self.body(torch.cat([local, global_feature], -1)).squeeze(-1)


class ArchivedHSD(nn.Module):
    

    def __init__(
        self, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, spec: ReferenceSpec
    ) -> None:
        super().__init__()
        self.geo, self.task = geo, task
        k = spec.hsd_k
        self.adapters = nn.ModuleList(nn.Linear(d, 1) for d in dims)
        self.register_buffer("phi0", torch.from_numpy(basis(geo, "0", k)))
        self.register_buffer("phi1", torch.from_numpy(basis(geo, "1", k)))
        self.register_buffer("phi2", torch.from_numpy(basis(geo, "2", k)))
        
        
        m01 = torch.from_numpy(
            (self.phi1.cpu().numpy().T @ (geo.d0 @ self.phi0.cpu().numpy())).astype(np.float32)
        )
        m12 = torch.from_numpy(
            (self.phi2.cpu().numpy().T @ (geo.d1 @ self.phi1.cpu().numpy())).astype(np.float32)
        )
        self.amp = nn.ModuleList(SpectralAmp(k) for _ in range(3))
        self.physics = PhysicsEncoder(m01, m12)
        last = 7 * k
        layers = []
        for h in spec.hsd_hidden:
            layers.append(PhysicsGMLP(last, h))
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, k)
        self.raster = Raster2d(geo, spec.grid)
        self.res_adapter = nn.Linear(dims[0], 2)
        self.residual = ArchivedFNO2d(spec.hsd_fno_width, spec.hsd_fno_layers, spec.hsd_fno_modes)
        self.interaction = CouplingErrorMLP(7 * k)
        self.register_buffer("query_coords", torch.from_numpy(entity_coordinates(geo, task)))
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, forms: list[Tensor]) -> Tensor:
        c0 = torch.einsum("bn,nk->bk", self.adapters[0](forms[0]).squeeze(-1), self.phi0)
        c1 = torch.einsum("bn,nk->bk", self.adapters[1](forms[1]).squeeze(-1), self.phi1)
        c2 = torch.einsum("bn,nk->bk", self.adapters[2](forms[2]).squeeze(-1), self.phi2)
        enhanced = self.physics(self.amp[0](c0), self.amp[1](c1), self.amp[2](c2))
        latent = self.body(enhanced)
        phi = {"0": self.phi0, "1": self.phi1, "2": self.phi2}[self.task]
        base = torch.einsum("bk,nk->bn", self.head(latent), phi)
        fno_res = self.raster.query(
            self.residual(self.raster.splat(self.res_adapter(forms[0]))), self.task
        ).squeeze(-1)
        raw_res = fno_res + self.interaction(self.query_coords, enhanced)
        projection = torch.einsum("bn,nk->bk", raw_res, phi)
        orthogonal = raw_res - torch.einsum("bk,nk->bn", projection, phi)
        return base + self.res_scale * orthogonal


def make_reference_model(
    name: str, geo: DarcyGeometry, dims: tuple[int, int, int], task: str, reference: str
) -> nn.Module:
    spec = SPECS[reference]
    if name == "GNO":
        return ArchivedGNO(geo, dims, task, spec)
    if name == "FNO":
        return ArchivedFNO(geo, dims, task, spec)
    if name == "MGN":
        return ArchivedMGN(geo, dims, task, spec)
    if name == "DeepONet":
        return ArchivedDeepONet(geo, dims, task, spec)
    if name == "GeoFNO":
        return ArchivedGeoFNO(geo, dims, task, spec)
    if name == "HSD":
        return ArchivedHSD(geo, dims, task, spec)
    raise KeyError(name)
