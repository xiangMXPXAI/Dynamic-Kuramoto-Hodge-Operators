"""Train the magnetostatics model and structural ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cavity_magnetostatics_v1.pkl"
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


@dataclass(frozen=True)
class Features:
    name: str
    boundary: bool = False
    lpe: bool = False
    heat: bool = False
    qp_local: bool = False


CONFIGS = {
    "native": Features("native"),
    "boundary": Features("boundary", boundary=True),
    "boundary_spectral": Features("boundary_spectral", boundary=True, lpe=True),
    "boundary_diffusion_spectral": Features(
        "boundary_diffusion_spectral", boundary=True, lpe=True, heat=True
    ),
    "conditioned": Features("conditioned", boundary=True, lpe=True, heat=True, qp_local=True),
}
STRUCTURE_VARIANTS = ("full", "no_dirac", "no_phase", "no_harmonic")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def resolve_device(requested):
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA requested but unavailable; falling back to CPU.", flush=True)
    return torch.device("cpu")


def splits(n):
    a = np.arange(n)
    tv, te = train_test_split(a, test_size=0.2, random_state=42)
    tr, va = train_test_split(tv, test_size=0.15, random_state=42)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def norm(x):
    return ((x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)).astype("float32")


def complex_from_tets(tets, n0):
    faces = np.asarray(
        sorted(
            {
                tuple(sorted(f))
                for t in tets
                for f in (
                    (t[0], t[1], t[2]),
                    (t[0], t[1], t[3]),
                    (t[0], t[2], t[3]),
                    (t[1], t[2], t[3]),
                )
            }
        ),
        dtype=np.int64,
    )
    es = sorted(
        {
            tuple(sorted((int(a), int(b))))
            for f in faces
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
        }
    )
    edges = np.asarray(es, dtype=np.int64)
    em = {x: i for i, x in enumerate(es)}
    fm = {tuple(face): i for i, face in enumerate(faces)}
    e = np.arange(len(es))
    d0 = sparse.csr_matrix(
        (np.r_[-np.ones(len(e)), np.ones(len(e))], (np.r_[e, e], np.r_[edges[:, 0], edges[:, 1]])),
        shape=(len(e), n0),
        dtype=np.float32,
    )
    rr = []
    cc = []
    vv = []
    for i, (a, b, c) in enumerate(faces):
        for u, v, s in ((b, c, 1), (a, c, -1), (a, b, 1)):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            rr.append(i)
            cc.append(em[key])
            vv.append(s if (u, v) == key else -s)
    d1 = sparse.csr_matrix((np.asarray(vv, np.float32), (rr, cc)), shape=(len(faces), len(es)))
    assert np.max(np.abs((d1 @ d0).data), initial=0) < 1e-6
    
    
    
    rr = []
    cc = []
    vv = []
    for ti, tet in enumerate(tets):
        tet = [int(value) for value in tet]
        for omitted in range(4):
            raw = [tet[j] for j in range(4) if j != omitted]
            key = tuple(sorted(raw))
            inversions = sum(raw[i] > raw[j] for i in range(3) for j in range(i + 1, 3))
            sign = (-1 if omitted % 2 else 1) * (-1 if inversions % 2 else 1)
            rr.append(ti)
            cc.append(fm[key])
            vv.append(sign)
    d2 = sparse.csr_matrix((np.asarray(vv, np.float32), (rr, cc)), shape=(len(tets), len(faces)))
    assert np.max(np.abs((d2 @ d1).data), initial=0) < 1e-6
    return edges, faces, d0, d1, d2


class Geo:
    def __init__(self, p, t, inner, outer, cache, k):
        self.p = p.astype("float32")
        self.t = t.astype(np.int64)
        self.n0 = len(p)
        self.edges, self.faces, self.d0, self.d1, self.d2 = complex_from_tets(self.t, self.n0)
        self.n1 = len(self.edges)
        self.n2 = len(self.faces)
        self.n3 = len(self.t)
        self.tail, self.head = self.edges.T
        self.fe = self.d1.indices.reshape(self.n2, 3).astype(np.int64)
        self.fs = self.d1.data.reshape(self.n2, 3).astype("float32")
        self.ep = (self.p[self.tail] + self.p[self.head]) / 2
        self.ev = self.p[self.head] - self.p[self.tail]
        self.el = np.linalg.norm(self.ev, axis=1, keepdims=True).astype("float32")
        self.ed = self.ev / np.maximum(self.el, 1e-8)
        a, b, c = (self.p[self.faces[:, i]] for i in range(3))
        cr = np.cross(b - a, c - a)
        n = np.linalg.norm(cr, axis=1, keepdims=True)
        self.fa = (n * 0.5).astype("float32")
        self.fn = (cr / np.maximum(n, 1e-8)).astype("float32")
        self.fp = ((a + b + c) / 3).astype("float32")
        self.p0, self.p1, self.p2 = norm(self.p), norm(self.ep), norm(self.fp)
        r = np.linalg.norm(self.p, axis=1, keepdims=True)
        self.rad = ((r - r.mean()) / (r.std() + 1e-6)).astype("float32")
        self.vol = self._vol()
        self.bound = self._bound(inner, outer)
        self.cache = cache
        self.k = k
        self.lpe = None

    def _vol(self):
        q = self.p[self.t]
        v = (
            np.abs(
                np.einsum(
                    "ij,ij->i", q[:, 1] - q[:, 0], np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])
                )
            )
            / 6
        )
        o = np.zeros(self.n0, np.float32)
        for j in range(4):
            np.add.at(o, self.t[:, j], v.astype("float32") / 4)
        return o / max(o.sum(), 1e-8)

    def _bound(self, inner, outer):
        mi = np.zeros(self.n0, np.float32)
        mo = mi.copy()
        mi[inner] = 1
        mo[outer] = 1
        di = cKDTree(self.p[inner]).query(self.p)[0]
        do = cKDTree(self.p[outer]).query(self.p)[0]
        sc = max(np.linalg.norm(self.p.max(0) - self.p.min(0)), 1e-8)
        rad = self.p / np.maximum(np.linalg.norm(self.p, axis=1, keepdims=True), 1e-8)
        return np.c_[mi, mo, di / sc, do / sc, rad * (mo - mi)[:, None]].astype("float32")

    def ensure_lpe(self):
        
        f = self.cache / f"lpe_unweighted_full_l2_v2_k{self.k}.npz"
        if f.exists():
            q = np.load(f)
            candidate = (q["a"], q["b"], q["c"])
            if (
                candidate[0].shape == (self.n0, self.k)
                and candidate[1].shape == (self.n1, self.k)
                and candidate[2].shape == (self.n2, self.k)
            ):
                self.lpe = candidate
                return
        self.cache.mkdir(parents=True, exist_ok=True)
        print("[preprocess] calculate unweighted 0/1/2 LPE", flush=True)

        def basis(L):
            n = L.shape[0]
            request = min(max(self.k + 8, 2 * self.k + 4), n - 1)
            while True:
                va, ve = eigsh(L.astype("float64"), k=request, which="SM", tol=1e-4, maxiter=20000)
                o = np.argsort(va)
                va, ve = va[o], ve[:, o]
                ix = np.flatnonzero(va > 1e-7)[: self.k]
                if len(ix) == self.k:
                    v = ve[:, ix]
                    sign = np.sign(v[np.abs(v).argmax(0), np.arange(v.shape[1])])
                    v *= np.where(sign == 0, 1.0, sign)[None, :]
                    return v.astype("float32")
                if request >= n - 1:
                    raise RuntimeError(
                        f"insufficient positive LPE modes: requested={self.k}, found={len(ix)}, n={n}"
                    )
                request = min(n - 1, max(request + 8, 2 * request))

        self.lpe = (
            basis(self.d0.T @ self.d0),
            basis(self.d0 @ self.d0.T + self.d1.T @ self.d1),
            basis(self.d1 @ self.d1.T + self.d2.T @ self.d2),
        )
        np.savez_compressed(f, a=self.lpe[0], b=self.lpe[1], c=self.lpe[2])


class Op(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.n0 = g.n0
        self.n1 = g.n1
        self.n2 = g.n2
        self.register_buffer("tail", torch.from_numpy(g.tail))
        self.register_buffer("head", torch.from_numpy(g.head))
        self.register_buffer("fe", torch.from_numpy(g.fe))
        self.register_buffer("fv", torch.from_numpy(g.faces))
        self.register_buffer("fs", torch.from_numpy(g.fs))
        assert int(g.faces.min()) >= 0 and int(g.faces.max()) < g.n0
        assert int(g.fe.min()) >= 0 and int(g.fe.max()) < g.n1

    def d0(self, x):
        return x[:, self.head] - x[:, self.tail]

    def de1(self, x):
        o = torch.zeros(x.shape[0], self.n0, x.shape[-1], device=x.device, dtype=x.dtype)
        o.index_add_(1, self.tail, -x)
        o.index_add_(1, self.head, x)
        return o

    def d1(self, x):
        return (x[:, self.fe] * self.fs[None, :, :, None]).sum(2)

    def de2(self, x):
        o = torch.zeros(x.shape[0], self.n1, x.shape[-1], device=x.device, dtype=x.dtype)
        for j in range(3):
            o.index_add_(1, self.fe[:, j], x * self.fs[None, :, j, None])
        return o


def M(fin, h, fout):
    return nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, fout))


class FB(nn.Module):
    def __init__(self, g, c):
        super().__init__()
        self.g = g
        self.c = c
        self.op = Op(g)
        if c.lpe:
            g.ensure_lpe()
        a = {
            "p0": g.p0,
            "p1": g.p1,
            "p2": g.p2,
            "ed": g.ed,
            "el": g.el,
            "fn": g.fn,
            "fa": g.fa,
            "bd": g.bound,
            "rad": g.rad,
            "vol": g.vol[:, None],
        }
        if c.lpe:
            a |= {"l0": g.lpe[0], "l1": g.lpe[1], "l2": g.lpe[2]}
        for k, v in a.items():
            self.register_buffer(k, torch.from_numpy(v))
        d0, d1, d2 = 5, 8, 8
        if c.boundary:
            d0 += 7
            d1 += 7
            d2 += 7
        if c.lpe:
            d0 += g.k
            d1 += g.k
            d2 += g.k
        if c.heat:
            d0 += 4
            d1 += 4
            d2 += 4
        if c.qp_local:
            d0 += 4
            d1 += 4
            d2 += 4
        self.dims = (d0, d1, d2)

    def ex(self, x, b):
        return x.unsqueeze(0).expand(b, -1, -1)

    def qp(self, r):
        return torch.cat(
            [
                (r * self.vol[:, 0]).sum(1, keepdim=True),
                (r[:, :, None] * self.vol[None] * self.p0[None]).sum(1),
            ],
            -1,
        )

    def forward(self, r):
        b = len(r)
        u = r[:, :, None]
        gr = self.op.d0(u)
        f0 = [u, self.ex(self.p0, b), self.ex(self.rad, b)]
        f1 = [gr, self.ex(self.p1, b), self.ex(self.ed, b), self.ex(self.el, b)]
        f2 = [
            torch.zeros(b, self.g.n2, 1, device=r.device),
            self.ex(self.p2, b),
            self.ex(self.fn, b),
            self.ex(self.fa, b),
        ]
        if self.c.boundary:
            q = self.ex(self.bd, b)
            f0 += [q]
            f1 += [0.5 * (q[:, self.op.tail] + q[:, self.op.head])]
            f2 += [q[:, self.op.fv].mean(2)]
        if self.c.lpe:
            f0 += [self.ex(self.l0, b)]
            f1 += [self.ex(self.l1, b)]
            f2 += [self.ex(self.l2, b)]
        if self.c.heat:
            hs = []
            h = u
            for _ in range(4):
                h = h - 0.025 * self.op.de1(self.op.d0(h))
                hs.append(h)
            f0 += hs
            f1 += [self.op.d0(v) for v in hs]
            f2 += [torch.zeros(b, self.g.n2, 4, device=r.device)]
        qp = self.qp(r)
        if self.c.qp_local:
            f0 += [qp[:, None].expand(-1, self.g.n0, -1)]
            f1 += [qp[:, None].expand(-1, self.g.n1, -1)]
            f2 += [qp[:, None].expand(-1, self.g.n2, -1)]
        return [torch.cat(x, -1) for x in (f0, f1, f2)], qp


class DK(nn.Module):
    def __init__(self, h, c, m):
        super().__init__()
        self.c = c
        self.m = m
        self.om = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.g = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.l = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.s = nn.Parameter(torch.zeros(3))
        self.dt = nn.Parameter(torch.tensor(-1.35))
        self.ph = nn.ModuleList([nn.Linear(2 * c, h) for _ in range(3)])
        self.u = nn.ModuleList([M(2 * h, h, h) for _ in range(3)])
        self.skip = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(3)])
        self.n = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])

    def forward(self, z, zi, t, o):
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt)
        s = F.softplus(self.s) + 1e-4
        for _ in range(self.m):
            z0, z1, z2 = z
            t0, t1, t2 = t
            g01 = o.d0(t0) - torch.tanh(self.g[1](z1)) * t1
            a10 = o.de1(t1) + torch.tanh(self.g[0](z0)) * t0
            g12 = o.d1(t1) - torch.tanh(self.g[2](z2)) * t2
            a21 = o.de2(t2) + torch.tanh(self.g[1](z1)) * t1
            l0, l1, l2 = [0.5 * torch.tanh(self.l[i](z[i])) for i in range(3)]
            t = [
                t0 + dt * (-self.om[0](z0) - s[0] * o.de1(torch.sin(g01 - l1))),
                t1
                + dt
                * (
                    -self.om[1](z1)
                    - s[1] * (o.d0(torch.sin(a10 - l0)) + o.de2(torch.sin(g12 - l2)))
                ),
                t2 + dt * (-self.om[2](z2) - s[2] * o.d1(torch.sin(a21 - l1))),
            ]
        return [
            self.n[i](
                z[i]
                + self.u[i](
                    torch.cat(
                        [z[i], self.ph[i](torch.cat([torch.cos(t[i]), torch.sin(t[i])], -1))], -1
                    )
                )
                + self.skip[i](zi[i])
            )
            for i in range(3)
        ], t


class NoDiracDK(DK):
    

    def forward(self, z, zi, t, o):
        del o
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt)
        s = F.softplus(self.s) + 1e-4
        for _ in range(self.m):
            t = [
                t[i]
                + dt
                * (
                    -self.om[i](z[i])
                    - s[i]
                    * torch.sin(
                        torch.tanh(self.g[i](z[i])) * t[i] - 0.5 * torch.tanh(self.l[i](z[i]))
                    )
                )
                for i in range(3)
            ]
        return [
            self.n[i](
                z[i]
                + self.u[i](
                    torch.cat(
                        [z[i], self.ph[i](torch.cat([torch.cos(t[i]), torch.sin(t[i])], -1))], -1
                    )
                )
                + self.skip[i](zi[i])
            )
            for i in range(3)
        ], t


class FeatureDirac(nn.Module):
    

    def __init__(self, h, c):
        super().__init__()
        self.c = c
        self.om = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.g = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.l = nn.ModuleList([nn.Linear(h, c) for _ in range(3)])
        self.s = nn.Parameter(torch.zeros(3))
        self.dt = nn.Parameter(torch.tensor(-1.35))
        self.read = nn.ModuleList([nn.Linear(2 * c, h) for _ in range(3)])
        self.u = nn.ModuleList([M(2 * h, h, h) for _ in range(3)])
        self.skip = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(3)])
        self.n = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])

    def carrier(self, z):
        return [
            torch.sigmoid(self.g[i](z[i])) * torch.tanh(self.om[i](z[i]))
            + torch.tanh(self.l[i](z[i]))
            for i in range(3)
        ]

    def decoder_channels(self, x, i):
        carrier = torch.sigmoid(self.g[i](x)) * torch.tanh(self.om[i](x)) + torch.tanh(self.l[i](x))
        return torch.cat([carrier, torch.tanh(self.g[i](x))], -1)

    def forward(self, z, zi, o):
        x0, x1, x2 = self.carrier(z)
        s = F.softplus(self.s) + 1e-4
        msg = [s[0] * o.de1(x1), s[1] * (o.d0(x0) + o.de2(x2)), s[2] * o.d1(x1)]
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt)
        return [
            self.n[i](
                z[i]
                + dt
                * self.u[i](
                    torch.cat([z[i], self.read[i](torch.cat([msg[i], (x0, x1, x2)[i]], -1))], -1)
                )
                + self.skip[i](zi[i])
            )
            for i in range(3)
        ]


class Model(nn.Module):
    def __init__(self, g, c, h, l, ch, m, init, harm, variant="full"):
        super().__init__()
        if variant not in STRUCTURE_VARIANTS:
            raise ValueError(f"unknown structure variant: {variant}")
        self.init = init
        self.variant = variant
        self.uses_phase = variant in ("full", "no_dirac", "no_harmonic")
        self.harm = harm and variant != "no_harmonic"
        self.f = FB(g, c)
        self.e = nn.ModuleList([M(d, h, h) for d in self.f.dims])
        layer = (
            DK
            if variant in ("full", "no_harmonic")
            else NoDiracDK if variant == "no_dirac" else FeatureDirac
        )
        self.layers = nn.ModuleList(
            [layer(h, ch, m) if layer is not FeatureDirac else layer(h, ch) for _ in range(l)]
        )
        self.dec = M(h + 2 * ch, h, 3)
        if self.uses_phase and init == "deterministic":
            self.i = nn.ModuleList([nn.Linear(h, ch) for _ in range(3)])
        if self.uses_phase and init == "random_learnable":
            self.logsig = nn.Parameter(torch.full((3,), -1.2))
        if self.harm:
            self.top = M(4, h, 3)

    def forward(self, r):
        fs, qp = self.f(r)
        z = [e(x) for e, x in zip(self.e, fs)]
        zi = list(z)
        if self.uses_phase:
            if self.init == "zero":
                t = [
                    torch.zeros(
                        x.shape[0], x.shape[1], self.layers[0].c, device=x.device, dtype=x.dtype
                    )
                    for x in z
                ]
            elif self.init == "deterministic":
                t = [math.pi * torch.tanh(a(x)) for a, x in zip(self.i, z)]
            else:
                t = [
                    F.softplus(self.logsig[j])
                    * torch.randn(
                        x.shape[0], x.shape[1], self.layers[0].c, device=x.device, dtype=x.dtype
                    )
                    for j, x in enumerate(z)
                ]
            for a in self.layers:
                z, t = a(z, zi, t, self.f.op)
            decoded = torch.cat([z[0], torch.cos(t[0]), torch.sin(t[0])], -1)
        else:
            for a in self.layers:
                z = a(z, zi, self.f.op)
            decoded = torch.cat([z[0], self.layers[-1].decoder_channels(z[0], 0)], -1)
        y = self.dec(decoded)
        return (
            y
            if not self.harm
            else y - y.mean(1, keepdim=True) + self.top(qp)[:, None] / math.sqrt(self.f.g.n0)
        )


def rloss(p, y):
    return torch.sqrt(
        (p.sub(y).square().sum((1, 2)) / y.square().sum((1, 2)).clamp_min(1e-10))
    ).mean()


def hstatus(y, train):
    a = y[train].mean(1) * math.sqrt(y.shape[1])
    c = float(np.sqrt(np.mean((a - a.mean(0)) ** 2)))
    f = float(np.sqrt(np.mean(y[train] ** 2)))
    q = c / max(f, 1e-12)
    return {
        "basis": "H0=span{1/sqrt(n0)} per xyz channel",
        "coefficient_rms": c,
        "field_rms": f,
        "relative_variation": q,
        "threshold": 1e-5,
        "constant_mode": q < 1e-5,
        "harmonic_head_active": not q < 1e-5,
    }


@torch.no_grad()
def predict(m, x, b):
    m.eval()
    return torch.cat([m(x[i : i + b]) for i in range(0, len(x), b)])


def npar(m):
    return sum(x.numel() for x in m.parameters())


def figs(run, p, y, yp, h):
    d = run / "figures"
    d.mkdir(parents=True, exist_ok=True)
    i = int(np.argmax(np.mean((yp - y) ** 2, (1, 2))))
    mid = np.abs(p[:, 2] - np.median(p[:, 2]))
    ix = np.flatnonzero(mid < np.quantile(mid, 0.12))[:: max(1, len(p) // 800)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, v, t in zip(ax, (y[i], yp[i], yp[i] - y[i]), ("target", "TDK-HO", "error")):
        q = a.quiver(
            p[ix, 0],
            p[ix, 1],
            v[ix, 0],
            v[ix, 1],
            np.linalg.norm(v[ix], axis=1),
            cmap="viridis",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.003,
        )
        a.set(title=t, aspect="equal")
        fig.colorbar(q, ax=a)
    fig.tight_layout()
    fig.savefig(d / "vector_comparison.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(h["train"], label="train")
    ax.plot(h["val"], label="val")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set(xlabel="epoch", ylabel="relative vector L2")
    fig.tight_layout()
    fig.savefig(d / "learning_curve.png", dpi=180)
    plt.close(fig)


def run(cfg, args, g, data, hs, out, variant="full"):
    rd = out / "form_2" / cfg.name / variant / f"seed_{args.seed}"
    rp = rd / "result.json"
    if rp.exists() and not args.rerun:
        return json.loads(rp.read_text(encoding="utf8"))
    rd.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    sp = splits(len(data["X_data"]))
    xx = np.asarray(data["X_data"], np.float32)
    yy = np.asarray(data["Y_data"], np.float32)
    scale = np.r_[sp["train"], sp["val"]]
    xs = float(abs(xx[scale]).max() + 1e-9)
    ys = float(abs(yy[scale]).max() + 1e-9)
    dev = resolve_device(args.device)
    x = torch.from_numpy(xx / xs).to(dev)
    y = torch.from_numpy(yy / ys).to(dev)
    m = Model(
        g,
        cfg,
        args.hidden,
        args.layers,
        args.channels,
        args.microsteps,
        args.phase_init,
        hs["harmonic_head_active"],
        variant,
    ).to(dev)
    dl = DataLoader(
        TensorDataset(x[sp["train"]], y[sp["train"]]), batch_size=args.batch_size, shuffle=True
    )
    op = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=args.epochs)
    hist = {"train": [], "val": []}
    best = 1e9
    state = None
    be = 0
    st = time.time()
    log_path = rd / "training.log"
    progress_path = rd / "progress.json"
    log_path.write_text("", encoding="utf8")
    log_every = max(1, len(dl) // 10)
    label = f"{cfg.name}/{variant}"

    def emit(message, payload):
        print(message, flush=True)
        with open(log_path, "a", encoding="utf8") as f:
            f.write(message + "\\n")
        progress_path.write_text(json.dumps(payload, indent=2), encoding="utf8")

    emit(
        f"[{label}] start: parameters={npar(m)} batches_per_epoch={len(dl)} batch_size={args.batch_size}",
        {
            "status": "running",
            "config": cfg.name,
            "structure_variant": variant,
            "parameters": npar(m),
            "epoch": 0,
            "batch": 0,
            "batches_per_epoch": len(dl),
            "elapsed_seconds": 0.0,
        },
    )
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = 0.0
        for bi, (xb, yb) in enumerate(dl, 1):
            op.zero_grad(set_to_none=True)
            loss = rloss(m(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            op.step()
            tot += float(loss.detach())
            if bi == 1 or bi % log_every == 0 or bi == len(dl):
                emit(
                    f"[{label}] epoch {ep:03d}/{args.epochs} batch {bi:03d}/{len(dl)} running_train={tot/bi:.6f} elapsed={time.time()-st:.1f}s",
                    {
                        "status": "running",
                        "config": cfg.name,
                        "structure_variant": variant,
                        "epoch": ep,
                        "batch": bi,
                        "batches_per_epoch": len(dl),
                        "running_train_loss": tot / bi,
                        "elapsed_seconds": time.time() - st,
                    },
                )
        tr = tot / len(dl)
        va = float(rloss(predict(m, x[sp["val"]], args.eval_batch), y[sp["val"]]))
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best:
            best = va
            be = ep
            state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        sch.step()
        emit(
            f"[{label}] epoch {ep:03d}/{args.epochs} complete train={tr:.6f} val={va:.6f} elapsed={time.time()-st:.1f}s",
            {
                "status": "running",
                "config": cfg.name,
                "structure_variant": variant,
                "epoch": ep,
                "batch": len(dl),
                "train_loss": tr,
                "val_loss": va,
                "best_val_loss": best,
                "best_epoch": be,
                "elapsed_seconds": time.time() - st,
            },
        )
    m.load_state_dict(state)
    yp = predict(m, x[sp["test"]], args.eval_batch).cpu().numpy() * ys
    yt = yy[sp["test"]]
    np.save(rd / "prediction_test.npy", yp.astype("float32"))
    np.save(rd / "target_test.npy", yt.astype("float32"))
    np.save(rd / "test_indices.npy", sp["test"])
    torch.save(
        {
            "state_dict": m.state_dict(),
            "features": asdict(cfg),
            "args": portable_arguments(args),
            "harmonic_diagnosis": hs,
            "structure_variant": variant,
        },
        rd / "best.pt",
    )
    (rd / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf8")
    figs(rd, g.p, yt, yp, hist)
    r = {
        "config": cfg.name,
        "structure_variant": variant,
        "uses_phase": m.uses_phase,
        "parameters": npar(m),
        "best_val_relative_l2": best,
        "best_epoch": be,
        "epochs_completed": args.epochs,
        "wall_seconds": time.time() - st,
        "phase_init": args.phase_init if m.uses_phase else None,
        "hodge_star": False,
        "mass_matrix": False,
        "loss": "relative_vector_l2_only",
        "x_scale_train_val": xs,
        "y_scale_train_val": ys,
        **hs,
        "harmonic_head_active": m.harm,
    }
    rp.write_text(json.dumps(r, indent=2), encoding="utf8")
    emit(
        f"[{label}] finished best_epoch={be} best_val={best:.6f} elapsed={time.time()-st:.1f}s",
        {"status": "finished", **r},
    )
    return r


def collect(out):
    rows = []
    for n in CONFIGS:
        for p in sorted((out / "form_2" / n).glob("**/result.json")):
            rows.append(json.loads(p.read_text(encoding="utf8")))
    mp = out / "metrics.json"
    met = json.loads(mp.read_text(encoding="utf8")) if mp.exists() else {}
    for r in rows:
        
        
        if r.get("structure_variant", "full") == "full":
            r.update(met.get(r["config"], {}))
        r["run_label"] = (
            r["config"]
            if r.get("structure_variant", "full") == "full"
            else f"{r['config']}/{r['structure_variant']}"
        )
    with open(out / "summary.csv", "w", newline="", encoding="utf8") as f:
        ks = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=ks)
        w.writeheader()
        w.writerows(rows)
    return {r["run_label"]: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument(
        "--variants", default="full", help="comma-separated: full,no_dirac,no_phase,no_harmonic"
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--microsteps", type=int, default=2)
    p.add_argument("--lpe-dim", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument(
        "--phase-init", choices=("zero", "deterministic", "random_learnable"), default="zero"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--output", default=str(ROOT / "runs" / "dkho" / "small"))
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.lpe_dim <= 0:
        p.error("--lpe-dim must be a positive integer")
    variants = tuple(x.strip() for x in a.variants.split(",") if x.strip())
    if not variants or any(x not in STRUCTURE_VARIANTS for x in variants):
        p.error(f"--variants must be drawn from {STRUCTURE_VARIANTS}")
    torch.set_num_threads(a.threads)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(DATA, "rb") as f:
        data = pickle.load(f)
    g = Geo(
        data["nodes"],
        data["elements"],
        data["inner_boundary"],
        data["outer_boundary"],
        out / "cache",
        a.lpe_dim,
    )
    hs = hstatus(np.asarray(data["Y_data"], np.float32), splits(len(data["X_data"]))["train"])
    (out / "harmonic_diagnosis.json").write_text(json.dumps(hs, indent=2), encoding="utf8")
    man = {
        "dataset": portable_path(DATA),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "device": str(resolve_device(a.device)),
        "args": portable_arguments(a),
        "mesh": {"n0": g.n0, "n1": g.n1, "n2": g.n2, "tets": len(g.t)},
        "split": {k: len(v) for k, v in splits(len(data["X_data"])).items()},
        "harmonic": hs,
        "protocol": "serial, exact HSD split/scale, unweighted 0/1/2 DK, vector-field loss only",
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf8")
    if "no_harmonic" in variants and not hs["harmonic_head_active"]:
        p.error("no_harmonic is undefined: the diagnosed full model has no explicit H0 head.")
    for n in a.configs.split(","):
        if n not in CONFIGS:
            raise KeyError(n)
        for variant in variants:
            run(CONFIGS[n], a, g, data, hs, out, variant)
        collect(out)
    print(json.dumps(collect(out), indent=2), flush=True)


if __name__ == "__main__":
    main()
