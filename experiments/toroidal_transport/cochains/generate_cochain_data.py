"""Build C0, C1, and C2 targets from toroidal transport trajectories."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "torus_transport_v1.pkl"


def oriented_complex(
    faces: np.ndarray, n0: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    
    edge_to_id: dict[tuple[int, int], int] = {}
    face_edges = np.empty((len(faces), 3), np.int64)
    face_signs = np.empty((len(faces), 3), np.float32)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for fi, (a0, b0, c0) in enumerate(np.asarray(faces, np.int64)):
        for li, (a, b) in enumerate(((int(a0), int(b0)), (int(b0), int(c0)), (int(c0), int(a0)))):
            key = (min(a, b), max(a, b))
            edge_to_id.setdefault(key, len(edge_to_id))
            ei = edge_to_id[key]
            sign = 1.0 if (a, b) == key else -1.0
            face_edges[fi, li] = ei
            face_signs[fi, li] = sign
            rows.append(fi)
            cols.append(ei)
            values.append(sign)
    edges = np.asarray(list(edge_to_id), np.int64)
    d0 = sparse.csr_matrix(
        (
            np.tile(np.asarray([-1.0, 1.0], np.float32), len(edges)),
            (np.repeat(np.arange(len(edges)), 2), edges.reshape(-1)),
        ),
        shape=(len(edges), n0),
    )
    d1 = sparse.csr_matrix(
        (np.asarray(values, np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(len(faces), len(edges)),
    )
    residual = d1 @ d0
    if residual.nnz and np.max(np.abs(residual.data)) > 1e-6:
        raise RuntimeError("Invalid orientation: d1 @ d0 is not zero")
    return edges, face_edges, face_signs, d0, d1


def tangential_velocity(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    raw = np.stack((-points[:, 1], points[:, 0], np.zeros(len(points))), axis=1)
    return (raw - (raw * normals).sum(1, keepdims=True) * normals).astype(np.float32)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "torus_multiform_v1",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Optional prefix length for a smoke archive; 0 preserves all source samples.",
    )
    p.add_argument("--diffusivity", type=float, default=0.01)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse()
    source, out = args.source.resolve(), args.out.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists; pass --overwrite to replace this generated archive"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with source.open("rb") as f:
        raw = pickle.load(f)
    traj = np.asarray(raw["trajectories"], dtype=np.float32)
    if args.samples:
        if args.samples < 1:
            raise ValueError("--samples must be positive when specified")
        traj = traj[: min(int(args.samples), len(traj))]
    if traj.ndim != 3:
        raise ValueError(f"Expected (sample,time,node) trajectories, got {traj.shape}")
    ns, nt, n0 = traj.shape
    points = np.asarray(raw["points"], np.float32)
    faces = np.asarray(raw["faces"], np.int64)
    normals = np.asarray(raw["normals"], np.float32)
    if len(points) != n0:
        raise ValueError("trajectory node count differs from mesh")
    edges, face_edges, face_signs, d0, d1 = oriented_complex(faces, n0)
    edge_vec = points[edges[:, 1]] - points[edges[:, 0]]
    edge_mid = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    edge_length = np.linalg.norm(edge_vec, axis=1).astype(np.float32)
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normal_raw = np.cross(b - a, c - a)
    doubled = np.linalg.norm(normal_raw, axis=1, keepdims=True)
    area = (0.5 * doubled[:, 0]).astype(np.float32)
    face_normal = (normal_raw / np.maximum(doubled, 1e-12)).astype(np.float32)
    face_mid = ((a + b + c) / 3).astype(np.float32)
    node_area = np.zeros(n0, np.float32)
    for col in range(3):
        np.add.at(node_area, faces[:, col], area / 3)
    v0 = tangential_velocity(points, normals)
    v1 = (0.5 * (v0[edges[:, 0]] + v0[edges[:, 1]]) * edge_vec).sum(1).astype(np.float32)
    
    
    u0 = np.lib.format.open_memmap(out / "u0.npy", mode="w+", dtype=np.float32, shape=(ns, n0))
    ut = np.lib.format.open_memmap(out / "uT0.npy", mode="w+", dtype=np.float32, shape=(ns, n0))
    q1 = np.lib.format.open_memmap(
        out / "qT1.npy", mode="w+", dtype=np.float32, shape=(ns, len(edges))
    )
    m2 = np.lib.format.open_memmap(
        out / "mT2.npy", mode="w+", dtype=np.float32, shape=(ns, len(faces))
    )
    started = time.time()
    for start in range(0, ns, args.chunk):
        stop = min(ns, start + args.chunk)
        terminal = traj[start:stop, -1]
        u0[start:stop] = traj[start:stop, 0]
        ut[start:stop] = terminal
        ue = 0.5 * (terminal[:, edges[:, 0]] + terminal[:, edges[:, 1]])
        
        q1[start:stop] = ue * v1[None] - args.diffusivity * (
            terminal[:, edges[:, 1]] - terminal[:, edges[:, 0]]
        )
        
        m2[start:stop] = terminal[:, faces].mean(2) * area[None]
        rate = stop / max(time.time() - started, 1e-8)
        eta = (ns - stop) / max(rate, 1e-8)
        print(
            f"[forms] {stop:4d}/{ns} ({100*stop/ns:6.2f}%) {rate:6.1f} samples/s ETA {eta:5.1f}s",
            flush=True,
        )
    for arr in (u0, ut, q1, m2):
        arr.flush()
    static = {
        "points": points,
        "faces": faces,
        "normals": normals,
        "edges": edges,
        "face_edges": face_edges,
        "face_signs": face_signs,
        "edge_vec": edge_vec.astype(np.float32),
        "edge_mid": edge_mid,
        "edge_length": edge_length[:, None],
        "face_area": area[:, None],
        "face_normal": face_normal,
        "face_mid": face_mid,
        "node_area": node_area[:, None],
        "velocity0": v0,
        "velocity1": v1[:, None],
    }
    for name, value in static.items():
        np.save(out / f"{name}.npy", value)
    
    sample = np.asarray(q1[: min(ns, 64)])
    curl = np.asarray(sample @ d1.T)
    metadata = {
        "revision": "torus_multiform_v1",
        "source": source.resolve().relative_to(ROOT).as_posix(),
        "samples": int(ns),
        "trajectory_steps": int(nt - 1),
        "n0": int(n0),
        "n1": int(len(edges)),
        "n2": int(len(faces)),
        "betti_expected": [1, 2, 1],
        "diffusivity": args.diffusivity,
        "task0": "terminal concentration u(T) in C0",
        "task1": "terminal covariant constitutive flux q^flat=u v^flat-kappa du in C1",
        "task2": "terminal face-integrated mass m=u(T)dA in C2",
        "source_velocity": "fixed tangential projection of (-y,x,0)",
        "d1d0_max": float(np.max(np.abs((d1 @ d0).data)) if (d1 @ d0).nnz else 0.0),
        "mean_abs_d1_q1": float(np.abs(curl).mean()),
        "note": "qT1/mT2 are terminal constitutive observables. Archived trajectories do not retain the original time-step edge flux, so exact stepwise DEC continuity is not asserted.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
