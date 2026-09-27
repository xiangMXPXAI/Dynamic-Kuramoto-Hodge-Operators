"""Generate Darcy samples and their cochain targets."""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from dec_darcy_holes import build_complex, compute_betti0, compute_beta1_and_harmonic
from anisotropy_darcy_holes import build_anisotropy
from source_field_darcy_holes import fourier_modes, sample_source_field
from solve_darcy_holes import build_basis, solve_darcy
from targets_darcy_holes import compute_p0_q1_omega2

SPLIT_SIZES = {
    "smoke": dict(train=64, val=16, test=32),
    "main": dict(train=4000, val=500, test=500),
}
BASE_SEED = 20260827  


def load_mesh(resolution):
    here = os.path.dirname(os.path.abspath(__file__))
    d = np.load(os.path.join(here, f"perforated_darcy_mesh_{resolution}_v1.npz"))
    return d


def generate(resolution: str, output_path: str | None = None, overwrite: bool = False):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(here), "data")
    filename = (
        "perforated_darcy_v1.npz"
        if resolution == "main"
        else "perforated_darcy_smoke_v1.npz"
    )
    out_path = output_path or os.path.join(data_dir, filename)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"output already exists: {out_path}; pass --overwrite to replace it")
    d = load_mesh(resolution)
    boundary = {
        "outer": d["boundary_edges_outer"],
        "hole1": d["boundary_edges_hole1"],
        "hole2": d["boundary_edges_hole2"],
    }
    cx = build_complex(d["points"], d["triangles"], boundary)
    n0 = cx["points"].shape[0]
    n1 = cx["edges"].shape[0]
    n2 = cx["triangles"].shape[0]

    beta0 = compute_betti0(cx["d0"], n0)
    beta1, harmonic_basis, harmonic_eigs = compute_beta1_and_harmonic(cx["d0"], cx["d1"], n1)
    print(f"[{resolution}] n0={n0} n1={n1} n2={n2} beta0={beta0} beta1={beta1}")
    assert beta0 == 1, f"expected a connected domain, got beta0={beta0}"
    assert beta1 == 2, f"expected two holes -> beta1=2, got beta1={beta1}"

    d1csr = cx["d1"].tocsr()
    face_edges_incidence = [d1csr.indices[d1csr.indptr[f] : d1csr.indptr[f + 1]] for f in range(n2)]

    aniso = build_anisotropy(n2, cx["edges"], face_edges_incidence, seed=BASE_SEED)
    mesh, basis = build_basis(cx["points"], cx["triangles"])
    modes = fourier_modes(3)

    sizes = SPLIT_SIZES[resolution]
    n_total = sizes["train"] + sizes["val"] + sizes["test"]
    rng = np.random.default_rng(BASE_SEED + 1)  

    f_samples = np.zeros((n_total, n0))
    g1_samples = np.zeros(n_total)
    g2_samples = np.zeros(n_total)
    p0_samples = np.zeros((n_total, n0))
    q1_samples = np.zeros((n_total, n1))
    omega2_samples = np.zeros((n_total, n2))

    t0 = time.time()
    for i in range(n_total):
        f_nodal = sample_source_field(cx["points"], rng, modes, domain_half_width=1.0)
        g1 = rng.uniform(0.5, 1.5)
        g2 = rng.uniform(0.5, 1.5)
        u = solve_darcy(
            cx["points"],
            cx["triangles"],
            aniso["kappa_face"],
            f_nodal,
            cx["node_component"],
            g1,
            g2,
        )
        p0, q1, omega2 = compute_p0_q1_omega2(
            basis,
            u,
            aniso["kappa_face"],
            cx["points"],
            cx["edges"],
            cx["d1"],
            cx["areas"],
            face_edges_incidence,
        )
        f_samples[i] = f_nodal
        g1_samples[i] = g1
        g2_samples[i] = g2
        p0_samples[i] = p0
        q1_samples[i] = q1
        omega2_samples[i] = omega2
        if (i + 1) % max(1, n_total // 10) == 0 or i == n_total - 1:
            elapsed = time.time() - t0
            print(
                f"[{resolution}] solved {i + 1}/{n_total} samples "
                f"({elapsed:.1f}s elapsed, {elapsed / (i + 1):.4f}s/sample)"
            )

    idx = rng.permutation(n_total)
    split_train = np.sort(idx[: sizes["train"]])
    split_val = np.sort(idx[sizes["train"] : sizes["train"] + sizes["val"]])
    split_test = np.sort(idx[sizes["train"] + sizes["val"] :])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        points=cx["points"],
        triangles=cx["triangles"],
        edges=cx["edges"],
        areas=cx["areas"],
        node_component=cx["node_component"],
        edge_component=cx["edge_component"],
        d0_row=cx["d0"].tocoo().row,
        d0_col=cx["d0"].tocoo().col,
        d0_data=cx["d0"].tocoo().data,
        d0_shape=np.array(cx["d0"].shape),
        d1_row=cx["d1"].tocoo().row,
        d1_col=cx["d1"].tocoo().col,
        d1_data=cx["d1"].tocoo().data,
        d1_shape=np.array(cx["d1"].shape),
        beta0=beta0,
        beta1=beta1,
        harmonic_basis_1=harmonic_basis,
        harmonic_eigs_1=harmonic_eigs,
        phi_face=aniso["phi_face"],
        kappa_face=aniso["kappa_face"],
        kappa_parallel=aniso["kappa_parallel"],
        kappa_perp=aniso["kappa_perp"],
        c_e=aniso["c_e"],
        c_f=aniso["c_f"],
        f_samples=f_samples.astype(np.float32),
        g1=g1_samples,
        g2=g2_samples,
        p0=p0_samples.astype(np.float32),
        q1=q1_samples.astype(np.float32),
        omega2=omega2_samples.astype(np.float32),
        split_train=split_train,
        split_val=split_val,
        split_test=split_test,
        hole1_center=d["hole1_center"],
        hole1_radius=d["hole1_radius"],
        hole2_center=d["hole2_center"],
        hole2_radius=d["hole2_radius"],
        domain_half_width=d["domain_half_width"],
    )
    print(f"[{resolution}] wrote {out_path}")

    manifest_path = (
        os.path.join(data_dir, "manifest.json" if resolution == "main" else f"manifest_{resolution}.json")
        if output_path is None
        else os.path.splitext(out_path)[0] + "_manifest.json"
    )
    manifest = dict(
        resolution=resolution,
        n0=n0,
        n1=n1,
        n2=n2,
        beta0=int(beta0),
        beta1=int(beta1),
        harmonic_eigs_1=[float(v) for v in harmonic_eigs],
        n_samples=dict(train=sizes["train"], val=sizes["val"], test=sizes["test"], total=n_total),
        seed_phi_face=BASE_SEED,
        seed_sampling=BASE_SEED + 1,
        kappa_parallel=aniso["kappa_parallel"],
        kappa_perp=aniso["kappa_perp"],
        hole1_center=list(map(float, d["hole1_center"])),
        hole1_radius=float(d["hole1_radius"]),
        hole2_center=list(map(float, d["hole2_center"])),
        hole2_radius=float(d["hole2_radius"]),
        domain="[-1,1]^2 minus 2 circular holes (TNO-style anisotropic Darcy)",
        f_generator="truncated Fourier random field, max_freq=3, sigma0=0.6, self-built (see source_field_darcy_holes.py)",
        g1_g2_distribution="U[0.5, 1.5] iid per sample",
        generation_time_seconds=time.time() - t0,
        p0_range=[float(p0_samples.min()), float(p0_samples.max())],
        q1_range=[float(q1_samples.min()), float(q1_samples.max())],
        omega2_range=[float(omega2_samples.min()), float(omega2_samples.max())],
        file_sha256=None,
    )
    with open(out_path, "rb") as fbin:
        manifest["file_sha256"] = hashlib.sha256(fbin.read()).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as fjson:
        json.dump(manifest, fjson, indent=2, ensure_ascii=False)
    print(f"[{resolution}] wrote {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["smoke", "main"], required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(args.resolution, args.output, args.overwrite)
