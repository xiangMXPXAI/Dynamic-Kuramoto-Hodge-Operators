"""Construct anisotropic permeability fields for the Darcy dataset."""

from pathlib import Path

import numpy as np

KAPPA_PARALLEL = 4.0
KAPPA_PERP = 1.0


def build_anisotropy(n_faces, edges, face_edges_signed, seed=20260827):
    
    rng = np.random.default_rng(seed)
    phi_f = rng.uniform(0.0, np.pi, size=n_faces)

    cos2 = np.cos(2.0 * phi_f)
    sin2 = np.sin(2.0 * phi_f)
    
    
    trace_half = 0.5 * (KAPPA_PARALLEL + KAPPA_PERP)
    diff_half = 0.5 * (KAPPA_PARALLEL - KAPPA_PERP)
    kappa_f = np.zeros((n_faces, 2, 2), dtype=np.float64)
    kappa_f[:, 0, 0] = trace_half + diff_half * cos2
    kappa_f[:, 1, 1] = trace_half - diff_half * cos2
    kappa_f[:, 0, 1] = diff_half * sin2
    kappa_f[:, 1, 0] = diff_half * sin2

    n_edges = edges.shape[0]
    c_f = cos2  
    c_e_sum = np.zeros(n_edges)
    c_e_count = np.zeros(n_edges)
    for f in range(n_faces):
        for eidx in face_edges_signed[f]:
            c_e_sum[eidx] += c_f[f]
            c_e_count[eidx] += 1
    c_e = c_e_sum / np.maximum(c_e_count, 1)

    return dict(
        phi_face=phi_f,
        kappa_face=kappa_f,
        c_e=c_e,
        c_f=c_f,
        kappa_parallel=KAPPA_PARALLEL,
        kappa_perp=KAPPA_PERP,
    )


if __name__ == "__main__":
    import numpy as np
    from dec_darcy_holes import build_complex

    d = np.load(Path(__file__).resolve().parent / "perforated_darcy_mesh_smoke_v1.npz")
    boundary = {
        "outer": d["boundary_edges_outer"],
        "hole1": d["boundary_edges_hole1"],
        "hole2": d["boundary_edges_hole2"],
    }
    cx = build_complex(d["points"], d["triangles"], boundary)

    
    d1 = cx["d1"].tocsr()
    face_edges = [d1.indices[d1.indptr[f] : d1.indptr[f + 1]] for f in range(d1.shape[0])]

    aniso = build_anisotropy(cx["triangles"].shape[0], cx["edges"], face_edges)
    print("phi_face range:", aniso["phi_face"].min(), aniso["phi_face"].max())
    print("c_f range:", aniso["c_f"].min(), aniso["c_f"].max(), "mean:", aniso["c_f"].mean())
    print("c_e range:", aniso["c_e"].min(), aniso["c_e"].max(), "mean:", aniso["c_e"].mean())
    
    eigvals = np.linalg.eigvalsh(aniso["kappa_face"])
    print(
        "kappa eigenvalue check: min/max over all faces of sorted eigs:",
        eigvals.min(axis=0),
        eigvals.max(axis=0),
    )
    assert np.allclose(np.sort(eigvals, axis=1), np.array([1.0, 4.0]), atol=1e-10)
    print("anisotropy_darcy_holes.py self-test PASSED")
