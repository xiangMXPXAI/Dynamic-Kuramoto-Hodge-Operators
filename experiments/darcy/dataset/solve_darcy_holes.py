"""Assemble and solve the Darcy equation on perforated meshes."""

from pathlib import Path

import numpy as np
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm, condense, solve
from skfem.helpers import grad


def build_basis(points, triangles):
    mesh = MeshTri(points.T.copy().astype(np.float64), triangles.T.copy().astype(np.int64))
    basis = Basis(mesh, ElementTriP1())
    return mesh, basis


@BilinearForm
def _aniso_stiffness(u, v, w):
    gu, gv = grad(u), grad(v)
    kxx, kxy, kyy = w["kxx"], w["kxy"], w["kyy"]
    return kxx * gu[0] * gv[0] + kxy * (gu[0] * gv[1] + gu[1] * gv[0]) + kyy * gu[1] * gv[1]


@LinearForm
def _load(v, w):
    return w["f"] * v


def solve_darcy(points, triangles, kappa_face, f_nodal, node_component, g1_value, g2_value):
    
    mesh, basis = build_basis(points, triangles)
    n_elem = triangles.shape[0]

    kxx = kappa_face[:, 0, 0].reshape(n_elem, 1)
    kxy = kappa_face[:, 0, 1].reshape(n_elem, 1)
    kyy = kappa_face[:, 1, 1].reshape(n_elem, 1)

    A = _aniso_stiffness.assemble(basis, kxx=kxx, kxy=kxy, kyy=kyy)
    b = _load.assemble(basis, f=basis.interpolate(f_nodal))

    x = basis.zeros()
    dofs_outer = np.where(node_component == 1)[0]
    dofs_hole1 = np.where(node_component == 2)[0]
    dofs_hole2 = np.where(node_component == 3)[0]
    x[dofs_outer] = 0.0
    x[dofs_hole1] = g1_value
    x[dofs_hole2] = g2_value
    D = np.concatenate([dofs_outer, dofs_hole1, dofs_hole2])

    Acond, bcond, xcond, I = condense(A, b, x=x, D=D)
    xcond[I] = solve(Acond, bcond)
    return xcond


if __name__ == "__main__":
    
    from dec_darcy_holes import build_complex
    from skfem.models.poisson import laplace as skfem_laplace

    d = np.load(Path(__file__).resolve().parent / "perforated_darcy_mesh_smoke_v1.npz")
    boundary = {
        "outer": d["boundary_edges_outer"],
        "hole1": d["boundary_edges_hole1"],
        "hole2": d["boundary_edges_hole2"],
    }
    cx = build_complex(d["points"], d["triangles"], boundary)
    n0, n2 = cx["points"].shape[0], cx["triangles"].shape[0]

    kappa_iso = np.zeros((n2, 2, 2))
    kappa_iso[:, 0, 0] = 1.0
    kappa_iso[:, 1, 1] = 1.0
    f_zero = np.zeros(n0)

    u = solve_darcy(
        cx["points"],
        cx["triangles"],
        kappa_iso,
        f_zero,
        cx["node_component"],
        g1_value=1.0,
        g2_value=1.0,
    )
    print("isotropic harmonic solution: min,max =", u.min(), u.max())
    assert -1e-8 <= u.min() and u.max() <= 1.0 + 1e-8, "maximum principle violated"

    mesh, basis = build_basis(cx["points"], cx["triangles"])
    A_mine = _aniso_stiffness.assemble(
        basis, kxx=np.ones((n2, 1)), kxy=np.zeros((n2, 1)), kyy=np.ones((n2, 1))
    )
    A_skfem = skfem_laplace.assemble(basis)
    diff = np.abs((A_mine - A_skfem).toarray()).max()
    print("max |A_mine - A_skfem(laplace)| for isotropic kappa=I:", diff)
    assert (
        diff < 1e-10
    ), "custom anisotropic bilinear form does not reduce to skfem's laplace for kappa=I"
    print("solve_darcy_holes.py self-test PASSED")
