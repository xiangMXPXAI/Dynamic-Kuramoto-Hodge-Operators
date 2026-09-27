"""Compute Darcy solution, flux, and circulation cochain targets."""

import numpy as np
from skfem import Basis


def face_constant_gradient(basis: Basis, u: np.ndarray) -> np.ndarray:
    
    field = basis.interpolate(u)
    grad_qp = field.grad  
    spread = np.abs(grad_qp - grad_qp[:, :, :1]).max()
    assert (
        spread < 1e-8
    ), f"P1 gradient not constant across quadrature points (spread={spread}); unexpected element type"
    return grad_qp[:, :, 0].T  


def flux_to_edge_cochain(points, edges, face_edges_incidence, areas, flux_face):
    
    n_edges = edges.shape[0]
    weighted_sum = np.zeros((n_edges, 2))
    weight_total = np.zeros(n_edges)
    for f, inc_edges in enumerate(face_edges_incidence):
        for e in inc_edges:
            weighted_sum[e] += areas[f] * flux_face[f]
            weight_total[e] += areas[f]
    avg_flux = weighted_sum / weight_total[:, None]
    edge_vec = points[edges[:, 1]] - points[edges[:, 0]]
    q1 = np.einsum("ij,ij->i", avg_flux, edge_vec)
    return q1


def compute_p0_q1_omega2(basis, u, kappa_face, points, edges, d1, areas, face_edges_incidence):
    grad_face = face_constant_gradient(basis, u)  
    flux_face = -np.einsum("fij,fj->fi", kappa_face, grad_face)  
    q1 = flux_to_edge_cochain(points, edges, face_edges_incidence, areas, flux_face)
    omega2 = d1 @ q1
    return u, q1, omega2
