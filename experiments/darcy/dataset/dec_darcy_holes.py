"""Build discrete exterior calculus operators for perforated Darcy meshes."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh


def _ensure_ccw(points, triangles):
    tris = triangles.copy()
    p = points
    x0, y0 = p[tris[:, 0], 0], p[tris[:, 0], 1]
    x1, y1 = p[tris[:, 1], 0], p[tris[:, 1], 1]
    x2, y2 = p[tris[:, 2], 0], p[tris[:, 2], 1]
    signed_area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    assert np.all(np.abs(signed_area2) > 1e-14), "degenerate (zero-area) triangle found"
    flip = signed_area2 < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    areas = np.abs(signed_area2) / 2.0
    return tris, areas


def build_complex(points, triangles, boundary_edges_by_component):
    
    tris, areas = _ensure_ccw(points, triangles)
    n0 = points.shape[0]

    edge_index = {}
    edges = []
    face_edges = []  
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        fe = []
        for p_, q_ in ((a, b), (b, c), (c, a)):
            key = (min(p_, q_), max(p_, q_))
            if key not in edge_index:
                edge_index[key] = len(edges)
                edges.append(key)
            eidx = edge_index[key]
            sign = 1 if (p_, q_) == key else -1
            fe.append((eidx, sign))
        face_edges.append(fe)
    edges = np.array(edges, dtype=np.int64)
    n1 = len(edges)
    n2 = len(tris)

    rows = np.repeat(np.arange(n1), 2)
    cols = edges.reshape(-1)
    vals = np.tile(np.array([-1.0, 1.0]), n1)
    d0 = coo_matrix((vals, (rows, cols)), shape=(n1, n0)).tocsr()

    rows2, cols2, vals2 = [], [], []
    for f, fe in enumerate(face_edges):
        for eidx, sign in fe:
            rows2.append(f)
            cols2.append(eidx)
            vals2.append(float(sign))
    d1 = coo_matrix((vals2, (rows2, cols2)), shape=(n2, n1)).tocsr()

    residual = d1 @ d0
    max_residual = np.abs(residual).max() if residual.nnz > 0 else 0.0
    assert max_residual < 1e-9, f"d1 @ d0 != 0 (max entry {max_residual}); orientation bug"

    
    node_component = np.zeros(n0, dtype=np.int64)
    edge_component = np.zeros(n1, dtype=np.int64)
    for comp_id, (name, pairs) in enumerate(boundary_edges_by_component.items(), start=1):
        nodes_here = np.unique(pairs.reshape(-1))
        overlap = node_component[nodes_here] != 0
        assert not np.any(overlap), f"node boundary components overlap at component '{name}'"
        node_component[nodes_here] = comp_id
        for p_, q_ in pairs:
            key = (min(int(p_), int(q_)), max(int(p_), int(q_)))
            eidx = edge_index.get(key)
            assert eidx is not None, f"boundary edge {key} of component '{name}' is not a mesh edge"
            assert (
                edge_component[eidx] == 0
            ), f"edge boundary components overlap at component '{name}'"
            edge_component[eidx] = comp_id

    return dict(
        points=points,
        triangles=tris,
        areas=areas,
        edges=edges,
        d0=d0,
        d1=d1,
        node_component=node_component,
        edge_component=edge_component,
        component_names=["interior"] + list(boundary_edges_by_component.keys()),
    )


def compute_betti0(d0, n0):
    
    n1 = d0.shape[0]
    edge_ij = np.array([d0.indices[d0.indptr[e] : d0.indptr[e + 1]] for e in range(n1)])
    graph = coo_matrix((np.ones(len(edge_ij)), (edge_ij[:, 0], edge_ij[:, 1])), shape=(n0, n0))
    n_components, _ = connected_components(graph, directed=False)
    return n_components


def compute_beta1_and_harmonic(d0, d1, n1, tol=1e-8, k_probe=8):
    
    delta1 = (d0 @ d0.T) + (d1.T @ d1)
    delta1 = delta1.tocsr()
    k = min(k_probe, n1 - 2)
    shift = 1e-8  
    
    while True:
        vals, vecs = eigsh(delta1, k=k, sigma=shift, which="LM")
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        n_zero = int(np.sum(vals < tol))
        if n_zero < k or k >= n1 - 2:
            break
        k = min(k * 2, n1 - 2)
    beta1 = n_zero
    psi = vecs[:, :beta1]
    
    if beta1 > 0:
        psi, _ = np.linalg.qr(psi)
    return beta1, psi, vals[: max(beta1, 1)]


if __name__ == "__main__":

    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    boundary = {
        "outer": np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
    }
    cx = build_complex(pts, tris, boundary)
    b0 = compute_betti0(cx["d0"], pts.shape[0])
    b1, psi, vals = compute_beta1_and_harmonic(cx["d0"], cx["d1"], cx["edges"].shape[0])
    print("self-test: n0,n1,n2 =", pts.shape[0], cx["edges"].shape[0], tris.shape[0])
    print("self-test: beta0 =", b0, "(expected 1)")
    print("self-test: beta1 =", b1, "(expected 0, simply-connected square)")
    assert b0 == 1 and b1 == 0
    print("dec_darcy_holes.py self-test PASSED")
