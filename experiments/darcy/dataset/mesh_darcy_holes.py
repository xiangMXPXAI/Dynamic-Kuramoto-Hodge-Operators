"""Create perforated-domain meshes for Darcy data generation."""

import argparse
from pathlib import Path

import numpy as np
import gmsh

DOMAIN_HALF_WIDTH = 1.0  
HOLE1_CENTER = (-0.45, 0.30)
HOLE1_RADIUS = 0.22
HOLE2_CENTER = (0.35, -0.40)
HOLE2_RADIUS = 0.35

RESOLUTIONS = {
    "smoke": (0.30, 0.08),
    "main": (0.05, 0.015),
    "refine": (0.03, 0.008),
}

def _classify_curve(dim, tag):
    
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    if abs(cx - HOLE1_CENTER[0]) < 1e-6 and abs(cy - HOLE1_CENTER[1]) < 1e-6:
        return "hole1"
    if abs(cx - HOLE2_CENTER[0]) < 1e-6 and abs(cy - HOLE2_CENTER[1]) < 1e-6:
        return "hole2"
    return "outer"

def build_mesh(resolution: str):
    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r}, expected one of {list(RESOLUTIONS)}")
    h_far, h_near = RESOLUTIONS[resolution]

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"darcy_holes_{resolution}")

    L = DOMAIN_HALF_WIDTH
    outer = gmsh.model.occ.addRectangle(-L, -L, 0, 2 * L, 2 * L, tag=1)
    hole1 = gmsh.model.occ.addDisk(
        HOLE1_CENTER[0], HOLE1_CENTER[1], 0, HOLE1_RADIUS, HOLE1_RADIUS, tag=2
    )
    hole2 = gmsh.model.occ.addDisk(
        HOLE2_CENTER[0], HOLE2_CENTER[1], 0, HOLE2_RADIUS, HOLE2_RADIUS, tag=3
    )
    domain, _ = gmsh.model.occ.cut([(2, outer)], [(2, hole1), (2, hole2)], removeTool=True)
    gmsh.model.occ.synchronize()

    curves = gmsh.model.getEntities(1)
    groups = {"outer": [], "hole1": [], "hole2": []}
    for dim, tag in curves:
        groups[_classify_curve(dim, tag)].append(tag)

    assert (
        len(groups["hole1"]) > 0
    ), "hole1 boundary curve not found -- geometry classification failed"
    assert (
        len(groups["hole2"]) > 0
    ), "hole2 boundary curve not found -- geometry classification failed"
    assert (
        len(groups["outer"]) == 4
    ), f"expected 4 outer rectangle edges, got {len(groups['outer'])}"

    gmsh.model.addPhysicalGroup(1, groups["outer"], tag=101, name="outer")
    gmsh.model.addPhysicalGroup(1, groups["hole1"], tag=102, name="hole1")
    gmsh.model.addPhysicalGroup(1, groups["hole2"], tag=103, name="hole2")
    surf_tags = [t for (d, t) in domain if d == 2]
    gmsh.model.addPhysicalGroup(2, surf_tags, tag=201, name="domain")

    
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", groups["hole1"] + groups["hole2"])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_near)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
    points = node_coords.reshape(-1, 3)[:, :2].copy()

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, -1)
    triangles = None
    for et, en in zip(elem_types, elem_node_tags):
        if et == 2:  
            tri = en.reshape(-1, 3).astype(np.int64)
            tri = np.vectorize(tag_to_index.get)(tri)
            triangles = tri if triangles is None else np.vstack([triangles, tri])
    assert triangles is not None, "no triangle elements found"

    def boundary_edges_of(curve_tags):
        pairs = []
        for ct in curve_tags:
            etypes, _, enodes = gmsh.model.mesh.getElements(1, ct)
            for et, en in zip(etypes, enodes):
                if et == 1:  
                    seg = en.reshape(-1, 2).astype(np.int64)
                    seg = np.vectorize(tag_to_index.get)(seg)
                    pairs.append(seg)
        return np.vstack(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)

    edges_outer = boundary_edges_of(groups["outer"])
    edges_hole1 = boundary_edges_of(groups["hole1"])
    edges_hole2 = boundary_edges_of(groups["hole2"])

    gmsh.finalize()

    return dict(
        points=points,
        triangles=triangles,
        boundary_edges_outer=edges_outer,
        boundary_edges_hole1=edges_hole1,
        boundary_edges_hole2=edges_hole2,
        hole1_center=np.array(HOLE1_CENTER),
        hole1_radius=np.float64(HOLE1_RADIUS),
        hole2_center=np.array(HOLE2_CENTER),
        hole2_radius=np.float64(HOLE2_RADIUS),
        domain_half_width=np.float64(DOMAIN_HALF_WIDTH),
    )


def main(resolutions: tuple[str, ...], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for res in resolutions:
        output = output_dir / f"perforated_darcy_mesh_{res}_v1.npz"
        if output.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
        data = build_mesh(res)
        n0, n2 = len(data["points"]), len(data["triangles"])
        print(
            f"[{res}] nodes={n0} triangles={n2} "
            f"outer_edges={len(data['boundary_edges_outer'])} "
            f"hole1_edges={len(data['boundary_edges_hole1'])} "
            f"hole2_edges={len(data['boundary_edges_hole2'])}"
        )
        np.savez(output, **data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=(*RESOLUTIONS, "all"), default="main")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = tuple(RESOLUTIONS) if args.resolution == "all" else (args.resolution,)
    main(selected, args.output_dir, args.overwrite)
