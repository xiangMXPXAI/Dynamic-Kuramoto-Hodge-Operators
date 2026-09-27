"""Evaluate saved DKHO and baseline predictions for toroidal transport."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

import train as tdk


ROOT = Path(__file__).resolve().parent


RUNS_ROOT = ROOT / "runs" / "dkho"

HSD_DIR = ROOT / "baselines"
OUT = RUNS_ROOT / "small" / "form_0"
sys.path.insert(0, str(HSD_DIR))
from spectral_operators import HighOrderSpectralOperators  
from dataset import DataManager  
from models import (
    GNO,
    FNO3d,
    LightweightMGN,
    DeepONet,
    GeoFNO,
    SpectralPhysicsAwareOperator,
    HSDSpectralFNO,
)  


def configure_style() -> None:
    
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#fbfcfe",
            "axes.facecolor": "#fbfcfe",
            "savefig.facecolor": "#fbfcfe",
            "axes.edgecolor": "#bbc6d2",
            "axes.labelcolor": "#2b4058",
            "xtick.color": "#5a6f85",
            "ytick.color": "#5a6f85",
            "grid.color": "#d9e2eb",
            "grid.alpha": 0.75,
            "axes.titleweight": "semibold",
        }
    )


def clean(axis, grid: str = "x") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid, zorder=0)


def presentation_label(name: str) -> str:
    
    labels = {
        "native": "native",
        "flow": "flow",
        "flow_diffusion": "flow + diffusion",
        "flow_diffusion_spectral": "flow + diffusion + spectral",
        "flow_global_spectral": "flow + global spectral",
        "conditioned": "conditioned",
    }
    if name.startswith("TDK-small"):
        return "TDK-HO small: " + labels.get(name.rsplit("/", 1)[-1], name.rsplit("/", 1)[-1])
    if name.startswith("TDK-large"):
        return "TDK-HO large: " + labels.get(name.rsplit("/", 1)[-1], name.rsplit("/", 1)[-1])
    return name


class LegacyMGN(torch.nn.Module):
    

    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int, hidden: int, layers: int):
        super().__init__()
        self.node_encoder = torch.nn.Linear(node_in_dim, hidden)
        self.edge_encoder = torch.nn.Linear(edge_in_dim, hidden)

        def block(in_dim: int):
            return torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden),
                torch.nn.LayerNorm(hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
                torch.nn.LayerNorm(hidden),
            )

        self.edge_mlps = torch.nn.ModuleList([block(hidden * 3) for _ in range(layers)])
        self.node_mlps = torch.nn.ModuleList([block(hidden * 2) for _ in range(layers)])
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, out_dim)
        )
        self.num_layers = layers

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr = self.node_encoder(x), self.edge_encoder(edge_attr)
        src, dst = edge_index
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            edge_attr = edge_mlp(torch.cat([x[src], x[dst], edge_attr], 1))
            aggregate = torch.zeros_like(x)
            aggregate.index_add_(0, dst, edge_attr)
            x = x + node_mlp(torch.cat([x, aggregate], 1))
        return self.decoder(x)


def split(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tv, test = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    train, val = train_test_split(tv, test_size=0.15, random_state=42)
    return np.sort(train), np.sort(val), np.sort(test)


def node_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = points[faces]
    a = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    out = np.zeros(len(points), dtype=np.float32)
    for local in range(3):
        np.add.at(out, faces[:, local], a / 3)
    return out


def adjacency(faces: np.ndarray, n: int) -> csr_matrix:
    rows, cols = [], []
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            rows += [a, b]
            cols += [b, a]
    return csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))


def fidelity_from_energy(ep: np.ndarray, ey: np.ndarray) -> float:
    return float(np.mean(1 / (1 + np.abs(np.log((ep + 1e-10) / (ey + 1e-10))))))


def level_metrics(
    pred: np.ndarray, target: np.ndarray, graph: csr_matrix, area: np.ndarray
) -> tuple[float, float]:
    def components(mask: np.ndarray) -> int:
        
        active = np.flatnonzero(mask)
        if len(active) == 0:
            return 0
        
        
        induced = graph[active, :][:, active]
        return int(connected_components(induced, directed=False)[0])

    beta_scores, ious = [], []
    for p, y in zip(pred, target):
        ymin, ymax = float(y.min()), float(y.max())
        for level in (0.2, 0.5, 0.8):
            threshold = ymin + level * (ymax - ymin)
            cp, cy = components(p >= threshold), components(y >= threshold)
            beta_scores.append(np.exp(-1.5 * abs(cp - cy) / max(cp, cy, 1)))
        local = []
        weights = []
        for level in np.linspace(ymin, ymax, 12)[1:-1]:
            pm, ym = p >= level, y >= level
            union = np.sum(area * (pm | ym))
            local.append(np.sum(area * (pm & ym)) / max(union, 1e-12))
            pos = (level - ymin) / max(ymax - ymin, 1e-12)
            weights.append(np.exp(-4 * (pos - 0.5) ** 2))
        ious.append(np.sum(np.asarray(local) * np.asarray(weights) / np.sum(weights)))
    return float(np.mean(beta_scores)), float(np.sqrt(np.mean(ious)))


def common_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    host: HighOrderSpectralOperators,
    points: np.ndarray,
    faces: np.ndarray,
) -> dict:
    b1 = host.B1.tocsr()  
    grad_p, grad_y = b1.dot(pred.T).T, b1.dot(target.T).T
    
    
    dot = np.sum(grad_p * grad_y, axis=1)
    grad_fid = np.mean(
        (1 + dot / (np.linalg.norm(grad_p, axis=1) * np.linalg.norm(grad_y, axis=1) + 1e-10)) / 2
    )
    e_p, e_y = np.sum(grad_p**2, axis=1), np.sum(grad_y**2, axis=1)
    lp, ly = b1.T.dot(grad_p.T).T, b1.T.dot(grad_y.T).T
    z_p, z_y = np.sum(lp**2, axis=1), np.sum(ly**2, axis=1)
    area = node_areas(points, faces)
    phi = host.Phi0[:, :20].astype(np.float64)
    eig = np.array([v @ (host.L0 @ v) / max(v @ v, 1e-12) for v in phi.T])
    cp, cy = pred @ (phi * area[:, None]), target @ (phi * area[:, None])
    w = 1 / (eig + 0.1)
    w /= w.sum()
    spec_rel = np.sqrt(np.sum(w * (cp - cy) ** 2, axis=1)) / (
        np.sqrt(np.sum(w * cy**2, axis=1)) + 1e-10
    )
    b0, iou = level_metrics(pred, target, adjacency(faces, len(points)), area)
    
    
    
    
    
    coo = b1.tocoo()
    tails = np.empty(b1.shape[0], dtype=np.int64)
    heads = tails.copy()
    tails[coo.row[coo.data < 0]] = coo.col[coo.data < 0]
    heads[coo.row[coo.data > 0]] = coo.col[coo.data > 0]
    raw_velocity = np.c_[-points[:, 1], points[:, 0], np.zeros(len(points))]
    radial_xy = np.linalg.norm(points[:, :2], axis=1)
    major = float(radial_xy.mean())
    normals = np.c_[
        points[:, 0] * (1 - major / np.maximum(radial_xy, 1e-12)),
        points[:, 1] * (1 - major / np.maximum(radial_xy, 1e-12)),
        points[:, 2],
    ]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    velocity = raw_velocity - np.sum(raw_velocity * normals, axis=1, keepdims=True) * normals
    edge_vector = points[heads] - points[tails]
    velocity_edge = np.sum(0.5 * (velocity[tails] + velocity[heads]) * edge_vector, axis=1)

    def transport_flux(field: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return 0.5 * (field[:, tails] + field[:, heads]) * velocity_edge[None] - 0.01 * gradient

    jp, jy = transport_flux(pred, grad_p), transport_flux(target, grad_y)
    div_p, div_y = b1.T.dot(jp.T).T, b1.T.dot(jy.T).T
    curl_p, curl_y = host.B2.tocsr().dot(jp.T).T, host.B2.tocsr().dot(jy.T).T
    div_rel = np.linalg.norm(div_p - div_y, axis=1) / (np.linalg.norm(div_y, axis=1) + 1e-10)
    vort_rel = np.linalg.norm(curl_p - curl_y, axis=1) / (np.linalg.norm(curl_y, axis=1) + 1e-10)
    return {
        "MSE": float(np.mean((pred - target) ** 2)),
        "Div_Fid": float(np.mean(np.exp(-div_rel))),
        "Curl_MSE": float(np.mean((curl_p - curl_y) ** 2)),
        "Vort_Fid": float(np.mean(np.exp(-vort_rel))),
        "Grad_Fid": float(grad_fid),
        "Enst_Fid": fidelity_from_energy(z_p, z_y),
        "Energy_Fid": fidelity_from_energy(e_p, e_y),
        "Spec_Fid": float(np.mean(np.exp(-2 * spec_rel))),
        "S_beta0": b0,
        "IoU": iou,
    }


@torch.no_grad()
def baseline_predictions(
    data: dict, host: HighOrderSpectralOperators, x: np.ndarray, device: torch.device
) -> dict[str, np.ndarray]:
    pts, faces = data["points"], data["faces"]
    old_out = ROOT / "runs" / "baseline" / "legacy" / "scalar_native"
    xscale = float(
        np.max(
            np.abs(
                np.asarray(data["trajectories"], dtype=np.float32)[
                    np.r_[split(len(data["trajectories"]))[0], split(len(data["trajectories"]))[1]],
                    0,
                ]
            )
        )
        + 1e-9
    )
    yscale = float(
        np.max(
            np.abs(
                np.asarray(data["trajectories"], dtype=np.float32)[
                    np.r_[split(len(data["trajectories"]))[0], split(len(data["trajectories"]))[1]],
                    -1,
                ]
            )
        )
        + 1e-9
    )
    xt = torch.from_numpy(x / xscale).float().to(device)
    dm = DataManager(len(pts), pts, device, faces=faces, grid_res=16)
    models = {
        "GNO": GNO(4, 1, 120, 68, 3, 0.15),
        "FNO": FNO3d(2, 1, 20, (4, 4, 4), 3),
        "MGN": LegacyMGN(4, 3, 1, 72, 8),
        "DeepONet": DeepONet(len(pts), 3, 1, [96, 96, 64], [68, 68, 64], 64),
        "GeoFNO": GeoFNO(6, 8, 4, 16),
        "HSD": HSDSpectralFNO(
            SpectralPhysicsAwareOperator(host.Md0, host.Md1, 64, 64, 64, (32, 32)),
            dm,
            torch.from_numpy(host.Phi0[:, :64].astype(np.float32)).to(device),
            (4, 4, 4),
            12,
            6,
        ),
    }
    for name, model in models.items():
        model.load_state_dict(
            torch.load(old_out / f"model_{name.lower()}.pt", map_location=device), strict=True
        )
        model.to(device).eval()
    pred: dict[str, np.ndarray] = {}
    
    
    xin, _ = dm.prepare_gno_batch(xt, None)
    pred["GNO"] = models["GNO"](xin, dm.pts).squeeze(1).cpu().numpy() * yscale
    vals = []
    for i in range(0, len(xt), 16):
        grid, _ = dm.prepare_fno_batch(xt[i : i + 16], None)
        vals.append(dm.decode_fno_output(models["FNO"](grid)).squeeze(-1).cpu().numpy())
    pred["FNO"] = np.concatenate(vals) * yscale
    values = []
    for i in range(len(xt)):
        feat = torch.cat([xt[i : i + 1].T, dm.pts], dim=1)
        values.append(models["MGN"](feat, dm.edge_index, dm.edge_attr).squeeze(-1).cpu().numpy())
    pred["MGN"] = np.asarray(values) * yscale
    pred["DeepONet"] = models["DeepONet"](xt, dm.pts).squeeze(-1).cpu().numpy() * yscale
    vals = []
    for i in range(0, len(xt), 16):
        coords, feat = dm.prepare_geofno_batch(xt[i : i + 16])
        vals.append(models["GeoFNO"](coords, feat).squeeze(-1).cpu().numpy())
    pred["GeoFNO"] = np.concatenate(vals) * yscale
    
    
    
    state_hsd = torch.load(old_out / "model_hsd.pt", map_location="cpu")
    phi0_saved = state_hsd["Phi0"].numpy().astype(np.float32)
    md0_saved = state_hsd["Md0"].numpy().astype(np.float32)
    c0_hsd = torch.from_numpy(xt.cpu().numpy() @ phi0_saved).to(device)
    
    
    c1_hsd = torch.matmul(c0_hsd, torch.from_numpy(md0_saved.T).to(device))
    c2_hsd = torch.zeros(len(xt), 64, device=device)
    vals = []
    for i in range(0, len(xt), 12):
        vals.append(
            models["HSD"](
                c0_hsd[i : i + 12], c1_hsd[i : i + 12], c2_hsd[i : i + 12], xt[i : i + 12]
            )[0]
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    pred["HSD"] = np.concatenate(vals) * yscale
    return pred


@torch.no_grad()
def tdk_predictions(data: dict, device: torch.device, lpe_dim: int) -> dict[str, np.ndarray]:
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    _, _, test = split(len(trajectories))
    geo = tdk.Geometry(data["points"], data["faces"], data["normals"], lpe_dim=lpe_dim)
    out = {}
    for name, config in tdk.FEATURE_CONFIGS.items():
        directory = OUT / name / "full" / "seed_42"
        checkpoint = directory / "best.pt"
        
        if not checkpoint.exists():
            continue
        payload = torch.load(checkpoint, map_location=device)
        args = payload["args"]
        
        
        
        
        harmonic = payload.get("harmonic", payload.get("harmonic_diagnosis", {}))
        state_dict = payload["state_dict"]
        architecture = payload.get("architecture", {})
        
        
        slow_initial_skip = architecture.get(
            "slow_initial_skip", "layers.0.initial_skip.0.weight" in state_dict
        )
        model = tdk.TDKHO(
            geo,
            config,
            args["hidden"],
            args["layers"],
            args["channels"],
            args["microsteps"],
            device,
            phase_init=args.get("phase_init", "zero"),
            harmonic_active=harmonic.get("harmonic_head_active", False),
            slow_initial_skip=slow_initial_skip,
        ).to(device)
        model.load_state_dict(state_dict)
        scale = json.loads((directory / "test_metrics.json").read_text())["x_scale"]
        x = torch.from_numpy(trajectories[test, 0] / scale).float().to(device)
        out[f"TDK-{name}"] = (
            tdk.predict(model, x, batch=192).cpu().numpy()
            * json.loads((directory / "test_metrics.json").read_text())["y_scale"]
        ).astype(np.float32)
    return out


def population_diagnostics(
    preds: dict[str, np.ndarray], target: np.ndarray, metrics: dict, folder: Path
) -> None:
    
    best_tdk = min(
        (name for name in metrics if name.startswith("TDK-")), key=lambda name: metrics[name]["MSE"]
    )
    selected = [best_tdk, "HSD", "FNO", "GNO", "MGN", "DeepONet", "GeoFNO"]
    selected = [name for name in selected if name in preds]
    colors = {
        best_tdk: "#173f6c",
        "HSD": "#d47a55",
        "FNO": "#c88e39",
        "GNO": "#b55a43",
        "MGN": "#9a7652",
        "DeepONet": "#8a5a93",
        "GeoFNO": "#5b8c85",
    }
    labels = {best_tdk: "TDK-HO (best saved)", **{name: name for name in selected}}
    relative = {
        name: np.linalg.norm(preds[name] - target, axis=1)
        / (np.linalg.norm(target, axis=1) + 1e-12)
        for name in selected
    }
    target_strength = np.sqrt(np.mean(target**2, axis=1))
    fig = plt.figure(figsize=(18.5, 10.2), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        2, 2, left=0.07, right=0.97, bottom=0.10, top=0.86, hspace=0.34, wspace=0.22
    )
    ax = fig.add_subplot(grid[0, 0])
    for name in selected:
        values = np.sort(relative[name])
        q = np.linspace(0, 1, len(values))
        ax.plot(values, q, color=colors[name], lw=2.25, label=labels[name])
    ax.set(
        xscale="log",
        xlabel="per-sample relative $L_2$",
        ylabel="empirical CDF",
        title="Error distribution over all held-out trajectories",
    )
    clean(ax, "both")
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="lower right")
    ax = fig.add_subplot(grid[0, 1])
    violin = ax.violinplot(
        [np.log10(np.maximum(relative[name], 1e-9)) for name in selected],
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, name in zip(violin["bodies"], selected):
        body.set_facecolor(colors[name])
        body.set_edgecolor("white")
        body.set_alpha(0.82)
    violin["cmedians"].set_color("#172e48")
    violin["cmedians"].set_linewidth(1.6)
    ax.set_xticks(
        range(1, len(selected) + 1), [labels[name] for name in selected], rotation=22, ha="right"
    )
    ax.set(ylabel=r"$\log_{10}$ relative $L_2$", title="Error spread and median; lower is better")
    clean(ax, "y")
    ax = fig.add_subplot(grid[1, :])
    cuts = np.quantile(target_strength, np.linspace(0, 1, 9))
    centers = 0.5 * (cuts[:-1] + cuts[1:])
    for name in selected:
        medians = [
            np.median(relative[name][(target_strength >= lo) & (target_strength <= hi)])
            for lo, hi in zip(cuts[:-1], cuts[1:])
        ]
        ax.plot(centers, medians, marker="o", ms=4, lw=2.1, color=colors[name], label=labels[name])
    ax.set(
        xlabel="target-field RMS amplitude (equal-count bins)",
        ylabel="median relative $L_2$",
        title="Robustness across solution amplitude",
    )
    clean(ax, "both")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Toroidal transport: population-level reliability",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.07,
        0.915,
        "All panels use the same 600 trajectories; no visual sample selection is involved.",
        fontsize=9.5,
        color="#596d82",
    )
    fig.savefig(
        folder / "03_population_error_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(fig)

    radar_models = [best_tdk, "HSD", "FNO", "GNO"]
    radar_models = [name for name in radar_models if name in metrics]
    candidates = (
        ("Div_Fid", "Div"),
        ("Vort_Fid", "Vort"),
        ("Enst_Fid", "Enst"),
        ("Grad_Fid", "Grad"),
        ("Spec_Fid", "Spec"),
        ("Energy_Fid", "Energy"),
        ("S_beta0", "$S_{\\beta_0}$"),
        ("IoU", "IoU"),
    )
    
    
    
    candidates = [
        item for item in candidates if all(item[0] in metrics[name] for name in radar_models)
    ]
    keys, tick_labels = tuple(item[0] for item in candidates), tuple(item[1] for item in candidates)
    theta = np.linspace(0, 2 * np.pi, len(keys), endpoint=False)
    theta = np.r_[theta, theta[0]]
    fig, ax = plt.subplots(
        figsize=(10.5, 9), subplot_kw={"projection": "polar"}, facecolor="#fbfcfe"
    )
    for name in radar_models:
        values = np.r_[[metrics[name][key] for key in keys], metrics[name][keys[0]]]
        ax.plot(theta, values, color=colors[name], lw=2.5, label=labels.get(name, name))
        ax.fill(theta, values, color=colors[name], alpha=0.09)
    ax.set_xticks(theta[:-1], tick_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks((0.2, 0.4, 0.6, 0.8, 1.0))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(color="#cfd9e3")
    ax.set_title(
        "Physics-consistency portrait", fontsize=17, fontweight="bold", color="#15283f", pad=24
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.14), frameon=False)
    fig.text(
        0.5,
        0.04,
        "All axes are fidelity-like quantities; higher is better. Raw MSE and Curl MSE remain in the metric table.",
        ha="center",
        fontsize=9,
        color="#596d82",
    )
    fig.savefig(
        folder / "04_physics_consistency_radar.png", dpi=300, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(fig)


def audit_deeponet_parameters(metrics: dict, output: Path) -> dict:
    

    def count(path: Path) -> tuple[int, list[int]]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        return sum(value.numel() for value in state.values()), list(
            state["branch_net.0.weight"].shape
        )

    records = {}
    original = ROOT / "runs" / "baseline" / "legacy" / "scalar_native" / "model_deeponet.pt"
    n, shape = count(original)
    if metrics["DeepONet"]["Params"] != n:
        raise RuntimeError("DeepONet table count disagrees with its saved checkpoint")
    records["original_base"] = {
        "checkpoint": str(original.relative_to(ROOT)),
        "parameters": n,
        "first_branch_weight": shape,
    }
    controls = ROOT / "runs" / "controls" / "deeponet" / "form_0"
    for checkpoint in sorted(controls.glob("*/normal/seed_42/best_val.pt")):
        n, shape = count(checkpoint)
        records[checkpoint.parents[1].name] = {
            "checkpoint": str(checkpoint.relative_to(ROOT.parent)),
            "parameters": n,
            "first_branch_weight": shape,
        }
    (output / "deeponet_parameter_audit.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def figures(
    preds: dict[str, np.ndarray],
    target: np.ndarray,
    points: np.ndarray,
    faces: np.ndarray,
    metrics: dict,
    folder: Path | None = None,
    metric_filename: str = "metric_heatmap.png",
    pareto_filename: str = "parameter_vs_mse.png",
    sample_panels: bool = True,
) -> None:
    folder = folder or OUT / "fair_comparison_figures"
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(metrics, key=lambda name: metrics[name]["MSE"])
    labels = [presentation_label(name) for name in names]
    colors = [
        (
            "#173f6c"
            if name.startswith("TDK-small")
            else "#4d91c8" if name.startswith("TDK-large") else "#d47a55"
        )
        for name in names
    ]
    all_keys = [
        "Div_Fid",
        "Curl_MSE",
        "Vort_Fid",
        "Enst_Fid",
        "Grad_Fid",
        "Spec_Fid",
        "Energy_Fid",
        "S_beta0",
        "IoU",
    ]
    available = [key for key in all_keys if all(key in metrics[name] for name in names)]
    headers = {
        "Div_Fid": "Div",
        "Curl_MSE": "Curl*",
        "Vort_Fid": "Vort",
        "Enst_Fid": "Enst",
        "Grad_Fid": "Grad",
        "Spec_Fid": "Spec",
        "Energy_Fid": "Energy",
        "S_beta0": "$S_{\\beta_0}$",
        "IoU": "IoU",
    }
    heat = np.asarray(
        [
            [
                metrics[name][key] if key != "Curl_MSE" else np.exp(-metrics[name][key])
                for key in available
            ]
            for name in names
        ]
    )
    fig = plt.figure(figsize=(22, max(8.5, 0.43 * len(names) + 3)), facecolor="#fbfcfe")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.3, 3.7), left=0.10, right=0.93, top=0.82, bottom=0.10, wspace=0.10
    )
    y = np.arange(len(names))
    mse = np.asarray([metrics[name]["MSE"] for name in names])
    ax = fig.add_subplot(grid[0, 0])
    ax.hlines(y, mse.min() * 0.66, mse, color="#d7e1ea", lw=1.35, zorder=1)
    ax.scatter(mse, y, s=86, c=colors, edgecolor="white", linewidth=1.15, zorder=3)
    for yy, value in zip(y, mse):
        ax.annotate(
            f"{value:.2e}",
            (value, yy),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#425972",
        )
    ax.set(
        xscale="log",
        xlim=(mse.min() * 0.60, mse.max() * 3.0),
        yticks=y,
        yticklabels=labels,
        xlabel="physical MSE  ·  lower is better",
    )
    ax.invert_yaxis()
    clean(ax, "x")
    ax.set_title("Accuracy", loc="left", pad=15, fontsize=13, color="#15283f")
    hm = fig.add_subplot(grid[0, 1])
    im = hm.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    hm.set(
        xticks=np.arange(len(available)),
        xticklabels=[headers[key] for key in available],
        yticks=y,
        yticklabels=[""] * len(names),
    )
    hm.xaxis.tick_top()
    hm.tick_params(axis="x", pad=9, labelsize=9)
    for yy in range(len(names)):
        for xx in range(len(available)):
            value = heat[yy, xx]
            hm.text(
                xx,
                yy,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.62 else "#163047",
            )
    for spine in hm.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=hm, shrink=0.77, pad=0.025)
    bar.set_label("fidelity  ·  higher is better  (Curl* = exp(−Curl MSE))", fontsize=9)
    fig.suptitle(
        "Toroidal transport · unified physical metric atlas",
        x=0.10,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#15283f",
    )
    fig.text(
        0.10,
        0.915,
        "All displayed models are evaluated on the same fixed 600 trajectories. Blue: TDK-HO; orange: archived baselines.",
        fontsize=10,
        color="#596d82",
    )
    fig.savefig(folder / metric_filename, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.2, 7.5), constrained_layout=True, facecolor="#fbfcfe")
    for name, value in metrics.items():
        is_tdk = name.startswith("TDK-")
        color = "#173f6c" if name.startswith("TDK-small") else "#4d91c8" if is_tdk else "#d47a55"
        ax.scatter(
            value["Params"],
            value["MSE"],
            s=62 if is_tdk else 82,
            color=color,
            marker="o" if is_tdk else "s",
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
    frontier = []
    best = np.inf
    for name, item in sorted(metrics.items(), key=lambda item: item[1]["Params"]):
        if item["MSE"] < best:
            frontier.append((name, item))
            best = item["MSE"]
    ax.plot(
        [row["Params"] for _, row in frontier],
        [row["MSE"] for _, row in frontier],
        "--",
        color="#172e48",
        lw=2.1,
        alpha=0.82,
        zorder=2,
    )
    shown = [name for name, _ in frontier] + [
        name for name in metrics if name in {"HSD", "FNO", "DeepONet"}
    ]
    for name in dict.fromkeys(shown):
        row = metrics[name]
        ax.annotate(
            presentation_label(name),
            (row["Params"], row["MSE"]),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=8.5,
            color="#2d425a",
            bbox={"boxstyle": "round,pad=.22", "fc": "#fbfcfe", "ec": "none", "alpha": 0.85},
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="trainable parameters",
        ylabel="physical MSE",
        title="Accuracy–capacity frontier",
    )
    clean(ax, "both")
    ax.text(
        0.015,
        0.025,
        "Lower-left is preferable. Dashed line: non-dominated models.",
        transform=ax.transAxes,
        fontsize=9,
        color="#596d82",
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="TDK-HO small",
            markerfacecolor="#173f6c",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="TDK-HO large",
            markerfacecolor="#4d91c8",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="archived baseline",
            markerfacecolor="#d47a55",
            markersize=8,
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        title="model family",
        title_fontsize=8.5,
        fontsize=8.5,
        loc="upper right",
    )
    fig.savefig(folder / pareto_filename, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    if not sample_panels:
        return
    preferred = "TDK-flow_diffusion_spectral"
    reference = (
        preferred if preferred in preds else next(name for name in preds if name.startswith("TDK-"))
    )
    full = preds[reference]
    errors = np.linalg.norm(full - target, axis=1) / (np.linalg.norm(target, axis=1) + 1e-12)
    sample_ids = {
        "best": int(errors.argmin()),
        "median": int(np.argsort(errors)[len(errors) // 2]),
        "worst": int(errors.argmax()),
    }
    tri = matplotlib.tri.Triangulation(points[:, 0], points[:, 1], faces)
    show = list(metrics)
    for label, idx in sample_ids.items():
        fields = {"Ground truth": target[idx], **{name: preds[name][idx] for name in show}}
        vabs = max(float(np.abs(value).max()) for value in fields.values())
        eabs = max(
            float(np.abs(value - target[idx]).max())
            for name, value in fields.items()
            if name != "Ground truth"
        )
        for kind in ("prediction", "error"):
            fig, axes = plt.subplots(4, 4, figsize=(15, 14))
            axes = axes.ravel()
            for ax, (name, value) in zip(axes, fields.items()):
                shown = value if kind == "prediction" else value - target[idx]
                if kind == "prediction":
                    im = ax.tripcolor(
                        tri, shown, shading="gouraud", cmap="RdBu_r", vmin=-vabs, vmax=vabs
                    )
                else:
                    im = ax.tripcolor(
                        tri, shown, shading="gouraud", cmap="coolwarm", vmin=-eabs, vmax=eabs
                    )
                ax.set_title(
                    name, fontsize=9, fontweight="bold" if name.startswith("TDK-") else "normal"
                )
                ax.set_aspect("equal")
                ax.axis("off")
            for ax in axes[len(fields) :]:
                ax.axis("off")
            cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.68, pad=0.02)
            cbar.set_label("scalar value" if kind == "prediction" else "prediction - target")
            fig.suptitle(
                f"Same test sample ({label}, index={idx}): {kind}; {reference} relL2={errors[idx]:.4f}",
                y=0.995,
                fontsize=14,
                fontweight="bold",
            )
            fig.subplots_adjust(
                left=0.02,
                right=0.89,
                bottom=0.02,
                top=0.94,
                wspace=0.06,
                hspace=0.10,
            )
            fig.savefig(folder / f"same_sample_{label}_{kind}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


def write_evaluation_summary(metrics: dict, output: Path) -> None:
    

    lines = [
        "# Toroidal transport evaluation output",
        "",
        "This directory was generated on one fixed held-out split.",
        "`fair_paper_metrics.json` contains physical-unit metrics and "
        "`fair_predictions.npz` contains aligned target/prediction arrays.",
        "",
        "| Model | Parameters | MSE | Div Fid | Curl MSE | Vort Fid | Grad Fid | Enst Fid | Energy Fid | Spec Fid | S_beta0 | IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(metrics.items(), key=lambda row: row[1]["MSE"]):
        lines.append(
            f"| {name} | {item['Params']:,} | {item['MSE']:.4e} | {item['Div_Fid']:.4f} | "
            f"{item['Curl_MSE']:.4e} | {item['Vort_Fid']:.4f} | {item['Grad_Fid']:.4f} | "
            f"{item['Enst_Fid']:.4f} | {item['Energy_Fid']:.4f} | {item['Spec_Fid']:.4f} | "
            f"{item['S_beta0']:.4f} | {item['IoU']:.4f} |"
        )
    lines += [
        "",
        "`fair_comparison_figures/` contains the metric heatmap, accuracy--parameter "
        "trade-off, population diagnostics, and shared-scale sample panels.",
        "`deeponet_parameter_audit.json` records the parameter count reconstructed "
        "from the exact DeepONet state dictionary used for inference.",
    ]
    (output / "EVALUATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved TDK-HO and legacy baselines on one common test set."
    )
    parser.add_argument(
        "--tdk-output",
        default="small/form_0",
        help="DKHO C0 root relative to runs/dkho/, e.g. small/form_0.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--lpe-dim",
        type=int,
        default=None,
        help="Override LPE dimension; otherwise read it from the training manifest (default: 8).",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate figures from fair_predictions.npz and fair_paper_metrics.json without inference.",
    )
    parser.add_argument(
        "--tdk-only",
        action="store_true",
        help="Evaluate only saved TDK-HO weights.  This is a read-only audit helper for selecting a TDK checkpoint without re-running archived baselines.",
    )
    cli = parser.parse_args()
    configure_style()
    OUT = (RUNS_ROOT / cli.tdk_output).resolve()
    lpe_dim = cli.lpe_dim
    if lpe_dim is None:
        manifest_path = OUT / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                lpe_dim = int(json.load(f).get("args", {}).get("lpe_dim", 8))
        else:
            lpe_dim = 8
    if lpe_dim <= 0:
        parser.error("--lpe-dim must be a positive integer")
    if not OUT.is_dir():
        raise FileNotFoundError(f"TDK output directory not found: {OUT}")
    torch.set_num_threads(12)
    device = (
        torch.device("cuda")
        if cli.device in {"auto", "cuda"} and torch.cuda.is_available()
        else torch.device("cpu")
    )
    if cli.device == "cuda" and device.type != "cuda":
        print("[device] CUDA requested but unavailable; falling back to CPU.", flush=True)
    with open(tdk.DATA_PATH, "rb") as f:
        data = pickle.load(f)
    if cli.render_only:
        prediction_path, metric_path = OUT / "fair_predictions.npz", OUT / "fair_paper_metrics.json"
        if not prediction_path.exists() or not metric_path.exists():
            raise FileNotFoundError(
                "--render-only requires existing fair_predictions.npz and fair_paper_metrics.json"
            )
        cached = np.load(prediction_path)
        target = cached["target"]
        preds = {name: cached[name] for name in cached.files if name != "target"}
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        figures(preds, target, data["points"], data["faces"], metrics)
        population_diagnostics(preds, target, metrics, OUT / "fair_comparison_figures")
        audit_deeponet_parameters(metrics, OUT)
        if all("Div_Fid" in item for item in metrics.values()):
            write_evaluation_summary(metrics, OUT)
        else:
            print(
                "[render-only] retained legacy summary: cached metrics predate Div/Curl/Vort diagnostics"
            )
        print(f"[render-only] regenerated figures in {OUT / 'fair_comparison_figures'}")
        return
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    _, _, test = split(len(trajectories))
    target = trajectories[test, -1]
    host = HighOrderSpectralOperators(
        data["points"], data["faces"], k_list=(64, 64, 64), normalize_laplacian=True
    )
    preds = tdk_predictions(data, device, lpe_dim)
    if not cli.tdk_only:
        preds.update(baseline_predictions(data, host, trajectories[test, 0], device))
    metrics = {
        name: common_metrics(value, target, host, data["points"], data["faces"])
        for name, value in preds.items()
    }
    baseline_params = {}
    if not cli.tdk_only:
        baseline_params = {
            "GNO": 285857,
            "FNO": 309081,
            "MGN": 303193,
            "DeepONet": 273221,
            "GeoFNO": 224652,
            "HSD": 281575,
        }
        
        
        deeponet_state = torch.load(
            ROOT / "runs" / "baseline" / "legacy" / "scalar_native" / "model_deeponet.pt",
            map_location="cpu",
            weights_only=True,
        )
        baseline_params["DeepONet"] = sum(value.numel() for value in deeponet_state.values())
    for name in metrics:
        if name.startswith("TDK-"):
            run = OUT / name.removeprefix("TDK-") / "full" / "seed_42" / "test_metrics.json"
            metrics[name]["Params"] = int(json.loads(run.read_text())["parameters"])
        else:
            metrics[name]["Params"] = baseline_params[name]
    if cli.tdk_only:
        np.savez_compressed(OUT / "tdk_only_predictions.npz", target=target, **preds)
        (OUT / "tdk_only_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    np.savez_compressed(OUT / "fair_predictions.npz", target=target, **preds)
    (OUT / "fair_paper_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    figures(preds, target, data["points"], data["faces"], metrics)
    population_diagnostics(preds, target, metrics, OUT / "fair_comparison_figures")
    audit_deeponet_parameters(metrics, OUT)
    write_evaluation_summary(metrics, OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
