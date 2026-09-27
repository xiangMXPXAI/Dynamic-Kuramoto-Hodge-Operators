"""Train cochain models for Darcy solution, flux, and circulation targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (  
    ARCHIVE,
    FEATURE_CONFIGS,
    DarcyGeometry,
    DarcyMemmapDataset,
    FeatureConfig,
    IncidenceOps,
    config_dict,
    feature_profile_details,
    parameter_count,
    portable_path,
    prepare_cache,
    relative_l2,
    autocast_context,
    configure_runtime,
    make_grad_scaler,
    make_loader,
    maybe_compile,
    resolve_device,
    seed_all,
    task_stats,
    write_json,
)


PROTOCOL_REVISION = "darcy_matched_rel_l2_200e_patience40_v1"
STRUCTURE_VARIANTS = ("full", "no_dirac", "no_harmonic", "no_phase")


class MLP(nn.Module):
    def __init__(self, fin: int, hidden: int, fout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fin, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, fout)
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class DarcyFeatureBuilder(nn.Module):
    

    def __init__(self, geo: DarcyGeometry, config: FeatureConfig) -> None:
        super().__init__()
        self.geo, self.config = geo, config
        if config.lpe:
            geo.ensure_lpe()
        self.ops = IncidenceOps(geo)
        static = {
            "node_pos": geo.node_pos,
            "edge_pos": geo.edge_pos,
            "face_pos": geo.face_pos,
            "edge_dir": geo.edge_dir,
            "edge_length": geo.edge_length,
            "area": geo.areas[:, None],
            "kappa_node": geo.kappa_node,
            "kappa_edge_invariants": geo.kappa_edge_invariants,
            "kappa_face": geo.kappa_face,
            "node_area": geo.node_area,
            "node_boundary": geo.node_boundary_onehot,
            "edge_boundary": geo.edge_boundary_onehot,
            "face_boundary": geo.face_boundary,
            "harmonic": geo.harmonic_basis_1,
        }
        if geo.lpe is not None:
            static.update({"lpe0": geo.lpe[0], "lpe1": geo.lpe[1], "lpe2": geo.lpe[2]})
        for name, value in static.items():
            self.register_buffer(name, torch.from_numpy(value.astype(np.float32)))
        self.hole1 = torch.from_numpy((geo.node_component == 2).astype(np.float32))[None, :, None]
        self.hole2 = torch.from_numpy((geo.node_component == 3).astype(np.float32))[None, :, None]
        self.register_buffer("hole1_mask", self.hole1)
        self.register_buffer("hole2_mask", self.hole2)
        self.dims = self._feature_dims()

    def _feature_dims(self) -> tuple[int, int, int]:
        
        
        d0, d1, d2 = 1 + 1 + 2 + 4 + 4, 1 + 1 + 2 + 2 + 1 + 3 + 4, 1 + 2 + 1 + 4 + 4
        if self.config.lpe:
            d0 += self.geo.lpe_dim
            d1 += self.geo.lpe_dim
            d2 += self.geo.lpe_dim
        if self.config.heat:
            d0 += 3
            d1 += 3
            d2 += 3
        if self.config.global_broadcast:
            d0 += 6
            d1 += 6
            d2 += 6
        return d0, d1, d2

    @staticmethod
    def _expand(value: Tensor, batch: int) -> Tensor:
        return value.unsqueeze(0).expand(batch, -1, -1)

    def summary(self, f0: Tensor, g: Tensor) -> Tensor:
        
        weight = self.node_area[:, 0]
        mass = weight.sum().clamp_min(1e-12)
        q = (f0 * weight[None]).sum(1) / mass
        rms = torch.sqrt((f0.square() * weight[None]).sum(1) / mass + 1e-12)
        px = (f0 * weight[None] * self.node_pos[:, 0][None]).sum(1) / mass
        py = (f0 * weight[None] * self.node_pos[:, 1][None]).sum(1) / mass
        return torch.stack([g[:, 0], g[:, 1], q, px, py, rms], dim=-1)

    def forward(self, f0: Tensor, g: Tensor) -> tuple[list[Tensor], Tensor]:
        batch = f0.shape[0]
        u = f0[:, :, None]
        
        
        g_lift = g[:, None, 0:1] * self.hole1_mask + g[:, None, 1:2] * self.hole2_mask
        du, dg = self.ops.d0(u), self.ops.d0(g_lift)
        
        
        face_source = u[:, self.ops.faces].mean(2) * self.area[None]
        f0_parts = [
            u,
            g_lift,
            self._expand(self.node_pos, batch),
            self._expand(self.kappa_node, batch),
            self._expand(self.node_boundary, batch),
        ]
        f1_parts = [
            du,
            dg,
            self._expand(self.edge_pos, batch),
            self._expand(self.edge_dir, batch),
            self._expand(self.edge_length, batch),
            self._expand(self.kappa_edge_invariants, batch),
            self._expand(self.edge_boundary, batch),
        ]
        f2_parts = [
            face_source,
            self._expand(self.face_pos, batch),
            self._expand(self.area, batch),
            self._expand(self.kappa_face, batch),
            self._expand(self.face_boundary, batch),
        ]
        if self.config.lpe:
            f0_parts.append(self._expand(self.lpe0, batch))
            f1_parts.append(self._expand(self.lpe1, batch))
            f2_parts.append(self._expand(self.lpe2, batch))
        if self.config.heat:
            
            
            heat0: list[Tensor] = []
            current = u
            for step in range(1, 13):
                current = current - 0.01 * self.ops.delta1(self.ops.d0(current))
                if step in (1, 4, 12):
                    heat0.append(current)
            f0_parts.append(torch.cat(heat0, dim=-1))
            f1_parts.append(torch.cat([self.ops.d0(item) for item in heat0], dim=-1))
            f2_parts.append(
                torch.cat(
                    [item[:, self.ops.faces].mean(2) * self.area[None] for item in heat0], dim=-1
                )
            )
        summary = self.summary(f0, g)
        if self.config.global_broadcast:
            f0_parts.append(summary[:, None].expand(-1, self.geo.n0, -1))
            f1_parts.append(summary[:, None].expand(-1, self.geo.n1, -1))
            f2_parts.append(summary[:, None].expand(-1, self.geo.n2, -1))
        return [
            torch.cat(f0_parts, dim=-1),
            torch.cat(f1_parts, dim=-1),
            torch.cat(f2_parts, dim=-1),
        ], summary


class DiracKuramotoLayer(nn.Module):
    

    def __init__(self, hidden: int, channels: int, microsteps: int) -> None:
        super().__init__()
        self.channels, self.microsteps = channels, microsteps
        self.frequency = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.gate = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.lag = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.coupling = nn.Parameter(torch.zeros(3))
        self.dt_logit = nn.Parameter(torch.tensor(-1.35))
        self.phase_readout = nn.ModuleList(nn.Linear(2 * channels, hidden) for _ in range(3))
        self.update = nn.ModuleList(MLP(2 * hidden, hidden, hidden) for _ in range(3))
        self.skip = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))

    def forward(
        self, z: list[Tensor], z_initial: list[Tensor], theta: list[Tensor], ops: IncidenceOps
    ) -> tuple[list[Tensor], list[Tensor]]:
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt_logit)
        strength = F.softplus(self.coupling) + 1e-4
        for _ in range(self.microsteps):
            z0, z1, z2 = z
            t0, t1, t2 = theta
            grad01 = ops.d0(t0) - torch.tanh(self.gate[1](z1)) * t1
            adj10 = ops.delta1(t1) + torch.tanh(self.gate[0](z0)) * t0
            curl12 = ops.d1(t1) - torch.tanh(self.gate[2](z2)) * t2
            adj21 = ops.delta2(t2) + torch.tanh(self.gate[1](z1)) * t1
            lag0, lag1, lag2 = [0.5 * torch.tanh(self.lag[index](z[index])) for index in range(3)]
            theta = [
                t0
                + dt
                * (-self.frequency[0](z0) - strength[0] * ops.delta1(torch.sin(grad01 - lag1))),
                t1
                + dt
                * (
                    -self.frequency[1](z1)
                    - strength[1]
                    * (ops.d0(torch.sin(adj10 - lag0)) + ops.delta2(torch.sin(curl12 - lag2)))
                ),
                t2 + dt * (-self.frequency[2](z2) - strength[2] * ops.d1(torch.sin(adj21 - lag1))),
            ]
        z_next = []
        for index in range(3):
            phase = self.phase_readout[index](
                torch.cat([torch.cos(theta[index]), torch.sin(theta[index])], dim=-1)
            )
            z_next.append(
                self.norm[index](
                    z[index]
                    + self.update[index](torch.cat([z[index], phase], dim=-1))
                    + self.skip[index](z_initial[index])
                )
            )
        return z_next, theta


class NoDiracKuramotoLayer(DiracKuramotoLayer):
    

    def forward(
        self, z: list[Tensor], z_initial: list[Tensor], theta: list[Tensor], ops: IncidenceOps
    ) -> tuple[list[Tensor], list[Tensor]]:
        del ops
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt_logit)
        strength = F.softplus(self.coupling) + 1e-4
        for _ in range(self.microsteps):
            theta = [
                theta[index]
                + dt
                * (
                    -self.frequency[index](z[index])
                    - strength[index]
                    * torch.sin(
                        torch.tanh(self.gate[index](z[index])) * theta[index]
                        - 0.5 * torch.tanh(self.lag[index](z[index]))
                    )
                )
                for index in range(3)
            ]
        z_next = []
        for index in range(3):
            phase = self.phase_readout[index](
                torch.cat([torch.cos(theta[index]), torch.sin(theta[index])], dim=-1)
            )
            z_next.append(
                self.norm[index](
                    z[index]
                    + self.update[index](torch.cat([z[index], phase], dim=-1))
                    + self.skip[index](z_initial[index])
                )
            )
        return z_next, theta


class FeatureDiracLayer(nn.Module):
    

    def __init__(self, hidden: int, channels: int) -> None:
        super().__init__()
        
        
        self.frequency = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.gate = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.lag = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.channel_readout = nn.ModuleList(nn.Linear(2 * channels, hidden) for _ in range(3))
        self.coupling = nn.Parameter(torch.zeros(3))
        self.dt_logit = nn.Parameter(torch.tensor(-1.35))
        self.update = nn.ModuleList(MLP(2 * hidden, hidden, hidden) for _ in range(3))
        self.skip = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.norm = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))

    def carrier(self, z: list[Tensor]) -> list[Tensor]:
        return [
            torch.sigmoid(self.gate[index](z[index])) * torch.tanh(self.frequency[index](z[index]))
            + torch.tanh(self.lag[index](z[index]))
            for index in range(3)
        ]

    def decoder_channels(self, value: Tensor, index: int) -> Tensor:
        carrier = torch.sigmoid(self.gate[index](value)) * torch.tanh(
            self.frequency[index](value)
        ) + torch.tanh(self.lag[index](value))
        return torch.cat([carrier, torch.tanh(self.gate[index](value))], dim=-1)

    def forward(self, z: list[Tensor], z_initial: list[Tensor], ops: IncidenceOps) -> list[Tensor]:
        x0, x1, x2 = self.carrier(z)
        strength = F.softplus(self.coupling) + 1e-4
        messages = [
            strength[0] * ops.delta1(x1),
            strength[1] * (ops.d0(x0) + ops.delta2(x2)),
            strength[2] * ops.d1(x1),
        ]
        dt = 0.02 + 0.18 * torch.sigmoid(self.dt_logit)
        return [
            self.norm[index](
                z[index]
                + dt
                * self.update[index](
                    torch.cat(
                        [
                            z[index],
                            self.channel_readout[index](
                                torch.cat([messages[index], (x0, x1, x2)[index]], dim=-1)
                            ),
                        ],
                        dim=-1,
                    )
                )
                + self.skip[index](z_initial[index])
            )
            for index in range(3)
        ]


class TDKHO(nn.Module):
    def __init__(
        self,
        geo: DarcyGeometry,
        config: FeatureConfig,
        task: str,
        hidden: int,
        layers: int,
        channels: int,
        microsteps: int,
        phase_init: str,
        variant: str = "full",
    ) -> None:
        super().__init__()
        if variant not in STRUCTURE_VARIANTS:
            raise ValueError(f"unknown structural variant: {variant}")
        self.task, self.phase_init, self.variant = task, phase_init, variant
        self.uses_phase = variant in ("full", "no_dirac", "no_harmonic")
        
        
        self.harmonic_head = task == "1" and geo.beta1 > 0 and variant != "no_harmonic"
        self.features = DarcyFeatureBuilder(geo, config)
        self.encoder = nn.ModuleList(MLP(dim, hidden, hidden) for dim in self.features.dims)
        if variant in ("full", "no_harmonic"):
            self.layers = nn.ModuleList(
                DiracKuramotoLayer(hidden, channels, microsteps) for _ in range(layers)
            )
        elif variant == "no_dirac":
            self.layers = nn.ModuleList(
                NoDiracKuramotoLayer(hidden, channels, microsteps) for _ in range(layers)
            )
        elif variant == "no_phase":
            self.layers = nn.ModuleList(FeatureDiracLayer(hidden, channels) for _ in range(layers))
        self.decoder = MLP(
            hidden + (2 * channels if self.uses_phase or variant == "no_phase" else 0), hidden, 1
        )
        if self.uses_phase and phase_init == "deterministic":
            self.phase_encoder = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        elif self.uses_phase and phase_init == "random_learnable":
            self.phase_seed = nn.ParameterList(
                nn.Parameter(0.05 * torch.randn(1, 1, channels)) for _ in range(3)
            )
        if self.harmonic_head:
            self.topology_head = MLP(6, hidden, geo.beta1)
            self.register_buffer("psi", torch.from_numpy(geo.harmonic_basis_1.astype(np.float32)))

    def _initial_phase(self, z: list[Tensor]) -> list[Tensor]:
        if self.phase_init == "zero":
            return [
                torch.zeros(
                    item.shape[0],
                    item.shape[1],
                    self.layers[0].channels,
                    dtype=item.dtype,
                    device=item.device,
                )
                for item in z
            ]
        if self.phase_init == "deterministic":
            return [math.pi * torch.tanh(layer(item)) for layer, item in zip(self.phase_encoder, z)]
        return [
            seed.expand(item.shape[0], item.shape[1], -1) for seed, item in zip(self.phase_seed, z)
        ]

    def forward(self, f0: Tensor, g: Tensor) -> Tensor:
        features, summary = self.features(f0, g)
        z = [encoder(value) for encoder, value in zip(self.encoder, features)]
        z_initial = list(z)
        if self.uses_phase:
            theta = self._initial_phase(z)
            for layer in self.layers:
                z, theta = layer(z, z_initial, theta, self.features.ops)
        else:
            for layer in self.layers:
                z = layer(z, z_initial, self.features.ops)
        rank = int(self.task)
        if self.uses_phase:
            decoded = [z[rank], torch.cos(theta[rank]), torch.sin(theta[rank])]
        else:
            decoded = [z[rank], self.layers[-1].decoder_channels(z[rank], rank)]
        output = self.decoder(torch.cat(decoded, dim=-1)).squeeze(-1)
        if self.harmonic_head:
            
            
            local = output - torch.einsum(
                "em,bm->be", self.psi, torch.einsum("em,be->bm", self.psi, output)
            )
            output = local + torch.einsum("em,bm->be", self.psi, self.topology_head(summary))
        return output


@torch.inference_mode()
def predict(
    model: nn.Module, loader, device: torch.device, precision: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predicted, target = [], []
    for f0, g, y in loader:
        with autocast_context(device, precision):
            value = model(f0.to(device, non_blocking=True), g.to(device, non_blocking=True))
        predicted.append(value.float().cpu().numpy())
        target.append(y.float().cpu().numpy())
    return np.concatenate(predicted), np.concatenate(target)


@torch.inference_mode()
def validation_relative_l2(model: nn.Module, loader, device: torch.device, precision: str) -> float:
    
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    for f0, g, y in loader:
        f0, g, y = (
            f0.to(device, non_blocking=True),
            g.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )
        with autocast_context(device, precision):
            prediction = model(f0, g)
        per_sample = torch.linalg.vector_norm(
            prediction.float() - y.float(), dim=1
        ) / torch.linalg.vector_norm(y.float(), dim=1).clamp_min(1e-12)
        total += per_sample.sum()
        count += y.shape[0]
    return (total / count).item()


def metrics(
    task: str, prediction: np.ndarray, target: np.ndarray, y_scale: float, geo: DarcyGeometry
) -> dict[str, float]:
    pred, truth = prediction * y_scale, target * y_scale
    error = pred - truth
    result = {
        "mse": float(np.mean(error**2)),
        "relative_l1": float(
            np.mean(
                np.sum(np.abs(error), axis=1) / np.maximum(np.sum(np.abs(truth), axis=1), 1e-12)
            )
        ),
        "relative_l2": float(
            np.mean(
                np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(truth, axis=1), 1e-12)
            )
        ),
    }
    if task == "1":
        psi = geo.harmonic_basis_1
        gamma_pred, gamma_true = pred @ psi, truth @ psi
        result["harmonic_relative_l2"] = float(
            np.mean(
                np.linalg.norm(gamma_pred - gamma_true, axis=1)
                / np.maximum(np.linalg.norm(gamma_true, axis=1), 1e-12)
            )
        )
    return result


def train_one(
    task: str,
    config: FeatureConfig,
    variant: str,
    args: argparse.Namespace,
    geo: DarcyGeometry,
    stats: dict[str, float],
) -> dict:
    
    
    run = args.output_root / f"form_{task}" / config.name / variant / f"seed_{args.seed}"
    result_path = run / "result.json"
    if result_path.exists() and not args.rerun:
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            saved.get("protocol_revision") == PROTOCOL_REVISION
            and saved.get("structure_variant", "full") == variant
        ):
            return saved
    run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = resolve_device(args.device)
    configure_runtime(device, args.tf32)
    train_loader = make_loader(
        task, "train", stats, args.batch_size, True, device, args.resident_data, args.workers
    )
    val_loader = make_loader(
        task, "val", stats, args.eval_batch_size, False, device, args.resident_data, args.workers
    )
    test_loader = make_loader(
        task, "test", stats, args.eval_batch_size, False, device, args.resident_data, args.workers
    )
    
    
    model = TDKHO(
        geo,
        config,
        task,
        args.hidden,
        args.layers,
        args.channels,
        args.microsteps,
        args.phase_init,
        variant,
    ).to(device)
    execution_model = maybe_compile(model, device, args.compile)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(device, args.precision)
    best, best_epoch, best_state = float("inf"), 0, None
    bad_epochs, completed_epochs = 0, 0
    history: dict[str, list[float]] = {"train": [], "val": []}
    started = time.time()
    print(
        f"[TDK-HO form={task} {config.name} structure={variant}] parameters={parameter_count(model)} device={device} batches={len(train_loader)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        execution_model.train()
        total = torch.zeros((), device=device)
        for f0, g, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            f0, g, y = (
                f0.to(device, non_blocking=True),
                g.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            with autocast_context(device, args.precision):
                prediction = execution_model(f0, g)
            loss = relative_l2(prediction.float(), y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
        train_loss = (total / len(train_loader)).item()
        
        
        
        val_loss = validation_relative_l2(model, val_loader, device, args.precision)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        completed_epochs = epoch
        if val_loss < best - args.min_delta:
            best, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1
        scheduler.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[TDK-HO form={task} {config.name} structure={variant}] epoch={epoch:03d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"[TDK-HO form={task} {config.name} structure={variant}] early-stop epoch={epoch} best_epoch={best_epoch} patience={args.patience}",
                flush=True,
            )
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    pred, target = predict(model, test_loader, device, args.precision)
    np.save(run / "prediction_test_normalized.npy", pred.astype(np.float32))
    np.save(run / "target_test_normalized.npy", target.astype(np.float32))
    np.save(run / "test_indices.npy", np.asarray(DarcyMemmapDataset(task, "test", stats).indices))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": task,
            "config": config_dict(config),
            "input_protocol": feature_profile_details(config),
            "structure_variant": variant,
            "protocol_revision": PROTOCOL_REVISION,
            "args": {**vars(args), "output_root": portable_path(args.output_root)},
            "stats": stats,
        },
        run / "best.pt",
    )
    write_json(run / "history.json", history)
    result = {
        "family": "TDK-HO",
        "task": f"form_{task}",
        "feature_config": config.name,
        "input_protocol": feature_profile_details(config),
        "structure_variant": variant,
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "best_val_relative_l2": best,
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "optimizer_updates": completed_epochs * len(train_loader),
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "wall_seconds": time.time() - started,
        "phase_init": args.phase_init if model.uses_phase else "not_used",
        "uses_phase": model.uses_phase,
        "harmonic_head": bool(model.harmonic_head),
        "protocol_revision": PROTOCOL_REVISION,
        "loss": "mean_per_sample_relative_l2",
        "normalization": stats,
        "execution": {
            "resident_data": args.resident_data,
            "workers": args.workers,
            "precision": args.precision,
            "tf32": args.tf32,
            "compile": args.compile,
        },
        **metrics(task, pred, target, stats["y_scale"], geo),
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("0", "1", "2", "all"), default="all")
    parser.add_argument("--configs", default="all", help="comma-separated feature levels or 'all'")
    parser.add_argument(
        "--variants", default="full", help="comma-separated structural variants or 'full'"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--patience", type=int, default=40, help="0 disables validation early stopping"
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--microsteps", type=int, default=2)
    parser.add_argument("--lpe-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--phase-init", choices=("zero", "deterministic", "random_learnable"), default="zero"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU data-loader workers; ignored by GPU-resident data",
    )
    parser.add_argument("--resident-data", choices=("auto", "memmap", "gpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "auto", "amp-bf16", "amp-fp16"), default="auto"
    )
    parser.add_argument(
        "--tf32", action="store_true", help="opt in to Ampere+ TensorFloat-32 matmuls"
    )
    parser.add_argument(
        "--compile",
        choices=("off", "auto", "reduce-overhead", "max-autotune"),
        default="off",
        help="opt-in Inductor; test per GPU/model family",
    )
    parser.add_argument(
        "--output-root",
        default=str(HERE.parent / "runs" / "dkho" / "small"),
        help="path relative to this file unless absolute",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.lpe_dim <= 0:
        parser.error("--lpe-dim must be positive")
    if args.workers < 0 or args.patience < 0 or args.min_delta < 0:
        parser.error("workers, patience and min-delta must be non-negative")
    args.output_root = Path(args.output_root)
    args.output_root = (
        args.output_root if args.output_root.is_absolute() else HERE / args.output_root
    )
    prepare_cache()
    geo = DarcyGeometry(lpe_dim=args.lpe_dim)
    requested = (
        tuple(FEATURE_CONFIGS)
        if args.configs == "all"
        else tuple(item for item in args.configs.split(",") if item)
    )
    unknown = [name for name in requested if name not in FEATURE_CONFIGS]
    if not requested or unknown:
        parser.error(f"unknown configs: {unknown}; choices={list(FEATURE_CONFIGS)}")
    tasks = ("0", "1", "2") if args.task == "all" else (args.task,)
    variants = tuple(item for item in args.variants.split(",") if item)
    invalid_variants = [name for name in variants if name not in STRUCTURE_VARIANTS]
    if not variants or invalid_variants:
        parser.error(f"unknown variants: {invalid_variants}; choices={list(STRUCTURE_VARIANTS)}")
    if "no_harmonic" in variants and tasks != ("1",):
        parser.error("no_harmonic is defined only for the C1 task; run it with --task 1")
    results = []
    for task in tasks:
        stats = task_stats(task)
        for name in requested:
            for variant in variants:
                results.append(train_one(task, FEATURE_CONFIGS[name], variant, args, geo, stats))
    write_json(
        args.output_root / "summary.json",
        {
            "archive": portable_path(ARCHIVE),
            "protocol_revision": PROTOCOL_REVISION,
            "structure_variants": list(variants),
            "results": results,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
