"""Train cochain models for C0, C1, and C2 toroidal transport targets."""

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
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]


def portable_arguments(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            candidate = value
        elif isinstance(value, str) and Path(value).is_absolute():
            candidate = Path(value)
        else:
            values[key] = value
            continue
        try:
            values[key] = candidate.resolve().relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"path must be inside the repository: {candidate}") from error
    return values

try:
    from .. import train as scalar_tdk  
    from .common import (
        FEATURE_CONFIGS,
        FeatureConfig,
        FormDataset,
        IncidenceOps,
        TorusGeometry,
        relative_l2,
        resolve_device,
        seed_all,
        splits,
        task_stats,
        write_json,
    )
except ImportError:  
    sys.path.insert(0, str(HERE.parent))
    import train as scalar_tdk  
    from common import (
        FEATURE_CONFIGS,
        FeatureConfig,
        FormDataset,
        IncidenceOps,
        TorusGeometry,
        relative_l2,
        resolve_device,
        seed_all,
        splits,
        task_stats,
        write_json,
    )


STRUCTURE_VARIANTS = ("full", "no_dirac", "no_phase", "no_harmonic")


def mlp(fin: int, hidden: int, fout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(fin, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, fout)
    )


class Features(nn.Module):
    

    def __init__(self, geo: TorusGeometry, config: FeatureConfig) -> None:
        super().__init__()
        self.geo, self.config, self.ops = geo, config, IncidenceOps(geo)
        if config.lpe:
            geo.ensure_lpe()
        p0, p1, p2 = geo.node_pos, geo.edge_pos, geo.face_pos
        speed = np.linalg.norm(np.asarray(geo.velocity0), axis=1, keepdims=True).astype(np.float32)
        divv = (geo.d0.T @ np.asarray(geo.velocity1)).astype(np.float32)
        curlv = (geo.d1 @ np.asarray(geo.velocity1)).astype(np.float32)
        for name, value in {
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "v1": np.asarray(geo.velocity1),
            "speed0": speed,
            "divv0": divv,
            "curlv2": curlv,
            "area": np.asarray(geo.face_area),
            "normal": np.asarray(geo.face_normal),
            "node_area": np.asarray(geo.node_area),
            "fourier": geo.torus_fourier,
        }.items():
            self.register_buffer(
                name, torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
            )
        if geo.lpe is not None:
            for rank, value in enumerate(geo.lpe):
                self.register_buffer(
                    f"lpe{rank}", torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
                )
        self.dims = self._dims()

    def _dims(self) -> tuple[int, int, int]:
        d0, d1, d2 = 4, 4, 1
        if self.config.velocity:
            d0 += 3
            d1 += 2
            d2 += 9
        if self.config.lpe:
            d0 += self.geo.lpe_dim
            d1 += self.geo.lpe_dim
            d2 += self.geo.lpe_dim
        if self.config.heat:
            d0 += 2
        if self.config.global_broadcast:
            d0 += 2
            d1 += 2
            d2 += 2
        if self.config.torus_fourier:
            d0 += 8
            d1 += 8
            d2 += 8
        return d0, d1, d2

    @staticmethod
    def ex(x: Tensor, batch: int) -> Tensor:
        return x[None].expand(batch, -1, -1)

    def summary(self, u: Tensor) -> tuple[Tensor, Tensor]:
        w = self.node_area[:, 0] / self.node_area[:, 0].sum().clamp_min(1e-12)
        mean = (u * w).sum(1)
        centered = u - mean[:, None]
        energy = torch.sqrt((u.square() * w).sum(1).clamp_min(1e-12))
        scale = torch.sqrt((centered.square() * w).sum(1).clamp_min(1e-12))
        grad = self.ops.d0(u[..., None]).squeeze(-1)
        stats = torch.stack(
            (mean, energy, scale, torch.sqrt(grad.square().mean(1).clamp_min(1e-12))), -1
        )
        fourier = (u[..., None] * w[None, :, None] * self.fourier[None]).sum(1)
        return stats, torch.cat((stats, fourier), -1) if self.config.torus_fourier else stats

    def forward(self, value: Tensor) -> tuple[list[Tensor], Tensor]:
        b = value.shape[0]
        stats, topology = self.summary(value)
        u = value[..., None]
        grad = self.ops.d0(u)
        f0, f1, f2 = (
            [u, self.ex(self.p0, b)],
            [grad, self.ex(self.p1, b)],
            [torch.zeros(b, self.geo.n2, 1, dtype=u.dtype, device=u.device)],
        )
        if self.config.velocity:
            ue = 0.5 * (u[:, self.ops.tail] + u[:, self.ops.head])
            qadv = ue * self.ex(self.v1, b)
            divq = self.ops.delta1(qadv)
            f0 += [self.ex(self.divv0, b), divq, self.ex(self.speed0, b)]
            f1 += [self.ex(self.v1, b), qadv]
            curlq = self.ops.d1(qadv)
            f2 += [
                self.ex(self.curlv2, b),
                curlq,
                self.ex(self.p2, b),
                self.ex(self.area, b),
                self.ex(self.normal, b),
            ]
        if self.config.heat:
            h1 = u - 0.03 * self.ops.delta1(grad)
            h2 = h1 - 0.03 * self.ops.delta1(self.ops.d0(h1))
            f0 += [h1, h2]
        if self.config.lpe:
            f0 += [self.ex(self.lpe0, b)]
            f1 += [self.ex(self.lpe1, b)]
            f2 += [self.ex(self.lpe2, b)]
        if self.config.global_broadcast:
            local = stats[:, None, 1:3]
            f0 += [local.expand(-1, self.geo.n0, -1)]
            f1 += [local.expand(-1, self.geo.n1, -1)]
            f2 += [local.expand(-1, self.geo.n2, -1)]
        if self.config.torus_fourier:
            fourier = topology[:, None, 4:]
            f0 += [fourier.expand(-1, self.geo.n0, -1)]
            f1 += [fourier.expand(-1, self.geo.n1, -1)]
            f2 += [fourier.expand(-1, self.geo.n2, -1)]
        return [torch.cat(f0, -1), torch.cat(f1, -1), torch.cat(f2, -1)], topology


class TDKHOForms(nn.Module):
    def __init__(
        self,
        geo: TorusGeometry,
        config: FeatureConfig,
        task: str,
        hidden: int,
        layers: int,
        channels: int,
        microsteps: int,
        phase_init: str,
        harmonic_active: bool,
        variant: str,
    ) -> None:
        super().__init__()
        self.task, self.rank, self.variant = task, int(task), variant
        self.features = Features(geo, config)
        self.enc = nn.ModuleList(mlp(dim, hidden, hidden) for dim in self.features.dims)
        self.uses_phase = variant in ("full", "no_dirac", "no_harmonic")
        self.harmonic_active = harmonic_active and variant != "no_harmonic"
        kind = (
            scalar_tdk.DiracKuramotoLayer
            if variant in ("full", "no_harmonic")
            else (
                scalar_tdk.NoDiracKuramotoLayer
                if variant == "no_dirac"
                else scalar_tdk.FeatureDiracLayer
            )
        )
        self.layers = nn.ModuleList(
            (
                kind(hidden, channels, microsteps, True)
                if kind is not scalar_tdk.FeatureDiracLayer
                else kind(hidden, channels, True)
            )
            for _ in range(layers)
        )
        self.channels, self.phase_init = channels, phase_init
        if self.uses_phase and phase_init == "learnable":
            self.theta_init = nn.ModuleList(nn.Linear(hidden, channels) for _ in range(3))
        self.decoder = mlp(hidden + 2 * channels, hidden, 1)
        if self.harmonic_active:
            geo.ensure_harmonic()
            basis = geo.harmonic[self.rank]
            self.register_buffer(
                "harmonic_basis", torch.from_numpy(np.array(basis, dtype=np.float32, copy=True))
            )
            self.topology_head = mlp(12 if config.torus_fourier else 4, hidden, basis.shape[1])

    def forward(self, u: Tensor) -> Tensor:
        forms, topo = self.features(u)
        z = [encoder(x) for encoder, x in zip(self.enc, forms)]
        initial = list(z)
        if self.uses_phase:
            theta = (
                [
                    torch.zeros(
                        x.shape[0], x.shape[1], self.channels, device=x.device, dtype=x.dtype
                    )
                    for x in z
                ]
                if self.phase_init == "zero"
                else [math.pi * torch.tanh(h(x)) for h, x in zip(self.theta_init, z)]
            )
            for layer in self.layers:
                z, theta = layer(z, initial, theta, self.features.ops)
            decoded = torch.cat(
                (z[self.rank], torch.cos(theta[self.rank]), torch.sin(theta[self.rank])), -1
            )
        else:
            for layer in self.layers:
                z = layer(z, initial, self.features.ops)
            decoded = torch.cat(
                (z[self.rank], self.layers[-1].decoder_channels(z[self.rank], self.rank)), -1
            )
        local = self.decoder(decoded).squeeze(-1)
        if not self.harmonic_active:
            return local
        basis = self.harmonic_basis
        local = local - torch.einsum("bn,nk->bk", local, basis) @ basis.T
        return local + self.topology_head(topo) @ basis.T


def harmonic_diagnosis(root: Path, task: str, geo: TorusGeometry) -> dict:
    geo.ensure_harmonic()
    target = np.load(root / {"0": "uT0.npy", "1": "qT1.npy", "2": "mT2.npy"}[task], mmap_mode="r")
    idx = splits(len(target))["train"]
    basis = geo.harmonic[int(task)]
    coeff = np.asarray(target[idx]) @ basis
    scale = float(np.sqrt(np.mean(np.asarray(target[idx]) ** 2)) + 1e-12)
    std = np.std(coeff, axis=0)
    return {
        "rank": int(task),
        "beta": int(basis.shape[1]),
        "coefficient_std": std.tolist(),
        "relative_std": (std / scale).tolist(),
        "harmonic_head_active": bool(np.max(std) > 1e-6 * scale),
    }


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values = []
    for x, y in loader:
        values.append(relative_l2(model(x.to(device)), y.to(device)).item())
    return float(np.mean(values))


def run(
    config: FeatureConfig,
    task: str,
    variant: str,
    args: argparse.Namespace,
    geo: TorusGeometry,
    device: torch.device,
) -> dict:
    root = args.data.resolve()
    output = args.output.resolve() / f"form_{task}" / config.name / variant / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    stats = task_stats(root, task)
    diagnostic = harmonic_diagnosis(root, task, geo)
    seed_all(args.seed)
    train = DataLoader(
        FormDataset(root, task, "train", stats),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val = DataLoader(
        FormDataset(root, task, "val", stats),
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test = DataLoader(
        FormDataset(root, task, "test", stats),
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = TDKHOForms(
        geo,
        config,
        task,
        args.hidden,
        args.layers,
        args.channels,
        args.microsteps,
        args.phase_init,
        diagnostic["harmonic_head_active"],
        variant,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, state, best_epoch = float("inf"), None, 0
    hist = {"train": [], "val": []}
    started = time.time()
    print(
        f"[TDK-HO form={task} {config.name}/{variant}] params={sum(p.numel() for p in model.parameters())} harmonic={diagnostic['harmonic_head_active']} device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train:
            opt.zero_grad(set_to_none=True)
            pred = model(x.to(device))
            loss = relative_l2(pred, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        tr, va = total / len(train), validate(model, val, device)
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best:
            best, best_epoch, state = (
                va,
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        scheduler.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[TDK-HO form={task} {config.name}/{variant}] epoch={epoch:03d}/{args.epochs} train={tr:.6f} val={va:.6f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    model.load_state_dict(state)
    model.eval()
    pp = []
    yy = []
    print(f"[TDK-HO form={task}] evaluating {len(test.dataset)} held-out samples", flush=True)
    with torch.inference_mode():
        for x, y in test:
            pp.append(model(x.to(device)).cpu().numpy())
            yy.append(y.numpy())
    pred, target = np.concatenate(pp), np.concatenate(yy)
    scale = stats["y_scale"]
    err = (pred - target) * scale
    result = {
        "family": "TDK-HO",
        "task": f"form_{task}",
        "feature_config": config.name,
        "variant": variant,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_relative_l2": best,
        "epochs": args.epochs,
        "loss": "mean_per_sample_relative_l2",
        "normalization": stats,
        "mse": float(np.mean(err**2)),
        "relative_l2": float(
            np.mean(
                np.linalg.norm(err, axis=1)
                / np.maximum(np.linalg.norm(target * scale, axis=1), 1e-12)
            )
        ),
        "harmonic": diagnostic,
    }
    np.save(output / "prediction_test_normalized.npy", pred.astype(np.float32))
    np.save(output / "target_test_normalized.npy", target.astype(np.float32))
    np.save(output / "test_indices.npy", np.asarray(test.dataset.indices, dtype=np.int64))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": portable_arguments(args),
            "task": task,
            "feature_config": config.name,
            "variant": variant,
            "harmonic": diagnostic,
        },
        output / "best.pt",
    )
    write_json(output / "history.json", hist)
    write_json(output / "result.json", result)
    print(f"[TDK-HO form={task}] complete test_relative_l2={result['relative_l2']:.6f}", flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "data" / "torus_multiform_v1")
    p.add_argument("--task", choices=("0", "1", "2", "all"), default="all")
    p.add_argument("--configs", default="native,spectral,diffusion_spectral,conditioned")
    p.add_argument("--variants", default="full")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--microsteps", type=int, default=2)
    p.add_argument("--lpe-dim", type=int, default=8)
    p.add_argument("--phase-init", choices=("zero", "learnable"), default="zero")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--output", type=Path, default=HERE.parent / "runs" / "dkho" / "small")
    args = p.parse_args()
    requested = tuple(x for x in args.configs.split(",") if x)
    variants = tuple(x for x in args.variants.split(",") if x)
    if any(x not in FEATURE_CONFIGS for x in requested):
        p.error(f"unknown configs; choices={tuple(FEATURE_CONFIGS)}")
    if any(x not in STRUCTURE_VARIANTS for x in variants):
        p.error(f"unknown variants; choices={STRUCTURE_VARIANTS}")
    device = resolve_device(args.device)
    geo = TorusGeometry(args.data, args.lpe_dim)
    tasks = ("0", "1", "2") if args.task == "all" else (args.task,)
    results = [
        run(FEATURE_CONFIGS[c], t, v, args, geo, device)
        for t in tasks
        for c in requested
        for v in variants
    ]
    write_json(
        args.output / "summary.json",
        {
            "results": results,
            "geometry": {"n0": geo.n0, "n1": geo.n1, "n2": geo.n2, "betti": [1, 2, 1]},
            "model_revision": "torus_cochains_dkho_v1_unweighted",
        },
    )


if __name__ == "__main__":
    main()
