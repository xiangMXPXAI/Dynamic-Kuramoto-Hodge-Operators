"""Train baseline models on toroidal transport cochain targets."""

from __future__ import annotations

import argparse
import importlib.util
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
    from . import common
    from .common import (
        FEATURE_CONFIGS,
        FeatureConfig,
        FormDataset,
        TorusGeometry,
        relative_l2,
        resolve_device,
        seed_all,
        task_stats,
        write_json,
    )
    from .train_dkho import Features
except ImportError:  
    sys.path.insert(0, str(HERE))
    import common
    from common import (
        FEATURE_CONFIGS,
        FeatureConfig,
        FormDataset,
        TorusGeometry,
        relative_l2,
        resolve_device,
        seed_all,
        task_stats,
        write_json,
    )
    from train_dkho import Features


MODELS = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")


def load_native_models():
    
    path = HERE.parents[1] / "darcy" / "baselines" / "native_models.py"
    spec = importlib.util.spec_from_file_location("torus_cochain_native_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    
    sys.modules["common"] = common
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


    def pad3(value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"coordinate array must be rank 2, got {value.shape}")
        if value.shape[1] == 3:
            return np.array(value, dtype=np.float32, copy=True)
        if value.shape[1] == 2:
            return np.concatenate((value, np.zeros((len(value), 1), np.float32)), axis=1)
        raise ValueError(f"expected R^2 or R^3 coordinates, got {value.shape}")

    module.pad3 = pad3
    return module


REF = load_native_models()


class BaselineFeatures(nn.Module):
    

    def __init__(self, geo: TorusGeometry, config: FeatureConfig, name: str) -> None:
        super().__init__()
        self.geo, self.config, self.name = geo, config, name
        self.full = name == "HSD"
        self.tdk = Features(geo, config) if self.full else None
        if config.lpe:
            geo.ensure_lpe()
        self.register_buffer(
            "p0", torch.from_numpy(np.array(geo.node_pos, dtype=np.float32, copy=True))
        )
        self.register_buffer(
            "area", torch.from_numpy(np.array(geo.node_area, dtype=np.float32, copy=True))
        )
        self.register_buffer(
            "tail", torch.from_numpy(np.array(geo.tail, dtype=np.int64, copy=True))
        )
        self.register_buffer(
            "head", torch.from_numpy(np.array(geo.head, dtype=np.int64, copy=True))
        )
        self.register_buffer(
            "torus_basis",
            torch.from_numpy(np.array(geo.torus_fourier, dtype=np.float32, copy=True)),
        )
        if geo.lpe is not None:
            self.register_buffer(
                "lpe0", torch.from_numpy(np.array(geo.lpe[0], dtype=np.float32, copy=True))
            )
        self.dims = self.tdk.dims if self.full else (self.node_dim(), 1, 1)

    def node_dim(self) -> int:
        
        
        value = 4
        if self.config.lpe:
            value += self.geo.lpe_dim
        if self.config.heat:
            value += 2
        if self.config.global_broadcast:
            value += 2
        if self.config.torus_fourier:
            value += 8
        return value

    def forward(self, u: Tensor) -> list[Tensor]:
        if self.full:
            return self.tdk(u)[0]
        b = len(u)
        parts = [u[..., None], self.p0[None].expand(b, -1, -1)]
        if self.config.lpe:
            parts.append(self.lpe0[None].expand(b, -1, -1))
        if self.config.heat:
            grad = u[:, self.head] - u[:, self.tail]
            lap = torch.zeros_like(u)
            lap.index_add_(1, self.tail, -grad)
            lap.index_add_(1, self.head, grad)
            h1 = u - 0.03 * lap
            lap2 = torch.zeros_like(u)
            g2 = h1[:, self.head] - h1[:, self.tail]
            lap2.index_add_(1, self.tail, -g2)
            lap2.index_add_(1, self.head, g2)
            parts += [torch.stack((h1, h1 - 0.03 * lap2), -1)]
        if self.config.global_broadcast:
            w = self.area[:, 0] / self.area[:, 0].sum().clamp_min(1e-12)
            mean = (u * w).sum(1)
            rms = torch.sqrt((u.square() * w).sum(1).clamp_min(1e-12))
            parts += [torch.stack((mean, rms), -1)[:, None].expand(-1, self.geo.n0, -1)]
        if self.config.torus_fourier:
            w = self.area[:, 0] / self.area[:, 0].sum()
            modes = (u[..., None] * w[None, :, None] * self.torus_basis[None]).sum(1)
            parts += [modes[:, None].expand(-1, self.geo.n0, -1)]
        node = torch.cat(parts, -1)
        return [
            node,
            torch.zeros(b, self.geo.n1, 1, device=u.device),
            torch.zeros(b, self.geo.n2, 1, device=u.device),
        ]


def harmonic_metric(task: str, pred: np.ndarray, target: np.ndarray, geo: TorusGeometry) -> float:
    geo.ensure_harmonic()
    basis = geo.harmonic[int(task)]
    a, b = pred @ basis, target @ basis
    return float(
        np.mean(np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12))
    )


@torch.no_grad()
def valid(
    model: nn.Module, builder: BaselineFeatures, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    values = []
    for x, y in loader:
        values.append(relative_l2(model(builder(x.to(device))), y.to(device)).item())
    return float(np.mean(values))


def run(
    name: str,
    config: FeatureConfig,
    task: str,
    args: argparse.Namespace,
    geo: TorusGeometry,
    device: torch.device,
) -> dict:
    root = args.data.resolve()
    out = args.output.resolve() / f"form_{task}" / config.name / name / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    stats = task_stats(root, task)
    builder = BaselineFeatures(geo, config, name).to(device)
    model = REF.make_reference_model(name, geo, builder.dims, task, "toroidal").to(device)
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = float("inf")
    state = None
    epoch_best = 0
    hist = {"train": [], "val": []}
    started = time.time()
    print(
        f"[{name} form={task} {config.name}] native_ranks={REF.native_input_ranks(name)} params={sum(p.numel() for p in model.parameters())} device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train:
            opt.zero_grad(set_to_none=True)
            pred = model(builder(x.to(device)))
            loss = relative_l2(pred, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        tr = total / len(train)
        va = valid(model, builder, val, device)
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best:
            best, epoch_best, state = (
                va,
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
        sch.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{name} form={task} {config.name}] epoch={epoch:03d}/{args.epochs} train={tr:.6f} val={va:.6f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    model.load_state_dict(state)
    model.eval()
    pp = []
    yy = []
    with torch.inference_mode():
        for x, y in test:
            pp.append(model(builder(x.to(device))).cpu().numpy())
            yy.append(y.numpy())
    pred, target = np.concatenate(pp), np.concatenate(yy)
    scale = stats["y_scale"]
    err = (pred - target) * scale
    result = {
        "family": name,
        "task": f"form_{task}",
        "feature_config": config.name,
        "native_input_ranks": list(REF.native_input_ranks(name)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": epoch_best,
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
        "harmonic_relative_l2": harmonic_metric(task, pred, target, geo),
    }
    np.save(out / "prediction_test_normalized.npy", pred.astype(np.float32))
    np.save(out / "target_test_normalized.npy", target.astype(np.float32))
    np.save(out / "test_indices.npy", np.asarray(test.dataset.indices, dtype=np.int64))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": task,
            "model": name,
            "feature_config": config.name,
            "args": portable_arguments(args),
        },
        out / "best.pt",
    )
    write_json(out / "history.json", hist)
    write_json(out / "result.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "data" / "torus_multiform_v1")
    p.add_argument("--model", choices=(*MODELS, "all"), default="all")
    p.add_argument("--task", choices=("0", "1", "2", "all"), default="all")
    p.add_argument("--configs", default="native,spectral,diffusion_spectral,conditioned")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lpe-dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--output", type=Path, default=HERE.parent / "runs" / "baseline" / "main")
    args = p.parse_args()
    requested = tuple(x for x in args.configs.split(",") if x)
    if any(x not in FEATURE_CONFIGS for x in requested):
        p.error(f"unknown feature config; choices={tuple(FEATURE_CONFIGS)}")
    geo = TorusGeometry(args.data, args.lpe_dim)
    device = resolve_device(args.device)
    tasks = ("0", "1", "2") if args.task == "all" else (args.task,)
    names = MODELS if args.model == "all" else (args.model,)
    results = [
        run(n, FEATURE_CONFIGS[c], t, args, geo, device)
        for t in tasks
        for c in requested
        for n in names
    ]
    write_json(
        args.output / "summary.json",
        {
            "results": results,
            "geometry": {"n0": geo.n0, "n1": geo.n1, "n2": geo.n2, "betti": [1, 2, 1]},
            "protocol": "native_rank_baseline_torus_cochains_v1",
        },
    )


if __name__ == "__main__":
    main()
