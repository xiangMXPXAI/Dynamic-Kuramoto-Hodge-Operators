"""Train baseline models for Darcy cochain targets under defined input profiles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (  
    ARCHIVE,
    FEATURE_CONFIGS,
    DarcyGeometry,
    DarcyMemmapDataset,
    FeatureConfig,
    IncidenceOps,
    autocast_context,
    configure_runtime,
    feature_profile_details,
    make_grad_scaler,
    make_loader,
    maybe_compile,
    parameter_count,
    portable_path,
    prepare_cache,
    relative_l2,
    resolve_device,
    seed_all,
    task_stats,
    write_json,
)
from native_models import SPECS, make_reference_model, native_input_ranks  

MODELS = ("GNO", "FNO", "MGN", "DeepONet", "GeoFNO", "HSD")
ARCHITECTURE_REVISION = "darcy_2d_native_rank_v4"
PROTOCOL_REVISION = "darcy_matched_rel_l2_200e_patience40_v1"


class CanonicalFeatures(nn.Module):
    

    def __init__(
        self, geo: DarcyGeometry, config: FeatureConfig, input_ranks: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.geo, self.config, self.ops = geo, config, IncidenceOps(geo)
        self.input_ranks = tuple(sorted(set(input_ranks)))
        if not self.input_ranks or any(rank not in (0, 1, 2) for rank in self.input_ranks):
            raise ValueError(f"invalid native input ranks: {input_ranks}")
        if config.lpe:
            geo.ensure_lpe()
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
        }
        if geo.lpe is not None:
            static.update({"lpe0": geo.lpe[0], "lpe1": geo.lpe[1], "lpe2": geo.lpe[2]})
        for name, value in static.items():
            self.register_buffer(name, torch.from_numpy(value.astype(np.float32)))
        self.register_buffer(
            "hole1", torch.from_numpy((geo.node_component == 2).astype(np.float32))[None, :, None]
        )
        self.register_buffer(
            "hole2", torch.from_numpy((geo.node_component == 3).astype(np.float32))[None, :, None]
        )
        self.dims = self._dims()

    def _dims(self) -> tuple[int, int, int]:
        d0, d1, d2 = 12, 14, 12
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
        return tuple((d0, d1, d2)[rank] if rank in self.input_ranks else 0 for rank in range(3))

    @staticmethod
    def ex(value: Tensor, batch: int) -> Tensor:
        return value[None].expand(batch, -1, -1)

    def forward(self, f: Tensor, g: Tensor) -> tuple[list[Tensor], Tensor]:
        b = f.shape[0]
        u = f[:, :, None]
        gb = g[:, None, :1] * self.hole1 + g[:, None, 1:] * self.hole2
        f0 = [
            u,
            gb,
            self.ex(self.node_pos, b),
            self.ex(self.kappa_node, b),
            self.ex(self.node_boundary, b),
        ]
        parts: dict[int, list[Tensor]] = {0: f0}
        if 1 in self.input_ranks:
            du, dgb = self.ops.d0(u), self.ops.d0(gb)
            parts[1] = [
                du,
                dgb,
                self.ex(self.edge_pos, b),
                self.ex(self.edge_dir, b),
                self.ex(self.edge_length, b),
                self.ex(self.kappa_edge_invariants, b),
                self.ex(self.edge_boundary, b),
            ]
        if 2 in self.input_ranks:
            parts[2] = [
                u[:, self.ops.faces].mean(2) * self.area[None],
                self.ex(self.face_pos, b),
                self.ex(self.area, b),
                self.ex(self.kappa_face, b),
                self.ex(self.face_boundary, b),
            ]
        if self.config.lpe:
            parts[0].append(self.ex(self.lpe0, b))
            if 1 in parts:
                parts[1].append(self.ex(self.lpe1, b))
            if 2 in parts:
                parts[2].append(self.ex(self.lpe2, b))
        if self.config.heat:
            scales: list[Tensor] = []
            current = u
            for step in range(1, 13):
                current = current - 0.01 * self.ops.delta1(self.ops.d0(current))
                if step in (1, 4, 12):
                    scales.append(current)
            parts[0].append(torch.cat(scales, -1))
            if 1 in parts:
                parts[1].append(torch.cat([self.ops.d0(v) for v in scales], -1))
            if 2 in parts:
                parts[2].append(
                    torch.cat([v[:, self.ops.faces].mean(2) * self.area[None] for v in scales], -1)
                )
        weight, mass = self.node_area[:, 0], self.node_area[:, 0].sum().clamp_min(1e-12)
        mean = (f * weight[None]).sum(1) / mass
        rms = torch.sqrt((f.square() * weight[None]).sum(1) / mass + 1e-12)
        px = (f * weight[None] * self.node_pos[:, 0][None]).sum(1) / mass
        py = (f * weight[None] * self.node_pos[:, 1][None]).sum(1) / mass
        summary = torch.stack([g[:, 0], g[:, 1], mean, px, py, rms], -1)
        if self.config.global_broadcast:
            parts[0].append(summary[:, None].expand(-1, self.geo.n0, -1))
            if 1 in parts:
                parts[1].append(summary[:, None].expand(-1, self.geo.n1, -1))
            if 2 in parts:
                parts[2].append(summary[:, None].expand(-1, self.geo.n2, -1))
        return [torch.cat(parts[rank], -1) for rank in self.input_ranks], summary


@torch.inference_mode()
def inference(
    model: nn.Module, builder: CanonicalFeatures, loader, device: torch.device, precision: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for f, g, y in loader:
        f, g = f.to(device, non_blocking=True), g.to(device, non_blocking=True)
        with autocast_context(device, precision):
            forms, _ = builder(f, g)
            prediction = model(forms)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(y.float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


@torch.inference_mode()
def validation_relative_l2(
    model: nn.Module, builder: CanonicalFeatures, loader, device: torch.device, precision: str
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    for f, g, y in loader:
        f, g, y = (
            f.to(device, non_blocking=True),
            g.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
        )
        with autocast_context(device, precision):
            forms, _ = builder(f, g)
            prediction = model(forms)
        per_sample = torch.linalg.vector_norm(
            prediction.float() - y.float(), dim=1
        ) / torch.linalg.vector_norm(y.float(), dim=1).clamp_min(1e-12)
        total += per_sample.sum()
        count += y.shape[0]
    return (total / count).item()


def calculate_metrics(
    task: str, prediction: np.ndarray, target: np.ndarray, scale: float, geo: DarcyGeometry
) -> dict[str, float]:
    prediction, target = prediction * scale, target * scale
    error = prediction - target
    metrics = {
        "mse": float(np.mean(error * error)),
        "relative_l1": float(
            np.mean(np.abs(error).sum(1) / np.maximum(np.abs(target).sum(1), 1e-12))
        ),
        "relative_l2": float(
            np.mean(
                np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
            )
        ),
    }
    if task == "1":
        hp, hy = prediction @ geo.harmonic_basis_1, target @ geo.harmonic_basis_1
        metrics["harmonic_relative_l2"] = float(
            np.mean(np.linalg.norm(hp - hy, axis=1) / np.maximum(np.linalg.norm(hy, axis=1), 1e-12))
        )
    return metrics


def train_one(
    name: str,
    task: str,
    config: FeatureConfig,
    args: argparse.Namespace,
    geo: DarcyGeometry,
    stats: dict[str, float],
) -> dict:
    run = args.output_root / f"form_{task}" / config.name / name / f"seed_{args.seed}"
    result_file = run / "result.json"
    ranks = native_input_ranks(name)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = resolve_device(args.device)
    configure_runtime(device, args.tf32)
    builder = CanonicalFeatures(geo, config, ranks).to(device)
    model = make_reference_model(name, geo, builder.dims, task, args.reference).to(device)
    expected_parameters = parameter_count(model)
    if result_file.exists() and not args.rerun:
        saved = json.loads(result_file.read_text(encoding="utf-8"))
        if (
            saved.get("architecture_revision") == ARCHITECTURE_REVISION
            and saved.get("protocol_revision") == PROTOCOL_REVISION
            and saved.get("parameters") == expected_parameters
        ):
            return saved
        print(
            f"[{name} form={task} {config.name}] stale result detected; retraining native-rank protocol",
            flush=True,
        )
    execution_model = maybe_compile(model, device, args.compile)
    train = make_loader(
        task, "train", stats, args.batch_size, True, device, args.resident_data, args.workers
    )
    val = make_loader(
        task, "val", stats, args.eval_batch_size, False, device, args.resident_data, args.workers
    )
    test = make_loader(
        task, "test", stats, args.eval_batch_size, False, device, args.resident_data, args.workers
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = make_grad_scaler(device, args.precision)
    best, best_epoch, best_state, history, started = (
        float("inf"),
        0,
        None,
        {"train": [], "val": []},
        time.time(),
    )
    bad_epochs, completed_epochs = 0, 0
    print(
        f"[{name} form={task} {config.name}] native_ranks={ranks} parameters={expected_parameters} reference={args.reference} device={device} batches={len(train)}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        execution_model.train()
        total = torch.zeros((), device=device)
        for f, g, y in train:
            optimizer.zero_grad(set_to_none=True)
            f, g, y = (
                f.to(device, non_blocking=True),
                g.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            with autocast_context(device, args.precision):
                forms, _ = builder(f, g)
                prediction = execution_model(forms)
            loss = relative_l2(prediction.float(), y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.detach()
        train_loss = (total / len(train)).item()
        val_loss = validation_relative_l2(model, builder, val, device, args.precision)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        completed_epochs = epoch
        if val_loss < best - args.min_delta:
            best, best_epoch, best_state, bad_epochs = (
                val_loss,
                epoch,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            bad_epochs += 1
        scheduler.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"[{name} form={task} {config.name}] epoch={epoch:03d}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"[{name} form={task} {config.name}] early-stop epoch={epoch} best_epoch={best_epoch} patience={args.patience}",
                flush=True,
            )
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    prediction, target = inference(model, builder, test, device, args.precision)
    np.save(run / "prediction_test_normalized.npy", prediction.astype(np.float32))
    np.save(run / "target_test_normalized.npy", target.astype(np.float32))
    np.save(run / "test_indices.npy", np.asarray(DarcyMemmapDataset(task, "test", stats).indices))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "task": task,
            "model": name,
            "feature_config": config.name,
            "input_protocol": feature_profile_details(config),
            "native_input_ranks": ranks,
            "reference_architecture": args.reference,
            "architecture_revision": ARCHITECTURE_REVISION,
            "protocol_revision": PROTOCOL_REVISION,
            "args": {**vars(args), "output_root": portable_path(args.output_root)},
            "stats": stats,
        },
        run / "best.pt",
    )
    write_json(run / "history.json", history)
    result = {
        "family": name,
        "task": f"form_{task}",
        "feature_config": config.name,
        "input_protocol": feature_profile_details(config),
        "native_input_ranks": list(ranks),
        "parameters": expected_parameters,
        "reference_architecture": args.reference,
        "architecture_revision": ARCHITECTURE_REVISION,
        "protocol_revision": PROTOCOL_REVISION,
        "best_epoch": best_epoch,
        "best_val_relative_l2": best,
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "optimizer_updates": completed_epochs * len(train),
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "wall_seconds": time.time() - started,
        "loss": "mean_per_sample_relative_l2",
        "normalization": stats,
        "execution": {
            "resident_data": args.resident_data,
            "workers": args.workers,
            "precision": args.precision,
            "tf32": args.tf32,
            "compile": args.compile,
        },
        **calculate_metrics(task, prediction, target, stats["y_scale"], geo),
    }
    write_json(result_file, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--task", choices=("0", "1", "2", "all"), default="all")
    parser.add_argument(
        "--feature-configs", default="all", help="comma-separated feature levels or 'all'"
    )
    parser.add_argument(
        "--feature-config",
        choices=tuple(FEATURE_CONFIGS),
        default=None,
        help="deprecated single-level alias",
    )
    parser.add_argument(
        "--reference", choices=tuple(SPECS), default="toroidal", help="archived capacity schedule"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--patience", type=int, default=40, help="0 disables validation early stopping"
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lpe-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resident-data", choices=("auto", "memmap", "gpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "auto", "amp-bf16", "amp-fp16"), default="auto"
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument(
        "--compile",
        choices=("off", "auto", "reduce-overhead", "max-autotune"),
        default="off",
        help="opt-in Inductor; test per GPU/model family",
    )
    parser.add_argument(
        "--output-root",
        default=str(HERE.parent / "runs" / "baseline" / "main"),
        help="path relative to this file unless absolute",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.workers < 0 or args.patience < 0 or args.min_delta < 0:
        parser.error("workers, patience and min-delta must be non-negative")
    requested = (
        (args.feature_config,)
        if args.feature_config
        else (
            tuple(FEATURE_CONFIGS)
            if args.feature_configs == "all"
            else tuple(item for item in args.feature_configs.split(",") if item)
        )
    )
    unknown = [name for name in requested if name not in FEATURE_CONFIGS]
    if not requested or unknown:
        parser.error(f"unknown feature config(s): {unknown}; choices={list(FEATURE_CONFIGS)}")
    args.output_root = Path(args.output_root)
    args.output_root = (
        args.output_root if args.output_root.is_absolute() else HERE / args.output_root
    )
    prepare_cache()
    geo = DarcyGeometry(lpe_dim=args.lpe_dim)
    tasks = ("0", "1", "2") if args.task == "all" else (args.task,)
    models = MODELS if args.model == "all" else (args.model,)
    results = []
    for task in tasks:
        for config_name in requested:
            for name in models:
                results.append(
                    train_one(name, task, FEATURE_CONFIGS[config_name], args, geo, task_stats(task))
                )
    write_json(
        args.output_root / "summary.json",
        {
            "archive": portable_path(ARCHIVE),
            "reference_architecture": args.reference,
            "architecture_revision": ARCHITECTURE_REVISION,
            "protocol_revision": PROTOCOL_REVISION,
            "results": results,
        },
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
