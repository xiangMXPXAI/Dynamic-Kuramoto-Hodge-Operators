"""Validate toroidal transport cochain data and model interfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
try:
    from .common import FEATURE_CONFIGS, TorusGeometry, task_stats
    from .train_dkho import TDKHOForms, harmonic_diagnosis
    from .train_baselines import BaselineFeatures, MODELS, REF
except ImportError:  
    sys.path.insert(0, str(HERE))
    from common import FEATURE_CONFIGS, TorusGeometry, task_stats
    from train_dkho import TDKHOForms, harmonic_diagnosis
    from train_baselines import BaselineFeatures, MODELS, REF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "torus_multiform_v1")
    parser.add_argument(
        "--features", choices=tuple(FEATURE_CONFIGS), default="conditioned"
    )
    parser.add_argument("--lpe-dim", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.data.resolve()
    geo = TorusGeometry(root, args.lpe_dim)
    cfg = FEATURE_CONFIGS[args.features]
    d0, d1 = geo.d0, geo.d1
    product = d1 @ d0
    archive = {
        "n0": geo.n0,
        "n1": geo.n1,
        "n2": geo.n2,
        "d1d0_max": float(np.max(np.abs(product.data)) if product.nnz else 0.0),
        "targets": {},
    }
    x = torch.from_numpy(
        np.asarray(np.load(root / "u0.npy", mmap_mode="r")[0:1], np.float32)
        / task_stats(root, "0")["x_scale"]
    )
    for task, filename, n in (
        ("0", "uT0.npy", geo.n0),
        ("1", "qT1.npy", geo.n1),
        ("2", "mT2.npy", geo.n2),
    ):
        array = np.load(root / filename, mmap_mode="r")
        if array.ndim != 2 or array.shape[1] != n:
            raise RuntimeError(f"{filename} has {array.shape}; expected (sample,{n})")
        diag = harmonic_diagnosis(root, task, geo)
        model = TDKHOForms(
            geo, cfg, task, 16, 1, 1, 1, "zero", diag["harmonic_head_active"], "full"
        ).eval()
        with torch.inference_mode():
            out = model(x)
        if tuple(out.shape) != (1, n):
            raise RuntimeError(f"TDK-HO task{task}: output {tuple(out.shape)}, expected (1,{n})")
        archive["targets"][f"C{task}"] = {
            "shape": list(array.shape),
            "harmonic": diag,
            "tdk_parameters_smoke": sum(p.numel() for p in model.parameters()),
        }
    baselines = {}
    for name in MODELS:
        builder = BaselineFeatures(geo, cfg, name).eval()
        per_task = {}
        for task, n in (("0", geo.n0), ("1", geo.n1), ("2", geo.n2)):
            model = REF.make_reference_model(name, geo, builder.dims, task, "toroidal").eval()
            parameters = sum(p.numel() for p in model.parameters())
            if not 200_000 <= parameters <= 500_000:
                raise RuntimeError(
                    f"{name}/C{task} has {parameters} parameters outside [200000,500000]"
                )
            with torch.inference_mode():
                out = model(builder(x))
            if tuple(out.shape) != (1, n):
                raise RuntimeError(f"{name}/C{task}: output {tuple(out.shape)}, expected (1,{n})")
            per_task[f"C{task}"] = {
                "parameters": parameters,
                "native_input_ranks": list(REF.native_input_ranks(name)),
                "output_shape": list(out.shape),
            }
        baselines[name] = per_task
    report = {
        "archive": archive,
        "baselines": baselines,
        "feature_config": cfg.name,
        "unweighted": True,
        "expected_betti": [1, 2, 1],
        "pass": True,
    }
    out = args.out or root / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
