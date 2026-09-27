"""Render qualitative Darcy predictions and native-support error maps."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Arc, Rectangle
from matplotlib.transforms import Bbox
from matplotlib.tri import LinearTriInterpolator, Triangulation

from common import CACHE, TASK_TARGET, DarcyGeometry, prepare_cache


ROOT = Path(__file__).resolve().parent

TDK_ROOTS = {
    "DKHO-large": ROOT / "runs" / "dkho" / "large",
    "DKHO-small": ROOT / "runs" / "dkho" / "small",
}

PRIMARY_DKHO = "DKHO-large"

BASE = ROOT / "runs" / "baseline" / "main"

TDK_CONFIG = "conditioned"

BASELINE_MAIN_CONFIG = "native"

TDK_STRUCTURE = "full"

BASELINE_NAMES = (
    "GNO",
    "FNO",
    "MGN",
    "DeepONet",
    "GeoFNO",
    "HSD",
)

PAPER_INK = "#10213B"

PAPER_MESH = "#A9C4D8"

PAPER_LOOP = "#D56A2D"

SOURCE_CMAP = LinearSegmentedColormap.from_list(
    "source_publication",
    ["#1E5D9C", "#4B9DCC", "#B9DDF0", "#FCFBF6", "#F2B08D", "#C9414D"],
    N=256,
)

POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "potential_publication",
    [
        "#0C437F",
        "#3271c2",
        "#5588C7",
        "#446CC1B8",
        "#E6ED84",
        "#F5D95D",
    ],
    N=256,
)

FLUX_CMAP = LinearSegmentedColormap.from_list(
    "flux_reference",
    ["#1F78C1", "#20A9C7", "#37C8B0", "#8BD866", "#F2E552"],
    N=256,
)

FLUX_SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "flux_direction_publication",
    ["#357AB6", "#E9F3F4", "#DF8E69"],
    N=256,
)

CIRCULATION_CMAP = LinearSegmentedColormap.from_list(
    "circulation_publication",
    ["#513A91", "#9678C1", "#D9D1EA", "#FCF8EE", "#F0AF74", "#D54A4E"],
    N=256,
)

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "magma_reference_extended_dark",
    [
        (0.00, "#000004"),
        (0.22, "#08091F"),
        (0.42, "#251255"),
        (0.60, "#57106E"),
        (0.74, "#9C2C7C"),
        (0.85, "#D64E6C"),
        (0.94, "#F99D6C"),
        (1.00, "#FCFDBF"),
    ],
    N=256,
)

REFERENCE_POTENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "reference_potential_lifted",
    [
        "#185390",
        "#23679B",
        "#77AFC9",
        "#C3CBA1",
        "#E6D653",
        "#F4E66A",
    ],
    N=256,
)

PALETTE_LIBRARY: dict[str, dict[str, list[str]]] = {
    
    "reference": {},
    "legacy_reference": {},
    
    "editorial_bluegold": {
        "source": [
            "#3268A8",
            "#6299C8",
            "#A9CCE1",
            "#E8F0F3",
            "#FCFAF5",
            "#F7D7B9",
            "#EE9B72",
            "#D85D5D",
            "#B94A59",
        ],
        "potential": [
            "#2F6598",
            "#4D86B3",
            "#72AAC0",
            "#9BC9C5",
            "#C9DDB2",
            "#E9DF88",
            "#F5C85B",
        ],
        "flux": [
            "#2E68A3",
            "#2F8DBC",
            "#3BB2C1",
            "#6CCCB4",
            "#A6D88D",
            "#D9E17B",
            "#F3DC68",
        ],
        "flux_signed": [
            "#4B82B1",
            "#A8CBDB",
            "#F9F8F2",
            "#EDB79D",
            "#D87967",
        ],
        "circulation": [
            "#6A5AA5",
            "#8E84BC",
            "#BAB4D2",
            "#E7E3EC",
            "#FCF9F2",
            "#F5CDA9",
            "#E99264",
            "#C96052",
        ],
    },

    "ocean_sunrise": {
        "source": [
            "#235A93",
            "#4F8BB9",
            "#91BCD3",
            "#D4E6ED",
            "#FCFAF4",
            "#F8D6B1",
            "#F3A26E",
            "#E46B55",
            "#C84955",
        ],
        "potential": [
            "#1F628E",
            "#2F87A6",
            "#4CAAB6",
            "#79C7B7",
            "#AAD8AB",
            "#D4DF91",
            "#F0D76F",
            "#F6C95A",
        ],
        "flux": [
            "#265E9C",
            "#247FAE",
            "#1A9EB5",
            "#38BBAA",
            "#72CC95",
            "#ADD783",
            "#DEE278",
            "#F4E26D",
        ],
        "flux_signed": [
            "#397EA8",
            "#9CC8D2",
            "#FAF8F1",
            "#F0B78B",
            "#DB7659",
        ],
        "circulation": [
            "#21687C",
            "#4B91A0",
            "#8CBDBA",
            "#C8DDD3",
            "#FBF8EF",
            "#F6D1AC",
            "#EFA06D",
            "#D96F50",
            "#B95147",
        ],
    },

    "pearl_rose": {
        "source": [
            "#4B69A6",
            "#7394C1",
            "#A9C0D8",
            "#DEE7ED",
            "#FDFCFA",
            "#F7D6D7",
            "#EEA3A9",
            "#D87383",
            "#B44E69",
        ],
        "potential": [
            "#5067A1",
            "#6F8DBD",
            "#89B1C9",
            "#A7CFCC",
            "#C9DFB6",
            "#E7E8A0",
            "#F3D579",
        ],
        "flux": [
            "#4B66A4",
            "#4D89BE",
            "#45AABF",
            "#54C4B3",
            "#85D1A0",
            "#B9DA8B",
            "#E1DF7C",
        ],
        "flux_signed": [
            "#627EAF",
            "#B4CAD8",
            "#FBFAF6",
            "#EFB8B3",
            "#D37B83",
        ],
        "circulation": [
            "#3E7B78",
            "#6AA39B",
            "#A4C6B8",
            "#DCE1D2",
            "#FCF8F1",
            "#F5D0B8",
            "#EB9C84",
            "#D16C6B",
            "#AC4E62",
        ],
    },
    
    "violet_citrus": {
        "source": [
            "#5B5AA6",
            "#7F7BC1",
            "#B1ABD8",
            "#E1DEEE",
            "#FCFAF5",
            "#F6D2B1",
            "#EFA06C",
            "#D96E55",
            "#B65259",
        ],
        "potential": [
            "#5B55A5",
            "#6E75BD",
            "#6E9DCA",
            "#74BFC9",
            "#99D4B6",
            "#C6DFA0",
            "#EBD878",
            "#F6C85C",
        ],
        "flux": [
            "#495DA9",
            "#4D7CC0",
            "#3FA1C6",
            "#39BFB6",
            "#65CEA0",
            "#A6D789",
            "#DBDF78",
            "#F1DB64",
        ],
        "flux_signed": [
            "#6E72B4",
            "#BEC4DD",
            "#FAF9F4",
            "#F0B98D",
            "#D77A61",
        ],
        "circulation": [
            "#7A4F96",
            "#9B78AE",
            "#C1A9C8",
            "#E4DBE5",
            "#FCF8F1",
            "#F4CFA8",
            "#E99A6B",
            "#D26655",
            "#AC4B57",
        ],
    },
    

    "mint_saffron": {
        "source": [
            "#2D6D84",
            "#5596A6",
            "#94C1C3",
            "#D7E6DD",
            "#FCFAF2",
            "#F5D8A8",
            "#EDAA64",
            "#D47A49",
            "#AD5547",
        ],
        "potential": [
            "#2C7181",
            "#4A9896",
            "#75B7A5",
            "#A9CFAC",
            "#CFE0A0",
            "#E6DE83",
            "#F2CC5E",
        ],
        "flux": [
            "#2D628E",
            "#2D879E",
            "#30A9A3",
            "#52C19B",
            "#88D091",
            "#BBD887",
            "#E3DB72",
            "#F2D05D",
        ],
        "flux_signed": [
            "#4C8398",
            "#B0CED0",
            "#FBFAF4",
            "#EFC18E",
            "#D8875B",
        ],
        "circulation": [
            "#4C61A1",
            "#7583B8",
            "#AAB0CF",
            "#DEE0EA",
            "#FCF9F1",
            "#F3D0A0",
            "#E7A05D",
            "#CF734A",
            "#A84E47",
        ],
    },

    
    "sky_peach": {
        "source": [
            "#2E71B0",
            "#5A9AC9",
            "#9CC5DD",
            "#D8E8EF",
            "#FCFBF6",
            "#F8DDBF",
            "#F3B183",
            "#E77C63",
            "#C9585A",
        ],
        "potential": [
            "#3073A5",
            "#5598BD",
            "#7FB7C6",
            "#A8D0C2",
            "#CDE0AE",
            "#E9DF91",
            "#F4D06E",
        ],
        "flux": [
            "#2E6AA7",
            "#2E8DBB",
            "#37AEC0",
            "#5AC5B0",
            "#8FD39A",
            "#C3DC88",
            "#E7E27A",
        ],
        "flux_signed": [
            "#4F8AB9",
            "#AFD0DC",
            "#FBFAF5",
            "#F1C19B",
            "#DD8267",
        ],
        "circulation": [
            "#5366A4",
            "#7F8BBD",
            "#ADB5D1",
            "#DDE0E9",
            "#FCF9F3",
            "#F7D3B1",
            "#EDA178",
            "#D86C5B",
            "#B64C55",
        ],
    },

    
    "marine_lilac": {
        "source": [
            "#2C648E",
            "#5489AD",
            "#90B5CB",
            "#D0E0E7",
            "#FCFAF5",
            "#F1D5C1",
            "#E7A58A",
            "#CF735F",
            "#AA5260",
        ],
        "potential": [
            "#355E98",
            "#557EB3",
            "#729FBE",
            "#8DBEC1",
            "#B4D2B3",
            "#D8DD99",
            "#EFD16E",
        ],
        "flux": [
            "#415A9D",
            "#4C7EB5",
            "#46A3BC",
            "#48BEAF",
            "#73CD99",
            "#ABD789",
            "#D9DF79",
        ],
        "flux_signed": [
            "#657BAC",
            "#BBCBDC",
            "#FBFAF6",
            "#EAB9A2",
            "#CC7969",
        ],
        "circulation": [
            "#66508F",
            "#8970A6",
            "#B19DBF",
            "#D8CEDC",
            "#FCF8F2",
            "#F1D4B5",
            "#E5A67B",
            "#CC735E",
            "#A74E58",
        ],
    },

    
    "coral_lagoon": {
        "source": [
            "#2E6B9A",
            "#6097BC",
            "#9ABFD1",
            "#D7E6E9",
            "#FCFAF4",
            "#F9D4B0",
            "#F1A16C",
            "#DD6E56",
            "#BB4B52",
        ],
        "potential": [
            "#247A8B",
            "#3E9E9D",
            "#67B9A8",
            "#96CEAD",
            "#C4DDA6",
            "#E1DF8E",
            "#F0D36E",
        ],
        "flux": [
            "#2B6A9B",
            "#278AA9",
            "#25A9AE",
            "#44BEA4",
            "#79CD95",
            "#AED788",
            "#DBDE7A",
        ],
        "flux_signed": [
            "#4C84A5",
            "#ACCDD2",
            "#FAF9F3",
            "#F1B98B",
            "#D87A5E",
        ],
        "circulation": [
            "#5A55A0",
            "#807AB6",
            "#AAA6CB",
            "#D9D7E4",
            "#FCF9F3",
            "#F6D0A6",
            "#EB9A67",
            "#D66B50",
            "#B04C49",
        ],
    },

    
    "journal_bright": {
        "source": [
            "#245C9F",
            "#4F8FC7",
            "#93C2DE",
            "#D5E7EF",
            "#FBFAF5",
            "#F7D1B7",
            "#F3A077",
            "#E66E5B",
            "#C54A59",
        ],
        "potential": [
            "#245B8F",
            "#347DB3",
            "#56A3C1",
            "#86C5C8",
            "#BFDCC0",
            "#E5E6A3",
            "#F3D56C",
            "#F6C64F",
        ],
        "flux": [
            "#285B91",
            "#2E78B4",
            "#269BC3",
            "#35B8BE",
            "#63CDA6",
            "#A7DB7D",
            "#DCE577",
            "#F4E56B",
        ],
        "flux_signed": [
            "#3C78AE",
            "#91BDD2",
            "#F7F8F4",
            "#EAB08B",
            "#D87361",
        ],
        "circulation": [
            "#24647A",
            "#438A9A",
            "#83B8B7",
            "#C6DBD0",
            "#FBF8EF",
            "#F5D0A5",
            "#EEA16A",
            "#DF754B",
            "#B95745",
        ],
    },
    "luminous": {
        "source": ["#12305F", "#287CA7", "#C8E8EE", "#FFF9EC", "#F5A27A", "#C63F4E"],
        "potential": ["#15583F", "#219B64", "#73C96B", "#C8DD61", "#F5D657"],
        "flux": ["#45206A", "#794C9A", "#B889BC", "#E8B5B0", "#F5C46B"],
        "flux_signed": ["#593179", "#FBF5EE", "#D76D5D"],
        "circulation": ["#503B2C", "#9D7C5C", "#F7F1DF", "#B8C56C", "#597C47"],
    },
    "coastal": {
        "source": ["#5A244F", "#A44D73", "#E7B2B0", "#FFF4DE", "#E6A85A", "#8E4B25"],
        "potential": ["#102D6B", "#1F5FAE", "#268DCB", "#64BCE0", "#C9EAF2"],
        "flux": ["#6B175F", "#B53E87", "#DF7197", "#F0AD8A", "#F6DB91"],
        "flux_signed": ["#7B266E", "#FFF7ED", "#C96952"],
        "circulation": ["#075B68", "#148D8C", "#A8D8C7", "#F4F3D8", "#D2C954", "#6D8D45"],
    },
    "aurora": {
        "source": ["#394150", "#71859A", "#E7ECE4", "#F8E5C4", "#D9825C", "#A94346"],
        "potential": ["#412365", "#6A52AB", "#8988CD", "#81BBD5", "#D9EDF1"],
        "flux": ["#1D5A3D", "#368B50", "#83BD59", "#D0D566", "#F1D86B"],
        "flux_signed": ["#2D704B", "#F7F7E9", "#B89B43"],
        "circulation": ["#064B55", "#197A78", "#77B5A0", "#F4E8C6", "#D98946", "#A95232"],
    },
}


PALETTE_SHOWCASE = (
    "editorial_bluegold",
    "ocean_sunrise",
    "pearl_rose",
    "violet_citrus",
    "mint_saffron",
    "sky_peach",
    "marine_lilac",
    "coral_lagoon",
)

PALETTE_DISPLAY_NAMES = {
    "editorial_bluegold": "A  Editorial Blue–Gold",
    "ocean_sunrise": "B  Ocean Sunrise",
    "pearl_rose": "C  Pearl Rose",
    "violet_citrus": "D  Violet Citrus",
    "mint_saffron": "E  Mint Saffron",
    "sky_peach": "F  Sky Peach",
    "marine_lilac": "G  Marine Lilac",
    "coral_lagoon": "H  Coral Lagoon",
}


def _resolved_palette_name(name: str) -> str:
    
    if name == "reference":
        return "editorial_bluegold"
    return name

def _palette_cmap(name: str, field: str) -> LinearSegmentedColormap:
    
    resolved = _resolved_palette_name(name)
    if resolved == "legacy_reference":
        raise ValueError("legacy_reference uses Matplotlib built-in colormaps")
    colors = PALETTE_LIBRARY[resolved][field]
    return LinearSegmentedColormap.from_list(
        f"{field}_{resolved}",
        colors,
        N=256,
    )

def apply_palette(name: str) -> None:
    
    global SOURCE_CMAP, POTENTIAL_CMAP, FLUX_CMAP
    global FLUX_SIGNED_CMAP, CIRCULATION_CMAP

    if name == "legacy_reference":
        SOURCE_CMAP = plt.get_cmap("coolwarm")
        POTENTIAL_CMAP = REFERENCE_POTENTIAL_CMAP
        FLUX_CMAP = plt.get_cmap("YlGnBu_r")
        FLUX_SIGNED_CMAP = FLUX_CMAP
        CIRCULATION_CMAP = plt.get_cmap("coolwarm")
        return

    resolved = _resolved_palette_name(name)
    SOURCE_CMAP = _palette_cmap(resolved, "source")
    POTENTIAL_CMAP = _palette_cmap(resolved, "potential")
    FLUX_CMAP = _palette_cmap(resolved, "flux")
    FLUX_SIGNED_CMAP = _palette_cmap(resolved, "flux_signed")
    CIRCULATION_CMAP = _palette_cmap(resolved, "circulation")


def save_palette_preview(output: Path) -> Path:
    
    output.mkdir(parents=True, exist_ok=True)

    gradient = np.linspace(0.0, 1.0, 768, dtype=np.float64)[None, :]
    fields = (
        ("source", "(a) Source / signed"),
        ("potential", "(b) Potential"),
        ("flux", "(c) Flux magnitude"),
        ("circulation", "(d) Circulation / signed"),
    )

    fig, axes = plt.subplots(
        len(PALETTE_SHOWCASE),
        len(fields),
        figsize=(13.6, 11.2),
        squeeze=False,
    )
    fig.patch.set_facecolor("#FFFFFF")

    for row, palette_name in enumerate(PALETTE_SHOWCASE):
        for col, (field, title) in enumerate(fields):
            ax = axes[row, col]
            cmap = _palette_cmap(palette_name, field)
            ax.imshow(
                gradient,
                aspect="auto",
                interpolation="bilinear",
                cmap=cmap,
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_edgecolor("#A7B3BF")

            if row == 0:
                ax.set_title(
                    title,
                    fontsize=11.0,
                    fontweight="bold",
                    color=PAPER_INK,
                    pad=8.0,
                )

            if col == 0:
                ax.text(
                    -0.055,
                    0.5,
                    PALETTE_DISPLAY_NAMES[palette_name],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10.3,
                    fontweight="semibold",
                    color=PAPER_INK,
                    clip_on=False,
                )

    fig.suptitle(
        "Publication Palette Showcase — Left-side Physical Fields",
        y=0.986,
        fontsize=15.0,
        fontweight="bold",
        color=PAPER_INK,
    )
    fig.subplots_adjust(
        left=0.205,
        right=0.988,
        top=0.935,
        bottom=0.038,
        wspace=0.085,
        hspace=0.72,
    )

    preview_path = output / "Palette_Showcase_Left_Fields.png"
    fig.savefig(
        preview_path,
        dpi=300,
        facecolor="#FFFFFF",
        bbox_inches=None,
        pad_inches=0.0,
    )
    fig.savefig(
        preview_path.with_suffix(".pdf"),
        dpi=300,
        facecolor="#FFFFFF",
        bbox_inches=None,
        pad_inches=0.0,
    )
    plt.close(fig)
    return preview_path

@dataclass
class Run:
    

    name: str
    family: str
    task: str
    config: str
    parameters: int
    run: Path
    prediction: np.ndarray
    target: np.ndarray
    result: dict

def style() -> None:
    
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.0,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#8FA2B5",
            "axes.labelcolor": PAPER_INK,
            "xtick.color": PAPER_INK,
            "ytick.color": PAPER_INK,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def read_run(
    root: Path,
    task: str,
    config: str,
    name: str | None = None,
    label: str | None = None,
    structure: str = "full",
) -> Run:

    run = root / f"form_{task}" / config

    if name is None:
        run = run / structure

    if name is not None:
        run = run / name
 
    run = run / "seed_42"

    result_path = run / "result.json"
    checkpoint_path = run / "best.pt"

    if not result_path.exists():
        raise FileNotFoundError(f"missing result.json: {result_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing validation-best checkpoint: {checkpoint_path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    scale = float(result["normalization"]["y_scale"])

    prediction = (
        np.asarray(
            np.load(run / "prediction_test_normalized.npy"),
            dtype=np.float64,
        )
        * scale
    )

    target = (
        np.asarray(
            np.load(run / "target_test_normalized.npy"),
            dtype=np.float64,
        )
        * scale
    )

    return Run(
        name=label or name or root.name,
        family=result["family"],
        task=task,
        config=config,
        parameters=int(result["parameters"]),
        run=run,
        prediction=prediction,
        target=target,
        result=result,
    )


def verify_common_test_order(runs: list[Run]) -> None:

    if not runs:
        raise ValueError("no runs supplied for visualisation")
    expected = np.asarray(np.load(CACHE / "split_test.npy", mmap_mode="r"))
    task = runs[0].task
    if any(item.task != task for item in runs):
        raise ValueError("a visualisation panel must contain one cochain rank")
    archive_target = np.asarray(
        np.load(CACHE / f"{TASK_TARGET[task]}.npy", mmap_mode="r")[expected],
        dtype=np.float64,
    )

    for item in runs:
        index = np.asarray(np.load(item.run / "test_indices.npy"))

        if not np.array_equal(index, expected):
            raise ValueError(f"test-set ordering mismatch: {item.run}")

        if item.prediction.shape != item.target.shape:
            raise ValueError(f"prediction/target shape mismatch: {item.run}")

        if len(index) != len(item.prediction):
            raise ValueError(f"test index/prediction count mismatch: {item.run}")

        if item.target.shape != archive_target.shape or not np.allclose(
            item.target, archive_target, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(f"saved target does not match the test-set target: {item.run}")


def load_primary(task: str) -> list[Run]:

    runs = [
        read_run(
            root,
            task,
            TDK_CONFIG,
            label=label,
            structure=TDK_STRUCTURE,
        )
        for label, root in TDK_ROOTS.items()
    ]

    
    runs += [read_run(BASE, task, BASELINE_MAIN_CONFIG, name=name) for name in BASELINE_NAMES]

    verify_common_test_order(runs)
    return runs

def per_sample_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    

    return np.linalg.norm(pred - target, axis=1) / np.maximum(
        np.linalg.norm(target, axis=1),
        1e-12,
    )

def _best_dkho_samples(
    selected: dict[str, Run],
    requested_index: int | None,
) -> dict[str, int]:
    
    n_test = len(next(iter(selected.values())).target)

    if requested_index is not None:
        if not 0 <= requested_index < n_test:
            raise ValueError(f"--darcy-figure-index must be in [0, {n_test - 1}]")
        return {
            "0": requested_index,
            "1": requested_index,
            "2": requested_index,
        }

    samples: dict[str, int] = {}
    for task in ("0", "1", "2"):
        run = selected[task]
        scores = per_sample_l2(run.prediction, run.target)
        samples[task] = int(np.argmin(scores))

    return samples

def _dkho_sample_audit(
    selected: dict[str, Run],
    samples: dict[str, int],
) -> dict:
    
    audit = {}

    for task in ("0", "1", "2"):
        run = selected[task]
        sample = int(samples[task])
        scores = per_sample_l2(run.prediction, run.target)

        order = np.argsort(scores, kind="stable")
        rank = int(np.flatnonzero(order == sample)[0]) + 1
        argmin = int(np.argmin(scores))

        audit[task] = {
            "test_row": sample,
            "relative_l2": float(scores[sample]),
            "rank_within_dkho_test_samples": rank,
            "argmin_test_row": argmin,
            "n_test": int(len(scores)),
        }

    return audit


def _model_ranking(runs: list[Run]) -> list[tuple[Run, float]]:
    
    ranking = [
        (
            run,
            float(np.mean(per_sample_l2(run.prediction, run.target))),
        )
        for run in runs
    ]
    ranking.sort(key=lambda item: item[1])
    return ranking


def _top_four(runs: list[Run]) -> list[Run]:
    
    return [run for run, _score in _model_ranking(runs)[:4]]


def short_name(name: str) -> str:
    
    return name

def scalar_panel(
    ax: plt.Axes,
    tri: Triangulation,
    value: np.ndarray,
    cmap: str,
    norm=None,
):
    

    image = ax.tripcolor(
        tri,
        value,
        shading="gouraud",
        cmap=cmap,
        norm=norm,
    )
    ax.set(aspect="equal")
    ax.set_axis_off()
    return image

def face_panel(
    ax: plt.Axes,
    tri: Triangulation,
    value: np.ndarray,
    cmap: str,
    norm=None,
):
    

    image = ax.tripcolor(
        tri,
        facecolors=value,
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    ax.set(aspect="equal")
    ax.set_axis_off()
    return image


def edge_panel(
    ax: plt.Axes,
    geo: DarcyGeometry,
    value: np.ndarray,
    cmap: str,
    norm=None,
) -> LineCollection:
    

    coll = LineCollection(
        geo.points[geo.edges],
        cmap=cmap,
        norm=norm,
        linewidths=0.52,
        antialiased=True,
    )
    coll.set_array(value)
    ax.add_collection(coll)

    ax.set(
        xlim=(geo.points[:, 0].min(), geo.points[:, 0].max()),
        ylim=(geo.points[:, 1].min(), geo.points[:, 1].max()),
        aspect="equal",
    )
    ax.set_axis_off()
    return coll


def _fix_domain_view(ax: plt.Axes, geo: DarcyGeometry) -> None:
    
    xmin, xmax = float(geo.points[:, 0].min()), float(geo.points[:, 0].max())
    ymin, ymax = float(geo.points[:, 1].min()), float(geo.points[:, 1].max())
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_axis_off()


def _draw_domain_frame(ax: plt.Axes, geo: DarcyGeometry) -> None:
    
    xmin, xmax = float(geo.points[:, 0].min()), float(geo.points[:, 0].max())
    ymin, ymax = float(geo.points[:, 1].min()), float(geo.points[:, 1].max())
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            lw=0.42,
            ec="#B9C4CE",
            alpha=0.78,
            zorder=30,
        )
    )


def _draw_c0_topology(
    ax: plt.Axes,
    geo: DarcyGeometry,
    tri: Triangulation,
    value: np.ndarray,
    norm: Normalize,
) -> None:
    
    ax.triplot(
        tri,
        color="#B7C9D8",
        linewidth=0.105,
        alpha=0.24,
        zorder=4,
    )

    
    boundary = np.flatnonzero(geo.node_component != 0)
    if len(boundary):
        stride = max(1, len(boundary) // 180)
        pick = boundary[::stride]
        ax.scatter(
            geo.points[pick, 0],
            geo.points[pick, 1],
            s=2.2,
            facecolors="#FFFFFF",
            edgecolors="#4D647A",
            linewidths=0.24,
            alpha=0.72,
            zorder=6,
            rasterized=True,
        )


def _draw_oriented_edge_cochain(
    ax: plt.Axes,
    geo: DarcyGeometry,
    value: np.ndarray,
    norm: Normalize,
) -> LineCollection:
    
    abs_value = np.abs(value)
    q98 = max(float(np.quantile(abs_value, 0.98)), 1e-12)
    strength = np.clip(abs_value / q98, 0.0, 1.0)

    rgba = FLUX_SIGNED_CMAP(norm(value))
    rgba[:, 3] = 0.025 + 0.075 * strength

    coll = LineCollection(
        geo.points[geo.edges],
        colors=rgba,
        linewidths=0.045 + 0.090 * strength,
        antialiased=True,
        zorder=3,
    )
    coll.set_rasterized(True)
    ax.add_collection(coll)
    return coll


def _column_heading(ax: plt.Axes, letter: str, title: str) -> None:
    
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=PAPER_INK,
    )

def cochain_to_node_vector(
    geo: DarcyGeometry,
    value: np.ndarray,
) -> np.ndarray:
    
    tangent_component = value[:, None] * geo.edge_dir / np.maximum(geo.edge_length, 1e-12)

    vector = np.zeros((geo.n0, 2), dtype=np.float64)
    degree = np.zeros((geo.n0, 1), dtype=np.float64)

    np.add.at(vector, geo.tail, tangent_component)
    np.add.at(vector, geo.head, tangent_component)

    np.add.at(degree, geo.tail, 1.0)
    np.add.at(degree, geo.head, 1.0)

    return vector / np.maximum(degree, 1.0)

def _streamlines(
    ax: plt.Axes,
    geo: DarcyGeometry,
    tri: Triangulation,
    edge_cochain: np.ndarray,
):
    
    vector = cochain_to_node_vector(geo, edge_cochain)

    x = np.linspace(geo.points[:, 0].min(), geo.points[:, 0].max(), 220)
    y = np.linspace(geo.points[:, 1].min(), geo.points[:, 1].max(), 220)
    grid_x, grid_y = np.meshgrid(x, y)

    u = np.ma.masked_invalid(LinearTriInterpolator(tri, vector[:, 0])(grid_x, grid_y))
    v = np.ma.masked_invalid(LinearTriInterpolator(tri, vector[:, 1])(grid_x, grid_y))
    if u.count() == 0 or v.count() == 0:
        return None, None

    speed = np.ma.sqrt(u * u + v * v)
    valid = np.asarray(speed.compressed(), dtype=np.float64)

    hi = max(float(np.quantile(valid, 0.995)), 1e-12)
    lo = max(hi * 1e-2, float(np.quantile(valid, 0.02)), 1e-12)
    if lo >= hi:
        lo = max(hi * 1e-2, 1e-12)

    mag_norm = LogNorm(vmin=lo, vmax=hi)

    
    width = 0.34 + 0.82 * np.clip(speed / hi, 0.0, 1.0)

    stream = ax.streamplot(
        x,
        y,
        u,
        v,
        density=1.06,
        color=speed,
        cmap=FLUX_CMAP,
        norm=mag_norm,
        linewidth=width,
        arrowsize=0.58,
        arrowstyle="-|>",
        minlength=0.10,
        maxlength=2.7,
        integration_direction="both",
        zorder=8,
    )
    return stream, mag_norm

def _robust_norm(
    values: np.ndarray,
    *,
    symmetric: bool = False,
) -> Normalize:

    values = np.asarray(values, dtype=np.float64).ravel()

    if symmetric:
        bound = float(np.quantile(np.abs(values), 0.995))
        bound = max(bound, 1e-12)
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    lo, hi = np.quantile(values, (0.005, 0.995))

    if hi - lo < 1e-12:
        hi = lo + 1e-12

    return Normalize(vmin=float(lo), vmax=float(hi))


def _relative_error_field(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    scale = max(
        float(np.quantile(np.abs(target), 0.95)),
        1e-12,
    )
    return np.abs(prediction - target) / scale

def _error_norm(values: np.ndarray) -> Normalize:
    
    values = np.asarray(values, dtype=np.float64).ravel()
    positive = values[np.isfinite(values) & (values > 0)]

    if len(positive) == 0:
        return LogNorm(vmin=1e-6, vmax=1e-2)

    q_low = max(float(np.quantile(positive, 0.01)), 1e-12)
    q_high = max(float(np.quantile(positive, 0.995)), q_low * 10.0)

    
    high_decade = 10.0 ** math.ceil(math.log10(q_high))
    low_decade = 10.0 ** math.floor(math.log10(q_low))

    
    low_decade = min(low_decade, high_decade * 1e-4)
    low_decade = max(low_decade, 1e-12)

    return LogNorm(vmin=low_decade, vmax=high_decade)


def _panel_heading(
    ax: plt.Axes,
    letter: str,
    title: str,
) -> None:
    

    ax.text(
        0.0,
        1.075,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.6,
        fontweight="semibold",
        color=PAPER_INK,
    )

def _hole_descriptors(
    geo: DarcyGeometry,
) -> list[tuple[np.ndarray, float]]:
    

    descriptors: list[tuple[np.ndarray, float]] = []

    for component in np.unique(geo.node_component):
        if component == 0:
            continue

        nodes = geo.points[geo.node_component == component]

        if len(nodes) < 3:
            continue

        center = nodes.mean(axis=0)
        radius = float(np.median(np.linalg.norm(nodes - center, axis=1)))
        descriptors.append((center, radius))

    
    if len(descriptors) <= 2:
        return sorted(
            descriptors,
            key=lambda item: (item[0][1], item[0][0]),
        )

    
    outer = int(np.argmax([radius for _, radius in descriptors]))

    holes = [item for index, item in enumerate(descriptors) if index != outer]

    
    return sorted(
        holes,
        key=lambda item: (-item[0][1], item[0][0]),
    )


def _draw_boundary_loops(
    ax: plt.Axes,
    holes: list[tuple[np.ndarray, float]],
) -> None:
    
    for index, (center, radius) in enumerate(holes[:2], start=1):
        loop_radius = 1.26 * radius

        
        arc = Arc(
            center,
            2 * loop_radius,
            2 * loop_radius,
            theta1=28,
            theta2=326,
            lw=2.0,
            color=PAPER_LOOP,
            zorder=9,
        )
        ax.add_patch(arc)

        
        theta0 = math.radians(306)
        theta1 = math.radians(326)

        start = center + loop_radius * np.array([math.cos(theta0), math.sin(theta0)])
        end = center + loop_radius * np.array([math.cos(theta1), math.sin(theta1)])

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.75,
                "color": PAPER_LOOP,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=10,
        )

        
        offset = np.array([0.30 * loop_radius, 0.82 * loop_radius])
        ax.text(
            *(center + offset),
            rf"$\gamma_{index}$",
            color=PAPER_LOOP,
            fontsize=11.0,
            fontweight="bold",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=.14",
                "fc": "#FFF8EB",
                "ec": "none",
                "alpha": 0.92,
            },
        )

    
    if holes:
        x0 = min(center[0] - 1.45 * radius for center, radius in holes)
        y0 = min(center[1] - 1.65 * radius for center, radius in holes)

def _comparison_models(runs: list[Run]) -> list[Run]:
    
    return [run for run, _score in _model_ranking(runs)[:4]]


def _draw_error_row(
    fig: plt.Figure,
    *,
    task: str,
    models: list[Run],
    sample: int,
    geo: DarcyGeometry,
    tri: Triangulation,
    x0: float,
    y0: float,
    tile_w: float,
    gap_x: float,
    image_h: float,
    cbar_y: float,
    row_label: str,
) -> None:
    
    if len(models) != 4:
        raise ValueError(f"form {task} error row must contain four models, found {len(models)}")

    errors = [_relative_error_field(run.prediction[sample], run.target[sample]) for run in models]
    norm = _error_norm(np.concatenate(errors))

    artist = None
    for col, (run, error) in enumerate(zip(models, errors)):
        x = x0 + col * (tile_w + gap_x)
        axis = fig.add_axes([x, y0, tile_w, image_h])
        axis.set_facecolor("#05051A")

        if task == "0":
            artist = scalar_panel(axis, tri, error, ERROR_CMAP, norm)
        elif task == "1":
            mesh = LineCollection(
                geo.points[geo.edges],
                colors="#17152C",
                linewidths=0.050,
                alpha=0.20,
                zorder=1,
            )
            axis.add_collection(mesh)
            artist = edge_panel(axis, geo, error, ERROR_CMAP, norm)
            artist.set_linewidth(0.68)
            artist.set_alpha(0.99)
        else:
            artist = face_panel(axis, tri, error, ERROR_CMAP, norm)

        artist.set_rasterized(True)
        _fix_domain_view(axis, geo)

        fig.text(
            x + 0.5 * tile_w,
            y0 + image_h + 0.005,
            short_name(run.name),
            ha="center",
            va="bottom",
            fontsize=9.4,
            fontweight="bold",
            color=PAPER_INK,
        )

    
    row_w = 4 * tile_w + 3 * gap_x
    cax = fig.add_axes([x0, cbar_y, row_w, 0.012])
    if artist is not None:
        cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")
        lo_exp = int(round(math.log10(norm.vmin)))
        hi_exp = int(round(math.log10(norm.vmax)))
        mid_exp = int(round(0.5 * (lo_exp + hi_exp)))
        exponents = sorted(set((lo_exp, mid_exp, hi_exp)))
        ticks = [10.0**exponent for exponent in exponents]
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([rf"$10^{{{e}}}$" for e in exponents])
        cbar.ax.tick_params(
            labelsize=8.4,
            pad=1.5,
            length=2.4,
            width=0.55,
        )
        cbar.outline.set_linewidth(0.62)
        cbar.outline.set_edgecolor("#52677C")

    
    fig.text(
        x0 - 0.013,
        y0 + 0.5 * image_h,
        row_label,
        ha="right",
        va="center",
        rotation=90,
        fontsize=9.8,
        fontweight="semibold",
        color=PAPER_INK,
    )


def _format_scientific_tick(value: float) -> str:
    
    if not np.isfinite(value) or value <= 0:
        return "0"

    exponent = int(math.floor(math.log10(value)))
    mantissa = value / (10.0**exponent)

    if abs(mantissa - 1.0) < 0.05:
        return rf"$10^{{{exponent}}}$"

    if -2 <= exponent <= 1:
        return f"{value:.2g}"

    return rf"${mantissa:.1f}\times10^{{{exponent}}}$"


def _main_colorbar(
    fig: plt.Figure,
    artist,
    rect: tuple[float, float, float, float],
    label: str,
    ticks=None,
    ticklabels=None,
    label_y: float | None = None,
) -> None:
    
    left, bottom, width, height = rect
    cax = fig.add_axes((left, bottom, width, height))
    cbar = fig.colorbar(artist, cax=cax, orientation="horizontal")

    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

    cbar.ax.tick_params(
        labelsize=8.1,
        pad=1.7,
        length=2.5,
        width=0.58,
    )
    cbar.outline.set_linewidth(0.64)
    cbar.outline.set_edgecolor("#52677C")

    
    
    
    if label_y is None:
        label_y = bottom + height + 0.014
    fig.text(
        left + 0.5 * width,
        label_y,
        label,
        ha="center",
        va="center",
        fontsize=8.6,
        color=PAPER_INK,
    )

def render_darcy_cochain_comparison(
    runs_by_task: dict[str, list[Run]],
    geo: DarcyGeometry,
    output: Path,
    requested_index: int | None = None,
    palette_name: str = "luminous",
) -> None:
    
    required = {"0", "1", "2"}
    if set(runs_by_task) != required:
        raise ValueError("runs_by_task must contain exactly the ranks '0', '1', and '2'")

    selected = {
        task: next(run for run in runs if run.name == PRIMARY_DKHO)
        for task, runs in runs_by_task.items()
    }

    test_ids = np.asarray(np.load(CACHE / "split_test.npy", mmap_mode="r"))
    for run in selected.values():
        saved_ids = np.asarray(np.load(run.run / "test_indices.npy"))
        if not np.array_equal(saved_ids, test_ids):
            raise ValueError(f"C0/C1/C2 test-set ordering mismatch: {run.run}")

    
    samples = _best_dkho_samples(selected, requested_index)
    archive_ids = {task: int(test_ids[samples[task]]) for task in ("0", "1", "2")}

    
    
    archive_id_a = archive_ids["0"]
    source = np.asarray(np.load(CACHE / "f_samples.npy", mmap_mode="r"))[archive_id_a]
    g1 = float(np.asarray(np.load(CACHE / "g1.npy", mmap_mode="r"))[archive_id_a])
    g2 = float(np.asarray(np.load(CACHE / "g2.npy", mmap_mode="r"))[archive_id_a])

    tri = Triangulation(
        geo.points[:, 0],
        geo.points[:, 1],
        geo.faces,
    )

    sample0 = samples["0"]
    sample1 = samples["1"]
    sample2 = samples["2"]

    p_hat = selected["0"].prediction[sample0]
    p_true = selected["0"].target[sample0]

    q_hat = selected["1"].prediction[sample1]

    omega_hat = selected["2"].prediction[sample2]
    omega_true = selected["2"].target[sample2]

    source_norm = _robust_norm(source, symmetric=True)
    pressure_norm = _robust_norm(np.concatenate((p_hat, p_true)))
    circulation_norm = _robust_norm(
        np.concatenate((omega_hat, omega_true)),
        symmetric=True,
    )
    q_bound = max(float(np.quantile(np.abs(q_hat), 0.995)), 1e-12)
    q_norm = TwoSlopeNorm(vmin=-q_bound, vcenter=0.0, vmax=q_bound)

    holes = _hole_descriptors(geo)

    
    rankings = {task: _model_ranking(runs) for task, runs in runs_by_task.items()}
    error_models = {task: [run for run, _score in rankings[task][:4]] for task in ("0", "1", "2")}

    
    sample_audit = _dkho_sample_audit(selected, samples)

    if requested_index is None:
        for task in ("0", "1", "2"):
            if sample_audit[task]["rank_within_dkho_test_samples"] != 1:
                raise RuntimeError(
                    f"form {task} automatic sample selection failed: "
                    "the selected sample is not the minimum-relative-L2 TDK-HO sample"
                )

    for task in ("0", "1", "2"):
        info = sample_audit[task]
        print(
            f"[visualize] form{task} DKHO-large best sample: "
            f"test_row={info['test_row']}, "
            f"archive_id={archive_ids[task]}, "
            f"relative_l2={info['relative_l2']:.6e}, "
            f"rank={info['rank_within_dkho_test_samples']}",
            flush=True,
        )
    for task in ("0", "1", "2"):
        top4_text = ", ".join(f"{run.name} ({score:.6e})" for run, score in rankings[task][:4])
        print(
            f"[visualize] form{task} Top-4 by full-test mean relative-L2: " f"{top4_text}",
            flush=True,
        )
    
    FIG_W, FIG_H = 20.0, 8.0
    FIG_ASPECT = FIG_W / FIG_H
    fig = plt.figure(
        figsize=(FIG_W, FIG_H),
        facecolor="#FFFFFF",
    )

    GROUP_TITLE_Y = 0.944

    CONTENT_TOP = 0.878

    BOTTOM_CBAR_Y = 0.025

    MAIN_W = 0.134
    MAIN_H = MAIN_W * FIG_ASPECT

    
    X_LEFT = 0.095
    X_RIGHT = 0.249
    Y_TOP = CONTENT_TOP - MAIN_H
    Y_BOTTOM = 0.082

    A_MAIN = (X_LEFT, Y_TOP, MAIN_W, MAIN_H)
    B_MAIN = (X_RIGHT, Y_TOP, MAIN_W, MAIN_H)
    C_MAIN = (X_LEFT, Y_BOTTOM, MAIN_W, MAIN_H)
    D_MAIN = (X_RIGHT, Y_BOTTOM, MAIN_W, MAIN_H)
    
    MAIN_CBAR_W = 0.121
    MAIN_CBAR_H = 0.0095
    CBAR_DX = 0.5 * (MAIN_W - MAIN_CBAR_W)
    TOP_CBAR_Y = 0.492

    MAIN_CBAR_LABEL_DY = 0.019
    A_CBAR = (X_LEFT + CBAR_DX, TOP_CBAR_Y, MAIN_CBAR_W, MAIN_CBAR_H)
    B_CBAR = (X_RIGHT + CBAR_DX, TOP_CBAR_Y, MAIN_CBAR_W, MAIN_CBAR_H)
    C_CBAR = (X_LEFT + CBAR_DX, BOTTOM_CBAR_Y, MAIN_CBAR_W, MAIN_CBAR_H)
    D_CBAR = (X_RIGHT + CBAR_DX, BOTTOM_CBAR_Y, MAIN_CBAR_W, MAIN_CBAR_H)

    geometry_ax = fig.add_axes(A_MAIN)
    potential_ax = fig.add_axes(B_MAIN)
    flux_ax = fig.add_axes(C_MAIN)
    circulation_ax = fig.add_axes(D_MAIN)

    left_group_center = 0.5 * (X_LEFT + X_RIGHT + MAIN_W)
    fig.text(
        left_group_center,
        GROUP_TITLE_Y,
        "Geometric Domain & Physical Fields",
        ha="center",
        va="bottom",
        fontsize=15.4,
        fontweight="bold",
        color=PAPER_INK,
    )

    PANEL_TITLE_DY = 0.008
    title_specs = [
        (X_LEFT + 0.5 * MAIN_W, Y_TOP + MAIN_H + PANEL_TITLE_DY, r"(a) Domain and Conditions"),
        (
            X_RIGHT + 0.5 * MAIN_W,
            Y_TOP + MAIN_H + PANEL_TITLE_DY,
            r"(b) $\mathbf{C}^{0}$ Potential",
        ),
        (
            X_LEFT + 0.5 * MAIN_W,
            Y_BOTTOM + MAIN_H + PANEL_TITLE_DY,
            r"(c) $\mathbf{C}^{1}$ Oriented Flux",
        ),
        (
            X_RIGHT + 0.5 * MAIN_W,
            Y_BOTTOM + MAIN_H + PANEL_TITLE_DY,
            r"(d) $\mathbf{C}^{2}$ Face Circulation",
        ),
    ]

    lower_title_y = Y_BOTTOM + MAIN_H + PANEL_TITLE_DY
    if TOP_CBAR_Y - lower_title_y < 0.050:
        raise RuntimeError("vertical layout is too dense: insufficient clearance between the upper colorbar and lower title")
    for x, y, title in title_specs:
        fig.text(
            x,
            y,
            title,
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color=PAPER_INK,
        )
    
    ERROR_X0 = 0.442
    ERROR_TILE_W = 0.090
    ERROR_GAP_X = 0.015
    ERROR_H = ERROR_TILE_W * FIG_ASPECT
    ERROR_ROW_Y = {
        "0": CONTENT_TOP - ERROR_H,
        "1": 0.360,
        "2": 0.067,
    }
    ERROR_CBAR_Y = {
        "0": 0.615,
        "1": 0.322,
        "2": BOTTOM_CBAR_Y,
    }

    error_group_center = ERROR_X0 + 0.5 * (4 * ERROR_TILE_W + 3 * ERROR_GAP_X)
    fig.text(
        error_group_center,
        GROUP_TITLE_Y,
        r"Top-4 Relative Error Fields  $|\hat y-y|/Q_{95}(|y|)$",
        ha="center",
        va="bottom",
        fontsize=15.4,
        fontweight="bold",
        color=PAPER_INK,
    )

    source_artist = scalar_panel(
        geometry_ax,
        tri,
        source,
        SOURCE_CMAP,
        source_norm,
    )
    source_artist.set_rasterized(True)

    geometry_ax.triplot(
        tri,
        color="#8FB2CC",
        linewidth=0.13,
        alpha=0.32,
        zorder=4,
    )

    for center, radius in holes:
        geometry_ax.add_patch(
            plt.Circle(
                center,
                1.012 * radius,
                fill=False,
                lw=1.15,
                ec="#FFFFFF",
                alpha=0.98,
                zorder=6,
            )
        )

    for index, ((center, _radius), boundary_value) in enumerate(
        zip(holes[:2], (g1, g2)),
        start=1,
    ):
        geometry_ax.text(
            center[0],
            center[1],
            rf"$\Gamma_{index}$" + "\n" + rf"$g_{index}={boundary_value:.2f}$",
            ha="center",
            va="center",
            fontsize=9.8,
            color=PAPER_INK,
            bbox={
                "boxstyle": "round,pad=.23",
                "fc": "#FFFFFF",
                "ec": "#D5DFE8",
                "lw": 0.58,
                "alpha": 0.97,
            },
            zorder=10,
        )

    _fix_domain_view(geometry_ax, geo)

    sb = float(max(abs(source_norm.vmin), abs(source_norm.vmax)))
    _main_colorbar(
        fig,
        source_artist,
        A_CBAR,
        r"Source field  $f(x)$",
        ticks=[-sb, 0.0, sb],
        label_y=TOP_CBAR_Y + MAIN_CBAR_H + MAIN_CBAR_LABEL_DY,
    )

    pressure_artist = scalar_panel(
        potential_ax,
        tri,
        p_hat,
        POTENTIAL_CMAP,
        pressure_norm,
    )
    pressure_artist.set_rasterized(True)

    _draw_c0_topology(
        potential_ax,
        geo,
        tri,
        p_hat,
        pressure_norm,
    )

    levels = np.linspace(
        pressure_norm.vmin,
        pressure_norm.vmax,
        10,
    )[1:-1]

    potential_ax.tricontour(
        tri,
        p_true,
        levels=levels,
        colors="#FFFFFF",
        linewidths=0.82,
        alpha=0.96,
        zorder=7,
    )

    for center, radius in holes:
        potential_ax.add_patch(
            plt.Circle(
                center,
                1.01 * radius,
                fill=False,
                lw=1.12,
                ec="#16263E",
                alpha=0.95,
                zorder=10,
            )
        )

    _fix_domain_view(potential_ax, geo)

    p_ticks = np.linspace(
        pressure_norm.vmin,
        pressure_norm.vmax,
        3,
    )
    _main_colorbar(
        fig,
        pressure_artist,
        B_CBAR,
        r"Potential  $u(x)$",
        ticks=p_ticks,
        label_y=TOP_CBAR_Y + MAIN_CBAR_H + MAIN_CBAR_LABEL_DY,
    )


    vector = cochain_to_node_vector(geo, q_hat)
    magnitude = np.linalg.norm(vector, axis=1)

    
    mag_hi = max(float(np.quantile(magnitude, 0.995)), 1e-12)
    bg_norm = Normalize(vmin=0.0, vmax=mag_hi)

    flux_background = scalar_panel(
        flux_ax,
        tri,
        magnitude,
        FLUX_CMAP,
        bg_norm,
    )
    
    flux_background.set_alpha(0.92)
    flux_background.set_rasterized(True)

    
    flux_ax.triplot(
        tri,
        color="#B8D0DD",
        linewidth=0.070,
        alpha=0.16,
        zorder=2,
    )

    
    _draw_oriented_edge_cochain(
        flux_ax,
        geo,
        q_hat,
        q_norm,
    )
    _stream, stream_norm = _streamlines(
        flux_ax,
        geo,
        tri,
        q_hat,
    )

    for center, radius in holes:
        flux_ax.add_patch(
            plt.Circle(
                center,
                1.01 * radius,
                fill=False,
                lw=1.52,
                ec="#E1A43A",
                alpha=0.95,
                zorder=13,
            )
        )

    _fix_domain_view(flux_ax, geo)

    from matplotlib.cm import ScalarMappable

    if stream_norm is None:
        flux_map = ScalarMappable(norm=bg_norm, cmap=FLUX_CMAP)
        flux_ticks = None
    else:
        flux_map = ScalarMappable(norm=stream_norm, cmap=FLUX_CMAP)
        
        
        
        flux_ticks = np.geomspace(
            stream_norm.vmin,
            stream_norm.vmax,
            3,
        ).tolist()
    flux_map.set_array([])

    flux_ticklabels = (
        [_format_scientific_tick(v) for v in flux_ticks] if flux_ticks is not None else None
    )

    _main_colorbar(
        fig,
        flux_map,
        C_CBAR,
        r"Flux magnitude  $|q(x)|$",
        ticks=flux_ticks,
        ticklabels=flux_ticklabels,
        label_y=BOTTOM_CBAR_Y + MAIN_CBAR_H + MAIN_CBAR_LABEL_DY,
    )


    circulation_artist = face_panel(
        circulation_ax,
        tri,
        omega_hat,
        CIRCULATION_CMAP,
        circulation_norm,
    )
    circulation_artist.set_rasterized(True)

    circulation_ax.triplot(
        tri,
        color="#FFFFFF",
        linewidth=0.15,
        alpha=0.62,
        zorder=5,
    )

    _draw_boundary_loops(
        circulation_ax,
        holes,
    )

    for center, radius in holes:
        circulation_ax.add_patch(
            plt.Circle(
                center,
                1.01 * radius,
                fill=False,
                lw=1.10,
                ec="#16263E",
                alpha=0.95,
                zorder=11,
            )
        )

    _fix_domain_view(circulation_ax, geo)

    cb = float(
        max(
            abs(circulation_norm.vmin),
            abs(circulation_norm.vmax),
        )
    )
    _main_colorbar(
        fig,
        circulation_artist,
        D_CBAR,
        r"Face circulation  $\Phi(x)$",
        ticks=[-cb, 0.0, cb],
        label_y=BOTTOM_CBAR_Y + MAIN_CBAR_H + MAIN_CBAR_LABEL_DY,
    )


    _draw_error_row(
        fig,
        task="0",
        models=error_models["0"],
        sample=sample0,
        geo=geo,
        tri=tri,
        x0=ERROR_X0,
        y0=ERROR_ROW_Y["0"],
        tile_w=ERROR_TILE_W,
        gap_x=ERROR_GAP_X,
        image_h=ERROR_H,
        cbar_y=ERROR_CBAR_Y["0"],
        row_label=r"$\mathbf{C}^{0}$  Potential",
    )
    _draw_error_row(
        fig,
        task="1",
        models=error_models["1"],
        sample=sample1,
        geo=geo,
        tri=tri,
        x0=ERROR_X0,
        y0=ERROR_ROW_Y["1"],
        tile_w=ERROR_TILE_W,
        gap_x=ERROR_GAP_X,
        image_h=ERROR_H,
        cbar_y=ERROR_CBAR_Y["1"],
        row_label=r"$\mathbf{C}^{1}$  Flux",
    )
    _draw_error_row(
        fig,
        task="2",
        models=error_models["2"],
        sample=sample2,
        geo=geo,
        tri=tri,
        x0=ERROR_X0,
        y0=ERROR_ROW_Y["2"],
        tile_w=ERROR_TILE_W,
        gap_x=ERROR_GAP_X,
        image_h=ERROR_H,
        cbar_y=ERROR_CBAR_Y["2"],
        row_label=r"$\mathbf{C}^{2}$  Circulation",
    )

    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / f"darcy_cochain_comparison_{palette_name}.png"

    HORIZONTAL_CROP_MARGIN = 0.010  
    content_left = X_LEFT
    content_right = ERROR_X0 + 4.0 * ERROR_TILE_W + 3.0 * ERROR_GAP_X

    crop_left = max(0.0, content_left - HORIZONTAL_CROP_MARGIN)
    crop_right = min(1.0, content_right + HORIZONTAL_CROP_MARGIN)

    
    
    crop_bbox = Bbox.from_extents(
        crop_left * FIG_W,
        0.0,
        crop_right * FIG_W,
        FIG_H,
    )

    fig.savefig(
        figure_path,
        dpi=600,
        facecolor="#FFFFFF",
        bbox_inches=crop_bbox,
        pad_inches=0.0,
    )
    fig.savefig(
        figure_path.with_suffix(".pdf"),
        dpi=600,
        facecolor="#FFFFFF",
        bbox_inches=crop_bbox,
        pad_inches=0.0,
    )
    plt.close(fig)

    errors = {
        task: float(
            per_sample_l2(
                selected[task].prediction,
                selected[task].target,
            )[samples[task]]
        )
        for task in ("0", "1", "2")
    }

    provenance = {
        "figure": figure_path.name,
        "palette": palette_name,
        "selection": (
            "independent best DKHO-large sample for each form by per-sample relative-L2"
            if requested_index is None
            else "same manually requested test-row index used for all forms"
        ),
        "test_rows": samples,
        "archive_ids": archive_ids,
        "panel_a_uses_form0_conditions": True,
        "relative_l2": errors,
        "sample_audit": sample_audit,
        "model_ranking": {
            task: [
                {
                    "rank": rank,
                    "model": run.name,
                    "mean_relative_l2": score,
                }
                for rank, (run, score) in enumerate(rankings[task], start=1)
            ]
            for task in ("0", "1", "2")
        },
        "error_models": {
            task: [run.name for run in models] for task, models in error_models.items()
        },
        "error_visualization": (
            "dimensionless field |prediction-target|/Q95(|target|); "
            "shared decade-rounded LogNorm within each form"
        ),
        "rendering_note": (
            "Wide 20x8-inch design canvas: left 2x2 physical-field panels plus right 3x4 Top-4 error matrix; "
            "the left block is shifted inward, the error matrix is shifted left, and the top colorbar/lower-title gutter is explicitly guarded against overlap. "
            "Export uses a deterministic horizontal-only Bbox crop, not bbox_inches='tight', so large left/right blank margins are removed while the full vertical layout is preserved."
        ),
    }

    (output / f"darcy_cochain_comparison_{palette_name}.json").write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

def main() -> None:
    

    global TDK_ROOTS
    global BASE
    global TDK_CONFIG
    global BASELINE_MAIN_CONFIG
    global TDK_STRUCTURE

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "qualitative_comparison",
        help="Output directory for the cochain comparison figure.",
    )

    parser.add_argument(
        "--dkho-large-root",
        type=Path,
        default=TDK_ROOTS["DKHO-large"],
        help="Root directory for DKHO-large checkpoints and predictions.",
    )

    parser.add_argument(
        "--dkho-small-root",
        type=Path,
        default=TDK_ROOTS["DKHO-small"],
        help="Root directory for DKHO-small checkpoints and predictions.",
    )

    
    parser.add_argument(
        "--tdk-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=BASE,
        help="Root directory for native-rank baseline outputs.",
    )

    parser.add_argument(
        "--tdk-feature-config",
        default=TDK_CONFIG,
        help="Input-profile directory shared by the DKHO small and large runs.",
    )

    parser.add_argument(
        "--baseline-main-config",
        default=BASELINE_MAIN_CONFIG,
        help="Input-profile directory used by the baseline runs.",
    )

    parser.add_argument(
        "--tdk-structure",
        choices=(
            "full",
            "no_dirac",
            "no_phase",
            "no_harmonic",
        ),
        default="full",
        help="DKHO structural variant to render.",
    )

    parser.add_argument(
        "--darcy-figure-index",
        type=int,
        default=None,
        help=(
            "Optional row in the aligned test split. By default, each rank uses "
            "the DKHO-large sample with lowest relative L2 error."
        ),
    )

    parser.add_argument(
        "--palette",
        choices=(*PALETTE_LIBRARY.keys(), "showcase", "all"),
        default="editorial_bluegold",
        help=(
            "Publication palette. 'showcase' exports selected alternatives; 'all' "
            "exports every registered palette."
        ),
    )

    parser.add_argument(
        "--palette-preview",
        action="store_true",
        help=(
            "Render palette previews only; do not read datasets, checkpoints, or predictions."
        ),
    )

    args = parser.parse_args()

    def rooted(path: Path) -> Path:
        

        return path if path.is_absolute() else ROOT / path

    
    large_root = args.tdk_root if args.tdk_root is not None else args.dkho_large_root
    TDK_ROOTS = {
        "DKHO-large": rooted(large_root),
        "DKHO-small": rooted(args.dkho_small_root),
    }
    BASE = rooted(args.baseline_root)
    TDK_CONFIG = args.tdk_feature_config
    BASELINE_MAIN_CONFIG = args.baseline_main_config
    TDK_STRUCTURE = args.tdk_structure

    args.output = rooted(args.output).resolve()

    
    style()

    
    if args.palette_preview:
        preview_path = save_palette_preview(args.output)
        print(
            f"[visualize] palette preview written to: {preview_path}",
            flush=True,
        )
        return

    
    prepare_cache()

    
    
    geo = DarcyGeometry(lpe_dim=16)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    runs_by_task = {task: load_primary(task) for task in ("0", "1", "2")}

    if args.palette == "showcase":
        palette_names = list(PALETTE_SHOWCASE)
    elif args.palette == "all":
        palette_names = list(PALETTE_LIBRARY)
    else:
        palette_names = [args.palette]
    for palette_name in palette_names:
        apply_palette(palette_name)
        render_darcy_cochain_comparison(
            runs_by_task,
            geo,
            args.output,
            args.darcy_figure_index,
            palette_name,
        )
        print(
            "[visualize] wrote: " f"{args.output / f'darcy_cochain_comparison_{palette_name}.png'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
