"""Define paths and hyperparameters for toroidal transport baseline runs."""

from pathlib import Path

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]


class Config:
    DIFFUSIVITY = 0.01
    ADVECTION_SPEED = 1.0
    DT = 0.02
    N_TIMESTEPS = 35
    N_SAMPLES = 3000
    N_BLOBS_MIN = 2
    N_BLOBS_MAX = 5
    BLOB_SIGMA = 0.15
    TEST_RATIO = 0.2
    VAL_RATIO = 0.15
    K_EIGENS = 64
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-3
    LR_OURS = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 20
    GNO_HIDDEN_CHANNELS = 120
    GNO_PROJECTION_CHANNELS = 68
    GNO_N_LAYERS = 3
    GNO_RADIUS = 0.15
    FNO_MODES = (4, 4, 4)
    FNO_HIDDEN_CHANNELS = 20
    FNO_N_LAYERS = 3
    FNO_GRID_RES = 16
    HSD_FNO_MODES = (4, 4, 4)
    HSD_FNO_HIDDEN = 12
    HSD_FNO_LAYERS = 6
    MGN_HIDDEN_DIM = 72
    MGN_NUM_LAYERS = 8
    DEEPONET_BRANCH_LAYERS = [96, 96, 64]
    DEEPONET_TRUNK_LAYERS = [68, 68, 64]
    DEEPONET_BASIS_DIM = 64
    GEOFNO_MODES = 6
    GEOFNO_WIDTH = 8
    GEOFNO_LAYERS = 4
    GEOFNO_GRID_RES = 16
    SPECTRAL_HIDDEN_DIMS = (32, 32)
    DATA_DIR = str(TASK_ROOT / "data")
    PICKLE_FILE = "torus_transport_v1.pkl"
    LEGACY_OUTPUT_DIR = TASK_ROOT / "runs" / "baseline" / "legacy" / "scalar_native"
    DEFAULT_OUTPUT_DIR = TASK_ROOT / "runs" / "baseline" / "retrained" / "scalar_native"
    LOG_FILE = "training_log.txt"
    VIZ_OUTPUT = "scalar_field_comparison.png"

    def __init__(self, output_dir: str | Path | None = None):
        self.OUTPUT_DIR = str(Path(output_dir) if output_dir is not None else self.DEFAULT_OUTPUT_DIR)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
