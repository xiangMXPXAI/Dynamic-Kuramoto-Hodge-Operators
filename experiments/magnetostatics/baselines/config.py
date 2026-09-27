"""Define paths and hyperparameters for magnetostatics baseline runs."""

from pathlib import Path
import torch

TASK_ROOT = Path(__file__).resolve().parents[1]

class Config:
    DATA_DIR = str(TASK_ROOT / "data")
    PICKLE_FILE = "cavity_magnetostatics_v1.pkl"

    LEGACY_OUTPUT_DIR = TASK_ROOT / "runs" / "baseline" / "legacy"
    DEFAULT_OUTPUT_DIR = TASK_ROOT / "runs" / "baseline" / "retrained"
    LOG_FILE = "training_log.txt"

    def __init__(self, output_dir: str | Path | None = None):
        self.OUTPUT_DIR = str(Path(output_dir) if output_dir is not None else self.DEFAULT_OUTPUT_DIR)
    VIZ_OUTPUT = "flux_field_comparison.png"
    TEST_RATIO = 0.2
    VAL_RATIO = 0.15
    K_EIGENS = 64
    EPOCHS = 100
    BATCH_SIZE = 64
    LR = 1e-3
    LR_OURS = 1e-3
    WEIGHT_DECAY = 1e-6
    PATIENCE = 20
    LAMBDA_COEFF_REG = 1e-4
    LAMBDA_L1 = 1e-3
    GNO_HIDDEN_CHANNELS = 84
    GNO_PROJECTION_CHANNELS = 96
    GNO_N_LAYERS = 5
    GNO_RADIUS = 0.2
    FNO_MODES = (4, 4, 4)
    FNO_HIDDEN_CHANNELS = 21
    FNO_N_LAYERS = 2
    FNO_GRID_RES = 16
    MGN_HIDDEN_DIM = 58
    MGN_NUM_LAYERS = 10
    DEEPONET_BRANCH_LAYERS = [64, 64, 64]
    DEEPONET_TRUNK_LAYERS = [64, 64, 64]
    DEEPONET_BASIS_DIM = 74
    GEOFNO_MODES = 6
    GEOFNO_WIDTH = 12
    GEOFNO_LAYERS = 2
    GEOFNO_GRID_RES = 16
    SPECTRAL_HIDDEN_DIMS = (32, 32)
    HSD_FNO_MODES = (4, 4, 4)
    HSD_FNO_HIDDEN = 14
    HSD_FNO_LAYERS = 4
    FLUX_LOSS_WEIGHT = 1.0
    DIVERGENCE_LOSS_WEIGHT = 0.5
    VIZ_N_SAMPLES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
