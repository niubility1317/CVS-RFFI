"""Public configuration and matrix contract for Phase1 HCF-DG."""

from .config import (
    HCFDG_BATCH_SIZE,
    HCFDGConfig,
    MatrixRow,
    SELECTION_SEEDS,
    StageBudget,
    V1_AMP_ENABLED,
    V1_BACKBONE_LR,
    V1_BATCH_SIZE,
    V1_COSFACE_FINAL_MARGIN,
    V1_COSFACE_MARGIN_RAMP_FRACTION,
    V1_COSINE_MIN_LR,
    V1_HEAD_LR,
    V1_OPTIMIZER_UPDATES,
    V1_WARMUP_FRACTION,
    V2_OPTIMIZER_UPDATES,
    V2_STAGE_BUDGET,
    candidate_config,
    deep_screen_rows,
    quick_screen_rows,
    residual_rows,
)
from .losses import compose_hcfdg_loss
from .metrics import SameRowMetrics, rank_source_rows
from .model import HCFDGModel, HCFDGOutput
from .sampler import EpisodeDescriptor, HCFDGEpisodeBatchSampler
from .satellite import ChannelFactors, SingleViewBatch, build_single_view_batch
from .trainer import HCFDGTrainer, TrainState

__all__ = [
    "HCFDG_BATCH_SIZE",
    "HCFDGConfig",
    "MatrixRow",
    "SELECTION_SEEDS",
    "StageBudget",
    "V1_AMP_ENABLED",
    "V1_BACKBONE_LR",
    "V1_BATCH_SIZE",
    "V1_COSFACE_FINAL_MARGIN",
    "V1_COSFACE_MARGIN_RAMP_FRACTION",
    "V1_COSINE_MIN_LR",
    "V1_HEAD_LR",
    "V1_OPTIMIZER_UPDATES",
    "V1_WARMUP_FRACTION",
    "V2_OPTIMIZER_UPDATES",
    "V2_STAGE_BUDGET",
    "candidate_config",
    "ChannelFactors",
    "compose_hcfdg_loss",
    "deep_screen_rows",
    "EpisodeDescriptor",
    "HCFDGEpisodeBatchSampler",
    "HCFDGModel",
    "HCFDGOutput",
    "HCFDGTrainer",
    "quick_screen_rows",
    "rank_source_rows",
    "residual_rows",
    "SameRowMetrics",
    "SingleViewBatch",
    "TrainState",
    "build_single_view_batch",
]
