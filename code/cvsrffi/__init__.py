"""Internal CVS-RFFI training utilities.

The public script/module entrypoints stay at the repository root; this package
holds reusable implementation pieces shared by training, post-stage, and SGC/SSDG
entrypoints.
"""

from .meta_weight_bank import (
    BlockSpec,
    DeltaBankEntry,
    DeltaTaskKey,
    WEIGHT_DELTA_BANK_SCHEMA,
    WeightDeltaBank,
    compose_weight_delta,
    extract_block_delta,
    fit_weight_delta_bank,
    parameter_block_key,
)


__all__ = [
    "BlockSpec",
    "DeltaBankEntry",
    "DeltaTaskKey",
    "WEIGHT_DELTA_BANK_SCHEMA",
    "WeightDeltaBank",
    "compose_weight_delta",
    "extract_block_delta",
    "fit_weight_delta_bank",
    "parameter_block_key",
]
