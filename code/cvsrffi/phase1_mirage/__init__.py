"""Public role-safe data contracts for Phase1 MIRAGE-OWDG."""

from .data import (
    LabeledView,
    SourceInventoryRow,
    SourceProtocolError,
    SourceSplitManifest,
    UnlabeledView,
    ValidationView,
    build_source_split,
    materialize_labeled,
    materialize_unlabeled,
    materialize_validation,
)

__all__ = [
    "LabeledView",
    "SourceInventoryRow",
    "SourceProtocolError",
    "SourceSplitManifest",
    "UnlabeledView",
    "ValidationView",
    "build_source_split",
    "materialize_labeled",
    "materialize_unlabeled",
    "materialize_validation",
]
