"""Isolated D42 numerical runtimes for compact D92 E0 feature spaces.

The historical D42 module is intentionally fixed at 288 dimensions for legacy
experiments. Current D92 E0 uses either identity160 or identity160+FFT96.
This factory loads a private module instance for those compact spaces so their
support-only LDA, full/block covariance and residual INT8 compilation operate
on the actual active coordinates rather than a 288D zero-padded surrogate.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from types import ModuleType

from cvsrffi import stage2_d42_unified_shrinkage_lda as _legacy_d42


class CompactD42RuntimeError(ValueError):
    """Raised when a requested D92 E0 compact geometry is unsupported."""


_COMPACT_BLOCK_DIMS = {
    160: (160,),
    256: (160, 96),
}


@lru_cache(maxsize=None)
def d42_runtime_for_feature_dim(feature_dim: int) -> ModuleType:
    """Return an isolated D42 runtime whose feature axes match ``feature_dim``.

    The 288D historical runtime remains untouched. Compact runtimes are private
    copies because D92's support-only component builders temporarily install
    fit functions on the passed D42 module during registration.
    """

    dimension = int(feature_dim)
    if dimension == int(_legacy_d42.FEATURE_DIM):
        return _legacy_d42
    try:
        block_dims = _COMPACT_BLOCK_DIMS[dimension]
    except KeyError as exc:
        raise CompactD42RuntimeError(
            f"unsupported compact D42 feature dimension: {dimension}"
        ) from exc

    module_name = f"cvsrffi._stage2_d42_compact_{dimension}"
    existing = sys.modules.get(module_name)
    if isinstance(existing, ModuleType):
        return existing
    source = getattr(_legacy_d42, "__file__", None)
    if not isinstance(source, str) or not source:
        raise CompactD42RuntimeError("cannot locate the legacy D42 runtime")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise CompactD42RuntimeError("cannot load a compact D42 runtime")
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runtime
    spec.loader.exec_module(runtime)

    offsets = [0]
    for width in block_dims:
        offsets.append(offsets[-1] + int(width))
    runtime.FEATURE_DIM = dimension
    runtime.BLOCK_DIMS = tuple(int(width) for width in block_dims)
    runtime.BLOCK_SLICES = tuple(
        slice(start, stop) for start, stop in zip(offsets[:-1], offsets[1:])
    )
    return runtime


__all__ = [
    "CompactD42RuntimeError",
    "d42_runtime_for_feature_dim",
]
