"""Frozen ground-A CosFace head only; no package loading, fitting or calibration.

Metadata is an explicit caller declaration, not proof of a real export. Unknown
Dual/logit corrections must be resolved by the exporter before using this route.
"""
from dataclasses import dataclass
import math
import re

import torch
import torch.nn.functional as F


FEATURE_DIM = 160
GROUND_CLASS_COUNT = 6
WEIGHT_KEY = 'id_backbone.cls_head.head.weight'
NORM_EPS = 1e-4
SOURCE_ONLY_VERDICT = 'MATCHED_SOURCE_ONLY_SCRATCH'


def _checkpoint_identity(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Explicit original checkpoint SHA256 identity required')


@dataclass(frozen=True)
class GroundFeatureContract:
    """Bind raw cached feat_joint to the checkpoint that produced it."""
    checkpoint_sha256: str
    cache_key: str
    source_tensor: str
    representation: str
    dtype: str
    feature_dim: int

    def __post_init__(self):
        _checkpoint_identity(self.checkpoint_sha256)
        if (self.cache_key != 'z_id' or self.source_tensor != 'feat_joint'
                or self.representation != 'raw' or self.dtype != 'float32'
                or type(self.feature_dim) is not int or self.feature_dim != FEATURE_DIM):
            raise ValueError('Ground-A requires raw feat_joint z_id float32[N,160]')


@dataclass(frozen=True)
class GroundHeadMetadata:
    """Frozen declarations supplied by the actual head-package exporter.

    Neither s nor class order has a default. The existing source-only verdict is
    consumed as a declaration; this module does not verify checkpoint contents.
    """
    checkpoint_sha256: str
    checkpoint_weight_key: str
    ordered_classes: tuple[str, ...]
    scale: float
    norm_eps: float
    feature_contract: GroundFeatureContract
    source_only_verdict: str
    target_access_before_freeze: bool
    checkpoint_inheritance: tuple
    logit_corrections: str

    def __post_init__(self):
        _checkpoint_identity(self.checkpoint_sha256)
        if self.checkpoint_weight_key != WEIGHT_KEY:
            raise ValueError('Ground-A original CosFace weight key mismatch')
        if not isinstance(self.ordered_classes, (tuple, list)):
            raise TypeError('Explicit ordered ground class IDs required')
        classes = tuple(self.ordered_classes)
        if (len(classes) != GROUND_CLASS_COUNT or any(not isinstance(c, str) or not c for c in classes)
                or len(set(classes)) != len(classes)):
            raise ValueError('Ground-A requires six unique ordered ground class IDs')
        object.__setattr__(self, 'ordered_classes', classes)
        if (type(self.scale) not in (int, float) or not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError('Explicit finite positive actual CosFace scale required')
        if type(self.norm_eps) not in (int, float) or self.norm_eps != NORM_EPS:
            raise ValueError('Original CosFace normalization eps must be 1e-4')
        if (not isinstance(self.feature_contract, GroundFeatureContract)
                or self.feature_contract.checkpoint_sha256 != self.checkpoint_sha256):
            raise ValueError('Head and raw feature checkpoint identities differ')
        if (self.source_only_verdict != SOURCE_ONLY_VERDICT or self.target_access_before_freeze is not False
                or not isinstance(self.checkpoint_inheritance, (tuple, list)) or self.checkpoint_inheritance):
            raise ValueError('Explicit matched source-only scratch provenance declaration required')
        object.__setattr__(self, 'checkpoint_inheritance', ())
        if self.logit_corrections != 'none':
            raise ValueError('Unknown or enabled Dual/logit corrections are unsupported by the bare CosFace route')


class GroundClassifierA:
    """Independent label-free scores in the original ground head column order.

    This is deliberately not an nn.Module: there is no training, parameter
    update, checkpoint loader, prototype fallback or target-data fit interface.
    Weight bytes are copied once, so caller tensor mutations cannot change A.
    """
    __slots__ = ('_metadata', '_weight_bytes')

    def __init__(self, *, weight: torch.Tensor, metadata: GroundHeadMetadata):
        if not isinstance(metadata, GroundHeadMetadata):
            raise TypeError('Explicit GroundHeadMetadata required')
        if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.float32
                or weight.layout != torch.strided or tuple(weight.shape) != (GROUND_CLASS_COUNT, FEATURE_DIM)):
            raise ValueError('Original ground head weight must be float32[6,160]')
        if not bool(torch.isfinite(weight).all()):
            raise ValueError('Nonfinite ground head weight')
        # uint8.tolist() avoids a NumPy dependency and stores exact float32 bits.
        raw = weight.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
        object.__setattr__(self, '_weight_bytes', bytes(raw.tolist()))
        object.__setattr__(self, '_metadata', metadata)

    def __setattr__(self, name, value):
        raise AttributeError('GroundClassifierA is frozen')

    @property
    def metadata(self):
        return self._metadata

    @property
    def classes(self):
        return self._metadata.ordered_classes

    def score(self, *, z_id: torch.Tensor, feature_contract: GroundFeatureContract) -> torch.Tensor:
        if not isinstance(feature_contract, GroundFeatureContract) or feature_contract != self.metadata.feature_contract:
            raise ValueError('Input feature contract differs from the frozen ground head binding')
        if (not isinstance(z_id, torch.Tensor) or z_id.dtype != torch.float32 or z_id.layout != torch.strided
                or z_id.ndim != 2 or z_id.shape[1] != FEATURE_DIM):
            raise ValueError('Ground-A input must be raw float32[N,160] tensor')
        if z_id.device.type not in ('cpu', 'cuda'):
            raise ValueError('Ground-A supports explicit CPU/CUDA float32 inference only')
        if not bool(torch.isfinite(z_id).all()):
            raise ValueError('Nonfinite ground-A input')
        # Local writable storage never aliases frozen bytes or a caller weight.
        weight = torch.frombuffer(bytearray(self._weight_bytes), dtype=torch.float32).reshape(
            GROUND_CLASS_COUNT, FEATURE_DIM).to(device=z_id.device)
        with torch.no_grad(), torch.autocast(device_type=z_id.device.type, enabled=False):
            x_f = F.normalize(z_id.float(), dim=1, eps=self.metadata.norm_eps)
            w_f = F.normalize(weight.float(), dim=1, eps=self.metadata.norm_eps)
            scores = F.linear(x_f, w_f) * self.metadata.scale
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('Nonfinite ground-A scores')
        return scores

    def predict(self, *, z_id: torch.Tensor, feature_contract: GroundFeatureContract) -> tuple[str, ...]:
        scores = self.score(z_id=z_id, feature_contract=feature_contract)
        # torch.argmax selects the first original head column on exact ties.
        return tuple(self.classes[i] for i in scores.argmax(dim=1).cpu().tolist())
