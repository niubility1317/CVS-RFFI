"""Same-forward native contributions with bounded reference-conditioned residuals."""
from __future__ import annotations

import math
from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from . import star_open_world_model as oldow


SEED = 2026092701
PARAMETERS = 82113
ACTIVE_PARAMETERS = 315106
CONTRIBUTION_NAMES = ('time_frequency', 'PA', 'reference', 'signature')


class ContributionParts(oldow.OpenParts):
    def __init__(self, use_reference):
        # Construct only the new graph; do not allocate the old 512-input graph.
        nn.Module.__init__(self)
        self.use_reference = use_reference
        self.projector = nn.Sequential(nn.Linear(640, 64), nn.GELU(),
                                       nn.Linear(64, 64), nn.LayerNorm(64))
        self.identity_projection = nn.Linear(64, 160)
        self.residual_gain = nn.Parameter(torch.zeros(160))
        self.relation_scorer = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))
        self.conditioner = nn.Sequential(nn.Linear(29, 32), nn.GELU(), nn.Linear(32, 640))
        nn.init.zeros_(self.conditioner[2].weight)
        nn.init.zeros_(self.conditioner[2].bias)

    def encode(self, contributions, reference):
        condition = reference if self.use_reference else torch.zeros_like(reference)
        h = self.conditioner(condition).reshape(-1, 4, 160)
        radius = .1 * torch.linalg.vector_norm(contributions, dim=-1, keepdim=True) / math.sqrt(160)
        delta = radius * h.tanh()
        conditioned = contributions + delta
        z = F.normalize(self.projector(conditioned.flatten(1)), dim=1, eps=oldow.UNIT_EPS)
        return z, conditioned, delta


forward = oldow.forward
relation = oldow.relation
clear = oldow.clear


def install(native_model, use_reference, seed=SEED):
    """Patch the verified native reader without replacing its original head."""
    if type(use_reference) is not bool or type(seed) is not int or seed <= 0:
        raise ValueError('A boolean condition policy and positive integer seed are required')
    if not isinstance(native_model, nn.Module) or hasattr(native_model, 'open_world'):
        raise ValueError('One unmodified native model is required')
    try:
        reader = native_model.id_backbone.encoder
        head = reader.core.id_backbone.cls_head
        targets = (head.base_norm, head.pa_norm, reader.response_projection, reader.response, reader.sequence_pool)
        original = reader.features
        prototype = next(reader.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError('Complete native fusion and masked reference interfaces required') from exc
    if (not all(isinstance(m, nn.Module) for m in targets) or not callable(original)
            or not callable(getattr(reader, 'classify_features', None))
            or hasattr(reader, '_open_world_original_features')
            or not torch.is_tensor(getattr(head, 'gain', None)) or tuple(head.gain.shape) != (160,)
            or not torch.is_tensor(getattr(reader, 'response_gain', None)) or tuple(reader.response_gain.shape) != (160,)):
        raise ValueError('Original native fusion gains, modules and default-head interface required')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        parts = ContributionParts(use_reference)
    parts.to(device=prototype.device, dtype=torch.float32)
    native_model.add_module('open_world', parts)
    reader._open_world_original_features = original
    owner = weakref.ref(native_model)
    clear(native_model)

    def features(self, x):
        model = owner()
        if model is None:
            raise RuntimeError('The native owner no longer exists')
        clear(model)
        if (not torch.is_tensor(x) or x.ndim != 3 or not len(x) or tuple(x.shape[1:]) != (2, 256)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite native FP32[B,2,256] input required')
        count = len(x); captured = {}; handles = []

        def record(name, value):
            if name in captured:
                raise ValueError('Each native contribution must be captured once: ' + name)
            captured[name] = value

        try:
            for name, module in zip(('identity', 'physical', 'reference_projection', 'reference29', 'signature'), targets):
                handles.append(module.register_forward_hook(lambda m, inputs, value, n=name: record(n, value)))
            model._open_world_handles = handles
            with torch.autocast(device_type=x.device.type, enabled=False):
                baseline = self._open_world_original_features(x)
                oldow._matrix(baseline, (count, 160), 'raw original fused160')
                if set(captured) != {'identity', 'physical', 'reference_projection', 'reference29', 'signature'}:
                    raise ValueError('All five same-forward native captures are required')
                for name in ('identity', 'physical', 'reference_projection', 'signature'):
                    oldow._matrix(captured[name], (count, 160), name)
                reference = oldow._matrix(captured['reference29'], (count, 29), 'masked reference29')
                if bool(reference[:, 27].ne(0).any()):
                    raise ValueError('NoExplicitCFO reference coordinate 27 must remain zero')
                contributions = torch.stack((captured['identity'], head.gain.tanh() * captured['physical'],
                    self.response_gain.tanh() * captured['reference_projection'], captured['signature']), dim=1)
                parts = oldow._parts(model)
                z, conditioned, delta = parts.encode(contributions, reference)
                direction = F.normalize(parts.identity_projection(z), dim=1, eps=oldow.UNIT_EPS)
                raw = baseline + .1 * baseline.norm(dim=1, keepdim=True) * parts.residual_gain.tanh() * direction
                oldow._matrix(z, (count, 64), 'open64')
                oldow._matrix(raw, (count, 160), 'open residual fused160')
            model._open_world_last = dict(baseline_fused160=baseline, raw_fused160=raw, open64=z,
                contributions=contributions, conditioned_contributions=conditioned, branch_deltas=delta, reference29=reference)
            return raw
        except BaseException:
            model._open_world_last = None
            raise
        finally:
            for handle in handles:
                handle.remove()
            model._open_world_handles = []

    reader.features = MethodType(features, reader)
    return native_model.eval()


def parameter_report(model):
    """Keep active identity costs separate from the unused native domain heads."""
    parts = oldow._parts(model)
    if not isinstance(parts, ContributionParts):
        raise ValueError('The complete-contribution reference model is required')
    report = oldow.parameter_report(model)
    report.update(reference_conditioner_parameters=sum(p.numel() for p in parts.conditioner.parameters()),
        expected_open_world_parameters=PARAMETERS, expected_active_parameters=ACTIVE_PARAMETERS,
        trainable_open_world_parameters=sum(p.numel() for p in parts.parameters() if p.requires_grad),
        trainable_identity_parameters=sum(p.numel() for p in model.id_backbone.encoder.parameters() if p.requires_grad))
    return report
