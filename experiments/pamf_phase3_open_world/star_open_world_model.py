"""Differentiable open features and bounded residual on a verified native PAMF."""
from __future__ import annotations

from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F


SEED = 2026092701
PARAMETERS = 51841
STD_EPS = 1e-12
UNIT_EPS = 1e-12


class OpenParts(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(512, 64), nn.GELU(),
                                       nn.Linear(64, 64), nn.LayerNorm(64))
        self.identity_projection = nn.Linear(64, 160)
        self.residual_gain = nn.Parameter(torch.zeros(160))
        self.relation_scorer = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, temporal, pa, signature):
        mean = temporal.mean(dim=-1)
        std = torch.sqrt(temporal.var(dim=-1, unbiased=False) + STD_EPS)
        inputs = torch.cat((mean, std, pa, signature), dim=1)
        return F.normalize(self.projector(inputs), dim=1, eps=UNIT_EPS)


def _parts(model):
    parts = getattr(model, 'open_world', None)
    if not isinstance(parts, OpenParts):
        raise ValueError('Install the native open-world module first')
    return parts


def clear(model):
    """Release only call-local hooks and the last transient autograd capture."""
    for handle in getattr(model, '_open_world_handles', ()):
        handle.remove()
    model._open_world_handles = []
    model._open_world_last = None


def _matrix(value, shape, name):
    if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())):
        raise ValueError('Finite FP32 tensor with the native shape required: ' + name)
    return value


def install(native_model, seed=SEED):
    """Keep original parameter paths and patch only the reader feature method.

    The caller verifies and strict-loads the original parent before installation.
    Modes and grad tracking remain caller-owned after the initial eval setting.
    """
    if type(seed) is not int or seed <= 0:
        raise ValueError('Positive integer initialization seed required')
    if not isinstance(native_model, nn.Module) or hasattr(native_model, 'open_world'):
        raise ValueError('An unmodified native model is required for one installation')
    try:
        reader = native_model.id_backbone.encoder
        core = reader.core.id_backbone
        targets = core.t2, core, reader.sequence_pool
        original = reader.features
        prototype_parameter = next(reader.parameters())
    except (AttributeError, StopIteration) as exc:
        raise ValueError('The native complete reader, t2, PA and signature interfaces are required') from exc
    if (not all(isinstance(target, nn.Module) for target in targets)
            or not callable(original) or not callable(getattr(reader, 'classify_features', None))
            or hasattr(reader, '_open_world_original_features')):
        raise ValueError('The native original feature method and module taps are required')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        parts = OpenParts()
    parts.to(device=prototype_parameter.device, dtype=torch.float32)
    native_model.add_module('open_world', parts)
    # A bound method is not another registered Module or a copied identity head.
    reader._open_world_original_features = original
    owner = weakref.ref(native_model)
    clear(native_model)

    def features(self, x):
        model = owner()
        if model is None:
            raise RuntimeError('The native owner no longer exists')
        clear(model)
        if (not torch.is_tensor(x) or x.ndim != 3 or not len(x)
                or tuple(x.shape[1:]) != (2, 256) or x.dtype != torch.float32
                or not bool(torch.isfinite(x).all())):
            raise ValueError('Finite native FP32[B,2,256] input required')
        count = len(x)
        captured = {}

        def record(name, value):
            if name in captured:
                raise ValueError('Each native branch must be captured once: ' + name)
            captured[name] = value

        def core_hook(module, inputs, value):
            if not isinstance(value, dict) or 'feat_pa' not in value:
                raise ValueError('The actual native core PA capture is required')
            record('pa', value['feat_pa'])

        handles = []
        try:
            handles.append(core.t2.register_forward_hook(lambda m, i, value: record('t2', value)))
            handles.append(core.register_forward_hook(core_hook))
            handles.append(self.sequence_pool.register_forward_hook(lambda m, i, value: record('signature', value)))
            model._open_world_handles = handles
            with torch.autocast(device_type=x.device.type, enabled=False):
                baseline = self._open_world_original_features(x)
            if set(captured) != {'t2', 'pa', 'signature'}:
                raise ValueError('All three same-forward native captures are required')
            _matrix(baseline, (count, 160), 'raw original fused160')
            _matrix(captured['t2'], (count, 96, 32), 't2 capture')
            _matrix(captured['pa'], (count, 160), 'PA capture')
            _matrix(captured['signature'], (count, 160), 'bounded signature capture')
            parts = _parts(model)
            with torch.autocast(device_type=x.device.type, enabled=False):
                open64 = parts.encode(captured['t2'], captured['pa'], captured['signature'])
                direction = F.normalize(parts.identity_projection(open64), dim=1, eps=UNIT_EPS)
                residual = .1 * baseline.norm(dim=1, keepdim=True) * parts.residual_gain.tanh() * direction
                raw = baseline + residual
            _matrix(open64, (count, 64), 'open64')
            _matrix(raw, (count, 160), 'open residual fused160')
            model._open_world_last = dict(baseline_fused160=baseline, raw_fused160=raw, open64=open64)
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


def forward(model, x):
    """One complete reader/head pass; inherit the caller's mode and grad scope."""
    _parts(model)
    reader = model.id_backbone.encoder
    try:
        raw = reader.features(x)
        with torch.autocast(device_type=raw.device.type, enabled=False):
            logits = reader.classify_features(raw)
        _matrix(logits, (len(x), 6), 'complete original default-head logits')
        return dict(model._open_world_last, logits=logits)
    except BaseException:
        clear(model)
        raise


def relation(model, z, prototypes):
    """Shared prototype membership logits; no old identity selection occurs."""
    parts = _parts(model)
    if (not torch.is_tensor(z) or z.ndim != 2 or not len(z) or z.shape[1] != 64
            or not torch.is_tensor(prototypes) or prototypes.ndim != 2
            or not len(prototypes) or prototypes.shape[1] != 64):
        raise ValueError('Open features [N,64] and prototype slots [K,64] required')
    _matrix(z, (len(z), 64), 'relation features')
    _matrix(prototypes, (len(prototypes), 64), 'relation prototypes')
    if z.device != prototypes.device or z.device != parts.residual_gain.device:
        raise ValueError('Relation inputs and the installed parameters must share a device')
    inputs = torch.cat(((z[:, None] - prototypes[None]).abs(), z[:, None] * prototypes[None]), dim=-1)
    with torch.autocast(device_type=z.device.type, enabled=False):
        return parts.relation_scorer(inputs).squeeze(-1)


def parameter_report(model):
    """Separate active identity/open parameters from unused native containers."""
    parts = _parts(model)
    identity = sum(parameter.numel() for parameter in model.id_backbone.encoder.parameters())
    additional = sum(parameter.numel() for parameter in parts.parameters())
    container = sum(parameter.numel() for parameter in model.parameters())
    return dict(native_identity_parameters=identity,
                open_projector_parameters=sum(p.numel() for p in parts.projector.parameters()),
                identity_residual_parameters=sum(p.numel() for p in parts.identity_projection.parameters()) + parts.residual_gain.numel(),
                relation_parameters=sum(p.numel() for p in parts.relation_scorer.parameters()),
                open_world_parameters=additional, active_parameters=identity + additional,
                native_container_parameters=container - additional, training_container_parameters=container,
                unused_native_parameters=container - additional - identity,
                expected_native_identity_parameters=232993, expected_open_world_parameters=PARAMETERS,
                expected_active_parameters=284834)
