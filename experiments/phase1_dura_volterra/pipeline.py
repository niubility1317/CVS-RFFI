"""Frozen DURA anchor with input-conditioned dynamic memory correction."""
import torch

from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dynamic_volterra.train_source import load_components as load_blind_components


def load_components(root, device, deps):
    blind, gate, _, _ = load_blind_components(root, device, deps)
    dura = frozen_dura(root, device).eval().requires_grad_(False)
    return blind, gate, dura, None


@torch.no_grad()
def input_noise(x, blind, dura, eye, deps):
    _, logvar = blind(x)
    base = dura(x)
    if (base.shape != x.shape or logvar.shape != (len(x),)
            or not all(torch.isfinite(t).all() for t in (x, base, logvar))):
        raise FloatingPointError('Finite DURA output and blind noise estimate are required')
    return base, logvar


@torch.no_grad()
def predict_batch(x, model, blind, gate, dura, eye, reader, deps):
    scores = gate(x).reshape(-1)
    base = dura(x)
    if (scores.shape != (len(x),) or base.shape != x.shape
            or not all(torch.isfinite(t).all() for t in (x, scores, base))):
        raise FloatingPointError('Finite gate scores and DURA waveforms are required')
    active = scores.sigmoid() > 0.8
    output = base.clone()
    if active.any():
        _, logvar = blind(x[active])
        if logvar.shape != (int(active.sum()),) or not torch.isfinite(logvar).all():
            raise FloatingPointError('One finite blind noise value per active window is required')
        output[active] = model(x[active], base[active], logvar)
    choices = []
    for value in (x, base, output):
        logits = reader.classify_features(reader.features(value))
        if logits.shape != (len(x), 6) or not torch.isfinite(value).all() or not torch.isfinite(logits).all():
            raise FloatingPointError('Finite waveform and frozen-reader outputs are required')
        choices.append(logits.argmax(1))
    return base, output, active, choices
