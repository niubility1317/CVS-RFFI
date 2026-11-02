"""Per-example source IQ gradient projection for frozen-reader guidance."""
import torch
from experiments.phase1_volterra_aligned.objective import aligned_iq_error, source_loss

COUNTS = [0, 0, 0]

def projected_loss(wave_loss, recognition_loss, output):
    if wave_loss.ndim != 0 or recognition_loss.ndim != 0 or output.ndim < 2:
        raise ValueError('Two scalar losses and a batch of output IQ are required')
    wave = torch.autograd.grad(wave_loss, output, retain_graph=True)[0].detach()
    rec = torch.autograd.grad(recognition_loss, output, retain_graph=True)[0].detach()
    if not torch.isfinite(wave).all() or not torch.isfinite(rec).all():
        raise FloatingPointError('Finite source IQ gradients are required')
    w, r = wave.flatten(1), rec.flatten(1)
    dot = (w*r).sum(1)
    norm2 = w.square().sum(1)
    coeff = (dot / norm2.clamp_min(1e-20)).clamp(max=0)
    correction = -coeff.reshape((-1,)+(1,)*(output.ndim-1))*wave
    surrogate = ((output-output.detach())*correction).sum()
    return wave_loss+recognition_loss+surrogate, {'conflicts': int((dot<0).sum())}

def projected_source_objective(reader, output, anchor, clean, clean_features, groups):
    _, terms = source_loss(reader, output, clean, clean_features, groups)
    with torch.no_grad():
        anchor_error = aligned_iq_error(anchor.detach(), clean)
    excess = (terms['aligned']-anchor_error).clamp_min(0)
    waveform = terms['aligned']+0.05*terms['power']+5.0*excess
    recognition = 0.2*terms['feature']+0.2*terms['distill']+0.05*terms['margin']
    def balanced(values):
        return torch.stack([values[groups==group].mean() for group in groups.unique()]).mean()
    wave_loss, rec_loss = balanced(waveform), balanced(recognition)
    total, info = projected_loss(wave_loss, rec_loss, output)
    COUNTS[0] += 1
    COUNTS[1] += len(output)
    COUNTS[2] += info['conflicts']
    return total, dict(terms, conflicts=info['conflicts'])
