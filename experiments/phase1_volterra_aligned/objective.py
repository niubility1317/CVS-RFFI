"""Source waveform loss invariant to bounded delay and complex gain."""
import torch
from torch.nn import functional as F

from experiments.phase1_spectral_margin.objective import source_loss as reader_loss


def aligned_iq_error(output, clean):
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256):
        raise ValueError('Matching 256-point IQ windows are required')
    target = torch.complex(clean[:, 0, 64:-64], clean[:, 1, 64:-64]).detach()
    target_energy = target.abs().square().mean(1).clamp_min(1e-8)
    with torch.no_grad():
        padded = F.pad(output.detach(), (64, 64), mode='reflect')
        windows = torch.complex(padded[:, 0], padded[:, 1]).unfold(-1, 256, 1)[:, :, 64:-64]
        gain = (windows.conj()*target[:, None]).sum(-1)/windows.abs().square().sum(-1).clamp_min(1e-20)
        error = (gain[:, :, None]*windows-target[:, None]).abs().square().mean(-1)
        choice = error.argmin(1)
        selected_gain = gain[torch.arange(len(output), device=output.device), choice]
    padded = F.pad(output, (64, 64), mode='reflect')
    windows = torch.complex(padded[:, 0], padded[:, 1]).unfold(-1, 256, 1)
    selected = windows[torch.arange(len(output), device=output.device), choice, 64:-64]
    return (selected_gain[:, None]*selected-target).abs().square().mean(1)/target_energy


def source_loss(reader, output, clean, clean_features, groups):
    _, terms = reader_loss(reader, output, clean, clean_features, groups)
    aligned = aligned_iq_error(output, clean)
    reference_power = clean.detach().flatten(1).square().mean(1).clamp_min(1e-8)
    output_power = output.flatten(1).square().mean(1).clamp_min(1e-8)
    log_power_error = (output_power/reference_power).log().clamp(-8, 8).square()
    per = aligned+0.2*terms['feature']+0.2*terms['distill']+0.05*terms['margin']+0.05*log_power_error
    loss = torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
    return loss, dict(aligned=aligned, power=log_power_error, feature=terms['feature'],
                      distill=terms['distill'], margin=terms['margin'])
