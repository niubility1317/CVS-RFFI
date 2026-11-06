"""Combine loss and gradient finiteness checks before a source optimizer step."""
import torch


def checked_backward_and_clip(loss, model, required_parameters):
    """Backpropagate and clip once; never step an optimizer on invalid values."""
    loss.backward()
    if any(p.grad is None for p in required_parameters):
        raise ValueError('Required gradient is missing')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
    if not bool(torch.isfinite(loss) & torch.isfinite(norm)):
        raise FloatingPointError('nonfinite loss or gradient norm')
