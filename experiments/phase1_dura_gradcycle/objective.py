"""Balance source and cross-view output-layer gradients for training."""
import torch

GRAD_COUNTS = [0, 0, 0.0, 0]


def balanced_objective(main_loss, auxiliary_loss, model):
    if main_loss.ndim or auxiliary_loss.ndim:
        raise ValueError('Scalar training objectives are required')
    weight = model.gate_output.weight
    main_grad = torch.autograd.grad(main_loss, weight, retain_graph=True)[0].detach()
    aux_grad = torch.autograd.grad(auxiliary_loss, weight, retain_graph=True)[0].detach()
    if not torch.isfinite(main_grad).all() or not torch.isfinite(aux_grad).all():
        raise FloatingPointError('Finite objective gradients are required')
    main_norm = main_grad.norm()
    dot = (main_grad * aux_grad).sum()
    coefficient = torch.minimum(dot / main_norm.square().clamp_min(1e-20), dot.new_zeros(()))
    projected = aux_grad - coefficient * main_grad
    projected_norm = projected.norm()
    if main_norm <= 1e-12:
        coefficient = dot.new_zeros(())
        scale = dot.new_ones(())
    elif projected_norm <= 1e-12:
        scale = dot.new_zeros(())
    else:
        scale = (0.2 * main_norm / projected_norm).clamp(max=50.0)
    GRAD_COUNTS[0] += 1
    GRAD_COUNTS[1] += int(dot < 0)
    GRAD_COUNTS[2] += float(scale)
    GRAD_COUNTS[3] += int(scale >= 50)
    return main_loss + scale * (auxiliary_loss - coefficient * main_loss)
