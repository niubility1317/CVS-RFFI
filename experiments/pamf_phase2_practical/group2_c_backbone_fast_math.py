"""Real-source no-query equivalence check for the accelerated MoPC penalty."""
import torch

from .io import require
from .vendor_native.paper_reproduction.mopc_hr_non_exemplar_cil_sei.algorithm import hierarchical_regularization


def verify_regularization(model):
    current = {name: torch.nn.Parameter(value.detach().clone()) for name,value in model.named_parameters()}
    require(bool(current), 'Missing source parameters for MoPC penalty check')
    previous = {name: value.detach().clone().add_(0.001) for name,value in current.items()}
    total = next(iter(current.values())).new_zeros(())
    for index,(name,value) in enumerate(current.items()):
        total = total + (1.0-index/len(current))*torch.sum((value-previous[name].detach())**2)
    reference = torch.autograd.grad(total, tuple(current.values()))
    accelerated = hierarchical_regularization(current, previous)
    actual = torch.autograd.grad(accelerated, tuple(current.values()))
    require(torch.equal(total, accelerated) and all(torch.equal(a,b) for a,b in zip(reference,actual)),
            'Accelerated MoPC penalty differs from original')
    return dict(status='PASS',parameter_tensors=len(current),loss_equal=True,gradients_equal=True,
                source_only=True,query_read=False)
