import torch
from tools.r17_task_fidelity import frozen_task_fidelity


def clean_objective(output,clean,reader):
    if len(output)!=3:raise ValueError('Three fixed restoration stages required')
    restored=output[-1]
    if restored.shape!=clean.shape or restored.ndim!=3 or restored.shape[1:]!=(2,256):
        raise ValueError('Paired source-clean IQ2x256 required')
    reference=clean.detach()
    denominator=reference.square().mean((1,2)).clamp_min(1e-4)
    iq=(restored-reference).square().mean((1,2))/denominator
    task,terms=frozen_task_fidelity(restored,reference,reader)
    terms.update(iq_nmse=iq)
    return iq+task,terms
