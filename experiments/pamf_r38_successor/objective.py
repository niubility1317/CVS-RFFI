import torch
from . import design as d
from tools.r17_task_fidelity import frozen_task_fidelity


def objective(outputs,desired,clean,reader,arm):
    if arm not in d.ARMS or len(outputs)!=3 or len(desired)!=3 or any(x.shape!=clean.shape for x in (*outputs,*desired)):
        raise ValueError('Registered arm and matching source stages required before reader access')
    reference=clean.detach();energy=reference.square().mean((1,2)).clamp_min(1e-4)
    paired=(outputs[-1]-reference).square().mean((1,2))/energy
    fidelity=torch.stack([(x-y.detach()).square().mean((1,2))/energy for x,y in zip(outputs,desired)]).mean(0)
    loss=paired+fidelity;terms=dict(iq_nmse=paired,teacher_stage_nmse=fidelity)
    if d.TASK[arm]:
        if reader is None:raise ValueError('Frozen original reader required only for task arms')
        task,detail=frozen_task_fidelity(outputs[-1],reference,reader);loss=loss+task;terms.update(detail)
    elif reader is not None:raise ValueError('IQ control cannot receive classifier knowledge')
    return loss,terms
