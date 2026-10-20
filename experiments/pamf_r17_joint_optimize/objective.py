from .models import distillation_loss
from tools.r17_task_fidelity import frozen_task_fidelity


def joint_objective(output,desired,clean,reader,*,clean_reference):
    wave,terms=distillation_loss(output,desired,clean)
    task,task_terms=frozen_task_fidelity(output[-1],clean if clean_reference else desired[-1],reader)
    terms.update(task_terms)
    return wave+task+(terms['iq_nmse'] if clean_reference else 0.),terms
