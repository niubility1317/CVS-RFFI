import torch
from experiments.pamf_r17_compact.models import stages


def teacher_and_student_context(teacher,student,received):
    with torch.no_grad():
        teacher_context=teacher.frozen.estimate(received)
        desired=stages(teacher.prior,received,teacher_context)[0]
    # Must remain outside no_grad: optimize both coarse and action networks.
    return desired,student.estimate_context(received)
