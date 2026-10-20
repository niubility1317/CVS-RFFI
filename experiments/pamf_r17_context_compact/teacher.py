import torch
from experiments.pamf_r17_compact.models import stages


@torch.no_grad()
def teacher_and_student_context(teacher,student,received):
    # Student context is now compressed; it cannot define teacher targets.
    teacher_context=teacher.frozen.estimate(received)
    desired=stages(teacher.prior,received,teacher_context)[0]
    student_context=student.frozen.estimate(received)
    return desired,student_context
