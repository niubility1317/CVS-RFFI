import torch
from torch.nn import functional as F
from experiments.cvs_feature_mean_teacher50.objective import consistency as ungated,coefficient
from experiments.cvs_source_teacher_confidence.metrics import policy
def consistency(student,teacher,teacher_logits,student_logits):
 ungated(student,teacher) # Same feature/teacher/finite validation as fixed allU parent.
 if len(student)!=len(teacher_logits):raise ValueError('Same U packet feature/logit count required')
 keep=policy(teacher_logits,student_logits)['keep']
 per=.5*(F.normalize(student.float(),dim=1,eps=1e-8)-F.normalize(teacher.detach().float(),dim=1,eps=1e-8)).square().sum(1)
 return (per*keep).sum()/len(student)
