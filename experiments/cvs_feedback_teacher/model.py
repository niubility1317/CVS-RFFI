from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
from experiments.cvs_stochastic_ensemble.model import build_member
def build_role(c,name,device):
 d.validate(c)
 if name=='STUDENT':return native(c['source_loader_config'],device)
 if name=='TEACHER':return build_member(dict(source_loader_config=c['source_loader_config'],member_init_seed=c['feedback_teacher']['teacher_init_seed'],member=1),device)
 raise ValueError('Unregistered teacher/student role')
def build(c,device):return build_role(c,'STUDENT',device),build_role(c,'TEACHER',device)
def groups(model,c):return native_groups(model,c['source_loader_config'])
