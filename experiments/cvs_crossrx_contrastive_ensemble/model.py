from . import design as d
from experiments.cvs_stochastic_ensemble.model import build_member,groups as original_groups
def build(c,device):d.validate_member(c);return build_member(c,device)
def groups(model,c):d.validate_member(c);return original_groups(model,c)
