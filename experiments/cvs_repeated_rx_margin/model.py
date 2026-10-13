from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
def build(c,device):d.validate(c);return native(c['source_loader_config'],device)
def groups(model,c):return native_groups(model,c['source_loader_config'])
