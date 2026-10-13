from . import design as d
from experiments.cvs_nocfo_foundation.model import build as native,groups as native_groups
from experiments.cvs_e20_cross_style.style import _paths
from experiments.cvs_six_direction_mechanisms.sampling import replacements

def build(c,device):
    d.validate(c);model=native(c['source_loader_config'],device)
    if c['arm']!='ORDINARY_GLOBAL_GN':
        modules,paths=_paths(model);prefix=paths['time_down'].rsplit('.',1)[0];identity=modules[prefix]
        replacement=replacements(identity.t3,device)[c['sampling_layout']]
        identity.t2.pool=replacement[0];identity.t3=replacement[1]
    model._sampling_layout=c['sampling_layout']
    return model.to(device)

def groups(model,c):return native_groups(model,c['source_loader_config'])
