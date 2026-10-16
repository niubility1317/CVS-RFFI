from . import design as d
from experiments.cvs_raw_anchor20 import model as original
source_domain_count,classifier=original.source_domain_count,original.classifier
def install(m,c):
    return original.attach(original.F_configure(original.base.legacy.install(m,d.parent_config()),c),c,validator=d.validate)
def build(c,device):
    return original.attach(original.F_configure(original.base.legacy.build(d.parent_config(),'cpu'),c).to(device),c,validator=d.validate)
