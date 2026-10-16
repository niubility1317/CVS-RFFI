from . import design as d
from experiments.cvs_ordinary_fold20 import model as base
source_domain_count,classifier=base.source_domain_count,base.classifier
def install(m,c):return base.attach(base.legacy.install(m,d.parent_config()),c,validator=d.validate)
def build(c,device):return base.attach(base.legacy.build(d.parent_config(),device),c,validator=d.validate)
