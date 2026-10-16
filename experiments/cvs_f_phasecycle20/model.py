from types import MethodType
from . import design as d
from experiments.cvs_f_crossrx_practical20.model import hook
from experiments.cvs_f_crossrx_practical20.supcon import attach as attach_supcon
from experiments.cvs_raw_anchor20 import model as raw
source_domain_count,classifier=raw.source_domain_count,raw.classifier
def attach(m,c):
    m=raw.attach(m,c,validator=d.validate);m._source_labeled_feature_hook=MethodType(hook,m)
    m._pair_primary=None;m._pair_secondary=None
    m._feature_learning_report['baseline_CROSS_RX']=c['supcon']
    return attach_supcon(m,c)
def install(m,c):return attach(raw.F_configure(raw.base.legacy.install(m,d.parent_config()),c),c)
def build(c,device='cpu'):return attach(raw.F_configure(raw.base.legacy.build(d.parent_config(),'cpu'),c).to(device),c)
