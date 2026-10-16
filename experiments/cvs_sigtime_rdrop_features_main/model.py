from types import MethodType
from . import design as d
from experiments.cvs_sigtime_rdrop_legacy import model as original
from experiments.cvs_sigtime_rdrop_gain20 import mechanism as f
from experiments.cvs_sigtime_rdrop_gain20.model import hook
classifier,source_domain_count=original.classifier,original.source_domain_count
def attach(m,c):
 m._gain_arm,m._gain_calls,m._gain_audit=c['mode'],0,f.new_audit()
 m._gain_capture_aux,m._gain_last_aux=False,None
 m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['rdrop_gradient_gain']=c['training_mechanism']
 return m
def install(m,c):d.validate(c);return attach(original.install(m,d.native_config(c)),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(c),device),c)
