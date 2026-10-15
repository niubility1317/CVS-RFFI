from . import design as d
from experiments.cvs_evidence_responsibility_train20 import model as base
from experiments.cvs_sigtime_cleanpath20.mechanism import install as place_noise
from experiments.cvs_feature_sequence50.features import SequenceReference
classifier=base.classifier;source_domain_count=base.source_domain_count
def placement(model,c):
    refs=[m for m in model.modules() if type(m) is SequenceReference]
    if len(refs)!=1:raise ValueError('One actual original SIG_TIME reference required')
    model._feature_learning_report['t2_noise_placement']=place_noise(refs[0],c['arm']);return model
def install(model,c):
    d.validate(c);return placement(base.install(model,d.parent_config(c['model_seed'])),c)
def build(c,device):
    d.validate(c);return placement(base.build(d.parent_config(c['model_seed']),device),c)
