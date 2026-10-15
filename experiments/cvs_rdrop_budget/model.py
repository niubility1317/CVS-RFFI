from types import MethodType
from . import design as d
from experiments.cvs_evidence_responsibility_train20 import model as base
from experiments.cvs_rdrop_fold20.model import hook,terms
classifier=base.classifier;source_domain_count=base.source_domain_count
def attach(model,c):
 model._rdrop_alpha=c['training_mechanism']['alpha'];model._rdrop_calls=0;model._rdrop_extra_packets=0
 model._source_labeled_feature_hook=MethodType(hook,model)
 model._feature_learning_report['paired_dropout_objective']=c['training_mechanism'];return model
def install(model,c):
 d.validate(c);return attach(base.install(model,d.parent_config(c['model_seed'])),c)
def build(c,device):
 d.validate(c);return attach(base.build(d.parent_config(c['model_seed']),device),c)
