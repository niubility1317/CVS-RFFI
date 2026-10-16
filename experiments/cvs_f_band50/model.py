"""Exactly the original SIG_TIME/RDrop and unified coordinates; no new head."""
from experiments.cvs_rdrop_sinc_units import model as base
from . import design as d
classifier=base.classifier;source_domain_count=base.source_domain_count

def complete(m,c):
 m._feature_learning_report['original_SIGTIME_RDrop']=c['training_mechanism'];m._feature_learning_report['Sinc_band_step_policy']=c['band_step_policy'];m._feature_learning_report['training_budget']=dict(epochs=50,label=33,pseudo=17)
 return m
def install(m,c):d.validate(c);return complete(base.install(m,d.base_config(c['model_seed'])),c)
def build(c,device='cpu'):d.validate(c);return complete(base.build(d.base_config(c['model_seed']),device),c)
export=base.export
