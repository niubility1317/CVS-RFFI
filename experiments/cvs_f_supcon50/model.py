"""Unchanged actual F extractor and default head, with L-only training objective."""
from experiments.cvs_rdrop_sinc_units import model as base
from . import design as d
from .mechanism import attach
classifier=base.classifier;source_domain_count=base.source_domain_count
def complete(m,c):
 m._feature_learning_report['original_SIGTIME_RDrop']=c['training_mechanism'];m._feature_learning_report['paired_source_SupCon']=c['supcon']
 return attach(m,c)
def install(m,c):d.validate(c);return complete(base.install(m,d.base_config(c['model_seed'])),c)
def build(c,device='cpu'):d.validate(c);return complete(base.build(d.base_config(c['model_seed']),device),c)
export=base.export
