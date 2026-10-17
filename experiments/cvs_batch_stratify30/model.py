"""Exact original practical F model/objective;only source loader changes."""
from experiments.cvs_clock_factorial30 import model as original
from . import design as d
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export
def note(m,c):
 m._feature_learning_report['L_batch_repartition']=dict(mode=c['batch_mode'],policy=c['batch_policy'],new_parameters=0,new_forwards=0,new_losses=0,new_IQ_Ag=False);return m
def install(m,c):
 d.validate(c);return note(original.install(m,d.nominal_config()),c)
def build(c,device='cpu'):
 d.validate(c);return note(original.build(d.nominal_config(),device),c)
