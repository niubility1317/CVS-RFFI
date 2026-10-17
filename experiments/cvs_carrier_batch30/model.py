"""Unchanged F/default head/SupCon/normal4forward joint training."""
from . import design as d
from experiments.cvs_clock_factorial30 import model as previous
classifier,source_domain_count,export=previous.classifier,previous.source_domain_count,previous.export
def describe(m,c):
 m._feature_learning_report['E30_amendment']=dict(parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,optimizer='resetcos10',EMA='sameNoAsparent',no_new_parameters=True)
 m._feature_learning_report['carrier_organization']=c['carrier_policy']
 m._full_config['parent_source']=c['parent_source'];return m
def install(m,c):d.validate(c);return describe(previous.install(m,d.model_config(c)),c)
def build(c,device='cpu'):d.validate(c);return describe(previous.build(d.model_config(c),device),c)
