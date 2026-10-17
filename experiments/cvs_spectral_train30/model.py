"""Exact native four-view graph and default head;replace only the training IQ operator."""
from experiments.cvs_clock_factorial30 import model as base
from . import design as d
from .filter import FixedFilter
classifier,source_domain_count,export=base.classifier,base.source_domain_count,base.export
def attach(m,c):
 m._full_config=c;m._clock=base.Shifted(FixedFilter());m._feature_learning_report.pop('clock',None);m._feature_learning_report['clock_factorial']=c['factorial']
 m._feature_learning_report['first_fixed_FIR_training']=dict(policy=c['filter_policy'],source_parent=c['parent_source'],new_parameters=0,new_NN=0,default_head_joint=True,Ag_introduced_only_E21to30=True)
 return m
def build(c,device='cpu'):
 d.validate(c);return attach(base.build(d.nominal_config(c['arm']),device),c)
def install(m,c):
 d.validate(c);return attach(base.install(m,d.nominal_config(c['arm'])),c)
