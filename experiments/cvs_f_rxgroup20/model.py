"""The exact deployed six-forward F and default head;no new model state."""
from . import design as d
from experiments.cvs_crossrx_satmix20 import model as parent
classifier,source_domain_count,export=parent.classifier,parent.source_domain_count,parent.export
def attach(m,c):
 d.validate(c);m._feature_learning_report.update(ordinary_coupling=c['coupling'],deployed_structure_unchanged=True);return m
def build(c,device='cpu'):return attach(parent.build(d.template(),device),c)
def install(m,c):return attach(parent.install(m,d.template()),c)
