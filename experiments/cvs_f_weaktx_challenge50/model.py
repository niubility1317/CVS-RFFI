"""Exact F+CROSS_RX and existing default classifier, feature-only challenge."""
from experiments.cvs_f_supcon50 import model as original
from . import design as d,baseline
from .mechanism import attach
classifier=original.classifier;source_domain_count=original.source_domain_count;export=original.export
def install(m,c):d.validate(c);return attach(original.install(m,baseline.original_config(c['model_seed'])),c)
def build(c,device='cpu'):d.validate(c);return attach(original.build(baseline.original_config(c['model_seed']),device),c)
