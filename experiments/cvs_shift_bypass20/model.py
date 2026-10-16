"""Exact original F+CROSS_RX B00 model/objective,no new deployment module."""
from . import design as d
from experiments.cvs_crossrx_satmix20 import model as original
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export
def install(m,c):d.validate(c);return original.install(m,d.original_config())
def build(c,device='cpu'):d.validate(c);return original.build(d.original_config(),device)
