"""Existing noAs F/defaulthead plus neutral matched gradient measurement."""
from . import design as d
from experiments.cvs_clock_factorial30 import model as original
from experiments.cvs_view_pcgrad30 import model as view
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export
def build(c,device='cpu'):
 d.validate(c);return view.attach(original.build(d.nominal_config(),device),c)
def install(m,c):
 d.validate(c);return view.attach(original.install(m,d.nominal_config()),c)
