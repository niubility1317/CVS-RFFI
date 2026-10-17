"""Reuse the immutable F/default-head training math in this isolated process."""
from . import design as d
from experiments.cvs_fullsource_concat22 import model as base
base.d=d
source_domain_count,classifier,install,build,own_model=(getattr(base,k) for k in ('source_domain_count','classifier','install','build','own_model'))
