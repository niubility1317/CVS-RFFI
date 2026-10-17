"""Use the same F/joint160/defaultcos; new IQ routing adds no model state."""
from . import design as d
from experiments.cvs_fullsource_concat30_25 import model as base
base.d=d
source_domain_count,classifier,build,install,own_model=(getattr(base,k) for k in ('source_domain_count','classifier','build','install','own_model'))
