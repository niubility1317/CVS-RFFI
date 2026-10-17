"""Same legal full-source compatibility CP, never candidate initialization."""
from . import design as d,model
from experiments.cvs_fullsource_concat22 import preflight as base
base.d=d
smoke=base.smoke
