"""Fixed7 predictions then separate scorer; no model training or target feedback."""
from . import design as d,model
from experiments.cvs_fullsource_concat22 import evaluate as base
base.d=d
predict,score=base.predict,base.score
