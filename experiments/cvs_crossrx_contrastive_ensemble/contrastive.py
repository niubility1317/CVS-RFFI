from . import design as d
from experiments.cvs_crossrx_contrastive_clean.contrastive import capture,penalty as original
def penalty(maps,y,rx,c):return original(maps,y,rx,d.parent_config(c))
