"""Use proven lineage checks and additionally require the actual row native seed."""
from experiments.cvs_practical_budget50.protocol import check_source_metadata,cumulative
from experiments.cvs_practical_budget50.protocol import check_checkpoint as original_check_checkpoint

def check_checkpoint(cp,*,seed,**kw):
 original_check_checkpoint(cp,**kw)
 if cp['args']['seed']!=seed:raise ValueError('Native training seed differs from actual row')
 return cp
