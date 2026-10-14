"""Freeze-time source summary only; no fitted diagnostic heads."""
from . import design as d
from .provenance import source_proof

def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row))
 p=source_proof(c);e=d.read(p/'source_training_mechanism.json')
 result=dict(status='SOURCE_MECHANISM_FROZEN',config=c,mechanism=e,
             default_head_only=True,head_fit_updates=0,source_selection='fixed final50',
             target_access=False)
 out=p.parent/'diagnostics'
 if (out/'summary.json').exists():
  previous=d.read(out/'summary.json')
  if previous!=result:raise ValueError('Existing source diagnostic differs; preserve')
  return previous
 out.mkdir(exist_ok=False)
 d.write(out/'summary.json',result)
 return result
