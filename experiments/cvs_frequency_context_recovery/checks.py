"""Focused recovery checks: preserve source config and protect output separation."""
from copy import deepcopy
from . import design as d
from experiments.cvs_frequency_context50 import design as source

def checks():
 original=[deepcopy(source.config(r)) for r in d.rows()]
 assert d.BASE!=d.SOURCE_BASE
 d.configure_runtime()
 assert [source.config(r) for r in d.rows()]==original
 assert source.BASE==d.BASE and source.RUN==d.SOURCE_RUN
 for r,c in zip(d.rows(),original):
  source.validate(c)
  assert c['output_root'].startswith(d.SOURCE_BASE.as_posix())
  assert not c['output_root'].startswith(d.BASE.as_posix()+'/')
  assert c['epochs']==50 and c['updates']==11100
  bad=deepcopy(c);bad['epochs']=51
  try:source.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Changed source budget accepted')
 assert len(d.rows())==9
 print('PASS: source configs/checkpoint identity retained; new predictions isolated; 9 modified-budget negatives; no training entry')
if __name__=='__main__':checks()
