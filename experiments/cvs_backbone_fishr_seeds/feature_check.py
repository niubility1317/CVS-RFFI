from types import SimpleNamespace
from . import design as d,model as api
from experiments.cvs_signature_stats50.feature_check import run as shared
def run(seed=None,only_row=None):
 v=shared(seed,only_row,design=d,build_fn=api.build)
 if only_row is None and v['status']=='VERIFIED':
  shadow=SimpleNamespace(**vars(d));shadow.PRIMARY='BONE_RAW_FISHR'
  raw=shared(seed,only_row,design=shadow,build_fn=api.build)
  if raw['status']!='VERIFIED' or raw['models']!=v['models']:raise ValueError('Secondary signature feature witnesses differ')
  v['paired']+=raw['paired'];v['profile_passes']=2
 return v
