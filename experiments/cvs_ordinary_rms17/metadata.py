"""Native configuration compatibility and JSON-only logging of legacy defaults."""
import copy,math
from . import design as d

def validate_base(actual,seed):
 if seed not in d.SEEDS:raise ValueError('Unregistered frozen modelseed')
 expected=copy.deepcopy(d.read(d.ROOT/'experiments/cvs_ordinary_rms17/reference_aug.json'));expected['seed']=seed
 if actual!=expected:raise ValueError('Original ordinary config differs')

def native_args_for_log(value):
 nonfinite=[]
 def visit(v,path):
  if isinstance(v,float) and not math.isfinite(v):nonfinite.append(dict(path=path,kind='nan' if math.isnan(v) else '+inf' if v>0 else '-inf'));return None
  if isinstance(v,dict):return {k:visit(x,path+'.'+str(k) if path else str(k)) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [visit(x,path+'['+str(i)+']') for i,x in enumerate(v)]
  return v
 return visit(value,''),nonfinite
