"""Focused scope negatives and exact discrete mask/deepcopy/native fallback checks."""
import copy,json
import torch
from . import design as d
from .masks import Engine,install,SIZE
from .checkpoint_policy import load

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid scope/state accepted')
 base=d.config(d.rows()[0])
 for key,value in [('epochs',11),('updates',2442),('cumulative_epochs',31),('parent_source','other'),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('mask_mode','other'),('model_seed',2026092702),('As',False),('new_Ag',True)]:
  c=copy.deepcopy(base);c[key]=value;bad(lambda c=c:d.validate(c))
 for key,value in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
  c=copy.deepcopy(base);c['data_contract'][key]=value;bad(lambda c=c:d.validate(c))
 for key,value in [('seed',1),('ordinary_pair_independent',False),('As_pair_independent',False)]:
  c=copy.deepcopy(base);c['dropout_policy'][key]=value;bad(lambda c=c:d.validate(c))
 sentinel=object();calls=[]
 def actual(path,dev):calls.append(path);return 'FINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='FINAL'
 for path in ('/other/final','/r/latest','/p/best'):bad(lambda path=path:load(path,'cpu','/p/final','/r/final',sentinel,actual))
 state=torch.random.get_rng_state().clone();results={};size=200000
 for mode in d.MODES:
  e=Engine(mode,d.MASK_SEED);e.begin(1);values=[];x=torch.ones(size,requires_grad=True)
  for slot in ('O1','O2','S1','S2'):
   with e.view(slot):values.append(e.apply(0,x,.25))
  masks=[v.detach()>0 for v in values]
  for mask in masks:assert abs(float(mask.float().mean())-.75)<.005
  assert abs(float((masks[0]&masks[1]).float().mean())-.75**2)<.005
  assert abs(float((masks[2]&masks[3]).float().mean())-.75**2)<.005
  if mode=='ANTITHETIC':
   assert bool((masks[0]|masks[2]).all()) and bool((masks[1]|masks[3]).all())
   assert abs(float((masks[0]&masks[2]).float().mean())-.5)<.005
  else:assert abs(float((masks[0]&masks[2]).float().mean())-.75**2)<.005
  record=e.finish();assert record['random_elements']==4*size and not e.cache and e.step is None
  torch.stack(values).mean().backward();assert torch.isfinite(x.grad).all()
  results[mode]=dict(random_elements=record['random_elements'],rates=[float(m.float().mean()) for m in masks],cross_overlap=float((masks[0]&masks[2]).float().mean()))
 assert torch.equal(state,torch.random.get_rng_state())
 # Reflection is a bijection; enumerate an analogous complete finite support.
 for bits in (4,8,12):
  r=torch.arange(2**bits)
  for p in (.1,.25,.5,.75):
   k=round((1-p)*len(r));a=r<k;b=(len(r)-1-r)<k
   assert int(a.sum())==int(b.sum())==k and int((a&b).sum())==max(0,2*k-len(r))
 e=Engine('ANTITHETIC',d.MASK_SEED);bad(lambda:e.view('S1').__enter__());e.begin(1);bad(lambda:e.begin(1));bad(lambda:e.view('O2').__enter__())
 with e.view('O1'):e.apply(0,torch.ones(4),.25)
 bad(lambda:e.finish())
 def wrong_shape():
  with e.view('O2'):e.apply(0,torch.ones(5),.25)
 bad(wrong_shape);e.abort();assert not e.cache
 # Installed module deepcopy must point to its own shared engine, not student.
 m=torch.nn.Sequential(torch.nn.Dropout(.25),torch.nn.Dropout(.1));before=set(m.state_dict());install(m,'ANTITHETIC',d.MASK_SEED);teacher=copy.deepcopy(m)
 assert teacher._coupling_engine is not m._coupling_engine and all(v._coupling_engine is teacher._coupling_engine for v in teacher if isinstance(v,torch.nn.Dropout)) and set(m.state_dict())==before
 x=torch.ones(100)
 for training in (False,True):
  m.train(training);torch.manual_seed(7);a=m(x);torch.manual_seed(7);b=torch.nn.functional.dropout(torch.nn.functional.dropout(x,.25,training),.1,training);assert torch.equal(a,b)
 return dict(status='PASS',protocol_negatives=negative,mask_law=results,exact_reflection_bijection=True,within_pair_independent=True,same_engine_cost=True,global_RNG_preserved=True,native_U_eval_fallback=True,deepcopy_independent=True,real_IQ=0,CP=0,NN_model_forwards=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
