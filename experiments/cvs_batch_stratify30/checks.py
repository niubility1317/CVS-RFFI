"""Protocol negatives and actual loader alignment/RNG/exposure checks;zero neural calls."""
import copy,json,tempfile,gzip,random
from pathlib import Path
import torch
from . import design as d
from .sampler import RepartitionLoader,IsolatedLoader,repartition,join
from .checkpoint_policy import load
def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 base=d.config(d.rows()[0])
 for k,v in [('epochs',11),('updates',2442),('cumulative_epochs',31),('model_seed',2026092702),('new_Ag',True),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1),('parent_source','other')]:
  c=copy.deepcopy(base);c[k]=v;bad(lambda c=c:d.validate(c))
 for k,v in [('source_RX_holdout',1),('L',5580),('RX',[3,4,6,8]),('U_TX_truth_hidden',False)]:
  c=copy.deepcopy(base);c['data_contract'][k]=v;bad(lambda c=c:d.validate(c))
 for k,v in [('pool_batches',7),('method','U_BALANCED'),('L_loader_seed',0)]:
  c=copy.deepcopy(base);c['batch_policy'][k]=v;bad(lambda c=c:d.validate(c))
 sentinel=object();calls=[]
 def actual(path,device):calls.append(path);return 'OWNFINAL'
 assert load('/p/final','cpu','/p/final','/r/final',sentinel,actual) is sentinel and not calls
 assert load('/r/final','cpu','/p/final','/r/final',sentinel,actual)=='OWNFINAL'
 bad(lambda:load('/other/final','cpu','/p/final','/r/final',sentinel,actual))
 class Fake:
  num_workers=0
  def __len__(self):return 7
  def __iter__(self):
   order=torch.randperm(896)
   for offset in range(0,896,128):
    k=order[offset:offset+128];y=k%6;rx=torch.tensor([d.FIT[int(z)%5] for z in k]);day=k%3+1;x=torch.zeros(128,2,256);x[:,0,0]=k.float();meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=day,sig_i=k,base_index=k,native_string=[str(int(z)) for z in k]);yield x,y,torch.tensor([d.FIT.index(int(r))*3+int(z)-1 for r,z in zip(rx,day)]),meta
 state=torch.random.get_rng_state().clone();python=random.getstate();reference=None;allpool=[]
 with tempfile.TemporaryDirectory() as folder:
  for mode in d.MODES:
   path=Path(folder)/(mode+'.gz');sampler=RepartitionLoader(Fake(),mode,seed=d.REPARTITION_SEED,loader_seed=d.L_LOADER_SEED,output=path)
   before=torch.random.get_rng_state().clone()
   for epoch in range(2):
    iterator=iter(sampler)
    for step in range(222):
     x,y,domain,meta=next(iterator);k=meta['base_index'];assert torch.equal(x[:,0,0],k.float()) and torch.equal(y,k%6) and torch.equal(meta['sig_i'],k) and meta['native_string']==[str(int(z)) for z in k]
   sampler.close();assert sampler.emitted_batches==444 and sampler.pools==74 and sampler.input_occurrences==56832 and torch.equal(before,torch.random.get_rng_state())
   with gzip.open(path,'rt') as f:logs=[json.loads(s) for s in f]
   inputs=[v['original_L_ids'] for v in logs]
   if reference is None:reference=inputs
   else:assert inputs==reference
   for v in logs:assert sorted(v['permutation'])==list(range(768))
   allpool+=logs
  u=IsolatedLoader(Fake(),d.U_LOADER_SEED);a=next(iter(u));assert a[0].shape==(128,2,256) and torch.equal(before,torch.random.get_rng_state())
  try:u.advance(lambda:(_ for _ in ()).throw(RuntimeError('synthetic')))
  except RuntimeError:pass
  else:raise AssertionError('Expected synthetic error')
  assert torch.equal(before,torch.random.get_rng_state())
 assert torch.equal(state,torch.random.get_rng_state()) and random.getstate()==python
 return dict(status='PASS',protocol_negatives=negative,synthetic_output_batches=1332,synthetic_pools=222,complete_epoch_boundaries=6,input_pools_identical_all_modes=True,exact_occurrences_and_alignment=True,RNG_isolated_and_exception_restored=True,IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
