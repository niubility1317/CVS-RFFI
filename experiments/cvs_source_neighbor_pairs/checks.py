import copy,json
from types import SimpleNamespace
import torch
from . import design as d
from .order import order,physical_pairs
from experiments.cvs_source_gate_audit.gates import masks
from experiments.cvs_phase1_overlay.model import native_modules

def checks():
 torch.set_num_threads(2);native=native_modules();n=1032
 rx=torch.tensor([1]*516+[3]*516);day=torch.tensor(([1]*258+[2]*258)*2);sig=torch.arange(258).repeat(4)+70
 meta=dict(rx_i=rx,day_i=day,eq_i=torch.zeros(n,dtype=torch.long),sig_i=sig,base_index=torch.cat([torch.arange(258)+i*1000+70 for i in range(4)]))
 before=torch.random.get_rng_state().clone();random=order(meta,2026101410,'RANDOM');paired=order(meta,2026101410,'PAIRS')
 assert torch.equal(before,torch.random.get_rng_state())
 assert torch.equal(random,torch.randperm(n,generator=torch.Generator().manual_seed(2026101410)))
 assert torch.equal(paired,order(meta,2026101410,'PAIRS')) and torch.equal(paired.sort().values,torch.arange(n))
 args=d.old.make_args(d.source_config(d.rows()[0]),'cpu');geometry=[]
 for begin in range(0,n,256):
  ix=paired[begin:begin+256];logits=torch.zeros(len(ix),6);logits[:,0]=12
  domains=(meta['rx_i'][ix]==3).long()*3+meta['day_i'][ix]-1
  _,selected,_=masks(native,logits,logits,{k:v[ix] for k,v in meta.items()},domains,args)
  assert selected['GEOMETRIC_NEIGHBOR'].all() and selected['FULL_OLD'].all();geometry.append(len(ix))
 assert geometry==[256]*4+[8]
 negative=0
 def reject(fn):
  nonlocal negative
  try:fn()
  except ValueError:negative+=1
  else:raise AssertionError('Invalid pair/source input accepted')
 extra={**meta,'tx_i':torch.zeros(n,dtype=torch.long)};reject(lambda:order(extra,1,'PAIRS'))
 reject(lambda:order(meta,1,'UNKNOWN'))
 odd={k:v[:-1] for k,v in meta.items()};reject(lambda:order(odd,1,'PAIRS'))
 bad=copy.deepcopy(meta);bad['base_index'][1]=bad['base_index'][0];reject(lambda:physical_pairs(bad))
 bad=copy.deepcopy(meta);bad['eq_i'][1]=1;reject(lambda:physical_pairs(bad))
 bad=copy.deepcopy(meta);bad['sig_i'][1]+=10;reject(lambda:physical_pairs(bad))
 bad=copy.deepcopy(meta);bad['rx_i']=bad['rx_i'].float();reject(lambda:order(bad,1,'RANDOM'))
 c=d.config(d.rows()[0]);badc=copy.deepcopy(c);badc['U_IQ_read']=True;reject(lambda:d.validate(badc))
 badc=copy.deepcopy(c);badc['roles']=['U_s'];reject(lambda:d.validate(badc))
 badc=copy.deepcopy(c);badc['source_criterion']['each_V_full_precision_min']=0;reject(lambda:d.validate(badc))
 return dict(status='PASS',physical_metadata_only=True,all_packets_once=True,tail_pairs_preserved=True,original_gates_full_visibility=True,random_control_identical_to_original_order=True,private_generator_no_global_RNG_change=True,negative_checks=negative,synthetic_only=True,no_real_IQ_or_query=True,head_fit=False,optimizer_steps=0)
if __name__=='__main__':print(json.dumps(checks()))
