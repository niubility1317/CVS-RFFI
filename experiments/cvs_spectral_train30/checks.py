"""Focused fixed bank selection,DSP and matching parent/4L/E30 checks."""
import copy,json
import torch
from torch.nn import functional as F
from . import design as d
from .filter import FixedFilter
from .checkpoint_policy import load

def run():
 torch.set_num_threads(2);negative=0
 def bad(fn):
  nonlocal negative
  try:fn()
  except (ValueError,TypeError,KeyError):negative+=1
  else:raise AssertionError('Invalid contract accepted')
 for row in d.rows():
  c=d.config(row)
  for key,val in [('epochs',11),('updates',2442),('cumulative_epochs',31),('new_Ag',False),('parent_source','other'),('model_seed',2026092702),('new_losses',1),('new_U_rules',1),('new_neural_forwards',1)]:
   z=copy.deepcopy(c);z[key]=val;bad(lambda z=z:d.validate(z))
  for key,val in [('RX',[3,4,6,8]),('L',5580),('U_TX_truth_hidden',False),('source_RX_holdout',1)]:
   z=copy.deepcopy(c);z['data_contract'][key]=val;bad(lambda z=z:d.validate(z))
  z=copy.deepcopy(c);z['filter_policy']['pre_RMS_response_bound_dB']=1.;bad(lambda z=z:d.validate(z))
  assert d.Path(c['parent_source'])==d.parent_source(row['arm']) and c['new_Ag'] and c['As']==(row['arm']=='B11')
 bank={}
 for k in range(12):
  h=[0.]*33;h[16]=1.;h[15]=.002*(k+1)/12;h[17]=h[15]
  bank['structured_synthetic_'+str(k)]=dict(h_real=h,h_imag=[0.]*33,grid_response_max_abs_dB=.1)
 filt=FixedFilter(bank);bad(lambda:FixedFilter({}).load());bad(lambda:filt.transform(torch.zeros(2,2,256),[0,1]));bad(lambda:filt.transform(torch.ones(2,2,255),[0,1]));bad(lambda:filt.transform(torch.ones(2,2,256),[0,12]))
 gen=torch.Generator().manual_seed(20261017991);x=torch.randn(64,2,256,generator=gen);meta={k:[] for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
 for i in range(64):
  for k,v in zip(meta,(1,d.FIT[i%5],1+i%3,i,i)):meta[k].append(v)
 state=torch.get_rng_state().clone();before=x.clone();out=filt.apply(x,meta,21,4441);assert torch.equal(x,before) and torch.equal(state,torch.get_rng_state())
 y=[]
 for i in range(64):
  pid=tuple(meta[k][i] for k in meta);ki=filt.choice(pid,21,4441);h=filt._h[ki];xx=F.pad(x[i:i+1].double(),(16,16),mode='reflect');rr=h.real.flip(0)[None,None];im=h.imag.flip(0)[None,None];v=torch.cat((F.conv1d(xx[:,:1],rr)-F.conv1d(xx[:,1:],im),F.conv1d(xx[:,1:],rr)+F.conv1d(xx[:,:1],im)),1);v*=torch.sqrt(x[i:i+1].double().square().sum(1).mean(1)/v.square().sum(1).mean(1))[:,None,None];y.append(v.float())
 max_error=float((out-torch.cat(y)).abs().max());assert max_error<2e-6
 order=list(reversed(range(64)));other=FixedFilter(bank);again=other.apply(x[order],{k:[v[i] for i in order] for k,v in meta.items()},21,4441);assert torch.equal(again,out[order])
 for ep,st in [(20,4441),(21,4440),(22,4441),(31,6661)]:bad(lambda ep=ep,st=st:filt.apply(x,meta,ep,st))
 for key,val in [('eq_i',0),('rx_i',2),('day_i',4)]:
  z=copy.deepcopy(meta);z[key][0]=val;bad(lambda z=z:filt.apply(x,z,21,4441))
 sentinel=object();calls=[]
 def actual(p,dev):calls.append(p);return 'FINAL'
 assert load('/parent','cpu','/parent','/final',sentinel,actual) is sentinel and not calls
 assert load('/final','cpu','/parent','/final',sentinel,actual)=='FINAL' and calls==['/final']
 for p in ['/other','/best']:bad(lambda p=p:load(p,'cpu','/parent','/final',sentinel,actual))
 from experiments.cvs_phase1_overlay.model import native_modules
 native_modules()
 from . import model as api
 for row in d.rows():
  m=api.build(d.config(row),'cpu');m.eval()
  with torch.no_grad():o=m.forward_identity_only(torch.zeros(2,2,256))
  assert o['tx_logits'].shape==(2,6) and o['id_feat_joint'].shape==(2,160) and sum(v.numel() for v in m.parameters())==900722
 return dict(status='PASS',protocol_negatives=negative,synthetic_FIR_packets=128,direct_original_formula_replays=64,max_FIR_error=max_error,global_RNG_unchanged=True,input_unchanged=True,batch_permutation_invariant=True,actual_F_parameters=900722,synthetic_NN=4,real_IQ=0,CP=0,updates=0,identity_or_Ag_gain_proved=False)
if __name__=='__main__':
 import sys
 sys.path.insert(0,str(d.ROOT/'experiments/adv3b02_xuc/code'));print(json.dumps(run()))
