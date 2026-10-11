import argparse,json
import torch
from . import design as d
from .spectral import observations,fit_contrast,audit

def check():
 torch.manual_seed(104);torch.set_num_threads(2)
 tx=torch.randn(6,1,1,8,dtype=torch.float64);day=torch.randn(1,3,1,8,dtype=torch.float64);rx=torch.randn(1,1,5,8,dtype=torch.float64)
 means=tx+day+rx;halves=torch.stack([means,means]);r=audit(means,halves)
 assert r['relative_MSE']['prediction']<1e-20 and r['folds_better_than_zero']==18
 changed=means.clone();changed[0]+=17*torch.randn_like(changed[0]);changed[:,0]+=13*torch.randn_like(changed[:,0])
 assert torch.equal(fit_contrast(means,0,0),fit_contrast(changed,0,0))
 independent=tx+day+rx*torch.tensor([1.,-1.,1.,-1.,1.,-1.])[:,None,None,None]
 assert audit(independent,torch.stack([independent,independent]))['relative_MSE']['prediction']>1
 x=torch.randn(10,2,256);z=torch.complex(x[:,0],x[:,1])*complex(.6,.8);rotated=torch.stack([z.real,z.imag],1)
 for window in ('rectangular','hann'):
  original=observations(x,window);gain=observations(3*x,window);phase=observations(rotated,window)
  for k in original:
   assert torch.allclose(original[k],gain[k],atol=1e-6) and torch.allclose(original[k],phase[k],atol=1e-6)
 return dict(status='PASS',additive_RX_recovery=True,heldout_TX_and_day_excluded=True,TX_RX_interaction_counterexample=True,gain_phase_invariance=True,target_access=False)

def preflight():
 from .run import load_l
 result=check();L,p=load_l(d.config(d.rows()[0]));observations(L['x'][:8],'rectangular')
 result['actual_L_count']=len(L['ids']);result['checkpoint_loaded']=False
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();result=preflight() if a.remote_preflight else check();d.write(a.output,result);print(json.dumps(result))
