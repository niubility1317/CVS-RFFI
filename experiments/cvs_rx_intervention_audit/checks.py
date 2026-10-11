import argparse,json
import torch
from . import design as d
from .actions import fir,apply,coefficients

def check():
 torch.manual_seed(713);torch.set_num_threads(2);x=torch.randn(7,2,256);delta=torch.randn(7,8)
 for taps in (17,33):
  identity,_=apply(x,delta,0.,taps);assert torch.allclose(identity,x,atol=1e-6)
  result,info=apply(x,delta,.5,taps);assert torch.allclose(result.square().sum((1,2)),x.square().sum((1,2)),rtol=1e-5)
  h,_=fir(delta,.5,taps);z=torch.complex(x[:,0],x[:,1]);reference=torch.zeros_like(z);half=taps//2
  for j in range(taps):
   lag=j-half
   if lag>=0:reference[:,lag:]+=h[:,j:j+1]*z[:,:256-lag]
   else:reference[:,:256+lag]+=h[:,j:j+1]*z[:,-lag:]
  reference*=info['normalization_gain'][:,None]
  assert torch.allclose(torch.complex(result[:,0],result[:,1]),reference,atol=2e-6)
  assert info['interior_residual_fraction'].max()<1e-10
 # A local impulse cannot wrap to the opposite packet boundary.
 impulse=torch.zeros(1,2,256);impulse[:,0,0]=1;out,_=apply(impulse,delta[:1],1.,33);assert out[:,:,-32:].abs().max()<1e-6
 effects=torch.randn(6,3,5,8);state=dict(effects=effects,TX=torch.arange(6),day=torch.arange(3),RX=torch.tensor([1,3,4,6,8]));y=torch.tensor([0,5]);rx=torch.tensor([1,8]);day=torch.tensor([0,2])
 a=coefficients(state,y,rx,day,1,'physical');b=coefficients(state,y,rx,day,1,'rotated');assert torch.allclose(a.norm(dim=1),b.norm(dim=1))
 return dict(status='PASS',zero_action_identity=True,linear_convolution_matches_direct_sum=True,no_wraparound=True,per_packet_RMS_preserved=True,coefficient_control_norm_matched=True,target_access=False)

def preflight():
 from .run import load
 from experiments.cvs_stage2_teacher_factorial.model import build
 result=check();records=[]
 for row in d.rows():
  c=d.config(row);L,provenance=load(c);model=build(c['parent_build_config'],'cpu').eval()
  with torch.no_grad():assert model(L['x'][:8]).shape==(8,6)
  records.append(dict(row_id=row['row_id'],parent=provenance['parent'],L_count=len(L['ids'])))
 result['parents']=records;return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();result=preflight() if a.remote_preflight else check();d.write(a.output,result);print(json.dumps(result))
