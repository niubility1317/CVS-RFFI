import argparse,json,torch
from pathlib import Path
from torch import nn
from .core import origin_prediction,jacobian,errors
from .run import extract,snapshot
from . import design as d
from experiments.cvs_e20_response.checks import native
from experiments.cvs_nocfo_engineering.teacher import teacher_config
class Known(nn.Module):
 def __init__(self,device):
  super().__init__();self.register_buffer('anchor',torch.zeros((),device=device));self.state={k:dict(mean=torch.tensor(v,device=device),scale=torch.tensor(s,device=device)) for k,v,s in [('h',[1.,2.,3.],[2.,3.,4.]),('q',[.2,.4],[.7,.9]),('a',[0.,0.],[1.,1.]),('y',[2.,3.,4.],[3.,4.,5.])]}
 def forward(self,h,q,a):
  hn=(h-self.state['h']['mean'])/self.state['h']['scale'];qn=(q-self.state['q']['mean'])/self.state['q']['scale'];physical=hn*a[:,:1]+.3*hn+.5+qn[:,:1]*a[:,1:2]
  return (physical-self.state['y']['mean'])/self.state['y']['scale']
def run(device):
 torch.set_num_threads(2);c=d.config(d.rows()[0]);model=native(teacher_config(d.SEEDS[0]),device).eval().requires_grad_(False);before=snapshot(model);gen=torch.Generator().manual_seed(42);x=torch.randn(8,2,256,generator=gen)*.1
 U=dict(x=x,ids=[f'u{i}' for i in range(8)],role='U_s');obs=extract(model,U,c,device,'U_s');assert obs['source_packet_occurrences']==24 and obs['model_forward_calls']==3 and obs['head'] is None
 try:extract(model,dict(U,y=torch.zeros(8,dtype=torch.long)),c,device,'U_s')
 except ValueError:pass
 else:raise AssertionError('U labels accepted')
 after=snapshot(model);assert all(torch.equal(v,after[n]) for n,v in before.items())
 reader=Known(device);h=torch.randn(16,3,generator=gen).to(device);q=torch.randn(16,2,generator=gen).to(device);a=torch.tensor([[1.,0.]],device=device).expand(16,-1);hn=(h-reader.state['h']['mean'])/reader.state['h']['scale'];actual=hn
 target=(actual-reader.state['y']['mean'])/reader.state['y']['scale'];corrected=origin_prediction(reader,h,q,a);assert torch.allclose(target,corrected,atol=1e-6)
 j=jacobian(reader,h,q,a,43,8);expected=float((1/reader.state['y']['scale']).square().sum());assert abs(j['zero_origin']['h_frobenius_squared']-expected)<1e-6 and j['zero_origin']['q_frobenius_squared']==0
 parameter=torch.zeros(3,3,device=device);m=errors(reader,parameter,h.cpu(),h.cpu(),q.cpu(),a[0],actual.cpu(),actual.cpu(),torch.arange(16),device)
 assert float(m['zero_origin_target'].max())<1e-12 and float(m['reference_target'].mean())>0 and float(m['actual_response_drift'].max())==0
 from experiments.cvs_finite_action.operators import apply
 assert all(torch.equal(x.to(device),apply(x.to(device),k,[0.,0.],42)) for k in ('spectral','timing'))
 for row in d.rows():d.validate(d.config(row))
 return dict(status='PASS',device=device,synthetic_only=True,origin_physical_units_verified=True,normalized_Jacobian_verified=True,source_native_operations_verified=True,U_truth_rejected=True,frozen_native_state_unchanged=True,identity_updates=0,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
