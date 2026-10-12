import argparse,math,tempfile,torch
from pathlib import Path
from . import design as d
from .operators import apply,quality,rms
from .splits import make
from .predictors import fit,predict
from .observations import extract
from .metrics import metric,coverage
def synthetic_L():
 tx=[];rx=[];day=[]
 for t in range(6):
  for r in (1,3,4,6,8):
   for v in (1,2,3):tx.extend([t]*70);rx.extend([r]*70);day.extend([v]*70)
 return dict(role='L_s',ids=[f'L-{i:06}' for i in range(6300)],y=torch.tensor(tx),rx=torch.tensor(rx),day=torch.tensor(day))
def run(device):
 torch.set_num_threads(2);g=torch.Generator().manual_seed(33);x=torch.randn(16,2,256,generator=g).to(device);rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []
 for kind in d.KINDS:
  assert torch.allclose(apply(x,kind,[0.,0.],1),x,atol=1e-6,rtol=1e-6)
  for a in d.POINTS[kind]:
   y=apply(x,kind,a,11);assert y.shape==x.shape and torch.isfinite(y).all() and torch.equal(y,apply(x,kind,a,11))
   if kind!='noise':assert torch.allclose(rms(y),rms(x),atol=1e-6,rtol=1e-6)
 assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
 L=synthetic_L();c=d.config(d.rows()[0]);q=torch.randn(6300,3,generator=g);split,proof=make(L,q,c)
 assert len(split)==13 and len(split['PID80_20'][0])==5040
 for tx in range(6):assert tx not in L['y'][split[f'TX{tx}'][0]].tolist()
 h=torch.randn(256,8,generator=g);qq=torch.randn(256,3,generator=g);a=torch.tensor([[-1.,0.],[1.,0.]]*128);target=h*a[:,:1];var=torch.ones_like(target)*.02
 mini=dict(c,fit_batch_size=64,ridge_relative=1e-6);fitted=fit(h[:192],qq[:192],a[:192],target[:192],var[:192],mini,'noise',119,device)
 assert fitted['updates']==60 and torch.allclose(fitted['state']['h']['mean'].cpu(),h[:192].mean(0),atol=1e-7,rtol=1e-6)
 errors={}
 for method in c['methods']:
  mu,v=predict(fitted,h[192:],qq[192:],a[192:],method,device);m=metric(mu,v,target[192:],var[192:],'noise');assert all(value is None or not isinstance(value,float) or math.isfinite(value) for value in m.values());errors[method]=m['mean_prediction_MSE']
 assert errors['bilinear_ridge']<1e-8 and errors['bilinear_ridge']<errors['parameter_linear']
 clean=torch.randn(6300,8,generator=g);fit_ids,ev=split['PID80_20'];v=coverage(clean[fit_ids],clean,L,fit_ids,ev,6);assert v['rank']==6 and 0<=v['coverage_fraction']<=1 and v['fit_eval_PID_disjoint']
 try:coverage(clean[ev],clean,L,ev,ev,6)
 except ValueError:pass
 else:raise AssertionError('Coverage physical leakage accepted')
 # Actual native maps/known operations are all reachable without real data.
 from experiments.cvs_nocfo_foundation.model import build
 from .state import snapshot
 fc=c['source_data_config'];model=build(fc,device).eval().requires_grad_(False);before=snapshot(model)
 small=dict(role='L_s',ids=[f'synthetic-{i}' for i in range(16)],x=x.cpu(),y=torch.arange(16)%6)
 obs=extract(model,small,c,device,'L_s');assert obs['counts']['packet_occurrences']==240 and set(obs['h'])==set(d.LEVELS)
 u=dict(role='U_s',ids=small['ids'],x=small['x']);uv=extract(model,u,c,device,'U_s');assert uv['counts']['packet_occurrences']==112 and uv['clean_head_accuracy'] is None
 try:extract(model,dict(u,y=small['y']),c,device,'U_s')
 except ValueError:pass
 else:raise AssertionError('U truth accepted')
 after=snapshot(model);assert all(torch.equal(before[k],after[k]) for k in before) and not any(p.grad is not None for p in model.parameters())
 return dict(status='PASS',device=str(device),synthetic_only=True,operator_zero_rms_private_rng=True,splits=13,held_TX_fit_independent=True,coverage_PID_leak_rejected=True,strong_bilinear_MSE=errors['bilinear_ridge'],auxiliary_fit_updates=60,actual_native_L_readouts=240,actual_native_U_readouts=112,reference_state_unchanged=True,U_truth_rejected=True,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(v)
