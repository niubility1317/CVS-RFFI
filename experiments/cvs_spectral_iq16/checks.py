"""Synthetic IQ and additive/interaction falsification;no real data or model."""
import copy,math,torch
from . import design as d
from .spectral import basis,observe,fit,audit,grid
from .dsp import bank,apply,response
def run():
 torch.set_num_threads(2);g=torch.Generator().manual_seed(20261016301);negative=0
 for key,value in (('L',6300),('held_RX1_used',True),('training_updates',1),('total_NN_packets',1)):
  c=copy.deepcopy(d.config());c[key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Changed source scope/budget accepted')
 effects=torch.randn(4,8,generator=g,dtype=torch.float64)*.1;effects-=effects.mean(0)
 tx=torch.randn(6,1,1,8,generator=g,dtype=torch.float64);day=torch.randn(1,3,1,8,generator=g,dtype=torch.float64);m=tx+day+effects
 a=audit(m);assert a['relative_MSE']<1e-25 and a['better_folds']==18 and a['prediction_SSE']<a['control_SSE']
 changed=m.clone();changed[2]+=3;changed[:,1]-=2;assert torch.allclose(fit(changed,2,1),fit(m,2,1),atol=1e-14,rtol=0)
 interaction=torch.tensor([1.,1.,1.,-1.,-1.,-1.])[:,None,None,None]*effects[None,None].expand(6,3,4,8);assert audit(interaction)['relative_MSE']>1
 B=basis();assert torch.allclose(B.T@B,torch.eye(8,dtype=torch.float64),atol=1e-14,rtol=0)
 v=torch.randn(5040,8,generator=g,dtype=torch.float64);y=torch.arange(6).repeat_interleave(840);days=torch.tensor([1,2,3]).repeat_interleave(280).repeat(6);rx=torch.tensor(d.FIT).repeat_interleave(70).repeat(18)
 assert grid(v,y,rx,days).shape==(6,3,4,8)
 for yy,rr in ((torch.full_like(y,-1),rx),(y,torch.ones_like(rx))):
  try:grid(v,yy,rr,days)
  except ValueError:negative+=1
  else:raise AssertionError('Hidden labels/held RX accepted')
 raw=torch.randn(16,2,256,generator=g);before=raw.clone();raw_obs=observe(raw);raw_fft=torch.fft.fft(torch.complex(raw[:,0].double(),raw[:,1].double()));kernels=bank(effects)
 assert len(kernels)==24
 max_bound=max_rmse=max_interior=max_power=0.
 for name,item in kernels.items():
  out,metrics=apply(raw,item,raw_obs,raw_fft);meta=item['metadata'];max_bound=max(max_bound,meta['grid_response_max_abs_dB']);max_rmse=max(max_rmse,meta['kernel_relative_RMSE']);max_interior=max(max_interior,metrics['interior_relative_MSE']);max_power=max(max_power,metrics['RMS_relative_error_max'])
  assert out.shape==raw.shape and torch.equal(raw,before) and metrics['interior_relative_MSE']<1e-20 and metrics['RMS_relative_error_max']<1e-6 and meta['grid_response_max_abs_dB']<=.5+1e-12 and meta['grid_phase_max_abs_rad']<1e-12
  if name.startswith('structured'):
   control=kernels[name.replace('structured','control',1)]['metadata'];assert math.isclose(meta['perturbation_energy'],control['perturbation_energy'],rel_tol=1e-12,abs_tol=1e-20)
 zero=bank(torch.zeros(4,8,dtype=torch.float64));out,meta=apply(raw,zero[next(iter(zero))],raw_obs,raw_fft);assert torch.equal(out,raw) and meta['boundary_error_energy']<1e-24
 for bad in (raw[:,:,:128],torch.full_like(raw,float('nan')),torch.zeros_like(raw)):
  try:apply(bad,kernels[next(iter(kernels))],raw_obs,raw_fft)
  except ValueError:negative+=1
  else:raise AssertionError('Invalid filter IQ accepted')
 return dict(status='PASS',negative_cases=negative,additive_RX_recovered=True,excluded_TX_AND_day_not_fitted=True,interaction_counterexample_rejected=True,complex_linear_centered_convolution_matches_circular_interior=True,structured_control_kernel_perturbation_energy_matched=True,max_grid_abs_dB=max_bound,max_kernel_relative_RMSE_synthetic=max_rmse,max_interior_relative_MSE=max_interior,max_RMS_relative_error=max_power,identity_kernel_exact=True,source_IQ_reads=0,checkpoint_loads=0,model_forwards=0,optimizer_updates=0,synthetic_filter_packets=400,synthetic_oracle_packets=400)
if __name__=='__main__':
 import json
 value=run();d.write(d.ROOT/d.LOCAL/'local_checks.json',value);print('LOCAL_RESULT '+json.dumps(value))
