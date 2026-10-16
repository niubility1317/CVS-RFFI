"""Focused protocol negatives and analytic IQ counterexamples; no real source/model."""
import copy,math,random
import torch
from . import design as d
from .phase import apply,generate,project_bound
from .decision import qualify

def run():
 negatives=0;science=0;calls=0
 def reject(fn):
  nonlocal negatives
  try:fn()
  except (ValueError,TypeError,RuntimeError):negatives+=1;return
  raise AssertionError('Invalid input accepted')
 torch.set_num_threads(2);torch.manual_seed(202610173201);x=torch.randn(64,2,256);phi=torch.randn(64,256,dtype=torch.float64).mul(.006).cumsum(1)
 for bad in [None,torch.ones(2,256),torch.ones(1,1,256),torch.ones(1,2,255),torch.ones(0,2,256),x.double(),x.clone().requires_grad_(),x.clone().fill_(float('nan')),x.clone().fill_(float('inf')),torch.zeros_like(x)]:reject(lambda bad=bad:apply(bad,phi))
 for bad in [torch.ones(64,255,dtype=torch.float64),torch.ones(64,256),torch.ones(0,256,dtype=torch.float64),phi.clone().requires_grad_(),phi.clone().fill_(float('nan')),phi[:1]]:reject(lambda bad=bad:apply(x,bad))
 for k,val in [('target_access',True),('U_used',True),('held_RX1_used',True),('phase_peak_bound_rad',.2),('phase_increment_sigma_rad',.003),('affine_phase_removed',False),('fit_RX',[3,4,6]),('automatic_training',True)]:
  c=d.config();c[k]=val;reject(lambda c=c:d.validate(c))
 saved=x.clone();py=random.getstate();ts=torch.random.get_rng_state();out,s=apply(x,phi);calls+=64;assert torch.equal(x,saved) and torch.equal(ts,torch.random.get_rng_state()) and py==random.getstate();theta=torch.tensor(s['phase_rad'],dtype=torch.float64);t=torch.arange(256,dtype=torch.float64)-127.5
 # Affine input phase must give identity and not reintroduce CFO.
 affine=.7+.004*t;z,ss=apply(x,affine.repeat(64,1));calls+=64;assert torch.equal(z,x) and max(ss['phase_peak_rad'])<1e-13;science+=1
 z2,s2=apply(x,phi+affine);calls+=64;assert torch.allclose(z2,out,atol=1e-6,rtol=1e-6);science+=1
 # Exact known quadratic curvature, unit modulus, and no wrap/window edge.
 q=.04*((t/127.5)**2);th,sc=project_bound(q[None]);expected=q-q.mean();assert torch.max(torch.abs(th[0]-expected))<1e-15 and sc.item()==1;assert torch.max(torch.abs((2*th[:,1:-1]-th[:,:-2]-th[:,2:])+2*.04/127.5**2))<1e-15;science+=1
 assert max(s['phase_peak_rad'])<=.1+1e-12 and max(abs(v) for v in s['phase_mean_rad'])<1e-12 and max(abs(v) for v in s['phase_linear_slope_rad_per_sample'])<1e-12;assert max(abs(v) for v in s['relative_power_change'])<1e-6;science+=1
 # Positive gain commutes with this phase observation; labels are not consumed.
 z3,_=apply(x*1.7,phi);calls+=64;assert torch.allclose(z3,out*1.7,atol=1e-6,rtol=1e-6);science+=1
 # All extremes retained and scaled globally, zero path retained as identity.
 extreme=torch.stack([1e5*torch.sin(t),torch.zeros(256,dtype=torch.float64)]);z4,s4=apply(x[:2],extreme);calls+=2;assert s4['phase_scale'][0]<1 and s4['phase_scale'][1]==1 and torch.equal(z4[1],x[1]);science+=1
 # Reference-window zero energy is recorded, not silently dropped.
 xx=x[:2].clone();xx[:,:,80:160]=0;z5,s5=apply(xx,phi[:2]);calls+=2;assert s5['reference_power_zero']==[True,True] and torch.equal(z5[:,:,80:160],xx[:,:,80:160]);science+=1
 # Full256 rotation analytic oracle from independent real arithmetic.
 co,si=theta.cos(),theta.sin();oracle=torch.stack((x[:,0].double()*co-x[:,1].double()*si,x[:,0].double()*si+x[:,1].double()*co),1).float();assert torch.equal(oracle,out);science+=1
 # Generator draws only its declared256 normals and consumes no Python RNG.
 torch.manual_seed(901);before=torch.random.get_rng_state();py=random.getstate();z6,sm=generate(x);calls+=64;after=torch.random.get_rng_state();torch.random.set_rng_state(before);known=torch.randn((64,256),dtype=torch.float64).mul(.006).cumsum(1);oracle,_=apply(x,known);calls+=64;assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(z6,oracle) and py==random.getstate();science+=1
 raw=dict(accuracy=.99,macro_F1=.99,by_RX={str(v):dict(accuracy=.99) for v in d.FIT},by_TX={str(v):dict(accuracy=.99) for v in range(6)},by_day={str(v):dict(accuracy=.99) for v in (1,2,3)});good=copy.deepcopy(raw);good['raw_correct_retention']=.999
 def verdict(v,draws=None,iq=None):return qualify(raw,v,[copy.deepcopy(v) for _ in range(4)] if draws is None else draws,dict(relative_IQ_RMS_change=[.02]*20160 if iq is None else iq))['decision']
 assert verdict(good)=='PASS'
 for mode in ('acc','F1','retention','RX','TX','day','one_draw','zero_IQ','large_IQ'):
  v=copy.deepcopy(good);draws=None;iq=None
  if mode=='acc':v['accuracy']=.984
  elif mode=='F1':v['macro_F1']=.984
  elif mode=='retention':v['raw_correct_retention']=.989
  elif mode in ('RX','TX','day'):v['by_'+mode][next(iter(v['by_'+mode]))]['accuracy']=.979
  elif mode=='one_draw':draws=[copy.deepcopy(v) for _ in range(4)];draws[-1]['accuracy']=.98
  elif mode=='zero_IQ':iq=[0.]*20160
  elif mode=='large_IQ':iq=[.02]*20159+[.101]
  assert verdict(v,draws,iq)=='FAIL';science+=1
 for bad in ([],[float('nan')],[float('inf')],[-1.]):reject(lambda bad=bad:qualify(raw,good,[good]*4,dict(relative_IQ_RMS_change=bad)))
 answer=dict(status='PASS',protocol_input_config_negatives=negatives,scientific_counterexamples=science,synthetic_phase_views=calls,source_IQ_reads=0,checkpoint_loads=0,NN=0,optimizer_updates=0,target_access=False)
 print(__import__('json').dumps(answer));return answer
if __name__=='__main__':run()
