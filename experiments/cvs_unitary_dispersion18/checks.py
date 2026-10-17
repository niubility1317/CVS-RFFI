"""Source-scope negatives and independent cosine-basis operator checks."""
import copy,math,json
import torch
from . import design as d,dsp

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 c=d.config()
 for k,v in [('V_selected',1799),('U_used',True),('target_access',True),('source_RX_holdout',1),('training_updates',1),('teacher_NN',1),('automatic_training',True),('per_cell',19),('expected_NN',21600)]:
  t=copy.deepcopy(c);t[k]=v;bad(lambda t=t:d.validate(t))
 for k,v in [('order',9),('max_abs_eigenphase_rad',.2),('window',512),('new_noise',True)]:
  t=copy.deepcopy(c);t['operator'][k]=v;bad(lambda t=t:d.validate(t))
 x=torch.ones(2,2,256,dtype=torch.float64)
 for z,a in [(x[:,:,:255],0.),(x.long(),0.),(x,.10001),(x,float('nan')),(x,torch.zeros(3,dtype=torch.float64))]:bad(lambda z=z,a=a:dsp.apply(z,a))
 from .analyze import metric
 known=metric([0,0,1,2],[0,1,1,2]);assert known['correct']==3 and known['count']==4 and known['confusion'][0][1]==1 and abs(known['macro_F1']-7/18)<1e-12
 state=torch.random.get_rng_state().clone();g=torch.Generator().manual_seed(20261018062);errors={}
 for dtype,tol in ((torch.float64,2e-12),(torch.float32,2e-6)):
  x=torch.randn(7,2,256,generator=g,dtype=dtype);x[0]=0;x[1]=1;x[2]=0;x[2,0,0]=1;x[3]=0;x[3,1,-1]=1;old=x.clone();maxenergy=0.;maxinverse=0.;maxrelative=0.
  for a in (0.,-.1,.1):
   y=dsp.apply(x,a);z=dsp.apply(y,-a);power=x.double().square().sum((1,2));active=power>0
   maxenergy=max(maxenergy,float(((y.double().square().sum((1,2))-power).abs()/power.clamp_min(1.)).max()))
   maxinverse=max(maxinverse,float(((z.double()-x).square().sum((1,2))/power.clamp_min(1.)).sqrt().max()))
   maxrelative=max(maxrelative,float(((y.double()-x).square().sum((1,2))/power.clamp_min(1.)).sqrt().max()))
   if a==0.:assert torch.equal(y,x)
   assert bool((y[2,:,-16:]==0).all()) and bool((y[3,:,:16]==0).all())
  assert maxenergy<tol and maxinverse<tol and maxrelative<=2*math.sin(.05)+tol and torch.equal(x,old)
  theta=.63;r=lambda t:torch.stack((t[:,0]*math.cos(theta)-t[:,1]*math.sin(theta),t[:,0]*math.sin(theta)+t[:,1]*math.cos(theta)),1)
  assert float((dsp.apply(r(x),.1)-r(dsp.apply(x,.1))).abs().max())<tol
  errors[str(dtype)]=dict(energy=maxenergy,inverse=maxinverse,relative_change=maxrelative)
 # Independent dense DCT-II diagonalization, not a repeated stencil implementation.
 n=torch.arange(256,dtype=torch.float64);k=n[:,None];B=(2./256)**.5*torch.cos(math.pi*(n+.5)*k/256);B[0]/=2**.5
 assert float((B@B.T-torch.eye(256)).abs().max())<1e-12
 L=dsp.generator(torch.eye(256,dtype=torch.float64));q=dsp.profile();assert float((B@L@B.T-torch.diag(q)).abs().max())<1e-12 and abs(float(q.sum()))<1e-12
 x=torch.randn(3,2,256,generator=g,dtype=torch.float64);z=torch.complex(x[:,0],x[:,1]);C=B.to(torch.complex128);dense=((z@C.T)*torch.exp(.1j*q))@C
 expected=torch.stack((dense.real,dense.imag),1);error=float((dsp.apply(x,.1)-expected).abs().max());assert error<2e-12
 assert torch.equal(state,torch.random.get_rng_state())
 return dict(status='PASS',protocol_negatives=negative,synthetic_errors=errors,independent_DCT_error=error,Taylor_operator_error_bound=dsp.error_bound(),no_wrap=True,no_external_IQ=True,proper_complex_linear=True,identity_at_zero=True,global_RNG_unchanged=True,real_IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
