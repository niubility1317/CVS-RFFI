"""Focused scope negatives and independent dense-projector contraction check."""
import copy,json
import numpy as np
from . import design as d,dsp

def run():
 negative=0
 def bad(f):
  nonlocal negative
  try:f()
  except (ValueError,TypeError):negative+=1
  else:raise AssertionError('Invalid fixed scope accepted')
 c=d.config()
 for k,v in [('V_selected',1799),('U_used',True),('target_access',True),('source_RX_holdout',1),('training_updates',1),('teacher_NN',1),('automatic_training',True),('per_cell',19),('expected_NN',14400)]:
  t=copy.deepcopy(c);t[k]=v;bad(lambda t=t:d.validate(t))
 for k,v in [('max_strength',.2),('window',[60,160]),('cycles',5),('new_noise',True)]:
  t=copy.deepcopy(c);t['operator'][k]=v;bad(lambda t=t:d.validate(t))
 x=np.ones((2,2,256),dtype=np.float64)
 for z,a in [(x[:,:,:255],0.),(x.astype(np.int64),0.),(x,.10001),(x,-.01),(x,float('nan')),(x,np.zeros(3)),(x.tolist(),.1)]:bad(lambda z=z,a=a:dsp.apply(z,a))
 n=x.copy();n[0,0,0]=np.nan;bad(lambda:dsp.apply(n,.1))
 from .analyze import metric
 known=metric([0,0,1,2],[0,1,1,2]);assert known['correct']==3 and known['count']==4 and known['confusion'][0][1]==1 and abs(known['macro_F1']-7/18)<1e-12
 rng=np.random.default_rng(20261018102);state=np.random.get_state();errors={};synthetic=0
 # Dense 80x80 independent projector, no production reshape/mean/taper call.
 P=np.kron(np.ones((4,4))/4,np.eye(20));Q=np.eye(80)-P
 weights=np.array([np.sin(np.pi*(i%20)/19)**2 if i%20 not in (0,19) else 0. for i in range(80)])
 M=np.eye(80)-.1*np.diag(weights)@Q
 eigen=np.linalg.eigvalsh(M);assert eigen.min()>=.9-1e-12 and eigen.max()<=1+1e-12
 for dtype,tol in ((np.float64,1e-12),(np.float32,2e-6)):
  x=rng.normal(size=(9,2,256)).astype(dtype);x[0]=0;x[1]=1;x[2]=0;x[2,0,0]=1;x[3]=0;x[3,1,-1]=1
  # A genuinely periodic waveform is fixed even with nontrivial IQ values.
  repeat=rng.normal(size=(2,20)).astype(dtype);x[4,:,80:160]=np.tile(repeat,(1,4));old=x.copy();synthetic+=len(x)
  y=dsp.apply(x,.1);expected=x.astype(np.float64);expected[:,:,80:160]=expected[:,:,80:160]@M.T
  err=float(np.max(np.abs(y-expected)));stats=dsp.numerics(x,y,.1)
  assert err<tol and stats['max_energy_increase_relative']<tol and stats['max_cycle_mean_change_relative']<tol and stats['max_relative_change']<=.1+tol
  assert stats['outside_unchanged'] and stats['cycle_boundaries_unchanged'] and np.array_equal(x,old) and np.array_equal(dsp.apply(x,0.),x) and np.array_equal(y[4],x[4])
  permutation=np.array([8,0,7,1,6,2,5,3,4]);assert np.array_equal(dsp.apply(x[permutation],.1),y[permutation])
  # Same real operator on I/Q commutes with complex multiplication.
  theta=.63
  def rotate(z):return np.stack((z[:,0]*np.cos(theta)-z[:,1]*np.sin(theta),z[:,0]*np.sin(theta)+z[:,1]*np.cos(theta)),axis=1)
  assert float(np.max(np.abs(dsp.apply(rotate(x),.1)-rotate(y))))<tol
  perpacket=np.linspace(0.,.1,len(x));mixed=dsp.apply(x,perpacket)
  assert np.array_equal(mixed[0],x[0]) and float(np.linalg.norm(mixed.astype(float)-x)/max(np.linalg.norm(x),1e-24))<=.1+tol
  errors[str(dtype)]=dict(dense_error=err,**stats)
 after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
 return dict(status='PASS',protocol_negatives=negative,synthetic_input_packets=synthetic,synthetic_errors=errors,dense_eigenvalue_range=[float(eigen.min()),float(eigen.max())],cycle_mean_preserved=True,outside_and_cycle_boundaries_unchanged=True,complex_linear=True,identity_at_zero=True,global_RNG_unchanged=True,real_IQ=0,CP=0,NN=0,updates=0)
if __name__=='__main__':print(json.dumps(run()))
