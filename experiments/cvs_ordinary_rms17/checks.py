"""Focused normalization/config negatives and physical scalar invariants;no real IQ/NN."""
import copy,json,random
import torch
from . import design as d
from .calibration import restore,qualify

def run():
 rejects=0;count=0
 def reject(fn):
  nonlocal rejects
  try:fn()
  except (ValueError,TypeError,KeyError):rejects+=1;return
  raise AssertionError('Illegal case accepted')
 def op(x,o):
  nonlocal count
  y,s=restore(x,o);count+=len(x);return y,s
 for key,value in [('effective_ordinary_epoch',20),('original_label_epochs',20),('fit_RX',[1,3,4,6]),('L_unique',6300),('target_access',True),('U_truth_read',True),('optimizer_updates',1),('head_fit_updates',1),('automatic_training',True),('total_NN_packets',30242),('augmentation_seeds',[1,2,3,4])]:
  c=copy.deepcopy(d.config());c[key]=value;reject(lambda:d.validate(c))
 x=torch.randn(12,2,256,dtype=torch.float64,generator=torch.Generator().manual_seed(2026101717));saved=x.clone()
 reject(lambda:restore(x[:,:,:255],x[:,:,:255]));reject(lambda:restore(x[0],x[0]));reject(lambda:restore(x,x[:6]));reject(lambda:restore(x.long(),x.long()));reject(lambda:restore(x.to(torch.complex64),x.to(torch.complex64)));reject(lambda:restore(x,torch.full_like(x,float('inf'))));reject(lambda:restore(torch.zeros_like(x),x));reject(lambda:restore(x,torch.zeros_like(x)))
 for g in (.25,.9,1.,1.1,4.):
  o=g*x;before=o.clone();tr=torch.get_rng_state().clone();pr=random.getstate();y,s=op(x,o);assert torch.equal(tr,torch.get_rng_state()) and pr==random.getstate() and torch.equal(x,saved) and torch.equal(o,before)
  assert torch.allclose(y,x,rtol=1e-13,atol=1e-13) and max(s['relative_power_error'])<1e-13 and max(s['positive_scalar_shape_error'])<1e-13
 rotated=torch.stack((-x[:,1],x[:,0]),dim=1);y,s=op(x,3*rotated);assert torch.allclose(y,rotated,rtol=1e-13,atol=1e-13)
 o=x+x.square()*.01;y,s=op(x,o);scaled,ss=op(7*x,7*o);assert torch.allclose(scaled,7*y,rtol=1e-13,atol=1e-13) and max(s['relative_power_error'])<1e-13
 raw=dict(accuracy=.995,macro_F1=.995,by_RX={str(v):dict(accuracy=.995) for v in d.FIT},by_TX={str(v):dict(accuracy=.995) for v in range(6)},by_day={str(v):dict(accuracy=.995) for v in (1,2,3)})
 ordinary=copy.deepcopy(raw);ordinary['accuracy']=ordinary['macro_F1']=.990;normalized=copy.deepcopy(raw);normalized['accuracy']=normalized['macro_F1']=.994;normalized['raw_correct_retention']=.999
 stats=dict(gain=[1.1]*12,relative_power_error=[0.]*12,positive_scalar_shape_error=[0.]*12)
 assert qualify(raw,ordinary,normalized,[.3,.4,.2,.5],stats)['decision']=='PASS'
 for key,value in [('accuracy',.991),('macro_F1',.989),('raw_correct_retention',.98)]:
  bad=copy.deepcopy(normalized);bad[key]=value;assert qualify(raw,ordinary,bad,[.3,.4,.2,.5],stats)['decision']=='FAIL'
 assert qualify(raw,ordinary,normalized,[.3,.4,-.1,-.1],stats)['decision']=='FAIL'
 bad=copy.deepcopy(normalized);bad['by_RX']['6']['accuracy']=.98;assert qualify(raw,ordinary,bad,[.3,.4,.2,.5],stats)['decision']=='FAIL'
 for key,value in [('gain',[1.]*12),('relative_power_error',[1e-4]*12),('positive_scalar_shape_error',[1e-5]*12)]:
  bad=copy.deepcopy(stats);bad[key]=value;assert qualify(raw,ordinary,normalized,[.3,.4,.2,.5],bad)['decision']=='FAIL'
 return dict(status='PASS',negative_cases=rejects,scientific_counterexamples=8,synthetic_RMS_operations=count,NN=0,source_IQ_reads=0,updates=0,head_fit_updates=0,target_access=False,positive_scaling_phase_shape_preserved=True,packetpower_restored=True,input_and_RNG_unchanged=True)
if __name__=='__main__':print(json.dumps(run()))
