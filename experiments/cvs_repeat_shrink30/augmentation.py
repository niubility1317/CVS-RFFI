"""Private source-only IQ draws;no global RNG,TX/RX labels or view dropout use."""
import time
import numpy as np
import torch
from . import design as d,dsp
class Contraction:
 def __init__(self,enabled):self.enabled=bool(enabled);self.packets=0;self.calls=0;self.seconds=0.
 def apply(self,x,epoch,step):
  if x.shape!=(128,2,256) or not bool(torch.isfinite(x).all()):raise ValueError('One original labeled128 IQ batch required')
  t=time.perf_counter();rng=np.random.default_rng(np.random.SeedSequence([d.SAMPLING_SEED,int(epoch),int(step)]));draw=rng.uniform(0.,.1,len(x));a=np.asarray(x.detach().cpu().tolist(),dtype=np.float32)
  strength=draw if self.enabled else np.zeros(len(x),dtype=np.float64);b=dsp.apply(a,strength);z=torch.frombuffer(bytearray(b.tobytes(order='C')),dtype=torch.float32).reshape(b.shape).clone().to(x.device)
  self.calls+=1;self.packets+=len(x);elapsed=time.perf_counter()-t;self.seconds+=elapsed
  return z,dict(enabled=self.enabled,lambda_values=draw.tolist(),applied_lambda_mean=float(strength.mean()),max_relative_change=float(np.sqrt(np.square(b.astype(float)-a).sum((1,2))/np.maximum(np.square(a.astype(float)).sum((1,2)),1e-24)).max()),CPU_bridge_operator_seconds=elapsed)
 def summary(self):return dict(enabled=self.enabled,physical_input_packets=self.packets,nonzero_candidate_packets=self.packets if self.enabled else 0,calls=self.calls,CPU_bridge_operator_seconds=self.seconds,labels_used=False,target=False,IQ_points=256,range=[0.,.1])
