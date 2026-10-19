import torch

def descriptor(v):return torch.cat([v.mean(-1),v.square().mean(-1),v.abs().mean(-1)],1)

class Probe:
 def __init__(self,core):
  if type(core.time_down).__name__!='AvgPool1d' or core.time_down.kernel_size!=(2,) or core.time_down.stride!=(2,):raise ValueError('Actual first AvgPool2 required')
  self.pending=None;self.values=None;self.max_error=0.;self.calls=0
  self.handles=[core.time_fuse.register_forward_hook(self.before),core.time_down.register_forward_hook(self.after)]
 def before(self,module,args,output):
  if self.pending is not None or output.ndim!=3 or output.shape[1:]!=(48,256):raise ValueError('One real pre-pool 48x256 activation required')
  self.pending=output
 def after(self,module,args,output):
  g=self.pending
  if g is None or output.shape[1:]!=(48,128) or args[0] is not g:raise ValueError('Exact linked first pooling graph required')
  m=.5*(g[:,:,::2]+g[:,:,1::2]);d=.5*(g[:,:,::2]-g[:,:,1::2]);error=max(float((m-output).abs().max()),float((g[:,:,::2]-(m+d)).abs().max()),float((g[:,:,1::2]-(m-d)).abs().max()))
  if not torch.isfinite(g).all() or error>2e-5:raise ValueError('Actual pair mean/difference reconstruction differs')
  self.max_error=max(self.max_error,error);self.values=dict(pooled=descriptor(m).tolist(),removed=descriptor(d).tolist());self.pending=None;self.calls+=1
 def take(self):
  if self.pending is not None or self.values is None:raise ValueError('Missing actual linked observation')
  v=self.values;self.values=None;return v
 def close(self):
  for h in self.handles:h.remove()
