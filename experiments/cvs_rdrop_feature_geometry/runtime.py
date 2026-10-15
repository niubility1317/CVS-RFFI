import copy
import numpy as np
import torch
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def snapshot(m):
    return dict(state={k:v.detach().clone() for k,v in m.state_dict().items()},buffers={k:v.detach().clone() for k,v in m.named_buffers()},
        counters={k:getattr(m,k) for k in ('_feature_l_steps','_legacy_calls','_completion_calls')},audit=copy.deepcopy(m._completion_audit))

def require_unchanged(m,saved):
    if m.training or any(p.requires_grad or p.grad is not None for p in m.parameters()) or m._legacy_actual_L_input is not None or m._completion_last_aux is not None or m._completion_audit!=saved['audit']:
        raise ValueError('Frozen eval source model without grad slots required')
    current,buffers=m.state_dict(),dict(m.named_buffers())
    if set(current)!=set(saved['state']) or set(buffers)!=set(saved['buffers']) or any(not torch.equal(v,current[k]) for k,v in saved['state'].items()) or any(not torch.equal(v,buffers[k]) for k,v in saved['buffers'].items()) or any(getattr(m,k)!=v for k,v in saved['counters'].items()):
        raise ValueError('Diagnostic changed a parameter,buffer or feature counter')

def extract(m,x):
    if m.training or any(p.requires_grad for p in m.parameters()) or x.ndim!=3 or x.shape[1:]!=(2,256) or not 0<len(x)<=1404 or x.device.type!='cpu' or not torch.isfinite(x).all():raise ValueError('Finite CPU source IQ and frozen eval model required')
    saved=snapshot(m);outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            with fresh_parameter_caches(m):z=m.forward_identity_only(x[start:start+128])['id_feat_joint']
            if z.shape!=(len(x[start:start+128]),160) or not torch.isfinite(z).all():raise ValueError('Finite deployed joint160 required')
            outputs.append(z.detach().cpu().clone())
    require_unchanged(m,saved)
    return torch.cat(outputs)

def feature_array(z):
    # N607 Torch2.1 cannot use Tensor.numpy against its NumPy2 runtime.
    # Python float lists preserve every FP32 value without that bridge.
    if z.device.type!='cpu' or z.requires_grad or z.ndim!=2 or z.shape[1]!=160 or z.dtype not in (torch.float32,torch.float64) or not torch.isfinite(z).all():raise ValueError('Finite frozen CPU joint160 required')
    return np.asarray(z.tolist(),dtype=np.float32 if z.dtype==torch.float32 else np.float64)
