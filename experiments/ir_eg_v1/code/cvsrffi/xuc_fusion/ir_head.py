"""Replay the native MLP with isolated, fixed Dropout masks and weighted CE GGN."""
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from .ir_types import DomainCall, HeadBatch

def _layers(head):
    net=head if type(head) is nn.Sequential else getattr(head,'net',None)
    if not isinstance(net,nn.Sequential) or len(net)!=4 or tuple(type(x) for x in net)!=(nn.Linear,nn.ReLU,nn.Dropout,nn.Linear):
        raise ValueError('unsupported head: requires Linear/ReLU/Dropout/Linear')
    if net[0].bias is None or net[3].bias is None or list(head.buffers()) or len(list(head.parameters()))!=4:
        raise ValueError('unsupported head parameters/buffers')
    return net

class HeadCallTape:
    def __init__(self,head,layout_signature,context_signature):
        self.head=head;self.net=_layers(head);self.layout_signature=layout_signature
        self.context_signature=context_signature;self.records=[];self._calls={};self._pending=None
    def _pre(self,module,args):
        if self._pending is not None:raise ValueError('recursive head call unsupported')
        self._pending={'post_grl_input':args[0]}
    def _drop_pre(self,module,args):
        x=args[0]
        devices=[x.device.index] if x.is_cuda else []
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(x.device) if x.is_cuda else None
        # fork_rng restores the exact incoming random stream; native call consumes it once.
        with torch.random.fork_rng(devices=devices):
            mask=F.dropout(torch.ones_like(x),p=module.p,training=module.training,inplace=False)
        self._pending.update(scaled_dropout_mask=mask.detach().clone(),dropout_rng={'cpu':cpu.clone(),'cuda':None if cuda is None else cuda.clone()})
    def _post(self,module,args,output):
        self._pending['logits']=output;self.records.append(self._pending);self._pending=None
    def register(self,key,physical_ids,domain,sample_weight,grl_multiplier,*,call_index=None):
        index=len(self.records)-1 if call_index is None else call_index
        if index<0 or index>=len(self.records):raise ValueError('missing captured head call')
        if index in self._calls or any(c.key==key for c in self._calls.values()):raise ValueError('duplicate domain call')
        record=self.records[index];z=record['post_grl_input']
        call=DomainCall(str(key),tuple(map(str,physical_ids)),z.detach().clone(),domain.detach().clone(),
                        sample_weight.detach().clone(),record['scaled_dropout_mask'].detach().clone(),float(grl_multiplier))
        _validate_call(call,record['logits'].shape[-1])
        self._calls[index]=call
        return call
    def batch(self):
        return HeadBatch(tuple(self._calls[i] for i in sorted(self._calls)),self.layout_signature,self.context_signature)

@contextmanager
def capture_head_calls(head,layout_signature,context_signature):
    tape=HeadCallTape(head,layout_signature,context_signature)
    handles=[head.register_forward_pre_hook(tape._pre),tape.net[2].register_forward_pre_hook(tape._drop_pre),head.register_forward_hook(tape._post)]
    try:yield tape
    finally:
        for h in handles:h.remove()

def _validate_call(call,num_classes):
    n=call.z_detached.shape[0]
    if call.z_detached.ndim!=2 or call.domain.shape!=(n,) or call.sample_weight.shape!=(n,) or len(call.physical_ids)!=n:
        raise ValueError('invalid domain call batch/weight/physical IDs')
    if call.domain.dtype!=torch.long or (n and (call.domain.min()<0 or call.domain.max()>=num_classes)):
        raise ValueError('domain label outside registered output mapping')
    if not torch.isfinite(call.sample_weight).all() or (call.sample_weight<0).any():raise ValueError('invalid nonnegative CE weights')
    if not math.isfinite(call.grl_multiplier):raise ValueError('nonfinite GRL')
    if not torch.isfinite(call.scaled_dropout_mask).all():raise ValueError('nonfinite Dropout mask')

def head_logits(phi,call):
    if len(phi)!=4:raise ValueError('head requires four native parameter tensors')
    w1,b1,w2,b2=phi
    hidden=F.relu(F.linear(call.z_detached,w1,b1),inplace=False)
    if hidden.shape!=call.scaled_dropout_mask.shape:raise ValueError('head mask shape changed')
    return F.linear(hidden*call.scaled_dropout_mask,w2,b2)

def head_loss(phi,batch):
    loss=sum(p.sum()*0. for p in phi)
    for call in batch.calls:
        _validate_call(call,phi[2].shape[0])
        if call.z_detached.requires_grad:raise ValueError('HeadBatch must detach backbone inputs')
        if call.domain.numel()==0:continue
        loss=loss+(F.cross_entropy(head_logits(phi,call),call.domain,reduction='none')*call.sample_weight).sum()
    return loss

def head_gradient(phi,batch):
    leaves=tuple(p.detach().requires_grad_(True) for p in phi)
    gradient=torch.autograd.grad(head_loss(leaves,batch),leaves)
    return torch.cat([g.detach().reshape(-1) for g in gradient])

def ggn_matvec(phi,batch,vector):
    """J^T W (diag(p)-pp^T) Jv, not the nonlinear loss Hessian."""
    fixed=tuple(p.detach() for p in phi)
    if vector.ndim!=1 or vector.numel()!=sum(p.numel() for p in fixed):raise ValueError('GGN vector layout mismatch')
    tangents=[];offset=0
    for p in fixed:
        tangents.append(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    result=torch.zeros_like(vector)
    for call in batch.calls:
        _validate_call(call,phi[2].shape[0])
        if call.z_detached.requires_grad:raise ValueError('HeadBatch must detach backbone inputs')
        if call.domain.numel()==0:continue
        fn=lambda *parameters: head_logits(parameters,call)
        logits,jv=torch.func.jvp(fn,fixed,tuple(tangents))
        probabilities=logits.detach().softmax(-1)
        weighted=(call.sample_weight[:,None]*probabilities*(jv-(probabilities*jv).sum(-1,keepdim=True))).detach()
        _,vjp=torch.func.vjp(fn,*fixed)
        jt=vjp(weighted)
        result=result+torch.cat([x.detach().reshape(-1) for x in jt])
    return result
