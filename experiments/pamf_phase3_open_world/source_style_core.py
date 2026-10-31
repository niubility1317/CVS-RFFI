"""Native identity/open feature sharing and temporary source-only MixStyle."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_ssl_research.u_mixstyle_objective import mix_moments
from . import source_open_embedding_train as native
from . import star_signal_open_model as signal

SITES = ('core.id_backbone.time_down', 'core.id_backbone.t2.pool')
SITE_SHAPES = ((48,128), (96,32))
SINC_PREFIX = 'reader.core.id_backbone.sinc.'
HEAD_WEIGHT = 'reader.core.id_backbone.cls_head.weight'
L_SEED, PAIR_SEED, OE_SEED, STYLE_SEED = 2026103160,2026103170,2026103018,2026103171


class NativeOpen(nn.Module):
    """Original F0 reader plus pure-source r03 projection, with one feature pass.

    The body is F0, so no initial open-feature equivalence to r03 is claimed.
    The discarded construction wrapper is never registered or deployed here.
    """
    def __init__(self,reader,projection):
        super().__init__();self.reader=deepcopy(reader);self.projection=deepcopy(projection)
        self.eval().requires_grad_(True)
        for name,p in self.named_parameters():
            if name.startswith(SINC_PREFIX):p.requires_grad_(False)
        if (sum(p.numel() for p in self.reader.parameters())!=232993 or sum(p.numel() for p in self.projection.parameters())!=102688
                or sum(p.numel() for p in self.parameters() if p.requires_grad)!=335633):
            raise ValueError('Complete original reader232993 plus source projection102688 required')

    def train(self,mode=True):
        return super().train(False)

    def forward(self,x):
        if x.dtype!=torch.float32 or x.ndim!=3 or tuple(x.shape[1:])!=(2,256) or not len(x) or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite cached FP32 P1[N,2,256] required')
        self.eval();r=self.reader;h=r.core.id_backbone.cls_head;captured={};handles=[]
        def record(key,value):
            if key in captured:raise ValueError('Each native contribution must execute once')
            captured[key]=value
        try:
            for key,module in zip(('base','pa','ref_projection','signature'),(h.base_norm,h.pa_norm,r.response_projection,r.sequence_pool)):
                handles.append(module.register_forward_hook(lambda m,a,v,key=key:record(key,v)))
            with torch.autocast(device_type=x.device.type,enabled=False):
                raw=r.features(x);logits=r.classify_features(raw)
                if raw.shape!=(len(x),160) or set(captured)!={'base','pa','ref_projection','signature'} or any(v.shape!=(len(x),160) for v in captured.values()):
                    raise ValueError('Same-pass native raw160/four contributions required')
                contributions=torch.stack((captured['base'],h.gain.tanh()*captured['pa'],r.response_gain.tanh()*captured['ref_projection'],captured['signature']),1)
                u=signal.unit(raw);c=signal.contribution_unit(contributions)
                z=F.normalize(F.normalize(u,dim=1)+self.projection.network(F.normalize(c,dim=1)),dim=1,eps=1e-12)
            if logits.shape!=(len(x),6) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (z,logits)):
                raise ValueError('Finite open unit160 and unchanged-formula default6 logits required')
            return z,logits
        finally:
            for handle in handles:handle.remove()


class SourceSampler:
    def __init__(self,L,OE):
        m=L['metadata'];self.y=np.asarray(m['y']);self.rx=np.asarray(m['rx_i'])
        if self.y.ndim!=1 or self.rx.shape!=self.y.shape or len(self.y)<128:
            raise ValueError('Legal full source L labels/RX and 128 anchors required')
        self.rows=np.arange(len(self.y));self.candidates={}
        for cls,rx in set(zip(self.y.tolist(),self.rx.tolist())):
            eligible=np.flatnonzero((self.y==cls)&(self.rx!=rx))
            if not len(eligible):raise ValueError('Every source TX needs a different-RX partner')
            self.candidates[cls,rx]=eligible
        self.rng=np.random.default_rng(L_SEED);self.pair=np.random.default_rng(PAIR_SEED)
        self.oe=native.DomainSampler(OE['metadata']['domain'],OE_SEED)

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive source step required')
        a=self.rng.choice(len(self.rows),128,replace=False)
        b=np.asarray([self.pair.choice(self.candidates[int(self.y[i]),int(self.rx[i])]) for i in a],np.int64)
        return dict(A=a.astype(np.int64),B=b,OE=self.oe.sample(64))


def style_plan(anchors=128,oe_count=64):
    if type(anchors) is not int or anchors<1 or type(oe_count) is not int or oe_count<0:raise ValueError('Positive paired L and nonnegative OE counts required')
    n=2*anchors+oe_count;partner=np.arange(n,dtype=np.int64);partner[:anchors]+=anchors;partner[anchors:2*anchors]-=anchors
    valid=np.arange(n)<2*anchors
    return dict(partner=partner,valid=valid)


@contextmanager
def style_context(reader,plan,rng,*,force=None):
    """Hooks exist only during this source forward; OE is exactly unchanged."""
    partner=np.asarray(plan['partner']);valid=np.asarray(plan['valid']);n=len(partner)
    if partner.shape!=(n,) or partner.dtype!=np.int64 or valid.shape!=(n,) or valid.dtype!=np.bool_:
        raise ValueError('Aligned integer partners and boolean source-style validity required')
    handles=[];observed={};decisions={}
    for path in SITES:
        enabled=bool(rng.random()<.5) if force is None else bool(force)
        decisions[path]=(enabled,rng.beta(.1,.1,size=(n,1,1)).astype(np.float32))
    try:
        for path,shape in zip(SITES,SITE_SHAPES):
            def hook(module,inputs,output,path=path,shape=shape):
                if path in observed or not torch.is_tensor(output) or output.shape!=(n,*shape) or output.dtype!=torch.float32 or not bool(torch.isfinite(output).all()):
                    raise ValueError('Exactly one real FP32 source feature map with declared style shape required: '+path)
                enabled,values=decisions[path]
                p=torch.tensor(partner.tolist(),device=output.device,dtype=torch.long);v=torch.tensor(valid.tolist(),device=output.device,dtype=torch.bool)
                lam=torch.tensor(values.tolist(),device=output.device,dtype=torch.float32)
                result=mix_moments(output,p,v,lam,1e-6) if enabled else output
                passthrough=torch.equal(result[~v],output[~v])
                if not passthrough:raise ValueError('OE must pass through the style hook exactly')
                observed[path]=dict(enabled=enabled,invocations=1,shape=list(output.shape),valid_rows=int(v.sum()),
                    mixed_rows=int(v.sum()) if enabled else 0,OE_exact_passthrough=passthrough,
                    lambda_mean=float(lam.mean().cpu()),lambda_min=float(lam.min().cpu()),lambda_max=float(lam.max().cpu()),
                    input_mean=float(output.detach().mean().cpu()),input_population_std_mean=float(output.detach().var(-1,unbiased=False).add(1e-6).sqrt().mean().cpu()),
                    output_mean=float(result.detach().mean().cpu()),moments_detached=True,eps=1e-6)
                return result
            handles.append(reader.get_submodule(path).register_forward_hook(hook))
        yield observed
        if set(observed)!=set(SITES):raise ValueError('Both actual source style sites must execute')
    finally:
        for handle in handles:handle.remove()
