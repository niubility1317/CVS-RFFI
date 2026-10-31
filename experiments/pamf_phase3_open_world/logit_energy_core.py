"""Original native reader and source-only default-head energy objectives."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('UNIFORM','RXPAIR')
RECEIVERS=(1,3,4,6,8)
L_SEED,OE_SEED=2026103017,2026103018
READER_PARAMETERS,TRAINABLE_PARAMETERS,HEAD_PARAMETERS=232993,232945,960
SINC_PREFIX='reader.core.id_backbone.sinc.'
HEAD_WEIGHT='reader.core.id_backbone.cls_head.weight'


class NativeEnergy(nn.Module):
    """Only the complete original reader; no projection or replacement head."""
    def __init__(self,reader):
        super().__init__()
        if not isinstance(reader,nn.Module) or not callable(getattr(reader,'features',None)) or not callable(getattr(reader,'classify_features',None)):
            raise ValueError('Complete strictly loaded native F0 reader required')
        self.reader=deepcopy(reader);self.eval().requires_grad_(True)
        for name,p in self.named_parameters():
            if name.startswith(SINC_PREFIX):p.requires_grad_(False)
        params=dict(self.named_parameters());frozen={n for n,p in params.items() if not p.requires_grad}
        if (sum(p.numel() for p in self.parameters())!=READER_PARAMETERS or sum(p.numel() for p in self.parameters() if p.requires_grad)!=TRAINABLE_PARAMETERS
                or params[HEAD_WEIGHT].shape!=(6,160) or frozen!={SINC_PREFIX+'low_coordinate',SINC_PREFIX+'band_coordinate'}
                or any(params[n].shape!=(24,1) for n in frozen) or any(p.dtype!=torch.float32 for p in params.values())):
            raise ValueError('Original reader232993/default6 head960 and only Sinc48 frozen required')

    def train(self,mode=True):return super().train(False)

    def forward(self,x):
        if not torch.is_tensor(x) or x.ndim!=3 or not len(x) or tuple(x.shape[1:])!=(2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite cached P1 FP32[N,2,256] required')
        self.eval()
        with torch.autocast(device_type=x.device.type,enabled=False):
            raw=self.reader.features(x);logits=self.reader.classify_features(raw)
        if raw.shape!=(len(x),160) or logits.shape!=(len(x),6) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (raw,logits)):
            raise ValueError('Same-pass finite raw160/default6 FP32 outputs required')
        return raw,logits


def _logits(value,n=None):
    if not torch.is_tensor(value) or value.dtype!=torch.float32 or value.ndim!=2 or not len(value) or value.shape[1]!=6 or (n is not None and len(value)!=n) or not bool(torch.isfinite(value).all()):
        raise ValueError('Nonempty aligned finite complete-six FP32 logits required')
    return value


def _codes(value,n,device,upper,name):
    if not torch.is_tensor(value) or value.dtype!=torch.int64 or value.shape!=(n,) or value.device!=device or bool(((value<0)|(value>=upper)).any()):
        raise ValueError('Legal aligned integer '+name+' required')
    return value


def compute(logits_L,y_L,teacher_logits_L,domains_L,logits_OE,domains_OE,arm):
    if arm not in ARMS:raise ValueError('Only matched source UNIFORM/RXPAIR arms are allowed')
    L=_logits(logits_L);OE=_logits(logits_OE);teacher=_logits(teacher_logits_L,len(L))
    if any(v.device!=L.device for v in (OE,teacher)):raise ValueError('Source student and frozen teacher must share a device')
    y=_codes(y_L,len(L),L.device,6,'known L labels');ld=_codes(domains_L,len(L),L.device,15,'L RX/day domains');od=_codes(domains_OE,len(OE),L.device,15,'OE RX/day domains')
    ce=F.cross_entropy(L,y,label_smoothing=0.)
    kd=4*F.kl_div(F.log_softmax(L/2,dim=1),F.softmax(teacher.detach()/2,dim=1),reduction='batchmean')
    sL=torch.logsumexp(L,dim=1);sOE=torch.logsumexp(OE,dim=1);rank=L.new_zeros(());uniform=L.new_zeros(());count=0
    if arm=='UNIFORM':uniform=-F.log_softmax(OE,dim=1).mean()-math.log(6.)
    else:
        mask=ld[:,None]==od[None]
        if not bool(mask.any(1).all()):raise ValueError('Every source-L domain needs a matching public OE domain')
        rank=F.softplus((sOE[None]-sL[:,None])[mask]).mean();count=int(mask.sum())
    outside=uniform if arm=='UNIFORM' else rank
    return dict(total=ce+kd+outside,CE=ce,KD=kd,OE=outside,rank=rank,uniform=uniform,s_L=sL.mean(),s_OE=sOE.mean(),pair_count=count)


def _metadata(value,known):
    fields={'rx_i','day_i','domain'}|({'y'} if known else set())
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('Only source-known y or public OE RX/day/domain metadata are allowed')
    a={k:np.asarray(v) for k,v in value.items()};n=len(a['domain'])
    if not n or any(v.shape!=(n,) or v.dtype!=np.int64 for v in a.values()):raise ValueError('Aligned int64 source metadata required')
    if not np.isin(a['rx_i'],RECEIVERS).all() or not np.isin(a['day_i'],(1,2,3)).all():raise ValueError('Only original source RX/day namespace allowed')
    indices=np.asarray([RECEIVERS.index(int(v)) for v in a['rx_i']],np.int64)
    if not np.array_equal(a['domain'],indices*3+a['day_i']-1) or set(a['domain'].tolist())!=set(range(15)):
        raise ValueError('Exact public fifteen RX/day domain mapping required')
    if known and set(a['y'].tolist())!=set(range(6)):raise ValueError('Complete original six source identities required')
    return a


class SourceSampler:
    def __init__(self,L_metadata,OE_metadata):
        self.L=_metadata(L_metadata,True);self.OE=_metadata(OE_metadata,False)
        self.cells={(c,d):np.flatnonzero((self.L['y']==c)&(self.L['domain']==d)) for c in range(6) for d in range(15)}
        self.classes={c:np.flatnonzero(self.L['y']==c) for c in range(6)}
        self.outside={d:np.flatnonzero(self.OE['domain']==d) for d in range(15)}
        if any(not len(rows) for rows in self.cells.values()):raise ValueError('All six by fifteen labeled source cells must be nonempty')
        self.L_rng=np.random.default_rng(L_SEED);self.OE_rng=np.random.default_rng(OE_SEED)

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive source optimization step required')
        labeled=np.concatenate([np.concatenate([*(self.L_rng.choice(self.cells[c,d],2,replace=True) for d in range(15)),self.L_rng.choice(self.classes[c],2,replace=True)]) for c in range(6)]).astype(np.int64)
        outside=[self.OE_rng.choice(self.outside[d],4,replace=True) for d in range(15)]
        outside.append(np.asarray([self.OE_rng.choice(self.outside[int(d)]) for d in self.OE_rng.integers(15,size=4)],np.int64))
        return dict(L=labeled,OE=np.concatenate(outside).astype(np.int64))
