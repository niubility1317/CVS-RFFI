"""Named evidence-block masks and pattern-specific source-V calibration."""
from dataclasses import dataclass
import math
import torch
from .anchored_calibration import FinalProbabilityCalibrator,physical_weights


@dataclass(frozen=True)
class PatternSpec:
    block_names:tuple
    block_sizes:tuple

    def __post_init__(self):
        if self.block_names!=('t','f','pa') or len(self.block_sizes)!=3 or any(not isinstance(x,int) or x<=0 for x in self.block_sizes):
            raise ValueError('named actual t/f/pa block sizes required')

    @property
    def patterns(self):return ('full','missing_t','missing_f','missing_pa','all_missing')

    def mask(self,pattern,n,device='cpu'):
        if pattern not in self.patterns:raise ValueError('unsupported evidence pattern')
        result=torch.ones(n,sum(self.block_sizes),dtype=torch.bool,device=device)
        if pattern=='all_missing':return ~result
        start=0
        for name,width in zip(self.block_names,self.block_sizes):
            if pattern=='missing_'+name:result[:,start:start+width]=False
            start+=width
        return result


class PatternCalibrator:
    def __init__(self,quality_bins=3,min_group_samples=20,consistency_quantile=.95,min_confidence=.5):
        if quality_bins<1 or min_group_samples<1 or not 0<consistency_quantile<1 or not 0<=min_confidence<=1:raise ValueError('invalid pattern calibration config')
        self.quality_bins=int(quality_bins);self.min_group_samples=int(min_group_samples)
        self.consistency_quantile=float(consistency_quantile);self.min_confidence=float(min_confidence);self.fitted=False;self.groups={}

    def _validate(self,scores,mahal,count,quality,patterns):
        n=len(scores)
        if scores.ndim!=2 or n==0 or mahal.shape!=scores.shape or count.shape!=(n,) or quality.shape!=(n,3) or len(patterns)!=n:
            raise ValueError('invalid pattern calibration shapes')
        if not torch.isfinite(scores).all() or not torch.isfinite(mahal).all() or (mahal<0).any() or not torch.isfinite(quality).all():
            raise ValueError('nonfinite pattern calibration arrays')
        if not torch.isfinite(count).all() or (count<0).any() or (count!=count.floor()).any():raise ValueError('invalid observed counts')
        if any(not isinstance(p,str) or not p for p in patterns):raise ValueError('invalid pattern identity')
        return (quality.double().mean(-1).clamp(0,1)*self.quality_bins).long().clamp_max(self.quality_bins-1)

    @torch.no_grad()
    def fit(self,scores,mahal,count,quality,labels,patterns,physical_ids,*,role='V',system_identity='partial-pattern-v1'):
        if self.fitted:raise RuntimeError('pattern calibration is frozen')
        if role!='V':raise ValueError('pattern calibration requires source V')
        bins=self._validate(scores,mahal,count,quality,patterns)
        if labels.shape!=(len(scores),) or labels.dtype!=torch.long or (labels<0).any() or (labels>=scores.shape[1]).any():raise ValueError('invalid pattern labels')
        if len(physical_ids)!=len(scores):raise ValueError('pattern physical IDs mismatch')
        weights=physical_weights(physical_ids).to(scores.device)
        self.final=FinalProbabilityCalibrator().fit(scores.log_softmax(-1),labels,weights=weights,system_identity=system_identity)
        standardized=mahal.gather(1,labels[:,None]).squeeze(1).double()/count.clamp_min(1)
        for pattern in sorted(set(patterns)):
            for qbin in range(self.quality_bins):
                take=torch.tensor([p==pattern for p in patterns],device=scores.device)&(bins==qbin)&(count>0)
                ix=take.nonzero().flatten().tolist();physical_count=len({physical_ids[i] for i in ix})
                if not ix:continue
                dims=count[take].unique()
                if len(dims)!=1:raise ValueError('one pattern must have one effective dimension')
                key=f'{pattern}:{qbin}:nonempty'
                if physical_count>=self.min_group_samples:
                    # Repeated views do not multiply effective physical support.
                    ids=[physical_ids[i] for i in ix];w=physical_weights(ids).to(standardized)
                    values=standardized[take];order=values.argsort(stable=True)
                    rank=int(torch.searchsorted(w[order].cumsum(0),values.new_tensor(self.consistency_quantile)).clamp_max(len(values)-1))
                    self.groups[key]=dict(limit=float(values[order[rank]]),physical_count=physical_count,view_count=len(ix),dimension=int(dims[0]))
        self.fitted=True;return self

    @torch.no_grad()
    def predict(self,scores,mahal,count,quality,patterns):
        if not self.fitted:raise RuntimeError('fit V pattern calibration before prediction')
        bins=self._validate(scores,mahal,count,quality,patterns)
        final=self.final.predict(scores.log_softmax(-1),system_identity=self.final.system_identity)
        top=final['top_class'];available=torch.zeros(len(scores),dtype=torch.bool,device=scores.device)
        limit=scores.new_full((len(scores),),float('nan'))
        for i,pattern in enumerate(patterns):
            group=self.groups.get(f'{pattern}:{int(bins[i])}:nonempty')
            if group is not None and int(count[i])==group['dimension']:
                available[i]=True;limit[i]=group['limit']
        standardized=mahal.gather(1,top[:,None]).squeeze(1)/count.clamp_min(1)
        accepted=available&(count>0)&(standardized<=limit)&(final['confidence']>=self.min_confidence)
        mismatch=available&(count>0)&(mahal.amin(-1)/count.clamp_min(1)>limit)
        status=['identify' if bool(a) else 'model_mismatch_candidate' if bool(m) else 'defer' for a,m in zip(accepted,mismatch)]
        return dict(**{k:v for k,v in final.items() if k!='accepted'},accepted=accepted,status=status,
                    calibration_available=available,consistency_limit=limit)

    def state_dict(self):
        if not self.fitted:raise RuntimeError('unfitted pattern calibration')
        return dict(version=2,quality_bins=self.quality_bins,min_group_samples=self.min_group_samples,
                    consistency_quantile=self.consistency_quantile,min_confidence=self.min_confidence,
                    groups={k:dict(v) for k,v in self.groups.items()},final=self.final.state_dict())

    @classmethod
    def from_state_dict(cls,state):
        if set(state)!={'version','quality_bins','min_group_samples','consistency_quantile','min_confidence','groups','final'} or state['version']!=2:
            raise ValueError('invalid pattern calibration version/schema')
        obj=cls(**{k:state[k] for k in ('quality_bins','min_group_samples','consistency_quantile','min_confidence')})
        for key,group in state['groups'].items():
            pattern,qbin,level=key.rsplit(':',2)
            if not pattern or not 0<=int(qbin)<obj.quality_bins or level!='nonempty' or set(group)!={'limit','physical_count','view_count','dimension'}:
                raise ValueError('invalid pattern group')
            if not math.isfinite(group['limit']) or group['limit']<0 or group['physical_count']<obj.min_group_samples or group['view_count']<group['physical_count'] or group['dimension']<=0:
                raise ValueError('invalid pattern group support')
        obj.groups={k:dict(v) for k,v in state['groups'].items()};obj.final=FinalProbabilityCalibrator.from_state_dict(state['final']);obj.fitted=True
        return obj
