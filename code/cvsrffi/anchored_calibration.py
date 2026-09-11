"""Final-output-only source-V temperature and tied-confidence calibration."""
import math
import torch
from torch.nn import functional as F


def physical_weights(ids):
    if not ids or any(not isinstance(x,str) or not x for x in ids):raise ValueError('nonempty physical IDs required')
    counts={}
    for pid in ids:counts[pid]=counts.get(pid,0)+1
    return torch.tensor([1/len(counts)/counts[pid] for pid in ids],dtype=torch.double)


class FinalProbabilityCalibrator:
    def __init__(self,target_coverage=.9):
        if not 0<target_coverage<=1:raise ValueError('invalid calibration coverage')
        self.target_coverage=float(target_coverage);self.fitted=False

    @torch.no_grad()
    def fit(self,log_probs,labels,*,role='V',weights=None,system_identity):
        if self.fitted:raise RuntimeError('final calibration is frozen')
        if role!='V':raise ValueError('final calibration requires source V')
        if not isinstance(system_identity,str) or not system_identity:raise ValueError('frozen system identity required')
        x=log_probs.detach().double();y=labels.detach().to(x.device)
        if x.ndim!=2 or len(x)==0 or x.shape[1]<2 or not torch.isfinite(x).all():raise ValueError('finite final log probabilities required')
        if y.shape!=(len(x),) or y.dtype!=torch.long or (y<0).any() or (y>=x.shape[1]).any():raise ValueError('invalid calibration labels')
        w=torch.ones(len(x),dtype=x.dtype,device=x.device)/len(x) if weights is None else weights.detach().to(x)
        if w.shape!=(len(x),) or not torch.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('invalid calibration weights')
        w=w/w.sum();x=x.log_softmax(-1)
        def loss(log_t):return float((w*F.cross_entropy(x/math.exp(log_t),y,reduction='none')).sum())
        left,right=math.log(.05),math.log(20.)
        for _ in range(48):
            a=left+(right-left)/3;b=right-(right-left)/3
            if loss(a)<=loss(b):right=b
            else:left=a
        t=math.exp((left+right)/2);final=(x/t).log_softmax(-1)
        # A positive temperature calibrates confidence; the frozen decision is
        # retained separately when normalization rounds a unique gap to a tie.
        confidence=final.exp().amax(-1);order=confidence.argsort(descending=True,stable=True)
        index=int(torch.searchsorted(w[order].cumsum(0),torch.tensor(self.target_coverage,device=x.device,dtype=x.dtype)).clamp_max(len(x)-1))
        threshold=float(confidence[order[index]])
        self.temperature=t;self.threshold=threshold;self.system_identity=system_identity
        self.attained_source_coverage=float(w[confidence>=threshold].sum())
        self.nll_before=loss(0.);self.nll_after=loss(math.log(t));self.fitted=True
        return self

    def predict(self,log_probs,*,system_identity,decision=None):
        if not self.fitted:raise RuntimeError('fit source V before prediction')
        if system_identity!=self.system_identity:raise ValueError('frozen system identity mismatch')
        if log_probs.ndim!=2 or not torch.isfinite(log_probs).all():raise ValueError('finite final scores required')
        result=(log_probs.detach().double()/self.temperature).log_softmax(-1)
        decision=log_probs.argmax(-1) if decision is None else decision
        if decision.shape!=(len(log_probs),) or decision.dtype!=torch.long or (decision<0).any() or (decision>=log_probs.shape[1]).any():raise ValueError('invalid frozen decision')
        confidence=result.exp().amax(-1)
        return dict(log_probabilities=result,probabilities=result.exp(),confidence=confidence,
                    accepted=confidence>=self.threshold,top_class=decision)

    def state_dict(self):
        if not self.fitted:raise RuntimeError('cannot export unfitted final calibration')
        return dict(version=1,target_coverage=self.target_coverage,temperature=self.temperature,threshold=self.threshold,
                    system_identity=self.system_identity,attained_source_coverage=self.attained_source_coverage,
                    nll_before=self.nll_before,nll_after=self.nll_after)

    @classmethod
    def from_state_dict(cls,state):
        keys={'version','target_coverage','temperature','threshold','system_identity','attained_source_coverage','nll_before','nll_after'}
        if set(state)!=keys or state['version']!=1:raise ValueError('invalid final calibrator state')
        if not all(math.isfinite(state[k]) for k in keys-{'version','system_identity'}):raise ValueError('nonfinite calibration state')
        if not .05<=state['temperature']<=20 or not 0<=state['threshold']<=1 or not 0<state['attained_source_coverage']<=1+1e-8:
            raise ValueError('invalid final calibration values')
        if not isinstance(state['system_identity'],str) or not state['system_identity']:raise ValueError('missing frozen system identity')
        obj=cls(state['target_coverage'])
        for key in keys-{'version'}:setattr(obj,key,state[key])
        obj.fitted=True;return obj
