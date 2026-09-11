"""Actual-action utility fusion. All class information is permutation symmetric."""
import math
import torch

ALPHAS=(0.,.25,.5,.75,1.)


def _log_pairs(left,right):
    if left.ndim!=2 or left.shape!=right.shape or left.shape[1]<2 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('finite matching expert log probabilities required')
    # Normalize defensively; expert temperatures must already have been frozen.
    return left.detach().double().log_softmax(-1),right.detach().double().log_softmax(-1)


def mix_log_probs(log0,logg,alpha):
    log0,logg=_log_pairs(log0,logg)
    a=torch.as_tensor(alpha,dtype=log0.dtype,device=log0.device)
    if a.ndim==0:a=a.expand(len(log0))
    if a.shape!=(len(log0),) or not torch.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('alpha must be in [0,1]')
    mixed=torch.logaddexp(log0+torch.log1p(-a)[:,None],logg+torch.log(a)[:,None])
    mixed=torch.where((a==0)[:,None],log0,torch.where((a==1)[:,None],logg,mixed))
    return mixed-mixed.logsumexp(-1,keepdim=True)


@torch.no_grad()
def realize_actions(log0,logg,*,protection_threshold,valid=None):
    """Accept raw logits; convert to FP64 before normalization in every caller.

    Pre-normalizing FP32 scores changes boundary decisions and utility labels.
    """
    if log0.ndim!=2 or logg.shape!=log0.shape or not torch.isfinite(log0).all():raise ValueError('finite H0 scores required')
    expert_valid=torch.isfinite(logg).all(-1)
    logg=torch.where(expert_valid[:,None],logg,log0)
    raw_top=log0.argmax(-1);expert_top=logg.argmax(-1)
    raw_unique=(log0==log0.amax(-1,keepdim=True)).sum(-1)==1
    log0,logg=_log_pairs(log0,logg);p0,pg=log0.exp(),logg.exp();n,c=p0.shape
    if not math.isfinite(float(protection_threshold)) or protection_threshold<0:raise ValueError('invalid protection threshold')
    valid=torch.ones(n,dtype=torch.bool,device=p0.device) if valid is None else valid.to(p0.device)
    if valid.shape!=(n,) or valid.dtype!=torch.bool:raise ValueError('valid must be bool [N]')
    valid=valid&expert_valid.to(valid.device)
    top=raw_top;topval=p0.gather(1,top[:,None]);unique=raw_unique
    margin=p0.topk(2,dim=-1).values.diff(dim=-1).squeeze(-1).neg()
    protected=(margin>=protection_threshold)&unique&valid
    m=topval-p0;d=pg.gather(1,top[:,None])-pg
    negative=(d<0)&(m>0)
    boundary=torch.where(negative,m/(m-d).clamp_min(torch.finfo(p0.dtype).tiny),torch.full_like(m,float('inf'))).amin(-1)
    safe=torch.nextafter(boundary,torch.zeros_like(boundary)).clamp_max(1.)
    actions=[];alphas=[];reasons=[];predictions=[]
    for requested in ALPHAS:
        actual=p0.new_full((n,),requested)
        actual=torch.where(~valid|~unique,0.,actual)
        # Choose an existing discrete action strictly inside the flip boundary.
        for lower in reversed(ALPHAS):
            bad=protected&(actual>safe)
            actual=torch.where(bad&(lower<actual),torch.full_like(actual,lower),actual)
        mix=mix_log_probs(log0,logg,actual)
        mixp=mix.exp();unchanged=(mix.argmax(-1)==top)&((mixp==mixp.amax(-1,keepdim=True)).sum(-1)==1)
        for lower in reversed(ALPHAS[:-1]):
            bad=protected&~unchanged&(actual>0)
            actual=torch.where(bad&(lower<actual),torch.full_like(actual,lower),actual)
            mix=mix_log_probs(log0,logg,actual);mixp=mix.exp()
            unchanged=(mix.argmax(-1)==top)&((mixp==mixp.amax(-1,keepdim=True)).sum(-1)==1)
        # alpha=0 directly preserves H0 even under finite precision/ties.
        actual=torch.where(protected&~unchanged,0.,actual)
        mix=mix_log_probs(log0,logg,actual)
        reason=torch.zeros(n,dtype=torch.long,device=p0.device)
        reason[protected&(actual<requested)]=1;reason[~unique]=2;reason[~valid]=3
        mix=torch.where((actual==0)[:,None],log0,mix)
        # Normalization may erase a tiny unique logit gap. Endpoint decisions
        # remain the original experts' decisions, including their tie policy.
        prediction=torch.where(actual==0,raw_top,torch.where(actual==1,expert_top,mix.argmax(-1)))
        actions.append(mix);alphas.append(actual);reasons.append(reason)
        predictions.append(prediction)
    return dict(log_probabilities=torch.stack(actions,1),alpha=torch.stack(alphas,1),
                predictions=torch.stack(predictions,1),protected=protected,reason=torch.stack(reasons,1),requested_alpha=p0.new_tensor(ALPHAS))


def action_utilities(action_log_probs,labels,baseline_predictions,lambda_h=2.,*,predictions=None):
    if action_log_probs.ndim!=3 or action_log_probs.shape[1]!=len(ALPHAS) or not math.isfinite(lambda_h) or lambda_h<1:
        raise ValueError('invalid realized actions/utility cost')
    if labels.shape!=(len(action_log_probs),) or baseline_predictions.shape!=labels.shape:raise ValueError('utility label shape mismatch')
    predictions=action_log_probs.argmax(-1) if predictions is None else predictions
    if predictions.shape!=action_log_probs.shape[:2] or predictions.dtype!=torch.long or (predictions<0).any() or (predictions>=action_log_probs.shape[-1]).any():raise ValueError('invalid action predictions')
    correct=predictions.eq(labels[:,None]);base=baseline_predictions.eq(labels)[:,None]
    utility=(correct&~base).double()-lambda_h*(~correct&base).double()
    if utility[:,0].abs().any():raise ValueError('zero action must be the actual baseline')
    return utility.detach()


def utility_features(s0,sg,quality,coverage):
    log0,logg=_log_pairs(s0,sg);p0,pg=log0.exp(),logg.exp();n,c=p0.shape
    q=quality.detach().to(p0);coverage=coverage.detach().to(p0)
    if q.shape!=(n,3) or coverage.shape!=(n,) or not torch.isfinite(q).all() or not torch.isfinite(coverage).all() or ((coverage<0)|(coverage>1)).any():
        raise ValueError('invalid quality/coverage descriptors')
    t0=p0.sort(-1,descending=True).values;tg=pg.sort(-1,descending=True).values
    lm=torch.logaddexp(log0,logg)-math.log(2.)
    entropy0=-(p0*log0).sum(-1)/math.log(c);entropyg=-(pg*logg).sum(-1)/math.log(c)
    js=.5*((p0*(log0-lm)).sum(-1)+(pg*(logg-lm)).sum(-1))
    symmetric=(t0[:,:min(3,c)]-tg[:,:min(3,c)]).abs().mean(-1)
    values=torch.stack((t0[:,0]-t0[:,1],tg[:,0]-tg[:,1],entropy0,entropyg,
                        (p0.argmax(-1)!=pg.argmax(-1)).double(),js,symmetric),-1)
    return torch.cat((values,q,coverage[:,None]),-1).detach()


class UtilityGate:
    def __init__(self,ridge=.001,utility_margin=0.):
        if not math.isfinite(ridge) or ridge<=0 or not math.isfinite(utility_margin) or utility_margin<0:raise ValueError('invalid gate configuration')
        self.ridge=float(ridge);self.utility_margin=float(utility_margin);self.fitted=False

    @torch.no_grad()
    def fit(self,phi,utilities,*,role='L_s',weights=None):
        if self.fitted:raise RuntimeError('utility gate is frozen')
        if role!='L_s':raise ValueError('gate fitting permits only L_s cross-fitted rows')
        x=phi.detach().double();y=utilities.detach().to(x)
        if x.ndim!=2 or x.shape[1]!=11 or y.shape!=(len(x),5) or not torch.isfinite(x).all() or not torch.isfinite(y).all() or y[:,0].abs().any():
            raise ValueError('invalid utility training arrays')
        weights=torch.ones(len(x),device=x.device,dtype=x.dtype)/len(x) if weights is None else weights.detach().to(x)
        if weights.shape!=(len(x),) or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('invalid physical utility weights')
        weights=weights/weights.sum();mean=(weights[:,None]*x).sum(0)
        scale=(weights[:,None]*(x-mean).square()).sum(0).sqrt().clamp_min(1e-8)
        design=torch.cat((torch.ones_like(x[:,:1]),(x-mean)/scale),-1)
        regularizer=self.ridge*torch.eye(design.shape[1],device=x.device,dtype=x.dtype);regularizer[0,0]=0
        coef=torch.linalg.solve(design.T@(weights[:,None]*design)+regularizer,design.T@(weights[:,None]*y[:,1:]))
        self.mean=mean.cpu();self.scale=scale.cpu();self.coefficients=coef.cpu();self.fitted=True
        return self

    def predict_utility(self,phi):
        if not self.fitted:raise RuntimeError('fit utility gate before prediction')
        x=phi.detach().double()
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not torch.isfinite(x).all():raise ValueError('invalid utility features')
        normalized=(x-self.mean.to(x))/self.scale.to(x)
        pred=torch.cat((torch.ones_like(x[:,:1]),normalized),-1)@self.coefficients.to(x)
        return torch.cat((torch.zeros_like(pred[:,:1]),pred),-1)

    def choose(self,phi):
        estimates=self.predict_utility(phi);value,index=estimates[:,1:].max(-1)
        action=torch.where(value>self.utility_margin,index+1,0)
        return dict(action=action,estimated_utility=estimates)

    def state_dict(self):
        if not self.fitted:raise RuntimeError('cannot export unfitted gate')
        return dict(version=1,ridge=self.ridge,utility_margin=self.utility_margin,mean=self.mean.clone(),scale=self.scale.clone(),coefficients=self.coefficients.clone())

    @classmethod
    def from_state_dict(cls,state):
        if set(state)!={'version','ridge','utility_margin','mean','scale','coefficients'} or state['version']!=1:raise ValueError('invalid gate state')
        obj=cls(state['ridge'],state['utility_margin'])
        if state['mean'].shape!=(11,) or state['scale'].shape!=(11,) or state['coefficients'].shape!=(12,4):raise ValueError('invalid gate shapes')
        if any(not torch.isfinite(state[k]).all() for k in ('mean','scale','coefficients')) or (state['scale']<=0).any():raise ValueError('invalid gate parameters')
        obj.mean=state['mean'].clone();obj.scale=state['scale'].clone();obj.coefficients=state['coefficients'].clone();obj.fitted=True
        return obj


def protection_quantile(s0,q=.9):
    probabilities=s0.detach().double().softmax(-1);top=probabilities.topk(2,dim=-1).values
    margin=top[:,0]-top[:,1];unique=margin>0
    return float(torch.quantile(margin[unique],q,interpolation='higher')) if unique.any() else 1.
