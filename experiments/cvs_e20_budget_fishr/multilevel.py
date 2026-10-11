"""Backbone-only linear-readout Fishr with explicit main-CE gradient budgets."""
from contextlib import contextmanager
import math
import numpy as np

import torch
from torch.nn import functional as F
from experiments.cvs_early_geometry.multilevel import (
    MultiLevel as OriginalMultiLevel, LEVELS, pooled_real_map, weak_mixstyle)


def weak_mixstyle_u(value,labels,receivers,days,ids,seed,p=.5,alpha=.1,eta=.3):
    """U-specific variant accepting repeated batch occurrences of a real PID.

    DomainStream may cross a permutation boundary within one batch. Occurrence
    indices are never substituted for physical IDs: every nontrivial donor must
    still have a different PID. Unique-PID inputs preserve the original random
    draw order and exact numerical operations. L keeps its original strict check.
    """
    if not 0<=p<=1 or not 0<=eta<=1 or alpha<=0:raise ValueError('Invalid registered MixStyle settings')
    pooled_real_map(value)
    y,rx,day=[v.detach().cpu().tolist() for v in (labels,receivers,days)]
    if any(len(v)!=len(value) for v in (ids,y,rx,day)):
        raise ValueError('U style metadata must match batch occurrences')
    rng=np.random.default_rng(int(seed))
    eligible=[[j for j in range(len(ids)) if y[j]==y[i] and rx[j]!=rx[i]
               and day[j]==day[i] and ids[j]!=ids[i]] for i in range(len(ids))]
    detail=dict(enabled=False,eligible_packets=sum(bool(v) for v in eligible),
                changed_packets=0,changed_ratio=0.,movement_mean=0.,delta_ratio=0.,donors=[])
    if rng.random()>=p:return value,detail
    donor=[int(rng.choice(v)) if v else i for i,v in enumerate(eligible)]
    movement=eta*(1-rng.beta(alpha,alpha,len(ids)))
    movement=np.where(np.asarray([bool(v) for v in eligible]),movement,0.)
    a=torch.tensor(movement.tolist(),dtype=value.dtype,device=value.device)[:,None,None]
    mu=value.mean(-1,keepdim=True).detach()
    sigma=(value.var(-1,unbiased=False,keepdim=True)+1e-6).sqrt().detach()
    output=(value-mu)/sigma*((1-a)*sigma+a*sigma[donor])+(1-a)*mu+a*mu[donor]
    output=torch.where(a!=0,output,value)
    ratio=(output.detach()-value.detach()).flatten(1).norm(dim=1)/value.detach().flatten(1).norm(dim=1).clamp_min(1e-6)
    detail.update(enabled=True,changed_packets=int((ratio>0).sum()),changed_ratio=float((ratio>0).float().mean()),
                  movement_mean=float(a.mean()),movement_max=float(a.max()),delta_ratio=float(ratio.mean()),donors=donor)
    return output,detail


def linear_sample_gradients(features, head, labels):
    """True CE parameter-gradient VALUES; readout parameters are fixed for Fishr.

    The probabilities remain differentiable through features. W/b are detached,
    so minimizing the regularizer cannot change its readout to hide differences.
    Auxiliary CE, computed separately, still learns W/b normally.
    """
    if head.bias is None or head.out_features != 6:
        raise ValueError('Expected biased six-class linear readout')
    logits=F.linear(features,head.weight.detach(),head.bias.detach())
    delta=logits.softmax(1)-F.one_hot(labels,6).to(features)
    augmented=torch.cat((features,torch.ones_like(features[:,:1])),1)
    return delta[:,:,None]*augmented[:,None,:]


def fishr_variance(features,head,labels,receivers):
    gradients=linear_sample_gradients(features,head,labels)
    domains=receivers.unique()
    if len(domains)<2:raise ValueError('At least two actual source RX required')
    values=[]
    for rx in domains:
        g=gradients[receivers==rx]
        if len(g)<2:raise ValueError('Insufficient actual samples for source RX variance')
        values.append((g-g.mean(0,keepdim=True)).square().mean(0))
    variance=torch.stack(values)
    loss=(variance-variance.mean(0,keepdim=True)).square().mean()
    return loss,dict(domains=domains.cpu().tolist(),variance_mean=float(variance.detach().mean()),
                     variance_max=float(variance.detach().max()),raw_fishr=float(loss.detach()),
                     coordinates=6*(features.shape[1]+1),variance_denominator='n',
                     variance_normalized=False,readout_detached_for_regularizer=True,
                     gradients='true linear weight+bias CE parameter-gradient values')


def _parameter_groups(model):
    patterns={'time_down':'.time_fuse.','t2.pool':'.t2.'}
    return {level:tuple(p for n,p in model.named_parameters()
                        if p.requires_grad and pattern in '.'+n and 'id_backbone' in n
                        and not n.endswith(('low_hz_','band_hz_')))
            for level,pattern in patterns.items()}


def _norm(loss,parameters):
    if not parameters or not loss.requires_grad:return 0.
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))


def budget_fishr(model,main_ce,per_layer_terms,config,ramp):
    """Detached active-step coefficient; final two-level reduction is an explicit mean.

    The 10% request is per layer BEFORE the final mean; each layer's direct
    contribution is 5% after the mean. Middle-layer cross-gradients can also
    reach the shallow group, so the combined group ratio is measured separately.
    """
    mode=config.get('fishr_mode','none')
    if mode not in ('none','raw','budget'):raise ValueError('Unknown Fishr budget mode')
    zero=main_ce*0
    detail=dict(mode=mode,levels={},level_reduction='mean',weighted_loss=0.,EMA=False)
    if not per_layer_terms or mode=='none':return zero,detail
    if set(per_layer_terms)!=set(LEVELS):raise ValueError('Missing shallow/middle Fishr term')
    ramp=float(ramp)
    ratio=float(config.get('fishr_ratio',.1));eps=float(config.get('fishr_eps',1e-12))
    cap=float(config.get('fishr_cap',10000.));raw_weight=float(config.get('fishr_raw_weight',2.))
    if not 0<=ramp<=1 or ratio<0 or eps<=0 or cap<=0 or raw_weight<0:
        raise ValueError('Invalid fixed gradient-budget definition')
    groups=_parameter_groups(model);weighted=[]
    for level in LEVELS:
        term=per_layer_terms[level];parameters=groups[level]
        ce_norm=_norm(main_ce,parameters);fishr_norm=_norm(term,parameters)
        valid=bool(parameters and math.isfinite(ce_norm+fishr_norm) and ce_norm>eps and fishr_norm>eps)
        requested=ratio*ramp
        uncapped=requested*ce_norm/(fishr_norm+eps) if valid and mode=='budget' else raw_weight*ramp
        coefficient=(min(cap,uncapped) if mode=='budget' else uncapped) if valid else 0.
        reason=None if valid else 'missing_nonfinite_or_negligible_standard_parameter_gradient'
        # Python scalar coefficients deliberately have no derivative through the norms.
        weighted.append(term*coefficient if valid else zero)
        achieved=coefficient*fishr_norm/ce_norm if valid else 0.
        raw_loss=float(term.detach())
        detail['levels'][level]=dict(main_ce_gradient_norm=ce_norm if math.isfinite(ce_norm) else None,
            raw_fishr_gradient_norm=fishr_norm if math.isfinite(fishr_norm) else None,
            coefficient=coefficient,uncapped_coefficient=uncapped if valid else None,
            coefficient_detached=True,coefficient_cap=cap if mode=='budget' else None,
            cap_active=valid and mode=='budget' and uncapped>cap,
            requested_ratio_pre_level_mean=requested if mode=='budget' else None,
            achieved_ratio_pre_level_mean=achieved,achieved_direct_contribution_ratio=achieved/len(LEVELS),
            parameters=len(parameters),reason=reason,raw_loss=raw_loss if math.isfinite(raw_loss) else None)
    combined=torch.stack(weighted).mean()
    detail['weighted_loss']=float(combined.detach())
    detail['combined_group_gradient_ratios']={}
    for level,parameters in groups.items():
        reference=detail['levels'][level]['main_ce_gradient_norm']
        combined_norm=_norm(combined,parameters)
        detail['combined_group_gradient_ratios'][level]=dict(gradient_norm=combined_norm,
            ratio_to_main_ce=combined_norm/reference if reference is not None and reference>eps else None)
    return combined,detail


class MultiLevel(OriginalMultiLevel):
    @torch.no_grad()
    def extract_clean_features(self,model,x):
        """Legal source-L cache for warm-head optimization; does not fit the backbone."""
        modes=[(m,m.training) for m in model.modules()]
        devices=sorted({p.device.index for p in model.parameters() if p.device.type=='cuda'})
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices),self.capture() as (maps,_):model(x)
            return {level:pooled_real_map(maps[level]).detach() for level in LEVELS}
        finally:
            for module,mode in modes:module.training=mode

    def probe_ce(self,features,labels):
        if set(features)!=set(LEVELS):raise ValueError('Expected both cached source readout features')
        return torch.stack([F.cross_entropy(self.heads[self.key(level)](features[level]),labels)
                            for level in LEVELS]).mean()

    def _loss(self,model,source,step,device,config):
        zero=next(self.heads.parameters()).sum()*0
        empty=dict(aux_forwards=0,auxiliary_packets=0,fishr_steps=0,levels={})
        if not config.get('enabled',False) or not model.training or step%int(config.get('interval',4)):
            return zero,{},empty
        if config.get('batch_size',60)!=60 or tuple(config.get('levels',LEVELS))!=LEVELS:
            raise ValueError('Auxiliary source layout differs')
        index=self.balanced_indices(source,step,int(config.get('interval',4)))
        x=source['x'][index].to(device);y=source['y'][index].to(device)
        rx=source['rx'][index].to(device);day=source['day'][index].to(device)
        ids=[source['ids'][i] for i in index.tolist()]
        with self.capture(step=step) as (maps,_):model(x)
        features={level:pooled_real_map(maps[level]) for level in LEVELS}
        ces={level:F.cross_entropy(self.heads[self.key(level)](f),y) for level,f in features.items()}
        mode=config.get('fishr_mode','none');terms={}
        if mode not in ('none','raw','budget'):raise ValueError('Unknown Fishr mode')
        detail=dict(aux_forwards=1,auxiliary_packets=60,distinct_PID=60,day=int(day[0]),
                    fishr_steps=int(mode!='none'),levels={},native_CPU_CUDA_RNG_preserved=True,
                    source_RX=rx.unique().cpu().tolist(),EMA=False,head_Fishr_gradient=False)
        for level,feature in features.items():
            record=dict(clean_ce=float(ces[level].detach()),shape=list(maps[level].shape))
            if mode!='none':terms[level],record_f=fishr_variance(feature,self.heads[self.key(level)],y,rx);record.update(record_f)
            detail['levels'][level]=record
        if config.get('mixstyle_l',False):
            with self.capture(mix=True,labels=y,receivers=rx,days=day,ids=ids,step=step,config=config) as (mixed,styles):model(x)
            detail['aux_forwards']+=1
            for level in LEVELS:
                mce=F.cross_entropy(self.heads[self.key(level)](pooled_real_map(mixed[level])),y)
                ces[level]=.5*(ces[level]+mce)
                detail['levels'][level].update(mixed_ce=float(mce.detach()),style=styles[level])
        ce=torch.stack(list(ces.values())).mean()
        detail['aux_ce']=float(ce.detach())
        detail['fishr']=float(torch.stack(list(terms.values())).mean().detach()) if terms else 0.
        return ce,terms,detail

    @contextmanager
    def main_style(self,probabilities,receivers,days,ids,step,config):
        """Style only the live U strong forward, with no access to U truth.

        Keeps main-path dropout untouched. Style uses private NumPy RNG; the
        single normal main forward advances its normal Torch RNG exactly once.
        """
        base=self._base();detail=dict(role='U_strong',levels={},high_confidence=0,used_U_truth=False)
        if self._active:raise RuntimeError('Nested style/capture contexts prohibited')
        if base is None:raise RuntimeError('Identity base expired')
        if probabilities.ndim!=2 or probabilities.shape[1]!=6 or not torch.isfinite(probabilities).all():
            raise ValueError('Expected detached weak-student probabilities [B,6]')
        if (probabilities<0).any() or not torch.allclose(probabilities.sum(1),torch.ones_like(probabilities[:,0]),atol=1e-4):
            raise ValueError('Pseudo-style input must be probabilities, not logits')
        q,pseudo=probabilities.detach().max(1)
        threshold=float(config.get('style_confidence',.95))
        if not .95<=threshold<=1:raise ValueError('Pseudo-style requires confidence at least .95')
        high=q>=threshold
        # Every rejected item gets its own non-class integer; no rejected donor.
        labels=torch.where(high,pseudo,torch.arange(len(q),device=q.device)+6)
        detail['high_confidence']=int(high.sum())
        enabled=bool(config.get('mixstyle_u',False) and self.training and base.training)
        handles=[];self._active=True
        try:
            if enabled:
                modules=dict(base.named_modules())
                for index,level in enumerate(LEVELS):
                    def hook(module,inputs,value,level=level,index=index):
                        if level in detail['levels']:raise RuntimeError('Multiple forwards in U-style context')
                        output,stats=weak_mixstyle_u(value,labels,receivers,days,ids,
                            self.seed+int(step)*701+index*1709+990117,
                            p=config.get('mixstyle_p',.5),alpha=config.get('mixstyle_alpha',.1),
                            eta=config.get('mixstyle_strength',.3))
                        detail['levels'][level]=stats
                        return output
                    handles.append(modules[self.paths[level]].register_forward_hook(hook))
            yield detail
        finally:
            for handle in handles:handle.remove()
            self._active=False
