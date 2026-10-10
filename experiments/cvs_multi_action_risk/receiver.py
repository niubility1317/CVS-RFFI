"""Source-only receiver actions in five-dimensional competitor-margin space.

Physical hash roles separate descriptors, recipients, and destination statistics.
No paired-transmission interpretation, raw-z alignment, or target data is used.
"""
from collections import defaultdict
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_multi_action_audit.receiver import packet_time_statistics, _fixed
from experiments.cvs_multi_state_action.receiver import validate_data, _values, make_relations
from experiments.cvs_multi_disentangle.model import (
    FixedViews, intermediate, classify_intermediate,
)


def margin_vector(logits, y):
    if logits.shape[1] != 6:
        raise ValueError('Registered six-class R requires five competitor margins')
    labels = torch.as_tensor(y, device=logits.device, dtype=torch.long).expand(len(logits))
    mask = torch.arange(6, device=logits.device)[None] != labels[:, None]
    return (logits.gather(1, labels[:, None]) - logits)[mask].reshape(len(logits), 5)


def margin_statistics(margins, prior=None):
    """Small-bag estimator; shrink toward diagonal then coarser matched stats.

    `prior` has the same TX/RX/condition but may span source days. It never
    changes descriptor/recipient/destination roles or physical holdout partition.
    Error estimates are descriptive plug-in SE/split sensitivity, not CIs.
    """
    n = len(margins)
    if not n:
        raise ValueError('Empty margin bag')
    mean = margins.mean(0)
    centered = margins - mean
    covariance = centered.T @ centered / max(n - 1, 1)
    diagonal = torch.diag(torch.diagonal(covariance))
    shrink = 5. / (n + 5.)
    covariance = (1-shrink)*covariance + shrink*diagonal
    qs = margins.new_tensor([.05, .10, .25, .50])
    lower = torch.quantile(margins.min(1).values, qs)
    weight = n / (n+8.)
    if prior is not None:
        mean = weight*mean + (1-weight)*prior['mean'].detach()
        covariance = weight*covariance + (1-weight)*prior['covariance'].detach()
        lower = weight*lower + (1-weight)*prior['lower_tail'].detach()
    raw_var = margins.var(0, unbiased=n > 1)
    se = (raw_var/n).clamp_min(0).sqrt()
    # A singleton has no identifiable variance/tail sampling error.
    if n >= 4:
        halves = [margins[k::2] for k in (0, 1)]
        half_cov = [((v-v.mean(0)).T @ (v-v.mean(0)))/max(len(v)-1,1) for v in halves]
        cov_error = (half_cov[0]-half_cov[1]).square().mean().sqrt()
        tails = [torch.quantile(v.min(1).values, qs) for v in halves]
        tail_error = (tails[0]-tails[1]).square().mean().sqrt()
    else:
        cov_error = tail_error = None
    return dict(mean=mean, covariance=covariance, lower_tail=lower,
                mean_se=se if n > 1 else None, covariance_split_error=cov_error,
                tail_split_error=tail_error, distinct_ids=n, shrinkage=shrink,
                evidence_weight=weight, coarse_shrink=prior is not None)


def statistic_errors(pred, target):
    # Classifier temperature stays fixed at 30; physical units retained in logs.
    return dict(mean=(pred['mean']-target['mean']).square().mean()/900.,
                covariance=(pred['covariance']-target['covariance']).square().mean()/810000.,
                lower_tail=(pred['lower_tail']-target['lower_tail']).square().mean()/900.)


def fit_objective(pred, target, target_mode='distribution'):
    errors = statistic_errors(pred, target)
    if target_mode not in ('mean', 'distribution'):
        raise ValueError('Unknown receiver target mode')
    loss = errors['mean'] if target_mode == 'mean' else sum(errors.values())
    return loss, errors


def cvar(risk, alpha=.25):
    """Exact empirical upper-tail CVaR, including fractional boundary mass."""
    if not 0 < alpha <= 1 or risk.numel() == 0:
        raise ValueError('CVaR needs nonempty observations and alpha in (0,1]')
    ordered = risk.sort(descending=True).values
    mass = alpha * len(ordered)
    whole = int(math.floor(mass))
    total = ordered[:whole].sum()
    if whole < len(ordered):
        total = total + (mass-whole)*ordered[whole]
    return total/mass


def receiver_risk(identity, moved, y, alpha=.25, tail_weight=.25):
    logits = classify_intermediate(identity, moved)
    labels = torch.full((len(moved),), int(y), device=moved.device, dtype=torch.long)
    ce = F.cross_entropy(logits, labels)
    tail = cvar(F.softplus(-margin_vector(logits, y).min(1).values), alpha)
    return ce + tail_weight*tail, dict(ce=ce, cvar=tail)


def descriptors(x):
    from .physics import quality_proxies
    views = FixedViews().to(x.device).eval()
    value = torch.cat((views.receiver(x), packet_time_statistics(x)), 1)
    # FixedViews.receiver[29] was normalized log power, not a quality proxy.
    # Replace three moment slots while preserving the 49-dimensional contract.
    value[:, 29:32] = quality_proxies(x)
    return value


class ReceiverDistributionAction(nn.Module):
    """Same R functional form for independent and shared-trunk controls."""
    def __init__(self, width=49, dimension=349, shared_core=None):
        super().__init__()
        from .actions import SharedActionCore
        self.core = shared_core if shared_core is not None else SharedActionCore(hidden=64)
        self.encoder = nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,8))
        self.decoder = nn.Sequential(nn.Linear(self.core.hidden+8,64),nn.SiLU(),nn.Linear(64,dimension))
        nn.init.normal_(self.decoder[-1].weight,std=.001)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,h,v0,v1):
        q=(self.encoder(v1)-self.encoder(v0)).expand(len(h),-1)
        state=torch.cat((v0[:,:20],v1[:,:20]),1).expand(len(h),-1)
        s=self.core(h,state,'receiver')
        return self.decoder(torch.cat((s,q),1))-self.decoder(torch.cat((s,torch.zeros_like(q)),1))


def _digest(pid):
    return int(hashlib.sha256(str(pid).encode('utf-8')).hexdigest(),16)


def _distinct(entries):
    return list({e['pid']:e for e in entries}.values())


class OnlineReceiver(nn.Module):
    def __init__(self,max_age_steps=888,action=None,target_mode='distribution',tx_holdout=(0,),
                 action_radius=.1,tail_weight=.25,cvar_alpha=.25,shared_core=None):
        super().__init__()
        if target_mode not in ('mean','distribution'): raise ValueError('Invalid R fit target')
        self.action=action if action is not None else ReceiverDistributionAction(shared_core=shared_core)
        self.target_mode=target_mode;self.tx_holdout=set(map(int,tx_holdout))
        self.max_age_steps=int(max_age_steps);self.action_radius=float(action_radius)
        self.tail_weight=float(tail_weight);self.cvar_alpha=float(cvar_alpha)
        self.entries={};self.reference_version=None;self.expired=0;self.counter=0
        self.live_calibration={};self.used_ids=set();self.seen_ids=set()
        self.usage=dict(fit_used=0,identity_used=0,audit_used=0)

    def reset(self,version):
        self.entries.clear();self.reference_version=str(version);self.live_calibration.clear()

    def expire(self,step,version):
        if self.reference_version != str(version): self.reset(version)
        for key,e in list(self.entries.items()):
            age=int(step)-e['step']
            if age<0: raise ValueError('Receiver contribution clock moved backwards')
            if age>self.max_age_steps: del self.entries[key];self.expired+=1
        live_ids={e['pid'] for e in self.entries.values()}
        for condition,state in list(self.live_calibration.items()):
            state['ids'].intersection_update(live_ids)
            state['rows']=[row for row in state['rows'] if int(step)-row['observed_step']<=self.max_age_steps]
            if not state['rows']:del self.live_calibration[condition]

    def counts(self,available=0,**extra):
        return dict(available=available,**self.usage,unique_physical_ids=len(self.seen_ids),
                    unique_used_physical_ids=len(self.used_ids),live_unique_physical_ids=len({e['pid'] for e in self.entries.values()}),
                    count_semantics='available is current candidate count; *_used cumulative actual relation uses; unique IDs never sum repeats',**extra)

    def update(self,x,h,y,rx,day,ids,condition,step,version,reference_identity):
        self.expire(step,version)
        fields=validate_data(dict(x=x,y=y,rx=rx,day=day,ids=ids,condition=condition))
        with torch.no_grad():
            v=descriptors(x);logits=classify_intermediate(reference_identity,h)
        for i,pid in enumerate(fields['ids']):
            existing=[e for (p,_),e in self.entries.items() if p==pid]
            meta=(int(fields['y'][i]),int(fields['rx'][i]),int(fields['day'][i]))
            if any((e['y'],e['rx'],e['day'])!=meta for e in existing):
                raise ValueError('Physical ID metadata changed across updates')
            self.seen_ids.add(pid)
            key=(pid,fields['condition'][i])
            if key in self.entries: continue
            self.entries[key]=dict(pid=pid,condition=fields['condition'][i],y=meta[0],rx=meta[1],day=meta[2],
                step=int(step),role=_digest(pid)%3,partition=(_digest(pid)//3)%2,
                h=h[i].detach(),v=v[i].detach(),logits=logits[i].detach())
            if _digest(pid)%3==1 and (_digest(pid)//3)%2==1:
                # Audit-only IQ permits fresh current-student features from a
                # real multi-physical bag; frozen h is not a live distribution.
                self.entries[key]['audit_x']=x[i].detach().clone()
        return self.counts(contributions=len(self.entries),expired=self.expired,reference_version=str(version))

    def _groups(self,step,version,partition=0):
        self.expire(step,version);groups=defaultdict(lambda:[[],[],[]])
        for e in self.entries.values():
            if e['partition']==partition:
                groups[(e['y'],e['rx'],e['day'],e['condition'])][e['role']].append(e)
        return groups

    def _contexts(self,groups,source_key,fit=False):
        y,rx,day,condition=source_key
        if fit and y in self.tx_holdout: return
        for dest_key,roles in sorted(groups.items(),key=lambda kv:str(kv[0])):
            if dest_key[0]!=y or dest_key[1]==rx or dest_key[2:]!=(day,condition) or not roles[2]:continue
            donors=[k for k,bags in groups.items() if k[0]!=y and (not fit or k[0] not in self.tx_holdout)
                and k[1:]==(rx,day,condition) and bags[0]
                and (k[0],dest_key[1],day,condition) in groups and groups[(k[0],dest_key[1],day,condition)][0]]
            if not donors:continue
            d0=[e for k in donors for e in groups[k][0]]
            d1=[e for k in donors for e in groups[(k[0],dest_key[1],day,condition)][0]]
            v0=torch.stack([torch.stack([e['v'] for e in groups[k][0]]).mean(0) for k in donors]).mean(0)[None]
            v1=torch.stack([torch.stack([e['v'] for e in groups[(k[0],dest_key[1],day,condition)][0]]).mean(0) for k in donors]).mean(0)[None]
            delta=torch.stack([torch.stack([e['h'] for e in groups[(k[0],dest_key[1],day,condition)][0]]).mean(0)
                -torch.stack([e['h'] for e in groups[k][0]]).mean(0) for k in donors]).mean(0)
            dest=_distinct(roles[2])[:32]
            # Coarser destination-only prior across source days, same TX/RX/condition.
            coarse=_distinct([e for k,b in groups.items() if k[0:2]==dest_key[0:2]
                and k[3]==condition and k[2]!=day for e in b[2]])[:64]
            yield dict(key=dest_key,dest=dest,coarse=coarse,v0=v0/8,v1=v1/8,mean_delta=delta,
                donors=d0+d1,donor_tx=[k[0] for k in donors])

    def _relations(self,groups,fit=False):
        return [(key,_distinct(bags[1])[:32],context) for key,bags in sorted(groups.items(),key=lambda kv:str(kv[0]))
                if bags[1] for context in self._contexts(groups,key,fit=fit)]

    def _target(self,c,y):
        with torch.no_grad():
            prior=margin_statistics(margin_vector(torch.stack([e['logits'] for e in c['coarse']]),y)) if c['coarse'] else None
            return margin_statistics(margin_vector(torch.stack([e['logits'] for e in c['dest']]),y),prior)

    def _delta(self,h,c):
        raw=self.action(h,c['v0'],c['v1'])
        radius=self.action_radius*h.detach().norm(dim=1,keepdim=True).clamp_min(1e-3)
        # Smooth bounded radial map, exact zero action and no hard clipping kink.
        return raw*radius/(radius+raw.norm(dim=1,keepdim=True))

    def _mark(self,mode,source,c):
        bags=[{e['pid'] for e in entries} for entries in (c['donors'],source,c['dest']+c['coarse'])]
        if any(bags[a]&bags[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError('Receiver physical bag overlap')
        self.usage[mode]+=1;self.used_ids.update(set.union(*bags))

    def fit_loss(self,reference_identity,step,version):
        relations=self._relations(self._groups(step,version),fit=True)
        if not relations:return None,self.counts(active=False,reason='insufficient_disjoint_day_matched_bags')
        key,source,c=relations[self.counter%len(relations)];self.counter+=1
        h=torch.stack([e['h'] for e in source]);delta=self._delta(h,c)
        target=self._target(c,key[0])
        # Context manager freezes reference parameters while retaining dloss/dh.
        with _fixed(reference_identity):
            pred=margin_statistics(margin_vector(classify_intermediate(reference_identity,h+delta),key[0]))
            fit,parts=fit_objective(pred,target,self.target_mode)
        magnitude=delta.square().mean()/h.square().mean().clamp_min(1e-6)
        # Nearest recipient in the SAME source condition, no synthetic cross-TX pair.
        if len(h)>1:
            dist=torch.cdist(h.detach(),h.detach());dist.fill_diagonal_(float('inf'))
            ix=dist.argmin(1)
            smooth=(delta-delta[ix]).square().sum(1).div((h-h[ix]).square().sum(1).clamp_min(1e-4)).mean()
        else:smooth=delta.sum()*0
        weight=min(len(source)/(len(source)+8.),target['evidence_weight'])
        loss=weight*fit+.01*magnitude+.01*smooth
        self._mark('fit_used',source,c)
        stats=self.counts(len(relations),active=True,fit_used_this_call=1,recipient_packets=len(h),
            destination_distinct_ids=len(c['dest']),coarse_distinct_ids=len(c['coarse']),evidence_weight=weight,
            condition=key[3],recipient_tx=key[0],donor_tx=c['donor_tx'],target_mode=self.target_mode,
            magnitude_loss=float(magnitude.detach()),smoothness_loss=float(smooth.detach()),
            **{k+'_loss':float(v.detach()) for k,v in parts.items()},**self._uncertainty(target))
        return loss,stats

    @staticmethod
    def _uncertainty(target):
        return {k:None if target[k] is None else float(target[k].detach().square().mean().sqrt())
                for k in ('mean_se','covariance_split_error','tail_split_error')}

    @torch.no_grad()
    def _audit_row(self,identity,h,key,c):
        target=self._target(c,key[0]);record={}
        supported=dict(mean=True,covariance=min(len(h),len(c['dest']))>=2,
                       lower_tail=min(len(h),len(c['dest']))>=4)
        with _fixed(identity):
            for mode,moved in (('learned',h+self._delta(h,c)),('zero',h),('directed_other_tx_mean',h+c['mean_delta'])):
                pred=margin_statistics(margin_vector(classify_intermediate(identity,moved),key[0]))
                for space,error in statistic_errors(pred,target).items():
                    record[mode+'_'+space]=float(error) if supported[space] else None
        return dict(record,recipient_tx=key[0],source_rx=key[1],destination_rx=c['key'][1],day=key[2],
            condition=key[3],tx_heldout=key[0] in self.tx_holdout,
            evidence_weight=min(len(h)/(len(h)+8.),target['evidence_weight']),
            recipient_distinct_ids=len(h),supported_statistics=supported,
            destination_distinct_ids=len(c['dest']),**self._uncertainty(target))

    @staticmethod
    def _reliability(rows):
        spaces=('mean','covariance','lower_tail')
        means={};counts={}
        for space in spaces:
            supported=[r for r in rows if all(r[mode+'_'+space] is not None
                       for mode in ('learned','zero','directed_other_tx_mean'))]
            counts[space]=len(supported)
            for mode in ('learned','zero','directed_other_tx_mean'):
                means[mode+'_'+space]=(sum(r[mode+'_'+space] for r in supported)/len(supported)) if supported else None
        skills={space:(1-means['learned_'+space]/max(min(means['zero_'+space],means['directed_other_tx_mean_'+space]),1e-12)
                if counts[space] else None) for space in spaces}
        evidence=sum(r['evidence_weight'] for r in rows)/len(rows)
        complete=all(value is not None for value in skills.values())
        weight=max(0.,min(1.,*skills.values()))*evidence if complete else 0.
        return weight,dict(**means,skills=skills,supported_relation_counts=counts,
                           all_distribution_statistics_supported=complete,evidence_weight=evidence)

    @torch.no_grad()
    def calibrate(self,reference_identity,step,version,max_relations=32):
        relations=self._relations(self._groups(step,version,partition=1))
        if not relations:return self.counts(active=False,weights={},reason='insufficient_heldout_receiver_bags')
        start=self.counter%len(relations);selected=(relations[start:]+relations[:start])[:max_relations]
        records=[];by_condition=defaultdict(list)
        for key,source,c in selected:
            row=self._audit_row(reference_identity,torch.stack([e['h'] for e in source]),key,c)
            records.append(row);by_condition[key[3]].append(row);self._mark('audit_used',source,c)
        weights={};metrics={}
        for condition,rows in by_condition.items():weights[condition],metrics[condition]=self._reliability(rows)
        return self.counts(len(relations),active=True,weights=weights,metrics=metrics,records=records,
            audit_used_this_call=len(selected),physical_holdout=True,reference_version=str(version),
            auxiliary_tx_holdout=sorted(self.tx_holdout),tx_holdout_records=[r for r in records if r['tx_heldout']])

    def identity_loss(self,identity,h,y,rx,day,ids,condition,step,version,reliability=1.):
        groups=self._groups(step,version);n=len(h)
        ys,rxs,days,conditions,pids=[_values(v,n) for v in (y,rx,day,condition,ids)]
        current=defaultdict(list)
        for i,(yy,rr,dd,cc,pid) in enumerate(zip(ys,rxs,days,conditions,pids)):
            if min(int(yy),int(rr),int(dd))<0:raise ValueError('R identity requires source-L metadata')
            if _digest(pid)%3==1 and (_digest(pid)//3)%2==0:
                current[(int(yy),int(rr),int(dd),cc)].append(i)
        losses=[];ces=[];tails=[];count=0
        for key,indices in current.items():
            # repeated views/IDs do not inflate effective recipient size
            ix=list({pids[i]:i for i in indices}.values())[:32]
            for c in self._contexts(groups,key):
                live=h[ix].detach()
                with torch.no_grad():delta=self._delta(live,c)
                risk,parts=receiver_risk(identity,live+delta.detach(),key[0],self.cvar_alpha,self.tail_weight)
                evidence=min(len(ix)/(len(ix)+8.),len(c['dest'])/(len(c['dest'])+8.))
                losses.append(evidence*risk);ces.append(parts['ce']);tails.append(parts['cvar']);count+=1
                if reliability>0:self._mark('identity_used',[dict(pid=pids[i]) for i in ix],c)
        if not losses:return None,self.counts(active=False,reason='no_live_recipient_with_matched_destination')
        return torch.stack(losses).mean(),self.counts(count,active=True,identity_used_this_call=count if reliability>0 else 0,
            attempted_relations=count,ce=float(torch.stack(ces).mean().detach()),cvar=float(torch.stack(tails).mean().detach()),
            encoder_gradient=False,action_gradient=False,identity_target='CE+CVaR',target_mode=self.target_mode)

    @torch.no_grad()
    def calibrate_live(self,identity,h,y,rx,day,ids,condition,step,version,min_distinct_ids=16,ema_decay=.9):
        """Re-encode <=32 audit-only physical IQ packets in current coordinates.

        Small incoming batches trigger an accumulated, independent physical bag,
        not a singleton covariance. Every returned metric is recomputed now;
        no metric/feature EMA mixes successive student parameter versions.
        """
        groups=self._groups(step,version,partition=1);n=len(h)
        ys,rxs,days,conditions,pids=[_values(v,n) for v in (y,rx,day,condition,ids)]
        current=defaultdict(list)
        for i,(yy,rr,dd,cc,pid) in enumerate(zip(ys,rxs,days,conditions,pids)):
            if _digest(pid)%3==1 and (_digest(pid)//3)%2==1:
                current[(int(yy),int(rr),int(dd),cc)].append(i)
        eligible={key:_distinct([e for e in bags[1] if 'audit_x' in e])[:32]
                  for key,bags in groups.items()}
        eligible={key:source for key,source in eligible.items() if len(source)>=4}
        ordered=sorted(eligible,key=str)
        if ordered:
            start=int(step)%len(ordered);ordered=ordered[start:]+ordered[:start]
        # Prioritize incoming strata, then fill with other actual audit strata.
        ordered=[key for key in current if key in eligible]+[key for key in ordered if key not in current]
        rows=defaultdict(list);new_ids=defaultdict(set);budget=32;reencoded=0
        for key in ordered:
            source=eligible[key][:budget]
            if len(source)<4:break
            contexts=list(self._contexts(groups,key))
            if not contexts:continue
            with _fixed(identity):
                fresh=intermediate(identity,torch.stack([e['audit_x'] for e in source])).detach()
            budget-=len(source);reencoded+=len(source)
            for c in contexts:
                rows[key[3]].append(dict(self._audit_row(identity,fresh,key,c),observed_step=int(step)))
                new_ids[key[3]].update(e['pid'] for e in source)
                self._mark('audit_used',source,c)
        previous=self.live_calibration
        self.live_calibration={condition:dict(ids=new_ids[condition],rows=observed,
             updates=previous.get(condition,{}).get('updates',0)+1) for condition,observed in rows.items()}
        weights={};metrics={}
        for condition,state in self.live_calibration.items():
            weight,metric=self._reliability(state['rows'])
            sufficient=len(state['ids'])>=min_distinct_ids
            weights[condition]=weight if sufficient else 0.
            metrics[condition]=dict(metric,distinct_recipient_ids=len(state['ids']),updates=state['updates'],sufficient_distinct_ids=sufficient)
        return self.counts(active=bool(rows),weights=weights,metrics=metrics,records=dict(rows),physical_holdout=True,
            min_distinct_recipient_ids=min_distinct_ids,reference_version=str(version),
            reencoded_audit_packets=reencoded,audit_packet_budget=32,
            semantics='Fresh current-student IQ bag margins vs true frozen-reference destination bags; no stale feature or metric averaging',
            combination='min(reference reliability,current-student reliability); no truth from target')


def receiver_audit(identity,fit,audit,generator,steps=200,lr=2e-4,repeats=2,reference_version='risk_source_reference',**unused):
    """Frozen-source audit with full new statistics and TX-heldout records."""
    fm,am=validate_data(fit),validate_data(audit)
    if set(fm['ids'])&set(am['ids']):raise ValueError('Auxiliary fit/audit physical packet overlap')
    device=fit['x'].device
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        torch.manual_seed(generator.initial_seed()+8401)
        online=OnlineReceiver(max_age_steps=steps+1).to(device)
    def populate(data,partition):
        with torch.no_grad():
            for ix in range(0,len(data['x']),64):
                batch={k:(v[ix:ix+64] if not isinstance(v,str) else v) for k,v in data.items()}
                h=intermediate(identity,batch['x'])
                online.update(batch['x'],h,batch['y'],batch['rx'],batch['day'],batch['ids'],batch['condition'],0,reference_version,identity)
        # Explicit disjoint fit/audit inputs supersede hash partition, role still physical hash.
        wanted=set(_values(data['ids'],len(data['x'])))
        for e in online.entries.values():
            if e['pid'] in wanted:e['partition']=partition
    with _fixed(identity):
        populate(fit,0);populate(audit,1)
        optimizer=torch.optim.Adam(online.parameters(),lr=lr);logs=[]
        for step in range(steps):
            loss,stats=online.fit_loss(identity,step,reference_version)
            if loss is None:break
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            logs.append(dict(step=step+1,loss=float(loss.detach()),**stats))
        result=online.calibrate(identity,steps,reference_version,max_relations=10000)
    return dict(report=dict(result,status='VERIFIED' if logs and result['active'] else 'INSUFFICIENT_MATCHED_GROUPS',
        scope='source_L_only',target_access=False,identity_updated=False,independent_bags=True,
        fit_steps=len(logs),repeat_count=1,requested_repeats=repeats,
        repeat_note='Physical partition fixed; no artificial independent sample multiplication',
        auxiliary_parameters=sum(p.numel() for p in online.parameters())),
        state_dict={k:v.detach().cpu() for k,v in online.state_dict().items()},step_logs=logs)
