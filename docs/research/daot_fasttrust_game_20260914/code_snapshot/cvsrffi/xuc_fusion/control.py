"""C*: private source evidence, unique-observation confirmation and bounded actions."""
from copy import deepcopy
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from cvsrffi.game_tracking.runtime import extract
from cvsrffi.game_tracking.data import audit_indices, labeled_source_records
from cvsrffi.game_tracking.source_audit import isolated_rng
from cvsrffi.cross_response.tensor_ops import normalized_identity_interaction_loss, grid_decomposition
from cvsrffi.eval import apply_sat_channel_for_scenario

SCENES=('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
GEOMETRY=('identity','margin','tx_main_energy','unit_interaction','leo_cosine')


def recovery_quality(fit_before, fit_after, differences, groups, seed=392005):
    if len(differences)!=len(groups) or not groups:
        return dict(valid=False,reason='missing_paired_groups')
    keys=sorted(set(groups))
    means=np.array([np.mean([d for d,g in zip(differences,groups) if g==key]) for key in keys])
    if len(keys)<2 or not np.isfinite(means).all() or not math.isfinite(fit_before+fit_after):
        return dict(valid=False,reason='invalid_group_evidence')
    rng=np.random.default_rng(seed)
    estimates=means[rng.integers(0,len(means),size=(500,len(means)))].mean(1)
    upper=float(np.quantile(estimates,.975))
    valid=fit_after<fit_before and upper<=0
    return dict(valid=bool(valid),reason='VALID_RECOVERY' if valid else 'RECOVERY_UNRELIABLE',
                fit_ce_before=float(fit_before),fit_ce_after=float(fit_after),
                monitor_delta_group_mean=float(means.mean()),monitor_delta_upper95=upper,
                independent_groups=len(keys),resamples=500,bootstrap_seed=seed)


class SourceObserver:
    def __init__(self, source, args):
        self.source,self.args=source,args
        self.indices=audit_indices(source,4)
        records=labeled_source_records(source)
        cells={}
        for row in records: cells.setdefault((row['day'],row['tx'],row['rx']),[]).append(row['index'])
        self.geometry_indices=[];self.geometry_shapes=[]
        for day in sorted({key[0] for key in cells}):
            txs=sorted({k[1] for k in cells if k[0]==day});rxs=sorted({k[2] for k in cells if k[0]==day})
            k=min(4,min(len(cells[(day,t,r)]) for t in txs for r in rxs))
            ids=[i for t in txs for r in rxs for i in cells[(day,t,r)][:k]]
            self.geometry_indices.extend(ids);self.geometry_shapes.append((len(txs),len(rxs),k,len(ids)))

    def observe(self, model, *, step, epoch, adv_weight):
        started=time.perf_counter()
        flags={m:m.training for m in model.modules()}
        buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        try:
            with isolated_rng(self.args.seed+81103):
                model.eval()
                fit=extract(model,self.source.train,self.indices['domain_fit'],self.args.eval_batch_size)
                mon=extract(model,self.source.train,self.indices['domain_monitor'],self.args.eval_batch_size)
                if set(fit['groups']) & set(mon['groups']): raise ValueError('Recovery capture overlap')
                head=deepcopy(model.adv_head).to(fit['z'].device)
                head.eval()
                with torch.no_grad():
                    fit_before=float(F.cross_entropy(head(fit['z']),fit['d']))
                    online=F.cross_entropy(model.adv_head(mon['z']),mon['d'],reduction='none')
                opt=torch.optim.AdamW(head.parameters(),lr=.002)
                head.train()
                recovery_steps=0
                for _ in range(40):
                    opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(head(fit['z'].detach()),fit['d'])
                    if not torch.isfinite(loss): break
                    loss.backward();opt.step()
                    recovery_steps+=1
                head.eval()
                with torch.no_grad():
                    fit_after=float(F.cross_entropy(head(fit['z']),fit['d']))
                    recovered=F.cross_entropy(head(mon['z']),mon['d'],reduction='none')
                quality=recovery_quality(fit_before,fit_after,(recovered-online).cpu().tolist(),mon['groups'],self.args.seed)
                if recovery_steps!=40:quality.update(valid=False,reason='INCOMPLETE_RECOVERY')
                freq=torch.bincount(mon['d'],minlength=model.num_domains).float();freq=freq/freq.sum()
                entropy=float(-(freq[freq>0]*freq[freq>0].log()).sum())
                online_ce=float(online.mean());recovered_ce=float(recovered.mean())
                lag=max(online_ce-recovered_ce,0.)/max(entropy,1e-12) if quality['valid'] else None
                readable=max(entropy-recovered_ce,0.)/max(entropy,1e-12) if quality['valid'] else None
                direction=None
                if quality['valid'] and adv_weight>0:
                    out=model(mon['x'][:32],y_tx=mon['y'][:32],return_aux=True)
                    parameters=list(model.id_backbone.parameters())
                    left=-adv_weight*F.cross_entropy(model.adv_head(out['z_id']),mon['d'][:32])
                    right=-adv_weight*F.cross_entropy(head(out['z_id']),mon['d'][:32])
                    a=torch.autograd.grad(left,parameters,retain_graph=True,allow_unused=True)
                    b=torch.autograd.grad(right,parameters,allow_unused=True)
                    dot=sum(float((x*y).sum()) for x,y in zip(a,b) if x is not None and y is not None)
                    na=sum(float(x.square().sum()) for x in a if x is not None)**.5
                    nb=sum(float(x.square().sum()) for x in b if x is not None)**.5
                    if na*nb>0: direction=1.-max(-1.,min(1.,dot/(na*nb)))
                geo=extract(model,self.source.train,self.geometry_indices,self.args.eval_batch_size)
                units=[];energies=[];offset=0
                for p,q,k,count in self.geometry_shapes:
                    z=geo['z'][offset:offset+count].reshape(p,q,k,-1);offset+=count
                    value,diag=normalized_identity_interaction_loss(z,min_norm=1e-8)
                    if value is not None:
                        units.append(float(value));energies.append(float(grid_decomposition(diag['normalized_grid'])['tx'].square().sum(-1).mean()))
                with torch.no_grad():
                    logits=[]
                    for start in range(0,len(geo['x']),self.args.eval_batch_size):
                        logits.append(model(geo['x'][start:start+self.args.eval_batch_size],return_aux=True)['tx_logits'])
                    logits=torch.cat(logits);p=logits.softmax(1)
                    identity=float((p.argmax(1)==geo['y']).float().mean())
                    true=p.gather(1,geo['y'][:,None]).squeeze(1)
                    other=p.scatter(1,geo['y'][:,None],float('-inf')).max(1).values
                    margin=float((true-other).mean())
                    clean_ce=float(F.cross_entropy(logits,geo['y']))
                    excess={};cosine=None
                    for si,scene in enumerate(SCENES):
                        losses=[];cosines=[]
                        gen=torch.Generator(device=geo['x'].device).manual_seed(self.args.seed+1911+si)
                        for start in range(0,len(geo['x']),self.args.eval_batch_size):
                            x=geo['x'][start:start+self.args.eval_batch_size]
                            sat,_=apply_sat_channel_for_scenario(x,scene,self.args,gen=gen,return_meta=False)
                            out=model(sat,return_aux=True)
                            losses.extend(F.cross_entropy(out['tx_logits'],geo['y'][start:start+len(x)],reduction='none').tolist())
                            cosines.extend(F.cosine_similarity(out['z_id'],geo['z'][start:start+len(x)],dim=-1).tolist())
                        excess[scene]=float(np.mean(losses))-clean_ce
                        if si==0: cosine=float(np.mean(cosines))
                geometry=dict(identity=identity,margin=margin,tx_main_energy=float(np.mean(energies)) if energies else None,
                              unit_interaction=float(np.mean(units)) if units else None,leo_cosine=cosine)
                geometry_valid=len(units)==len(self.geometry_shapes) and all(isinstance(v,(float,int)) and math.isfinite(v) for v in geometry.values())
                return dict(observation_id=int(step),step=int(step),epoch=epoch,recovery=quality,
                            G_lag=lag,readability=readable,direction_imbalance=direction,
                            online_ce=online_ce,recovered_ce=recovered_ce,prior_entropy=entropy,
                            geometry=geometry,geometry_valid=geometry_valid,difficulty=excess,
                            recovery_steps=recovery_steps,elapsed_seconds=time.perf_counter()-started,
                            source_only=True,gradient_scope='online_vs_recovered_adversary_on_identity_parameters')
        finally:
            with torch.no_grad():
                for n,b in model.named_buffers():b.copy_(buffers[n])
            for m,flag in flags.items():m.training=flag


class ReliableController:
    def __init__(self):
        self.last_id=-1;self.latest=None;self.calibration={};self.thresholds=None
        self.geometry_streak=0;self.action_streak=0;self.last_kind=None
        self.high_lag=False;self.imbalanced=False;self.last_action=-1000000
        self.used=set();self.corrections=0;self.actions=0

    def observe(self, evidence):
        oid=int(evidence['observation_id'])
        if oid<=self.last_id:return False
        self.last_id=oid;self.latest=deepcopy(evidence)
        if oid in (0,250,500):self.calibration[oid]=deepcopy(evidence)
        if self.thresholds is None and set(self.calibration)=={0,250,500}:
            values=list(self.calibration.values())
            if all(v['geometry_valid'] for v in values):
                self.thresholds={k:float(np.quantile([v['geometry'][k] for v in values],.9 if k=='unit_interaction' else .1)) for k in GEOMETRY}
        acceptable=self.thresholds is not None and oid>500 and evidence['geometry_valid']
        if acceptable:
            acceptable=all(evidence['geometry'][k]<=v if k=='unit_interaction' else evidence['geometry'][k]>=v for k,v in self.thresholds.items())
        self.geometry_streak=self.geometry_streak+1 if acceptable else 0
        valid=evidence['recovery']['valid']
        if not valid:
            self.action_streak=0;self.last_kind=None;self.high_lag=False;self.imbalanced=False
            return True
        lag=evidence['G_lag'];imb=evidence.get('direction_imbalance')
        self.high_lag=lag>.005 if self.high_lag else lag>=.01
        self.imbalanced=(imb>.3 if self.imbalanced else imb>=.5) if imb is not None else False
        kind='CATCHUP' if self.high_lag else ('CORRECT' if lag<=.005 and self.imbalanced else None)
        if evidence.get('readability',0)<.1:kind=None
        self.action_streak=self.action_streak+1 if kind is not None and kind==self.last_kind else int(kind is not None)
        self.last_kind=kind
        return True

    def capability(self,step):
        return bool(self.latest and 0<=step-self.latest['step']<=10 and self.geometry_streak>=3)

    def decide(self,step,enabled=True):
        decision=dict(action='NORMAL',catchup_steps=0,reason='unavailable_or_unconfirmed')
        if not enabled or self.latest is None:return decision
        oid=self.latest['observation_id']
        if oid in self.used or step-self.last_action<250 or not self.capability(step):return decision
        if not self.latest['recovery']['valid'] or self.action_streak<2:return decision
        if self.last_kind=='CORRECT' and (self.corrections+1)>.2*(step+1):return decision
        if self.last_kind:
            decision.update(action=self.last_kind,catchup_steps=3 if self.last_kind=='CATCHUP' else 0,
                            reason='fresh_confirmed_source_evidence',observation_id=oid)
        return decision

    def commit(self,decision,step,accepted):
        if accepted and decision['action']!='NORMAL':
            self.used.add(decision['observation_id']);self.last_action=step;self.actions+=1
            self.corrections+=int(decision['action']=='CORRECT')

    def state_dict(self):
        result=deepcopy(vars(self));result['used']=sorted(result['used']);return result
