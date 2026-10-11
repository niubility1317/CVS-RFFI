"""Independent paired source moments and fixed, source-calibrated Fishr weights."""
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cvs_e20_budget_fishr.multilevel import (
    MultiLevel as PreviousMultiLevel,LEVELS,pooled_real_map,linear_sample_gradients,
    _parameter_groups,_norm)
from experiments.cvs_e20_moment_audit.fishr_stats import unbiased_variances,pooled_variances,penalty


class MultiLevel(PreviousMultiLevel):
    def pair_indices(self,source,step,calibration=False):
        cached=getattr(self,'_pair_source_cache',None)
        if cached is None or cached[0] is not source:
            if source.get('role','L_s') not in ('L_s','L'):raise ValueError('Only visible source L allowed')
            y,rx,day=(source[k].detach().cpu() for k in ('y','rx','day'));ids=source['ids']
            if len(ids)!=len(y) or len(set(ids))!=len(ids) or any(len(v)!=len(y) for v in (rx,day,source['x'])):raise ValueError('Invalid unique source physical pool')
            if set(y.tolist())!=set(range(6)) or set(rx.tolist())!={1,3,4,6,8} or len(day.unique())!=3:raise ValueError('Complete six-TX/five-RX/three-day source required')
            members={}
            for d in sorted(day.unique().tolist()):
                cells={(r,k):torch.where((day==d)&(rx==r)&(y==k))[0] for r in (1,3,4,6,8) for k in range(6)}
                if any(len(v)<2 for v in cells.values()):raise ValueError('Missing/inadequate day/TX/RX cell; no fallback allowed')
                members[d]=cells
            self._pair_source_cache=(source,members)
        members=self._pair_source_cache[1];days=sorted(members);day=days[(int(step)//4-1)%3];cells=members[day]
        # Separate deterministic streams, no exclusion of A's PID from B.
        base=self.seed+int(step)*99991+(710000003 if calibration else 230000009)
        def draw(offset):
            generator=torch.Generator().manual_seed(base+offset)
            return torch.cat([cell[torch.randperm(len(cell),generator=generator)[:2]] for cell in cells.values()])
        a,b=draw(117),draw(990137)
        if len(a)!=60 or len(set(a.tolist()))!=60 or len(b)!=60 or len(set(b.tolist()))!=60:raise ValueError('Invalid paired source sample')
        return a,b,day,{key:len(value) for key,value in cells.items()}

    def _pair(self,model,source,step,device,config,calibration=False):
        levels=tuple(config['fishr_levels']);estimator=config['fishr_estimator']
        if len(set(levels))!=len(levels) or any(k not in LEVELS for k in levels) or estimator not in ('old_pair','cross'):raise ValueError('Invalid fixed Fishr selection')
        if config['batch_size']!=120 or config['pair_batch_size']!=60 or config['pair_quota']!=2 or config['fishr_fixed_denominator']!=2:raise ValueError('Pair budget differs')
        ia,ib,day,sizes=self.pair_indices(source,step,calibration);packs=[]
        for index in (ia,ib):
            y=source['y'][index].to(device);rx=source['rx'][index].to(device)
            with self.capture(step=step) as (maps,_):logits=model(source['x'][index].to(device))
            features={level:pooled_real_map(maps[level]) for level in LEVELS}
            packs.append(dict(y=y,rx=rx,logits=logits,features=features,shapes={k:list(v.shape) for k,v in maps.items()}))
        ces=[F.cross_entropy(self.heads[self.key(level)](pack['features'][level].detach()),pack['y']) for pack in packs for level in LEVELS]
        aux_ce=torch.stack(ces).mean();terms={}
        detail=dict(aux_forwards=2,auxiliary_packets=120,distinct_PID=len(set(ia.tolist())|set(ib.tolist())),
            pair_A_distinct_PID=60,pair_B_distinct_PID=60,cross_batch_PID_overlap=len(set(ia.tolist())&set(ib.tolist())),
            day=day,source_RX=[1,3,4,6,8],fishr_steps=1,levels={},aux_ce_to_backbone=False,
            fishr_levels=list(levels),fishr_active_levels=list(levels),fishr_fixed_denominator=2,
            estimator=estimator,head_Fishr_gradient=False,EMA=False,native_CPU_CUDA_RNG_preserved=True,
            independent_AB=True,calibration_sampling=calibration,aux_ce=float(aux_ce.detach()))
        for level in LEVELS:
            info=dict(shape=packs[0]['shapes'][level],fishr_computed=level in levels,
                clean_ce_A=float(ces[LEVELS.index(level)].detach()),clean_ce_B=float(ces[2+LEVELS.index(level)].detach()))
            if level in levels:
                gradients=[linear_sample_gradients(pack['features'][level],self.heads[self.key(level)],pack['y']).flatten(1) for pack in packs]
                old=[pooled_variances(g,pack['rx']) for g,pack in zip(gradients,packs)]
                old_losses=[penalty(v) for v in old]
                if estimator=='old_pair':term=(old_losses[0]+old_losses[1])/2
                else:
                    corrected=[unbiased_variances(g,pack['y'],pack['rx'],sizes) for g,pack in zip(gradients,packs)]
                    term=penalty(corrected[0],corrected[1])
                    info.update(corrected_variance_mean_A=float(corrected[0].detach().mean()),corrected_variance_mean_B=float(corrected[1].detach().mean()),
                        stratified_correction='Bhat + mean_k[(1-5/(6*m)-1/(6*N_k))*s_k^2], s uses m-1')
                terms[level]=term
                info.update(old_loss_A=float(old_losses[0].detach()),old_loss_B=float(old_losses[1].detach()),old_pair_loss=float(((old_losses[0]+old_losses[1])/2).detach()),
                    raw_fishr=float(term.detach()),raw_loss_negative=bool(term.detach()<0),domains=[1,3,4,6,8],coordinates=gradients[0].shape[1],
                    variance_mean_A=float(old[0].detach().mean()),variance_mean_B=float(old[1].detach().mean()))
            else:info['reason']='inactive_by_design'
            detail['levels'][level]=info
        detail['fishr']=float((sum(terms.values())/2).detach()) if terms else 0.
        main_ce=F.cross_entropy(torch.cat([p['logits'] for p in packs]),torch.cat([p['y'] for p in packs])) if calibration else None
        return aux_ce,terms,detail,main_ce

    def _loss(self,model,source,step,device,config):
        if not config['enabled'] or not model.training or step%config['interval']:
            return next(self.heads.parameters()).sum()*0,{},dict(aux_forwards=0,auxiliary_packets=0,fishr_steps=0,levels={})
        ce,terms,detail,_=self._pair(model,source,step,device,config)
        return ce,terms,detail


def calibrate(c,model,L,device,out):
    path=Path(out)/'calibration.json'
    if path.exists():raise FileExistsError('Fixed calibration already exists')
    config=c['multilevel'];started=time.perf_counter()
    if not config['enabled']:
        result=dict(status='NOT_APPLICABLE',fixed_coefficients={},pairs=0,extra_source_readouts=0,seconds=0.,state_unchanged=True)
    else:
        if config['fishr_calibration_pairs']!=12 or config['fishr_ratio']!=.2:raise ValueError('Fixed calibration protocol differs')
        before={n:v.detach().clone() for n,v in model.state_dict().items()};modes=[(m,m.training) for m in model.modules()]
        selected=dict(config,fishr_levels=list(LEVELS),fishr_estimator='old_pair');groups=_parameter_groups(model);rows=[]
        devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices):
                for i in range(12):
                    tick=time.perf_counter();_,terms,detail,ce=model.multilevel._pair(model,L,4*(i+1),device,selected,calibration=True)
                    values={level:dict(main_ce_gradient_norm=_norm(ce,groups[level]),old_pair_gradient_norm=_norm(terms[level],groups[level])) for level in LEVELS}
                    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
                    rows.append(dict(pair=i,day=detail['day'],cross_batch_PID_overlap=detail['cross_batch_PID_overlap'],levels=values,main_ce=float(ce.detach()),seconds=time.perf_counter()-tick))
            changed=[n for n,v in model.state_dict().items() if not torch.equal(v,before[n])]
            if changed:raise ValueError('Calibration changed state: '+','.join(changed))
        finally:
            with torch.no_grad():
                for n,v in model.state_dict().items():v.copy_(before[n])
            for m,training in modes:m.training=training
        coefficients={};summary={}
        for level in LEVELS:
            ce=sum(r['levels'][level]['main_ce_gradient_norm'] for r in rows)/12
            old=sum(r['levels'][level]['old_pair_gradient_norm'] for r in rows)/12
            if not math.isfinite(ce+old) or min(ce,old)<=config['fishr_eps']:raise ValueError('Invalid fixed calibration gradient norm')
            uncapped=.2*ce/(old+config['fishr_eps']);coefficient=min(config['fishr_cap'],uncapped)
            coefficients[level]=coefficient;summary[level]=dict(mean_CE_gradient_norm=ce,mean_old_pair_gradient_norm=old,
                uncapped_coefficient=uncapped,coefficient=coefficient,cap_active=uncapped>config['fishr_cap'])
        model.multilevel.fixed_coefficients=coefficients
        result=dict(status='FIXED_CALIBRATION_COMPLETE',fixed_coefficients=coefficients,levels=summary,pairs=12,records=rows,
            extra_source_readouts=1440,model_forwards=24,seconds=time.perf_counter()-started,state_unchanged=True,RNG_preserved=True,
            source_role='L_s only',target_access=False,identity_updates=0,head_updates=0,all_layers_calibrated=True,
            coefficient_formula='min(cap,0.2*mean(pair_main_CE_gradient_norm)/(mean(old_pair_Fishr_gradient_norm)+eps))',
            sampling_namespace='calibration-private; distinct from training; cross-batch overlap allowed')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


def fixed_fishr(model,main_ce,terms,config,ramp):
    active=tuple(config['fishr_levels']);zero=main_ce*0
    if set(terms)!=set(active) or config['fishr_fixed_denominator']!=2 or not 0<=ramp<=1:raise ValueError('Fixed layer/ramp definition differs')
    detail=dict(mode='fixed' if active else 'none',levels={},level_reduction='mean',fixed_denominator=2,weighted_loss=0.,EMA=False,
        active_levels=list(active),inactive_levels=[k for k in LEVELS if k not in active],dynamic_rescaling=False)
    if not active:return zero,detail
    coefficients=model.multilevel.fixed_coefficients
    if set(coefficients)!=set(LEVELS):raise ValueError('Both fixed calibration coefficients required')
    groups=_parameter_groups(model);weighted=[]
    for level in LEVELS:
        on=level in active;base=float(coefficients[level]);coefficient=base*ramp if on else 0.
        if not math.isfinite(base) or not 0<=base<=config['fishr_cap']:raise ValueError('Invalid stored coefficient')
        term=terms[level] if on else zero;reference=_norm(main_ce,groups[level]);norm=_norm(term,groups[level])
        if not math.isfinite(reference+norm+float(term.detach())):raise FloatingPointError('Nonfinite fixed Fishr')
        achieved=coefficient*norm/reference if reference>config['fishr_eps'] else None
        weighted.append(term*coefficient)
        detail['levels'][level]=dict(active=on,reason=None if on else 'inactive_by_design',technical_failure=False,
            coefficient=coefficient,fixed_calibrated_coefficient=base,uncapped_coefficient=None,coefficient_detached=True,
            coefficient_cap=config['fishr_cap'],cap_active=on and base==config['fishr_cap'],main_ce_gradient_norm=reference,
            raw_fishr_gradient_norm=norm,raw_loss=float(term.detach()),parameters=len(groups[level]),
            requested_ratio_pre_level_mean=None,achieved_ratio_pre_level_mean=achieved,achieved_direct_contribution_ratio=achieved/2 if achieved is not None else None)
    loss=torch.stack(weighted).mean();detail['weighted_loss']=float(loss.detach())
    detail['combined_group_gradient_ratios']={}
    for level,parameters in groups.items():
        norm=_norm(loss,parameters);reference=detail['levels'][level]['main_ce_gradient_norm']
        detail['combined_group_gradient_ratios'][level]=dict(gradient_norm=norm,ratio_to_main_ce=norm/reference if reference>config['fishr_eps'] else None)
    return loss,detail
