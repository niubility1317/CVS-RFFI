"""Explicit head-only CE routing and fixed-denominator selected-layer Fishr."""
import torch
from torch.nn import functional as F
from experiments.cvs_e20_budget_fishr.multilevel import (
    MultiLevel as PreviousMultiLevel, budget_fishr as previous_budget,
    fishr_variance, pooled_real_map, LEVELS)


def _selection(config):
    levels=tuple(config['fishr_levels'])
    if len(set(levels))!=len(levels) or any(k not in LEVELS for k in levels):raise ValueError('Invalid selected Fishr layers')
    if config['fishr_fixed_denominator']!=2:raise ValueError('Fishr denominator must stay two')
    if bool(levels)!=(config['fishr_mode']!='none'):raise ValueError('Fishr mode/selection differ')
    if config.get('mixstyle_l'):raise ValueError('No L MixStyle in this design')
    return levels


def budget_fishr(model,main_ce,per_layer_terms,config,ramp):
    active=_selection(config)
    if set(per_layer_terms)!=set(active):raise ValueError('Missing or unregistered Fishr computation')
    padded={level:per_layer_terms[level] if level in active else main_ce*0 for level in LEVELS} if active else {}
    loss,detail=previous_budget(model,main_ce,padded,config,ramp)
    detail.update(active_levels=list(active),fixed_denominator=2,inactive_levels=[k for k in LEVELS if k not in active])
    for level,info in detail['levels'].items():
        info['active']=level in active
        if level not in active:
            info.update(reason='inactive_by_design',technical_failure=False,requested_ratio_pre_level_mean=0.,raw_loss=0.,coefficient=0.,uncapped_coefficient=None)
        else:info['technical_failure']=info['reason'] is not None
    return loss,detail


class MultiLevel(PreviousMultiLevel):
    def _loss(self,model,source,step,device,config):
        active=_selection(config)
        if config['aux_ce_to_backbone']:
            if active!=LEVELS:raise ValueError('Coupled control requires both Fishr layers')
            ce,terms,detail=super()._loss(model,source,step,device,config)
            detail.update(aux_ce_to_backbone=True,aux_ce_routing='ordinary feature graph',fishr_levels=list(active),fishr_active_levels=list(active),fishr_fixed_denominator=2)
            for level,info in detail['levels'].items():info['fishr_computed']=level in active
            return ce,terms,detail
        zero=next(self.heads.parameters()).sum()*0
        empty=dict(aux_forwards=0,auxiliary_packets=0,fishr_steps=0,levels={},aux_ce_to_backbone=False,fishr_levels=list(active))
        if not config['enabled'] or not model.training or step%config['interval']:return zero,{},empty
        if config['batch_size']!=60 or tuple(config['levels'])!=LEVELS:raise ValueError('Fixed source layout differs')
        index=self.balanced_indices(source,step,config['interval'])
        x=source['x'][index].to(device);y=source['y'][index].to(device);rx=source['rx'][index].to(device);day=source['day'][index]
        with self.capture(step=step) as (maps,_):model(x)
        features={level:pooled_real_map(maps[level]) for level in LEVELS}
        ces={level:F.cross_entropy(self.heads[self.key(level)](feature.detach()),y) for level,feature in features.items()}
        terms={};detail=dict(aux_forwards=1,auxiliary_packets=60,distinct_PID=60,day=int(day[0]),
            fishr_steps=int(bool(active)),levels={},native_CPU_CUDA_RNG_preserved=True,source_RX=rx.unique().cpu().tolist(),
            EMA=False,head_Fishr_gradient=False,aux_ce_to_backbone=False,aux_ce_routing='detached feature; head-only CE',
            fishr_levels=list(active),fishr_active_levels=list(active),fishr_fixed_denominator=2)
        for level,feature in features.items():
            info=dict(clean_ce=float(ces[level].detach()),shape=list(maps[level].shape),fishr_computed=level in active)
            if level in active:
                terms[level],statistics=fishr_variance(feature,self.heads[self.key(level)],y,rx);info.update(statistics)
            else:info['fishr_inactive_reason']='inactive_by_design'
            detail['levels'][level]=info
        ce=torch.stack(list(ces.values())).mean()
        detail.update(aux_ce=float(ce.detach()),fishr=float((sum(terms.values())/2).detach()) if terms else 0.)
        return ce,terms,detail
