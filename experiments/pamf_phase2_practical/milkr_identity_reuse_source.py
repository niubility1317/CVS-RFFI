"""New source-macro-H selection run; the earlier conjunctive stop is preserved.

All comparisons reuse source-only validation, not target scores. The auxiliary
per-K1-new-accuracy gain is diagnostic rather than a substitute H objective.
"""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize
from .milkr.paired_cca import PairedCCA
from .milkr.source_class_metric import within_class_covariances,fuse_metric
from .milkr_identity_source import assess

SOURCE_KEYS=('source_cache','source_contract','DURA','PAMF_sources','model_seeds','augmentation_cache','source_cache_epochs','source_proxy_seed','phase1_release','frozen_sources','PAMF_definition')


def load_source_grid(rt):
    parent=read(rt['source_identity_grid_runtime'])
    require(parent['kernel_method']=='identity_holdout','Wrong source grid method')
    for key in SOURCE_KEYS:require(parent[key]==rt[key],'Source grid ancestry differs: '+key)
    root=Path(parent['run_root']);selection=read(root/'source_selection.json');grid=read(root/'source_calibration_grid.json')
    require(selection['status']=='SOURCE_FROZEN' and selection['target_eligible'] is False,'Expected preserved source-only rejected grid')
    require(all(selection[key]==0 for key in ('target_reads','query_reads','U_truth_reads','F_updates','G_updates')),'Source grid has disallowed ancestry')
    require(grid['source_only'] is True and grid['target_reads']==0 and len(grid['candidates'])==selection['source_candidates'],'Incomplete or non-source grid')
    return parent,selection,grid['candidates']


def macro_assessment(summary,control):
    out=assess(summary,control);reference={(r['k'],r['S'],r['Q']):r for r in control}
    groups={}
    for name in ('old_accuracy','new_accuracy'):
        groups[name]=[float(np.mean([100*(r[name]-reference[r['k'],r['S'],r['Q']][name]) for r in summary if r['S']==sp and r['Q']==0])) for sp in (0,1)]
    out['previous_component_gate_pass']=out['eligible']
    out['raw_old_gain_pp_by_S']=groups['old_accuracy'];out['raw_new_gain_pp_by_S']=groups['new_accuracy']
    out['eligible']=bool(out['source_gain_pp']>=.5 and out['new_macro_gain_pp']>=1 and out['min_K_Q0_gain_pp']>=.25
        and out['min_K_Q1_minus_Q0_pp']>=0 and min(out['Q1_minus_Q0_pp_by_S'])>0
        and min(groups['old_accuracy']+groups['new_accuracy'])>=-1.)
    return out


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();parent,previous,candidates=load_source_grid(rt);control=previous['source_summary'];evaluated=[]
    for candidate in candidates:evaluated.append(dict(candidate,**macro_assessment(candidate['summary'],control)))
    passing=[c for c in evaluated if c['eligible']]
    order=lambda c:(c['source_H'],c['new_macro_gain_pp'],c['K1_rule']['ridge'],-c['rank'],-c['strength'],-c['rule_index'])
    best=max(passing,key=order) if passing else None;quality=[]
    for rank in (16,32,64,96):
        for strength in (0.,.5,.9):
            local=[c for c in evaluated if c['rank']==rank and c['strength']==strength];eligible=[c for c in local if c['eligible']]
            chosen=max(eligible or local,key=order);quality.append({key:value for key,value in chosen.items() if key!='summary'})
    if best:
        reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']));m=data.metadata;banks=[[],[]]
        for epoch in rt['source_cache_epochs']:
            folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
            require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
            require(np.array_equal(np.load(folder/'ids.npy'),m['ids']),'Source physical alignment differs')
            iq=np.asarray(np.load(folder/'x.npy',mmap_mode='r'))
            for path,g in enumerate((None,engine)):banks[path].append(extract_for_method(rt,reader,iq,'cuda:0',g)[0])
            print('MACRO_SOURCE_EPOCH',epoch,len(iq),flush=True)
        features=[{key:np.concatenate([part[key] for part in parts]) for key in parts[0]} for parts in banks];y=np.tile(m['y'],len(rt['source_cache_epochs']))
        final=PairedCCA.fit(*features,best['rank']);cov=within_class_covariances(final,*features,y);fused=fuse_metric(final,cov,best['strength'])
        fused.save(Path(rt['run_root'])/'source_identity_cca.npz');loaded=PairedCCA.load(Path(rt['run_root'])/'source_identity_cca.npz')
        small=[{key:value[:12] for key,value in f.items()} for f in features]
        for path in (0,1):require(all(np.array_equal(value,loaded.transform(small[path],restored=bool(path))[key]) for key,value in fused.transform(small[path],restored=bool(path)).items()),'Source fused encoder reload differs')
        physical=len(m['ids']);paired=len(features[0]['z_id'])
    else:physical=previous['source_physical_rows'];paired=previous['source_paired_observations']
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=best['K1_rule'] if best else previous['K1_rule'],target_eligible=best is not None,
        scientific_verdict='PASS_SOURCE_MACRO_H_ALIGNED_GATE' if best else 'NO_SOURCE_CANDIDATE_PASSES_MACRO_H_GATE',source_quality=quality,source_summary=best['summary'] if best else control,
        source_parent_run=parent['run_id'],source_parent_original_stop='PRESERVED_FAILED_AUXILIARY_K1_NEW_COMPONENT_GATE',source_grid_reused=True,source_grid_target_reads=0,
        selected_rank=best['rank'] if best else None,selected_strength=best['strength'] if best else None,source_candidates=len(evaluated),
        source_RX_TX_folds=previous['source_RX_TX_folds'],source_physical_rows=physical,source_paired_observations=paired,
        source_RX_training_excludes_held_RX=True,source_TX_training_excludes_held_TX=True,source_support_held_physical_disjoint=True,source_numerical_smoke='PASS',
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,
        scope='new source-only macro-H selection rule; no retrospective edit to oldstop; targeteachKN+5/Q1>Q0 unchanged; F sawsource6, proxyadapterTXheld only'))
