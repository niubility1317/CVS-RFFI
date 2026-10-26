"""Complete seed02/03 L-only masks in each seed's own E20 feature coordinates."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time
from experiments.cric_dura_comparison.io import read,write

RUN='20261027-diagnostic-PAMF-CovMask-s0203-source-r01'
RELEASE='20261027-diagnostic-PAMF-CovMask-s0203-source-release-r01'
SEEDS=(2026092701,2026092702,2026092703)
NEW_SEEDS=SEEDS[1:]
LEVELS=('time_down','t2.pool')


def validate_context(c):
    if (set(c)!={'project','python','frozen_sources','cases','channel_code','U_channel_cache','seed01_masks','source_contract','source_cache'}
        or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or len(c['cases'])!=3 or {r['model_seed'] for r in c['cases']}!=set(SEEDS)):
        raise ValueError('Three existing same-contract SIG1 source parents and signal-physics code only')
    for row in c['cases']:
        cfg=row['baseline_config']
        from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config
        validate_config(cfg)
        if (cfg['model_seed']!=row['model_seed'] or cfg['source_contract']!=c['source_contract'] or cfg['source_cache']!=c['source_cache']
            or cfg['paper_method']!='SIG1-PL' or 'source_G' in cfg or 'G' in cfg['frozen_sources']):raise ValueError('Same-seed actual plainPL source data required;no G')
    return c


def activate(c):
    # This is only the original channel simulation/metadata library;no G engine,
    # network,weight or restored array is imported or opened.
    physics=Path(c['channel_code'])
    if not physics.is_dir():raise ValueError('Immutable source-channel code required')
    sys.path.insert(0,str(physics));importlib.import_module('baseline_origin_sat_view')
    from .fonly_confirm import activate_raw
    activate_raw(c);sys.path.insert(0,str(physics))


def fit(c):
    import torch
    from experiments.pamf_channel_pl.sig1_matched_baselines import load_prefix,check_state
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_channel_pl.restored_source import source_channel,snapshot,unchanged,SCENES
    from .covariance_fit import capture
    from .selective_covariance import MaskStatistics,InsufficientMaskEvidence
    ctx=validate_context(read(c['context']));activate(ctx);seed=c['model_seed']
    if seed not in NEW_SEEDS:raise ValueError('Only missing02/03 masks are newly fitted')
    cfg=next(r['baseline_config'] for r in ctx['cases'] if r['model_seed']==seed)
    parent,payload=load_prefix(cfg);check_state(payload);contract=read(cfg['source_contract'])
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    write(root/'process.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,model_seed=seed,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    with reader_structure(cfg):model=reader_from_payload(cfg,payload,'cuda:0')
    del payload;data=SourceDataset(cfg['source_cache'],'L_s',contract)
    if len(data)!=6300 or data.role!='L_s':raise ValueError('Full training L only')
    before=snapshot(model)
    with torch.inference_mode():
        z,_,shapes=capture(model,data.x[:2].to('cuda:0'));q=model.classify_features(z)
        if q.shape!=(2,6) or not torch.isfinite(q).all() or len(model.sequence_pool.views)!=1:raise ValueError('Actual same-seed SIG1 E20/defaulthead smoke failed')
    write(root/'smoke.json',dict(status='PASS',model_seed=seed,actual_parent=parent['folder'],source_L=2,feature_shapes=shapes,optimizer_updates=0,U_truth_reads=0,target_reads=0))
    stats={};seen=0
    for step,batch in enumerate(torch.utils.data.DataLoader(data,batch_size=128,shuffle=False,num_workers=0),1):
        x=batch[0].to('cuda:0');y=batch[1];meta=batch[3]
        if not bool(meta['tx_label_visible'].all()) or bool((y<0).any()):raise ValueError('L truth only')
        with torch.inference_mode():
            _,raw,current=capture(model,x);views=[]
            for scene in SCENES:
                h=source_channel(x,meta,scene,contract,namespace='PAMF/selective_covariance/L/frozen_E20')
                _,values,other=capture(model,h)
                if current!=shapes or other!=shapes:raise ValueError('Actual temporal shapes changed')
                views.append(values)
        for level in LEVELS:
            if level not in stats:stats[level]=MaskStatistics(raw[level].shape[1])
            stats[level].update(raw[level],torch.stack([v[level] for v in views]),y,meta['rx_i'],meta['day_i'],role='L_s')
        seen+=len(x)
        progress=dict(model_seed=seed,step=step,L_seen=seen,H_seen=3*seen)
        import json
        with (root/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
        write(root/'progress.json',progress,exclusive=False);print('SEED_COV_MASK',seed,seen,flush=True)
    if seen!=6300 or not unchanged(model,before):raise ValueError('Full L or frozen state differs')
    masks={};diagnostics={};unsupported=[]
    for level,s in stats.items():
        try:masks[level],diagnostics[level]=s.masks(fraction=.1,random_seed=2026102623,support_limited=True)
        except InsufficientMaskEvidence as e:unsupported.append(level);diagnostics[level]=e.diagnostic
    recipe=dict(model_seed=seed,source_role='L_s',mask_mode='natural_support_capped',fraction=.1,identity_quantile=.5,sensitivity_quantile=.75,random_seed=2026102623)
    if not unsupported:
        with (root/'masks.pt').open('xb') as f:torch.save(dict(schema='pamf_sig1_selective_covariance_masks_v2_natural_support',code_commit=c['code_commit'],
            recipe=recipe,source_prefix=parent['folder'],source_contract=contract,fit_role='L_s',masks=masks,diagnostics=diagnostics,U_truth_read=False,V_used=False,target_used=False),f)
    write(root/'result.json',dict(status='SOURCE_MASK_COMPLETE' if not unsupported else 'SOURCE_MASK_UNSUPPORTED',model_seed=seed,
        actual_prefix=parent['folder'],source_L=6300,H_views=18900,recipe=recipe,diagnostics=diagnostics,unsupported_levels=unsupported,
        parameters_and_buffers_unchanged=True,optimizer_updates=0,U_reads=0,V_reads=0,target_reads=0,G_weight_reads=0,seconds=time.monotonic()-start))


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.practical_full_2x2.cache import atomic
    if rt['run_id']!=RUN:raise ValueError('One registered seed-mask extension only')
    ctx=validate_context(read(rt['context']));root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    import torch
    reused=torch.load(ctx['seed01_masks'],map_location='cpu',weights_only=False)
    original=next(r['baseline_config']['shared_SIG1_E20'] for r in ctx['cases'] if r['model_seed']==SEEDS[0])
    if (reused['schema']!='pamf_sig1_selective_covariance_masks_v2_natural_support' or reused['fit_role']!='L_s'
        or reused['recipe']['model_seed']!=SEEDS[0] or reused['source_prefix']!=original or reused['source_contract']!=read(ctx['source_contract'])
        or reused['U_truth_read'] or reused['V_used'] or reused['target_used']):raise ValueError('Existing01 masks mustmatch actual source basis')
    write(root/'reused_seed01.json',dict(model_seed=SEEDS[0],masks=ctx['seed01_masks'],counts={k:int(v['selective'].sum()) for k,v in reused['masks'].items()},new_F_forwards=0,new_training=0))
    jobs=[]
    for seed in NEW_SEEDS:
        path=root/'configs'/('s'+str(seed)+'.json');c=dict(rt,model_seed=seed,output_root=str(root/('s'+str(seed))));write(path,c)
        jobs.append(dict(id='s'+str(seed),kind='fit',config=str(path),completion=str(Path(c['output_root'])/'result.json'),execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.covariance_seed_masks'))
    state=dict(status='RUNNING',controller_pid=os.getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item) in list(active.items()):
            code=child.poll()
            if code is None:continue
            job=next(j for j in jobs if j['id']==key);good=code==0 and Path(job['completion']).exists()
            state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key]
        pending=processes()
        for job in jobs:
            if stop or state['jobs'][job['id']]['status']!='QUEUED':continue
            free=[]
            for gpu in rt['allowed_gpus']:
                pids,memory=capacity(gpu)
                if not (pids|{pid for pid,g in pending.items() if g==gpu}) and memory>=10000:free.append((-memory,gpu))
            if free:
                gpu=min(free)[1];child,item=launch(rt,job,gpu);active[job['id']]=(child,item);state['jobs'][job['id']]=item;pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='FAILED' if stop else 'THREE_SEED_MASKS_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:
        results={j['id']:read(j['completion']) for j in jobs}
        write(root/'complete.json',dict(status='THREE_SEED_MASKS_COMPLETE',model_seeds=list(SEEDS),new_seed_fits=list(NEW_SEEDS),reused_seed=SEEDS[0],new_training=0,
            all_masks_available=all(v['status']=='SOURCE_MASK_COMPLETE' for v in results.values()),new_seed_results=results))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else fit(c)
