"""Fit fixed correlation masks on full labeled source L; no U/V/target access."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from experiments.cric_dura_comparison.io import read,write

RUN='20261026-diagnostic-PAMF-SIG1-CovMask-L-s2026092701-r01'
RELEASE='20261026-diagnostic-PAMF-SIG1-CovMask-L-release-r01'
LEVELS=('time_down','t2.pool')
RECIPE=dict(source_role='L_s',physical_records=6300,model_seed=2026092701,parent_epoch=20,parent_updates=4440,
    levels=list(LEVELS),batch_size=128,views=['raw','practical_low_suburban','practical_mid_urban','practical_low_urban'],
    ridge=1e-3,fraction=.1,identity_quantile=.5,sensitivity_quantile=.75,random_mask_seed=2026102623,
    source_rxs=[1,3,4,6,8],source_days=[1,2,3],fit_updates=0,
    namespace='PAMF/selective_covariance/L/frozen_E20')


def validate(c):
    if (c.get('run_id')!=RUN or c.get('recipe')!=RECIPE or c.get('run_root')!=c['project']+'/runs/'+RUN
        or c.get('release_root')!=c['project']+'/releases/'+RELEASE or c.get('allowed_gpus')!=list(range(8))
        or any(k in c for k in ('target_inputs','target_truth','truth','tests','dataset_path','checkpoint','teacher_checkpoint'))):
        raise ValueError('One immutable labeled-source-only covariance fit required')
    return c


def capture(model,x):
    from .selective_covariance import correlation
    maps={};handles=[]
    def hook(name):
        def observe(module,args,out):
            if name in maps:raise ValueError('Repeated feature map within one physical view')
            maps[name]=out
        return observe
    try:
        for name in LEVELS:handles.append(model.get_submodule('core.id_backbone.'+name).register_forward_hook(hook(name)))
        embedding=model.features(x)
    finally:
        for handle in handles:handle.remove()
    if set(maps)!=set(LEVELS):raise ValueError('Actual shallow and middle feature maps missing')
    return embedding,{name:correlation(h,RECIPE['ridge']) for name,h in maps.items()}, {name:list(h.shape[1:]) for name,h in maps.items()}


def fit(c):
    validate(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    import torch
    from .source import config,load_prefix
    from .selective_covariance import MaskStatistics,InsufficientMaskEvidence
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_channel_pl.restored_source import source_channel,snapshot,unchanged,SCENES
    torch.set_num_threads(2);torch.manual_seed(2026102623);torch.cuda.manual_seed_all(2026102623)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(c['run_root']);start=time.monotonic();write(out/'process.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started_at=time.time()))
    ctx=read(c['context']);cfg=config(ctx,c['code_commit'],'PL-History');parent,payload=load_prefix(cfg)
    contract=read(cfg['source_contract'])
    with reader_structure(cfg):model=reader_from_payload(cfg,payload,'cuda:0')
    del payload
    pool=model.sequence_pool
    if (len(pool.views)!=1 or sum(p.numel() for p in pool.parameters())!=18976
        or tuple(model.core.id_backbone.cls_head.weight.shape)!=(6,160)):raise ValueError('Actual SIG1 and original default head required')
    data=SourceDataset(cfg['source_cache'],'L_s',contract)
    if data.role!='L_s' or len(data)!=6300 or 'y' not in data.metadata:raise ValueError('Complete labeled training L only')
    before=snapshot(model);stats={};seen=0;shapes=None;all_predictions_finite=True
    # First real-checkpoint smoke continues immediately into the full L pass.
    with torch.inference_mode():
        z,corr,shapes=capture(model,data.x[:2].to('cuda:0'));logits=model.classify_features(z)
        if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual default-head source smoke failed')
    write(out/'smoke.json',dict(status='PASS',actual_SIG_TIME_views=1,signature_parameters=18976,
        actual_default_head=[6,160],source_L=2,feature_shapes=shapes,optimizer_updates=0,U_truth_read=False,target_reads=0))
    loader=torch.utils.data.DataLoader(data,batch_size=128,shuffle=False,num_workers=0)
    for step,batch in enumerate(loader,1):
        x=batch[0].to('cuda:0');y=batch[1];meta=batch[3]
        if not bool(meta['tx_label_visible'].all()) or bool((y<0).any()):raise ValueError('True source L required')
        with torch.inference_mode():
            _,raw,current_shapes=capture(model,x);views=[]
            for scene in SCENES:
                iq=source_channel(x,meta,scene,contract,namespace=RECIPE['namespace'])
                _,corr,other_shapes=capture(model,iq)
                if other_shapes!=current_shapes or current_shapes!=shapes:raise ValueError('Feature channels/time changed across views')
                views.append(corr)
        for name in LEVELS:
            if name not in stats:stats[name]=MaskStatistics(raw[name].shape[1],device='cpu')
            stats[name].update(raw[name],torch.stack([v[name] for v in views]),y,meta['rx_i'],meta['day_i'],role=data.role)
        seen+=len(x)
        progress=dict(step=step,L_seen=seen,As_seen=3*seen,feature_shapes=shapes,elapsed_seconds=time.monotonic()-start)
        with (out/'fit_steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
        write(out/'progress.json',progress,exclusive=False)
        print('COV_MASK_SOURCE_L',seen,'of',len(data),flush=True)
    if seen!=6300 or not unchanged(model,before):raise ValueError('Full source L or frozen model state check failed')
    masks={};diagnostics={};failed=[]
    for name in LEVELS:
        if stats[name].samples!=6300 or stats[name].paired_views!=18900:raise ValueError('Complete actual same-packet pairs required')
        try:
            masks[name],diagnostics[name]=stats[name].masks(fraction=.1,random_seed=2026102623)
            diagnostics[name]['mask_rule_passed']=True
        except InsufficientMaskEvidence as error:
            diagnostics[name]=dict(error.diagnostic,mask_rule_passed=False,reason=str(error));failed.append(name)
    state={name:{k:getattr(stat,k) for k in ('channels','count','total','square','sensitivity','rx_count','samples','paired_views')} for name,stat in stats.items()}
    # Private source-derived states, never Git artifacts or deployment inputs.
    with (out/'statistics.pt').open('xb') as f:torch.save(dict(schema='pamf_covariance_L_statistics_v1',code_commit=c['code_commit'],source_contract=contract,statistics=state),f)
    if not failed:
        with (out/'masks.pt').open('xb') as f:torch.save(dict(schema='pamf_sig1_selective_covariance_masks_v1',code_commit=c['code_commit'],
            source_contract=contract,source_prefix=parent['folder'],recipe=RECIPE,masks=masks,diagnostics=diagnostics,fit_role='L_s',U_truth_read=False,V_used=False,target_used=False),f)
    write(out/'result.json',dict(status='SOURCE_MASK_FIT_COMPLETE',run_id=RUN,code_commit=c['code_commit'],recipe=RECIPE,
        source_prefix=parent['folder'],actual_checkpoint_epoch=20,actual_checkpoint_updates=4440,source_L_seen=seen,
        As_physical_views=18900,actual_feature_shapes=shapes,diagnostics=diagnostics,all_levels_mask_rule_passed=not failed,
        failed_levels=failed,mask_artifact='masks.pt' if not failed else None,
        parameters_and_registered_buffers_unchanged=True,F_optimizer_updates=0,G_weight_reads=0,G_updates=0,
        U_IQ_reads=0,U_truth_read=False,V_reads=0,target_reads=0,seconds=time.monotonic()-start,
        scope='training-L-only mask statistics;no trained SSL candidate or target performance'))
    print('COV_MASK_SOURCE_FIT_COMPLETE','mask_rule_passed',not failed,flush=True)


def dispatch(c):
    validate(c)
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.practical_full_2x2.cache import atomic
    root=Path(c['run_root']);root.mkdir(parents=True,exist_ok=False);path=root/'config.json';write(path,c)
    state=dict(status='QUEUED',controller_pid=os.getpid(),code_commit=c['code_commit']);child=None
    while child is None:
        pending=processes();available=[]
        for gpu in c['allowed_gpus']:
            pids,free=capacity(gpu)
            if not (pids|{pid for pid,g in pending.items() if g==gpu}) and free>=10000:available.append((-free,gpu))
        if available:
            gpu=min(available)[1];job=dict(id='fit',kind='fit',config=str(path),execution_release=c['release_root'],execution_python=c['python'],entrypoint='experiments.pamf_u_history.covariance_fit')
            child,item=launch(c,job,gpu);state.update(status='RUNNING',worker=item)
        atomic(root/'pipeline_state.json',state)
        if child is None:time.sleep(12)
    code=child.wait();good=code==0 and (root/'result.json').exists()
    state.update(status='SOURCE_MASK_FIT_COMPLETE' if good else 'FAILED',exit_code=code,finished_at=time.time());atomic(root/'pipeline_state.json',state)
    if not good:raise RuntimeError('Covariance source fit failed;preserve output,do not restart blindly')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);dispatch(c) if a.action=='dispatch' else fit(c)
