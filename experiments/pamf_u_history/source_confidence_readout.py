"""Uniform CPU source-V readout of twelve already-trained E50 models; zero fitting."""
import argparse
import os
from pathlib import Path,PurePosixPath
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write
from .covariance_seed_masks import SEEDS,activate
from .source_confidence_metrics import grouped

RUN='20261027-diagnostic-PAMF-SIG1-sourceConfidence-m12-r01'
RELEASE='20261027-diagnostic-PAMF-SIG1-sourceConfidence-release-r01'
BASELINE='20261026-phase1-PAMF-SIG1-matched-baselines-m6-r01'
MEAN='20261027-phase1-PAMF-RXMean-SIG1-threeSeed-m6-r02'
METHODS=('SIG1-PL','SIG1-Sup','RXMean-U','RXMean-LU')


def source_folder(c,method,seed):
    if method not in METHODS or seed not in SEEDS:raise ValueError('Four fixed methods and three existing model seeds')
    parent=BASELINE if method.startswith('SIG1-') else MEAN
    return str(PurePosixPath(c['project'])/'runs'/parent/'F'/('F0-'+method+'-s'+str(seed))/'source')


def validate_context(c):
    if set(c)!={'project','python','frozen_sources','channel_code','source_contract','source_cache'} or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'}:raise ValueError('Source-only fixed F handles; no target or G input')
    return c


def check_source(c,method,seed):
    folder=Path(source_folder(c,method,seed));done=read(folder/'complete.json');cfg=done['config']
    if method.startswith('SIG1-'):
        from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config
        validate_config(cfg)
    else:
        from .rx_mean_source import validate_train
        validate_train(cfg)
    if (done['status']!='SOURCE_FROZEN' or done['epoch']!=50 or done['updates']!=11100 or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
        or cfg['model_seed']!=seed or cfg['paper_method']!=method or cfg['source_cache']!=c['source_cache'] or read(cfg['source_contract'])!=read(c['source_contract'])
        or cfg['sig_time_views']!=1 or not cfg['default_head_joint']):raise ValueError('Actual matched fixed-final E50 source checkpoint required')
    return cfg


def evaluate(rt,c,method,seed,data,labeled):
    import torch
    from experiments.pamf_repl_completion.source import load_reader
    cfg=check_source(c,method,seed);folder=Path(source_folder(c,method,seed));out=Path(rt['run_root'])/'rows'/('F0-'+method+'-s'+str(seed));out.mkdir(parents=True,exist_ok=False)
    model,_=load_reader(folder,read(c['source_contract']),'cpu',require_no_unlabeled=method=='SIG1-Sup')
    before={k:v.detach().clone() for k,v in model.state_dict().items()};pool=model.sequence_pool;head=model.get_submodule('core.id_backbone.cls_head')
    with torch.inference_mode():p=model(labeled.x[:2])
    if p.shape!=(2,6) or not torch.isfinite(p).all() or len(pool.views)!=1 or sum(p.numel() for p in pool.parameters())!=18976 or tuple(head.weight.shape)!=(6,160):raise ValueError('Actual reader/defaulthead/singleSIG1 no-query smoke failed')
    write(out/'smoke.json',dict(status='PASS',source_L_samples=2,query_reads=0,reader_parameters=sum(p.numel() for p in model.parameters()),signature_parameters=18976,
        signature_views=1,default_head=[6,160],optimizer_updates=0,U_TX_truth_reads=0,device='cpu'))
    logits=np.empty((27000,6),np.float32);seen=0;start=time.monotonic()
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            p=model(batch[0]);n=len(batch[0])
            if p.shape!=(n,6) or not torch.isfinite(p).all():raise ValueError('Finite original default-head source logits required')
            logits[seen:seen+n]=np.asarray(p.tolist(),np.float32);seen+=n
            write(out/'progress.json',dict(V_seen=seen,total=27000,method=method,model_seed=seed),exclusive=False)
            print('SOURCE_CONFIDENCE_V',method,seed,seen,flush=True)
    if seen!=27000 or any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise ValueError('Full V population and immutable model required')
    with (out/'source_V_logits.npz').open('xb') as f:np.savez(f,logits=logits,y=data.metadata['y'],ids=data.metadata['ids'],rx=data.metadata['rx_i'],day=data.metadata['day_i'])
    rows=grouped(logits,data.metadata['y'],data.metadata['rx_i'],data.metadata['day_i'])
    write(out/'complete.json',dict(status='SOURCE_CONFIDENCE_ROW_COMPLETE',method=method,model_seed=seed,source_folder=str(folder),source_role='V',
        V_seen=27000,source_L_smoke_seen=2,source_contract=read(c['source_contract']),rows=rows,parameters_and_buffers_unchanged=True,
        optimizer_updates=0,U_TX_truth_reads=0,target_reads=0,G_weight_reads=0,G_forwards=0,device='cpu',FP32=True,seconds=time.monotonic()-start))


def dispatch(rt):
    c=validate_context(read(rt['context']))
    if rt['run_id']!=RUN or not rt.get('cpu_only'):raise ValueError('One uniform CPU source-only twelve-model diagnostic')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    # Correct frozen dependency activation must precede native-runtime imports.
    activate(c)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    contract=read(c['source_contract']);data=SourceDataset(c['source_cache'],'V',contract);l=SourceDataset(c['source_cache'],'L_s',contract)
    if len(data)!=27000 or data.role!='V' or 'y' not in data.metadata or len(l)!=6300:raise ValueError('Original visible-label source V and source-L smoke only')
    names=[(m,s) for m in METHODS for s in SEEDS];state=dict(status='RUNNING_CPU_SOURCE_READOUT',pid=os.getpid(),code_commit=rt['code_commit'],rows={m+'-s'+str(s):dict(status='QUEUED') for m,s in names});ticks=0
    while any(v['status']!='COMPLETE' for v in state['rows'].values()):
        worked=False
        for method,seed in names:
            key=method+'-s'+str(seed)
            if state['rows'][key]['status']!='QUEUED' or not (Path(source_folder(c,method,seed))/'complete.json').exists():continue
            state['rows'][key]=dict(status='RUNNING');write(root/'pipeline_state.json',state,exclusive=False)
            print('SOURCE_CONFIDENCE_ROW_STARTED',method,seed,flush=True)
            try:evaluate(rt,c,method,seed,data,l)
            except BaseException:
                state['status']='STOPPED_FOR_TECHNICAL_FAILURE';state['rows'][key]=dict(status='FAILED');write(root/'pipeline_state.json',state,exclusive=False);raise
            state['rows'][key]=dict(status='COMPLETE');worked=True;write(root/'pipeline_state.json',state,exclusive=False)
        if all(v['status']=='COMPLETE' for v in state['rows'].values()):break
        if not worked:
            state['status']='WAITING_FOR_FROZEN_SOURCE_ROWS';state['updated_at']=time.time();write(root/'pipeline_state.json',state,exclusive=False);ticks+=1
            if ticks%4==1:print('SOURCE_CONFIDENCE_WAITING_FOR_FROZEN_SOURCE_ROWS',flush=True)
            time.sleep(12)
        else:state['status']='RUNNING_CPU_SOURCE_READOUT'
    state['status']='SOURCE_CONFIDENCE_ALL_ROWS_COMPLETE';write(root/'pipeline_state.json',state,exclusive=False)
    write(root/'complete.json',dict(status='SOURCE_CONFIDENCE_ALL_ROWS_COMPLETE',methods=list(METHODS),model_seeds=list(SEEDS),models=12,source_V_per_model=27000,
        source_forward_packets=324024,source_role='V',device='cpu',new_training_updates=0,U_TX_truth_reads=0,target_reads=0,G_forwards=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True);a=p.parse_args();dispatch(read(a.config))
