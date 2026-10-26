"""F-free U/RX calibration then three frozen-E20 source-only paired readouts."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from experiments.cric_dura_comparison.io import read,write,tensor
from .bounded_response import RECIPE,Calibration,Correction,fit
from .receiver_response import Response,RXS
from .covariance_seed_masks import SEEDS,activate
from .covariance_three_seed import RUN as PREDECESSOR
from .fonly_results import metrics

RUN='20261027-diagnostic-PAMF-BoundedResponse-SIG1-source-m3-r01'
RELEASE='20261027-diagnostic-PAMF-BoundedResponse-SIG1-source-release-r01'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
VIEWS=('clean',*SCENES)


def validate_context(c):
    if set(c)!={'project','python','frozen_sources','cases','channel_code','source_contract','source_cache','predecessor_complete'} or set(c['frozen_sources'])!={'F','AF','BASE','FOLD'} or len(c['cases'])!=3 or {r['model_seed'] for r in c['cases']}!=set(SEEDS):raise ValueError('Three actual source E20 parents and signal physics only')
    if c['predecessor_complete']!=c['project']+'/runs/'+PREDECESSOR+'/complete.json':raise ValueError('Only resource dependency on original fixed matrix')
    from experiments.pamf_channel_pl.sig1_matched_baselines import validate_config,prefix
    for row in c['cases']:
        a=row['baseline_config'];validate_config(a)
        if (a['model_seed']!=row['model_seed'] or a['source_cache']!=c['source_cache'] or a['source_contract']!=c['source_contract'] or a['paper_method']!='SIG1-PL'
            or a['shared_SIG1_E20']!=prefix(c['project'],row['model_seed']) or 'source_G' in a or 'G' in a['frozen_sources']):raise ValueError('Own same-contract source-only SIG1 E20 required')
    return c


def calibrate(rt):
    import torch
    from experiments.cric_dura_comparison.f_data import SourceDataset
    c=validate_context(read(rt['context']));contract=read(c['source_contract']);data=SourceDataset(c['source_cache'],'U_s',contract)
    if len(data)!=56700 or 'y' in data.metadata or set(data.metadata['rx_i'])!=set(RXS):raise ValueError('Full TX-hidden original trainingU only')
    root=Path(rt['run_root'])/'calibration';root.mkdir(parents=True,exist_ok=False);response=Response();values=[];start=time.monotonic();torch.set_num_threads(2)
    for begin in range(0,len(data),256):
        end=min(begin+256,len(data));coeff,_=response(np.asarray(data.x[begin:end].tolist(),np.float64));values.append(coeff)
        if begin==0:write(root/'smoke.json',dict(status='PASS',actual_U=256,TX_truth_reads=0,F_G_weight_reads=0,target_reads=0))
        write(root/'progress.json',dict(U_seen=end,total=56700),exclusive=False)
        print('BOUNDED_CALIBRATION',end,flush=True)
    cal,diagnostic=fit(np.concatenate(values),data.metadata['rx_i'],role='U_s')
    if set(diagnostic['counts_by_RX'].values())!={11340}:raise ValueError('All five original RX musthave11340U')
    with (root/'calibration.npz').open('xb') as f:np.savez(f,center=cal.center,basis=cal.basis)
    write(root/'complete.json',dict(status='BOUNDED_SOURCE_CALIBRATED',recipe=RECIPE,diagnostic=diagnostic,source_contract=contract,
        source_role='U_s',U_seen=56700,model_seed=None,optimizer_updates=0,F_G_weight_reads=0,V_reads=0,target_reads=0,seconds=time.monotonic()-start))


def paired_results(logits,y,rx,day):
    if logits.shape!=(2,4,len(y),6) or not np.isfinite(logits).all() or set(rx)!=set(RXS):raise ValueError('Complete finite raw/corrected source4view sixclass predictions required')
    pred=logits.argmax(-1);masks=[('overall','ALL',np.ones(len(y),bool))]
    for dimension,a in [('TX',y),('RX',rx),('day',day)]:masks.extend((dimension,str(v),a==v) for v in np.unique(a))
    masks.extend(('RX_day',str(r)+'/'+str(d),(rx==r)&(day==d)) for r in np.unique(rx) for d in np.unique(day))
    rows=[]
    for v,view in enumerate(VIEWS):
        for dimension,stratum,mask in masks:
            if not mask.any():continue
            matrices=[]
            for j in range(2):matrices.append(np.bincount(6*y[mask].astype(np.int64)+pred[j,v,mask],minlength=36).reshape(6,6))
            raw,corrected=[metrics(cm) for cm in matrices];a=pred[0,v]==y;b=pred[1,v]==y
            rows.append(dict(view=view,dimension=dimension,stratum=stratum,count=int(mask.sum()),raw=raw,corrected=corrected,
                fixed=int((mask&~a&b).sum()),broken=int((mask&a&~b).sum()),delta_correct=corrected['correct']-raw['correct'],
                accuracy_delta_pp=corrected['accuracy']-raw['accuracy'],raw_confusion=matrices[0].tolist(),corrected_confusion=matrices[1].tolist()))
    return rows


def source_rule(rows):
    cells={(r['view'],r['dimension'],r['stratum']):r for r in rows}
    if len(cells)!=len(rows):raise ValueError('Unique source strata required')
    for view in VIEWS:
        if cells[view,'overall','ALL']['count']!=27000:raise ValueError('All sourceV27000 records required')
        for rx in RXS:
            if cells[view,'RX',str(rx)]['count']!=5400:raise ValueError('All five sourceRX5400 required')
    conditions=dict(clean_no_worse_than_27=cells['clean','overall','ALL']['delta_correct']>=-27,
        three_H_total_nondecreasing=sum(cells[v,'overall','ALL']['delta_correct'] for v in SCENES)>=0,
        every_H_no_worse_than_54=all(cells[v,'overall','ALL']['delta_correct']>=-54 for v in SCENES),
        every_RX_view_no_worse_than_point2=all(cells[v,'RX',str(r)]['accuracy_delta_pp']>=-.2 for v in VIEWS for r in RXS))
    return dict(conditions=conditions,source_necessary_pass=all(conditions.values()),scope='fixedF immediate preprocessing only;not trainedF compensation or target performance')


def check_reader_smoke(model,outputs):
    import torch
    pool=model.sequence_pool;head=model.get_submodule('core.id_backbone.cls_head')
    signature_parameters=sum(p.numel() for p in pool.parameters())
    if (any(p.shape!=(2,6) or not torch.isfinite(p).all() for p in outputs) or len(outputs)!=2 or len(pool.views)!=1
        or signature_parameters!=18976 or tuple(head.weight.shape)!=(6,160)):
        raise ValueError('Actual ownE20 raw/corrected defaulthead noquery smoke failed')
    return dict(reader_parameters=sum(p.numel() for p in model.parameters()),signature_parameters=signature_parameters,default_head_shape=[6,160],signature_views=1)


def evaluate(rt):
    import torch
    from experiments.pamf_channel_pl.sig1_matched_baselines import load_prefix
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.pamf_channel_pl.restored_source import source_channel,snapshot,unchanged
    c=validate_context(read(rt['context']));activate(c);seed=rt['model_seed']
    if seed not in SEEDS:raise ValueError('All three declared frozen models only')
    cfg=next(r['baseline_config'] for r in c['cases'] if r['model_seed']==seed);parent,payload=load_prefix(cfg);contract=read(c['source_contract'])
    calroot=Path(rt['run_root'])/'calibration';header=read(calroot/'complete.json')
    if header['recipe']!=RECIPE or header['source_contract']!=contract or header['source_role']!='U_s' or header['U_seen']!=56700 or header['V_reads'] or header['target_reads']:raise ValueError('Original U-only frozen calibration required')
    with np.load(calroot/'calibration.npz',allow_pickle=False) as a:correction=Correction(Calibration(a['center'],a['basis']))
    root=Path(rt['run_root'])/('s'+str(seed));root.mkdir(parents=True,exist_ok=False)
    with reader_structure(cfg):model=reader_from_payload(cfg,payload,'cuda:0')
    del payload;before=snapshot(model);l=SourceDataset(c['source_cache'],'L_s',contract)
    x=np.asarray(l.x[:2].tolist(),np.float32);can,_=correction(x)
    with torch.inference_mode():outputs=[model(tensor(v,'cuda:0')) for v in (x,can)]
    signature=check_reader_smoke(model,outputs)
    write(root/'smoke.json',dict(status='PASS',model_seed=seed,source_L=2,source_parent=parent['folder'],**signature,query_reads=0,optimizer_updates=0))
    del l;data=SourceDataset(c['source_cache'],'V',contract)
    if len(data)!=27000:raise ValueError('Full original sourceV required')
    logits=np.empty((2,4,27000,6),np.float32);seen=0;start=time.monotonic();timing=dict(simulation_seconds=0.,correction_CPU_seconds=0.,raw_F_seconds=0.,corrected_F_seconds=0.);quality={v:dict(records=0,gate_sum=0.,gain_bound_sum=0.,phase_bound_sum=0.,max_gain_db=0.,max_phase_rad=0.) for v in VIEWS}
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0):
            t=time.monotonic();views=[batch[0],*[source_channel(batch[0],batch[3],s,contract) for s in SCENES]];timing['simulation_seconds']+=time.monotonic()-t
            for j,v in enumerate(views):
                raw=np.asarray(v.tolist(),np.float32);t=time.monotonic();can,stats=correction(raw);timing['correction_CPU_seconds']+=time.monotonic()-t
                q=quality[VIEWS[j]];q['records']+=len(raw);q['gate_sum']+=len(raw)*stats['mean_quality_gate'];q['gain_bound_sum']+=len(raw)*stats['gain_bound_fraction'];q['phase_bound_sum']+=len(raw)*stats['phase_bound_fraction'];q['max_gain_db']=max(q['max_gain_db'],stats['max_abs_gain_db']);q['max_phase_rad']=max(q['max_phase_rad'],stats['max_abs_phase_rad'])
                for k,a in enumerate((raw,can)):
                    x=tensor(a,'cuda:0');torch.cuda.synchronize();t=time.monotonic();p=model(x);torch.cuda.synchronize();timing[('raw' if k==0 else 'corrected')+'_F_seconds']+=time.monotonic()-t
                    if p.shape!=(len(raw),6) or not torch.isfinite(p).all():raise ValueError('Finite original sixclass defaulthead source logits required')
                    logits[k,j,seen:seen+len(raw)]=np.asarray(p.cpu().tolist(),np.float32)
            seen+=len(batch[0]);write(root/'progress.json',dict(V_seen=seen,total=27000,model_seed=seed,target_reads=0),exclusive=False);print('BOUNDED_SOURCE',seed,seen,flush=True)
    if seen!=27000 or not unchanged(model,before):raise ValueError('Complete sourceV and immutable model required')
    np.savez(root/'source_predictions.npz',logits=logits,y=data.metadata['y'],ids=data.metadata['ids'],rx=data.metadata['rx_i'],day=data.metadata['day_i'],views=np.asarray(VIEWS))
    rows=paired_results(logits,data.metadata['y'],data.metadata['rx_i'],data.metadata['day_i']);decision=source_rule(rows)
    write(root/'complete.json',dict(status='BOUNDED_F_SOURCE_EVALUATED',model_seed=seed,rows=rows,decision=decision,quality=quality,timing=timing,seconds=time.monotonic()-start,
        source_V_seen=seen,parameters_and_buffers_unchanged=True,optimizer_updates=0,U_TX_truth_reads=0,G_weight_reads=0,target_reads=0))


def dispatch(rt):
    from experiments.pamf_repl_completion.runner import capacity,processes
    from experiments.pamf_peer_supervised.runner import launch
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.practical_full_2x2.cache import atomic
    from .covariance_queue import admissible
    if rt['run_id']!=RUN or not rt.get('cpu_only'):raise ValueError('One F-free calibration andthree frozen sourceF evaluations')
    c=validate_context(read(rt['context']));root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);jobs=[]
    for seed in (None,*SEEDS):
        key='calibration' if seed is None else 's'+str(seed);path=root/'configs'/(key+'.json');write(path,dict(rt,model_seed=seed))
        jobs.append(dict(id=key,kind='calibrate' if seed is None else 'evaluate',config=str(path),completion=str(root/('calibration' if seed is None else key)/'complete.json'),
            successful_statuses=['BOUNDED_SOURCE_CALIBRATED' if seed is None else 'BOUNDED_F_SOURCE_EVALUATED'],dependencies=[] if seed is None else [str(root/'calibration/complete.json'),c['predecessor_complete']],execution_release=rt['release_root'],execution_python=rt['python'],entrypoint='experiments.pamf_u_history.bounded_response_source'))
    write(root/'jobs.json',jobs);state=dict(status='RUNNING',pid=__import__('os').getpid(),code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs});active={};stop=False
    while True:
        for key,(child,item,job) in list(active.items()):
            code=child.poll()
            if code is None:continue
            good=code==0 and done(job);state['jobs'][key]=dict(item,status='COMPLETE' if good else 'FAILED',exit_code=code);stop|=not good;del active[key]
        pending=processes()
        for job in jobs:
            key=job['id']
            if stop or state['jobs'][key]['status']!='QUEUED' or not ready(job):continue
            gpu=None
            if job['kind']=='evaluate':
                # Dependency is completion only;source study never consumes target scores.
                previous=read(c['predecessor_complete'])
                if previous['status']!='THREE_SEED_COVARIANCE_COMPLETE' or previous['logical_models']!=12:raise ValueError('Prior matrix mustfinish without source-score selection')
                choices=[]
                for card in rt['allowed_gpus']:
                    pids,memory=capacity(card)
                    if admissible(card,pids,pending,{k:v[1] for k,v in active.items()},memory):choices.append((-memory,card))
                if not choices:continue
                gpu=min(choices)[1]
            child,item=launch(rt,job,gpu);active[key]=(child,item,job);state['jobs'][key]=item
            if gpu is not None:pending[child.pid]=gpu
        state.update(updated_at=time.time(),systemic_stop_new_launches=stop);atomic(root/'pipeline_state.json',state)
        if not active and (stop or all(v['status']=='COMPLETE' for v in state['jobs'].values())):break
        time.sleep(12)
    state['status']='FAILED' if stop else 'BOUNDED_RESPONSE_SOURCE_COMPLETE';atomic(root/'pipeline_state.json',state)
    if not stop:
        results=[read(root/('s'+str(s))/'complete.json') for s in SEEDS]
        write(root/'complete.json',dict(status='BOUNDED_RESPONSE_SOURCE_COMPLETE',model_seeds=list(SEEDS),source_necessary_pass=all(r['decision']['source_necessary_pass'] for r in results),
            decisions=[dict(model_seed=r['model_seed'],**r['decision']) for r in results],new_training=0,target_reads=0,full_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind',choices=['calibrate','evaluate']);p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config)
    if a.action=='dispatch':dispatch(c)
    elif a.kind=='calibrate':calibrate(c)
    else:evaluate(c)
