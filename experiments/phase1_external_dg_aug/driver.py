"""Two-GPU owner for the user-revised augmented supervised/PL matrix."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import subprocess
import time

from experiments.phase1_external_dg.driver import atomic, environment, parent_state, run_stage, failure_fingerprint, process_alive
from .data import read, write
from .plan import make_plan, validate_plan, SEEDS, VIEWS


def source_config(rt, row, handle):
    meta = read(Path(handle['source_cache'])/'metadata.json')
    return dict(row, source_cache=handle['source_cache'], source_contract_path=handle['source_contract'],
        classes=meta['classes'], num_classes=len(meta['classes']),
        num_domains=len(meta['source_contract']['source_rxs']),
        ibrff_model_root=rt['ibrff_model_root'], sdg_model_root=rt['sdg_model_root'],
        output_root=str(Path(rt['run_root'])/row['row_id']/'source'))


def worker(c):
    rt,row,h=c['runtime'],c['row'],c['handle']
    root=Path(rt['run_root'])/row['row_id'];root.mkdir(parents=True,exist_ok=False)
    sc=source_config(rt,row,h);write(root/'source_config.json',sc)
    env=environment(rt,os.environ['CUDA_VISIBLE_DEVICES'])
    subprocess.run([rt['python'],'-u','-m','experiments.phase1_external_dg_aug.source',
        '--config',str(root/'source_config.json')],env=env,check=True)
    tests={name:dict(views_root=t['views_root'],kind=t['kind']) for name,t in h['tests'].items()}
    pc=dict(row_id=row['row_id'],model_seed=row['model_seed'],source_output=sc['output_root'],
        expected_source_contract=h['source_contract'],output_root=str(root/'prediction'),tests=tests)
    write(root/'prediction_config.json',pc)
    subprocess.run([rt['python'],'-u','-m','experiments.phase1_external_dg_aug.evaluation','predict',
        '--config',str(root/'prediction_config.json')],env=env,check=True)
    score=dict(row_id=row['row_id'],model_seed=row['model_seed'],classes=sc['classes'],
        prediction_output=pc['output_root'],output_root=str(root/'scores'),tests=h['tests'])
    write(root/'score_config.json',score)
    subprocess.run([rt['python'],'-u','-m','experiments.phase1_external_dg_aug.evaluation','score',
        '--config',str(root/'score_config.json')],env=environment(rt),check=True)
    write(root/'complete.json',dict(status='ANALYZED',row_id=row['row_id'],model_seed=row['model_seed'],
        training_regime=row['training_regime'],satellite_train_augmentation=True,
        trained=True,tests=list(tests),views=list(VIEWS),default_head=True))
    print('ROW_ANALYZED',row['row_id'],flush=True)


def smoke(c):
    import torch
    from . import source
    from .augmentation import DefaultConcat
    from .objectives import step
    from experiments.phase1_dg_nativebudget import pipeline as native
    from experiments.phase1_external_dg.models import balanced_indices,FishrVariance
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    results=[]
    for row in c['rows']:
        meta=source.check_training(row);native.seed(row['model_seed']);net=source.build(row)
        batch=next(iter(source.array_loader(row,'L_s',12)))
        x,y,rx=batch;author=None;views=None;fishr=None;fed=None
        if row['family']=='resnet18' and row['route'] in ('mixstyle','mixstyle_id','fishr','coral'):
            xx,yy,rr=source.array_loader(row,'L_s').dataset.tensors
            ix=balanced_indices(yy,rr,row['num_classes'],row['num_domains'],150,
                torch.Generator().manual_seed(row['model_seed']))
            x,y,rx=xx[ix],yy[ix],rr[ix]
        if row['family']=='sdg' and row['variant']=='full':
            author=native.author_sdg(row)
            dataset=source.SDGDataset(source.array_loader(row,'L_s'),author)
            x,y,rx,views=next(iter(torch.utils.data.DataLoader(dataset,batch_size=12)))
        if row['family']=='riei':
            params=list(net.ec.parameters())+(list(net.rc.parameters()) if row['variant']=='full' else list(net.fed.parameters()))
            optimizer=torch.optim.SGD(params,lr=row['lr'])
            if row['variant']=='full':fed=torch.optim.SGD(net.fed.parameters(),lr=row['lr'])
        else:optimizer=torch.optim.Adam(net.parameters(),lr=row['lr'])
        if row.get('route')=='fishr':fishr=FishrVariance(row['num_domains'],row['fishr_ema'])
        u=u_rx=None
        if row['U_s_use']:
            u,u_rx=next(iter(source.array_loader(row,'U_s')));u,u_rx=u[:len(y)],u_rx[:len(y)]
        result=step(net,row,x,y,rx,optimizer,DefaultConcat(),.5,
            row.get('fishr_warmup',0),stage=2 if row['family']=='ibrff' else 1,
            fed_optimizer=fed,unlabeled=u,native_views=views,author=author,fishr=fishr,unlabeled_receivers=u_rx)
        path=root/(row['row_id']+'.pt')
        native.save(path,net,row,meta['source_contract'],0,0,0.)
        loaded=native.load(path,row,meta['source_contract']);copy=source.build(row)
        copy.load_state_dict(loaded['state_dict'],strict=True);copy.eval();net.eval()
        with torch.no_grad():
            a=source.forward(net,row,x);b=source.forward(copy,row,x)
        if not torch.equal(a,b) or not torch.isfinite(a).all():
            raise ValueError('Real-checkpoint original-head no-query smoke differs')
        results.append(dict(row_id=row['row_id'],status='PASS',classes=row['num_classes'],
            source_RX=row['num_domains'],joint_objective=result,query_access=False,
            default_head=True,used_as_training_initialization=False))
        del net,copy,optimizer,loaded,batch,x,y,rx,u,views
        gc.collect()
    write(root/'result.json',dict(status='PASS',rows=results,query_access=False,
        note='Real L/U source-only joint objective and strict checkpoint smoke; not a formal training initialization.'))


def dispatch(rt):
    plan=validate_plan(read(rt['matrix']))
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    Path(rt['log_root']).mkdir(parents=True,exist_ok=False)
    state=dict(status='STARTING',run_id=rt['run_id'],pid=os.getpid(),cwd=os.getcwd(),
        allowed_gpus=[0,1],rows={},expected_training=219,expected_supervised=192,expected_pl=27)
    write(root/'dispatcher.json',state)
    run_stage(rt,state,'prepare_data',[rt['python'],'-u','-m','experiments.phase1_external_dg_aug.driver',
        'prepare','--config',rt['runtime_file']])
    handles=read(root/'data_handles.json')
    chosen=[];seen=set()
    for row in plan['rows']:
        if row['model_seed']!=SEEDS[0]:continue
        c=source_config(rt,row,handles[row['protocol_id']])
        key=(c['family'],c['variant'],c.get('route'),c['num_classes'],c['num_domains'],c['U_s_use'])
        if key in seen:continue
        seen.add(key);chosen.append(c)
    write(root/'smoke_config.json',dict(rows=chosen,output_root=str(root/'smoke')))
    run_stage(rt,state,'smoke',[rt['python'],'-u','-m','experiments.phase1_external_dg_aug.driver',
        'smoke','--config',str(root/'smoke_config.json')])
    while True:
        verdict,parent=parent_state(rt)
        if verdict=='DONE':break
        state.update(status='WAITING_PARENT' if verdict=='LIVE' else 'PARENT_NEEDS_RECOVERY',parent_status=parent['status'])
        atomic(root/'pipeline_state.json',state)
        if verdict!='LIVE':return
        time.sleep(10)
    pending=list(plan['rows']);active={};failures=Counter()
    while pending or active:
        for gpu,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            rid=item['row']['row_id'];done=root/rid/'complete.json'
            row_state=state['rows'][rid]
            row_state.update(returncode=code,status='ANALYZED' if code==0 and done.is_file() else 'FAILED')
            item['log'].close()
            if row_state['status']=='FAILED':
                fingerprint=failure_fingerprint(row_state['log'],rid);failures[fingerprint]+=1
                row_state['failure']=fingerprint
            del active[gpu]
        if pending and failures and max(failures.values())>=2:
            state['unstarted_rows']=[r['row_id'] for r in pending];pending.clear()
            state['stop_reason']='two_matching_technical_failures'
        for gpu in (0,1):
            if gpu in active or not pending:continue
            occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if occupied:continue
            row=pending.pop(0);rid=row['row_id'];path=root/(rid+'.json')
            write(path,dict(runtime=rt,row=row,handle=handles[row['protocol_id']]))
            logfile=Path(rt['log_root'])/(rid+'.log');log=logfile.open('x')
            command=[rt['python'],'-u','-m','experiments.phase1_external_dg_aug.driver','worker','--config',str(path)]
            child=subprocess.Popen(command,cwd=rt['release_root'],env=environment(rt,gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=dict(process=child,row=row,log=log)
            state['rows'][rid]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,cwd=rt['release_root'],log=str(logfile))
        state['status']='RUNNING';atomic(root/'pipeline_state.json',state)
        if pending or active:time.sleep(10)
    state['status']='PARTIAL' if failures or state.get('unstarted_rows') else 'PERFORMANCE_TRAINING_AND_SCORING_DONE'
    atomic(root/'pipeline_state.json',state)
    run_stage(rt,state,'analysis',[rt['python'],'-u','-m','experiments.phase1_external_dg_aug.analysis','--config',rt['runtime_file']])


def prepare(rt):
    from .data import prepare as build_data
    return build_data(rt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('dispatch','worker','prepare','smoke'))
    parser.add_argument('--config',required=True);args=parser.parse_args()
    globals()[args.action](read(args.config))
