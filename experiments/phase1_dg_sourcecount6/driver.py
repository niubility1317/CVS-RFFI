"""Idle-card owner: prepare once, smoke, train/freeze/predict/score every row."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import time
from experiments.phase1_external_dg.driver import atomic,environment,run_stage,failure_fingerprint
from experiments.phase1_external_dg.data import read,write
from .plan import make_plan,validate_plan,SEEDS,VIEWS


def source_config(rt,row,h):
    meta=read(Path(h['source_cache'])/'metadata.json')
    return dict(row,source_cache=h['source_cache'],source_contract_path=h['source_contract'],
        classes=meta['classes'],num_classes=6,num_domains=len(meta['source_contract']['source_rxs']),
        ibrff_model_root=rt['ibrff_model_root'],sdg_model_root=rt['sdg_model_root'],
        output_root=str(Path(rt['run_root'])/row['row_id']/'source'))


def worker(c):
    from .runtime import bind
    bind()
    rt,row,h=c['runtime'],c['row'],c['handle']
    out=Path(rt['run_root'])/row['row_id'];out.mkdir(parents=True,exist_ok=False)
    env=environment(rt,os.environ.get('CUDA_VISIBLE_DEVICES',''))
    sc=source_config(rt,row,h);write(out/'source_config.json',sc)
    tests={k:dict(views_root=t['views_root'],kind=t['kind']) for k,t in h['tests'].items()}
    pc=dict(row_id=row['row_id'],model_seed=row['model_seed'],source_output=sc['output_root'],
        expected_source_contract=h['source_contract'],output_root=str(out/'prediction'),tests=tests)
    action='predict'
    if row['reuse_parent_row_id']:
        from .reuse import parent_source
        parent,cost=parent_source(rt,row,h)
        write(out/'reused_source.json',dict(parent_run=rt['parent_run_root'],parent_row=row['reuse_parent_row_id'],
            source_output=str(parent/'source'),source_contract=h['source_contract'],
            provenance='identical physical source roles and complete native recipe; source-only frozen',
            original_cost=cost,new_training_seconds=0.))
        pc.update(parent_prediction=str(parent/'prediction'),classes=sc['classes']);action='reuse_predict'
    else:
        subprocess.run([rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','train',
            '--config',str(out/'source_config.json')],env=env,check=True)
    write(out/'prediction_config.json',pc)
    subprocess.run([rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver',action,
        '--config',str(out/'prediction_config.json')],env=env,check=True)
    score=dict(row_id=row['row_id'],model_seed=row['model_seed'],classes=sc['classes'],
        prediction_output=pc['output_root'],output_root=str(out/'scores'),tests=h['tests'])
    write(out/'score_config.json',score)
    subprocess.run([rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','score',
        '--config',str(out/'score_config.json')],env=environment(rt),check=True)
    write(out/'complete.json',dict(status='ANALYZED',row_id=row['row_id'],model_seed=row['model_seed'],
        trained=not bool(row['reuse_parent_row_id']),reused=bool(row['reuse_parent_row_id']),
        training_regime=row['training_regime'],satellite_train_augmentation=True,
        tests=['main'],views=list(VIEWS),default_head=True))
    print('ROW_ANALYZED',row['row_id'],flush=True)


def smoke(c):
    from .runtime import bind
    bind()
    from experiments.phase1_external_dg_aug.driver import smoke as run
    run(c)


def dispatch(rt):
    plan=validate_plan(read(rt['matrix']))
    if plan!=make_plan() or rt['allowed_gpus']!=plan['allowed_gpus']:
        raise ValueError('Runtime matrix differs')
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    Path(rt['log_root']).mkdir(parents=True,exist_ok=False)
    state=dict(status='STARTING',run_id=rt['run_id'],pid=os.getpid(),cwd=os.getcwd(),
        allowed_gpus=rt['allowed_gpus'],rows={},expected_rows=432,expected_training=384,reused_rows=48)
    atomic(root/'pipeline_state.json',state)
    run_stage(rt,state,'prepare_data',[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver',
        'prepare','--config',rt['runtime_file']])
    handles=read(root/'data_handles.json');chosen=[];seen=set()
    for r in plan['rows']:
        if r['model_seed']!=SEEDS[0] or r['reuse_parent_row_id']:continue
        if r['source_count'] not in (1,4,6):continue
        key=(r['family'],r['variant'],r['U_s_use'],r['source_count'])
        if key in seen:continue
        seen.add(key);chosen.append(source_config(rt,r,handles[r['protocol_id']]))
    # One fixed-L/V real-source checkpoint also exercises its distinct contract.
    fixed=next(r for r in plan['rows'] if r['label_budget']=='fixed' and r['source_count']==4)
    chosen.append(source_config(rt,fixed,handles[fixed['protocol_id']]))
    write(root/'smoke_config.json',dict(rows=chosen,output_root=str(root/'smoke')))
    run_stage(rt,state,'smoke',[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver',
        'smoke','--config',str(root/'smoke_config.json')])
    pending=list(plan['rows']);active={};failures=Counter()
    while pending or active:
        for gpu,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            rid=item['row']['row_id'];record=state['rows'][rid]
            record.update(returncode=code,status='ANALYZED' if code==0 and (root/rid/'complete.json').is_file() else 'FAILED')
            item['log'].close()
            if record['status']=='FAILED':
                fingerprint=failure_fingerprint(record['log'],rid);failures[fingerprint]+=1;record['failure']=fingerprint
            del active[gpu]
        if pending and failures and max(failures.values())>=2:
            state['unstarted_rows']=[r['row_id'] for r in pending];pending.clear()
            state['stop_reason']='two_matching_technical_failures'
        for gpu in rt['allowed_gpus']:
            if gpu in active or not pending:continue
            occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True).strip()
            if occupied:continue
            row=pending.pop(0);rid=row['row_id'];config=root/(rid+'.json')
            write(config,dict(runtime=rt,row=row,handle=handles[row['protocol_id']]))
            logfile=Path(rt['log_root'])/(rid+'.log');log=logfile.open('x')
            command=[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver','worker','--config',str(config)]
            child=subprocess.Popen(command,cwd=rt['release_root'],env=environment(rt,gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=dict(process=child,row=row,log=log)
            state['rows'][rid]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,cwd=rt['release_root'],log=str(logfile))
        state.update(status='RUNNING' if active else 'WAITING_GPU',pending_rows=len(pending))
        atomic(root/'pipeline_state.json',state)
        if pending or active:time.sleep(30)
    state['status']='PARTIAL' if failures or state.get('unstarted_rows') else 'TRAINING_PREDICTIONS_SCORING_DONE'
    atomic(root/'pipeline_state.json',state)
    run_stage(rt,state,'analysis',[rt['python'],'-u','-m','experiments.phase1_dg_sourcecount6.driver',
        'analysis','--config',rt['runtime_file']])


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('dispatch','worker','prepare','smoke','train',
        'predict','reuse_predict','score','analysis'));p.add_argument('--config',required=True);a=p.parse_args()
    c=read(a.config)
    if a.action in ('dispatch','worker','smoke'):return globals()[a.action](c)
    if a.action=='prepare':
        from .data import prepare
        return prepare(c)
    from .runtime import bind
    source,evaluation=bind()
    if a.action=='train':return source.train(c)
    if a.action in ('predict','score'):return getattr(evaluation,a.action)(c)
    if a.action=='reuse_predict':
        from .reuse import predict
        return predict(c)
    from .analysis import analyze
    return analyze(c)


if __name__=='__main__':main()
