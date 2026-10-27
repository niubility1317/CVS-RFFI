"""One immutable owner; automatic per-row prediction, independent scoring and report."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.cric_dura_comparison.io import read,write
from .plan import RUN,SEEDS,ARMS,TASKS,READOUTS,make_plan,configuration

def activate(c):
    from experiments.pamf_repl_completion.runner import activate as start
    start(c)

def prepare(rt):
    ctx=read(rt['context']);root=Path(rt['run_root']);jobs=[];sources={}
    def add(kind,row,c,complete,status,deps=()):
        path=root/'configs'/(kind+'__'+row['row_id']+'.json');write(path,c)
        jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,config=str(path),completion=str(complete),
                         successful_statuses=[status],dependencies=list(deps)))
    for row in make_plan()['rows']:
        c=configuration(ctx,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        h=ctx['handles'][row['legacy_protocol_id']];old=None
        if row['reuse_key']:
            old=ctx['single_view_reuse'][row['reuse_key']];folder=Path(old['source']);d=read(folder/'complete.json');pc=d['config']
            if (d['status']!='SOURCE_FROZEN' or d['updates']!=11100 or d['target_access'] or d['U_truth_read'] or d['head_fit_updates']
                or read(pc['source_contract'])!=read(h['source_contract']) or pc['model_seed']!=row['model_seed']
                or pc['U_use'] is not (row['learning_regime']=='pl')):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: reuse')
            if row['family']=='branch' and (pc['ablation']!='F15' or pc['satellite_CE_weight']!=.68 or not pc['channel_augmentation']):raise ValueError('Only matching CE control may be reused')
            if row['family']=='main' and (pc['repl_row']['structure']!='SIG_TIME' or not pc['repl_row']['L_A'] or not pc['repl_row']['L_B']):raise ValueError('Only matching full-L single-view control may be reused')
        else:
            folder=Path(c['output_root']);add('train',row,c,folder/'complete.json','SOURCE_FROZEN')
        out=root/'readouts'/row['row_id']
        p=dict(run_id=RUN,row_id=row['row_id'],paper_row=row,protocol_id=row['legacy_protocol_id'],model_seed=row['model_seed'],
            source_output=str(folder),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
            restored_caches={k:v['path'] for k,v in ctx['caches'][row['legacy_protocol_id']].items()},
            views=row['views'],readouts=list(READOUTS),reuse_predictions=old['predictions'] if old else None,
            output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        add('predict',row,p,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(folder/'complete.json')])
        s=deepcopy(p);s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
        for key in ('source_output','restored_caches','frozen_sources','reuse_predictions'):s.pop(key,None)
        add('score',row,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        sources[row['row_id']]=dict(row=row,source_output=str(folder),predictions=p['output_root'],scores=s['output_root'],reused=bool(old))
    assert len(jobs)==348 and sum(j['kind']=='train' for j in jobs)==108
    write(root/'jobs.json',jobs);write(root/'sources.json',sources)
    return jobs

def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from .model import installed,load_reader
    ctx=read(rt['context']);plan=make_plan()['rows'];base=next(r for r in plan if r['family']=='main' and r['data_setting']=='main_split' and r['learning_regime']=='pl' and r['channel_augmentation'] and r['model_seed']==SEEDS[0])
    c=configuration(ctx,base,rt['project'],rt['code_commit']);old=ctx['single_view_reuse'][base['reuse_key']]
    contract=read(c['source_contract']);reader,_=load_reader(old['source'],contract,base,'cuda:0')
    from experiments.cric_dura_comparison.f_data import SourceDataset
    x=SourceDataset(c['source_cache'],'L_s',contract).x[:2].to('cuda:0')
    with torch.inference_mode():
        logits=reader.classify_features(reader.features(x))
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real matching checkpoint no-query smoke failed')
    del reader,logits,x
    chosen=[r for r in plan if r['model_seed']==SEEDS[0] and ((r['family']=='branch') or (r['family']=='main' and ((r['data_setting'] in ('receiver_split_1','ten_class') and r['learning_regime']=='supervised') or (r['data_setting']=='main_split' and not r['channel_augmentation']))))]
    checks=[]
    for row in chosen:
        c=configuration(ctx,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        torch.manual_seed(c['model_seed']);torch.cuda.manual_seed_all(c['model_seed'])
        with installed(c):
            loaders=f_train.context(c)
            if (loaders['unlabeled_loader'] is not None)!=c['U_use']:raise ValueError('Actual U loader permissions differ')
            with f_train.class_dimension(c),native_scope(f_train.original_config(c['model_seed'])) as native:
                args=f_train.arguments(c,'cuda:0');net=t1_model.install(native.build_baseline_model(args,'cuda:0'),c).to('cuda:0')
            net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader) if net._cd_U_loader else None
            batch=next(iter(loaders['train_loader']));x,y,d=[a.to('cuda:0') for a in batch[:3]];net.train();net._feature_l_steps=4440
            with net._phase1_labeled_feature_context():out=net(x,return_aux=True,domain_labels=d)
            ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=0. if row['family']=='branch' else .01)
            correction,metrics=net._source_labeled_feature_hook(model=net,labeled_output=out,labels=y,extra=(None,batch[3]),domains=d,
                optimizer_step=4441,identity_loss=ce,labeled_count=128,labeled_inputs=x)
            loss=ce+correction;loss.backward()
            head=net.id_backbone.encoder.core.id_backbone.cls_head.weight
            if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm())<=0
                or net._cd_U_seen!=(256 if c['U_use'] else 0) or metrics['T1_L_packets']!=(384 if row['family']=='branch' else 768)
                or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):raise ValueError('Actual source L/U objective/backward differs')
            if row['family']=='branch' and (metrics['CRIC_supcon_weight']!=0 or metrics['T1_ordinary_KL_weight']!=0):raise ValueError('CE-only branch executed an auxiliary objective')
            ref=net.id_backbone.encoder
            if row['arm']!='no_signature' and sum(float(p.grad.square().sum()) for p in ref.sequence_pool.parameters() if p.grad is not None)<=0:raise ValueError('Single descriptor did not receive gradients')
            checks.append(dict(row_id=row['row_id'],parameters=sum(p.numel() for p in ref.parameters()),actual_U=net._cd_U_seen,L_forward_packets=metrics['T1_L_packets'],finite_joint_backward=True))
            del net,ref,head,loss,ce,correction,out,x,y,d,loaders
            torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',real_checkpoint=True,query_reads=0,U_truth_reads=0,optimizer_updates=0,checks=checks))

def analyze(rt,jobs):
    import csv
    import statistics
    root=Path(rt['run_root']);records=[];groups={}
    for job in jobs:
        if job['kind']!='score':continue
        c=read(job['config']);r=c['paper_row'];scores=read(Path(c['output_root'])/'scores.json')
        if not scores['truth_last'] or not scores['target_feedback_forbidden']:raise ValueError('Independent score required')
        for metric in ('accuracy','macro_f1'):
            for t in TASKS:
                for rd in READOUTS:
                    for view in c['views']+(['six_practical_mean'] if len(c['views'])==7 else []):
                        vv=[x[metric] for x in scores['results'] if x['dimension']=='overall' and x['test']==t and x['readout']==rd and (x['view']!='clean' if view=='six_practical_mean' else x['view']==view)]
                        if len(vv)!=(6 if view=='six_practical_mean' else 1):raise ValueError('Missing scored view')
                        value=100.*statistics.mean(vv)
                        rec=dict(data_setting=r['split_name'],family=r['family'],structure=r['arm_name'],learning=r['learning_name'],channel_augmentation=r['channel_augmentation'],
                            test={'cross_date':'Unseen date','cross_rx':'Unseen receivers','cross_both':'Unseen receivers and date'}[t],readout=rd,view=view,metric=metric,model_seed=r['model_seed'],percent=value)
                        records.append(rec);key=tuple(rec[k] for k in list(rec)[:-2]);groups.setdefault(key,[]).append((r['model_seed'],value))
    out=root/'analysis';out.mkdir(exist_ok=False);summary=[]
    for key,values in groups.items():
        if sorted(s for s,v in values)!=list(SEEDS):raise ValueError('All three registered seeds required')
        cols=list(records[0])[:-2];v=[n for s,n in values]
        summary.append(dict(zip(cols,key),seed_n=3,mean=statistics.mean(v),sd=statistics.stdev(v)))
    for name,data in [('per_seed.csv',records),('summary.csv',summary)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Single-view PAMF paper experiments','', '120 logical classifiers; 108 new, 12 exact matching reuses. No new DURA training.','',
        'All rows completed fixed-final E50 and their declared independent truth-last evaluation.','',
        '|Split|Objective family|Structure|Learning|Augmentation|Test|Readout|View|Accuracy %|','|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        if r['metric']=='accuracy' and r['view'] in ('clean','six_practical_mean'):
            lines.append('|'+ '|'.join(str(r[k]) for k in ('data_setting','family','structure','learning','channel_augmentation','test','readout','view'))+f"|{r['mean']:.4f} ± {r['sd']:.4f}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'complete.json',dict(status='ANALYZED',logical_F=120,new_F=108,reused_F=12,new_G=0,summary_rows=len(summary),all_three_seeds=True))

def worker(c,kind):
    if kind=='score':
        from .evaluation import score
        return score(c)
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from .source import train
        return train(c)
    if kind=='predict':
        from .evaluation import predict
        return predict(c)
    raise ValueError('Unknown worker kind')

def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.pamf_repl_completion.runner import processes,capacity,foreign_CPU_score
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();jobs=prepare(rt);by={j['id']:j for j in jobs};active={};failed=[]
    state=dict(status='SMOKE_PENDING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,code_commit=rt['code_commit'],owner=rt['launch_owner'],jobs={j['id']:dict(status='COMPLETE' if rt.get('reuse_completed_jobs') and done(j) else 'QUEUED') for j in jobs})
    def launch(job,gpu):
        path=root/'logs'/(job['id']+'.log');cmd=[rt['python'],'-u','-m','experiments.pamf_single_view_paper.runner','worker','--kind',job['kind'],'--config',job['config']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with path.open('x') as f:p=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        return p,dict(status='RUNNING',pid=p.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(path),started_at=time.time())
    atomic(root/'pipeline_state.json',state)
    while True:
        foreign=processes();gpu=None
        for g in rt['allowed_gpus']:
            pids,free=capacity(g)
            if len(pids|{p for p,v in foreign.items() if v==g})<2 and free>=12000:gpu=g;break
        if gpu is not None:break
        time.sleep(15)
    child,item=launch(dict(id='smoke',kind='smoke',config=rt['runtime_file']),gpu);state['smoke']=item;atomic(root/'pipeline_state.json',state)
    status=child.wait();state['smoke'].update(exit_code=status,status='PASS' if status==0 else 'FAILED')
    if status or read(root/'smoke.json')['status']!='PASS':
        state['status']='FAILED_NO_QUERY_SMOKE';atomic(root/'pipeline_state.json',state);return
    state['status']='RUNNING';turn=0
    while True:
        for key,(p,item) in list(active.items()):
            if p.poll() is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:failed.append(key)
            del active[key]
        block_failed_dependencies(jobs,state,[])
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED'];candidates=[j for j in pending if ready(j)] if not failed else []
        scorer=next((j for j in candidates if j['kind']=='score'),None)
        if scorer and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({p.pid for p,item in active.values()}):
            p,item=launch(scorer,None);active[scorer['id']]=(p,item);state['jobs'][scorer['id']]=item;candidates.remove(scorer)
        foreign=processes()
        for gpu in rt['allowed_gpus']:
            if any(item['gpu']==gpu for p,item in active.values()):continue
            pids,free=capacity(gpu);reserved={p for p,g in foreign.items() if g==gpu}
            if len(pids|reserved)>=2 or free<12000:continue
            available=[j for j in candidates if j['kind']!='score']
            if not available:continue
            wanted=('train','predict','train')[turn%3];turn+=1;job=next((j for j in available if j['kind']==wanted),available[0])
            p,item=launch(job,gpu);active[job['id']]=(p,item);state['jobs'][job['id']]=item;candidates.remove(job);foreign[p.pid]=gpu
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),failed_jobs=failed,stop_new_dispatch=bool(failed),GPU_workers=sum(item['gpu'] is not None for p,item in active.values()))
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or failed):break
        time.sleep(15)
    if failed or any(v['status']!='COMPLETE' for v in state['jobs'].values()):state['status']='PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status']='ANALYZING';atomic(root/'pipeline_state.json',state);analyze(rt,jobs);state['status']='ANALYZED'
    atomic(root/'pipeline_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
