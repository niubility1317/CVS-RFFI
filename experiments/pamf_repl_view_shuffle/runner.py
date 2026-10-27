import argparse,json,os,subprocess,sys,time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from .plan import RUN,SEEDS,TASKS,VIEWS,make_plan,configuration

def activate(c):
    from experiments.pamf_repl_completion.runner import activate as original
    original(c)

def prepare(rt):
    ctx=read(rt['context']);root=Path(rt['run_root']);jobs=[]
    three_view_scores=[str(root/'readouts'/r['row_id']/'scores/complete.json') for r in make_plan()['rows'] if r['view_count']==3]
    for row in make_plan()['rows']:
        c=configuration(ctx,row,rt['project'],rt['code_commit']);c['frozen_sources']=rt['frozen_sources']
        source=c['output_root'];out=root/'readouts'/row['row_id'];h=ctx['handles']['F0']
        p=dict(run_id=RUN,row_id=row['row_id'],model_seed=row['model_seed'],view_comparison=row,source_output=source,
            source_contract=c['source_contract'],classes=h['classes'],source_days=[1,2,3],readouts=['raw'],
            tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
        s=deepcopy(p);s.update(tests={t:h['tests'][t] for t in TASKS},prediction_output=p['output_root'],output_root=str(out/'scores'))
        for key in ('source_output','frozen_sources'):s.pop(key,None)
        for kind,cfg,completion,status,deps in [('train',c,Path(source)/'complete.json','SOURCE_FROZEN',three_view_scores if row['view_count']==1 else []),
            ('predict',p,out/'predictions/complete.json','ALL_PREDICTIONS_COMPLETE',[str(Path(source)/'complete.json')]),
            ('score',s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])]:
            path=root/'configs'/(kind+'__'+row['row_id']+'.json');write(path,cfg)
            jobs.append(dict(id=kind+'__'+row['row_id'],kind=kind,config=str(path),completion=str(completion),successful_statuses=[status],dependencies=deps))
    write(root/'jobs.json',jobs);return jobs

def smoke(rt):
    import torch
    from torch.nn import functional as F
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack.runtime import installed as native_scope
    from experiments.pamf_repl_completion.source import load_reader as historical_reader
    from .source import installed,assert_structure
    ctx=read(rt['context']);checks=[];device='cuda:0'
    old=ctx['historical_shuffle']['2026092701']['source'];contract=read(ctx['handles']['F0']['source_contract'])
    reader,_=historical_reader(old,contract,device)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    x=SourceDataset(ctx['handles']['F0']['source_cache'],'L_s',contract).x[:2].to(device)
    with torch.inference_mode():
        logits=reader.classify_features(reader.features(x))
        if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):raise ValueError('Real historical checkpoint smoke failed')
    del reader,logits,x;torch.cuda.empty_cache()
    # Both dimensions and the new final seed are exercised without target access.
    for seed in (SEEDS[0],SEEDS[-1]):
        for views in (3,1):
            row=next(r for r in make_plan()['rows'] if r['model_seed']==seed and r['view_count']==views)
            c=configuration(ctx,row,rt['project'],rt['code_commit']);c.update(frozen_sources=rt['frozen_sources'],hard_pl_smoke=True,epoch_offset=20,step_offset=4440,segment_epochs=10,parent_source=None)
            # Diagnostic warm weights are source-only smoke inputs, never training initializers.
            warm=ctx['historical_shuffle']['2026092701'] if views==3 else ctx['historical_single_smoke']
            parent=read(Path(warm['source'])/'complete.json')['segments'][0]['folder']
            payload=torch.load(Path(parent)/'final_ssdg.pth',map_location='cpu',weights_only=False)
            holder={};torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with installed(c,holder,True):
                loaders=f_train.context(c)
                with f_train.class_dimension(c),native_scope(f_train.original_config(seed)) as native:
                    a=f_train.arguments(c,device);a.input_len=256;a.baseline_ckpt=a.teacher_ckpt='';a.from_scratch=a.a1_scratch_only=True
                    if a.seed!=seed or a.use_unlabeled or a.use_ema_teacher:raise ValueError('Actual model seed/native U permissions differ')
                    net=t1_model.install(native.build_baseline_model(a,device),c).to(device)
                shape=assert_structure(net,c);net.load_state_dict(payload['model'],strict=True);net.train()
                net._cd_U_loader=loaders['unlabeled_loader'];net._cd_U_iter=iter(net._cd_U_loader)
                batch=next(iter(loaders['train_loader']));x,y,d=[v.to(device) for v in batch[:3]]
                meta={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
                with net._phase1_labeled_feature_context():out=net(x,return_aux=True,domain_labels=d)
                ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                correction,m=net._source_labeled_feature_hook(model=net,labeled_output=out,labels=y,extra=(None,meta),domains=d,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
                loss=ce+correction;loss.backward();obj=holder['objective'];st=obj.state_dict();head=net.id_backbone.encoder.core.id_backbone.cls_head.weight
                if (not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.norm())<=0 or st['calls']!=1 or st['seen']!=256
                    or not st['accepted'] or st['student_views']!=1536 or st['EMA_updates'] or st['U_truth_read'] or m['T1_L_packets']!=768
                    or st['U_gradient_guard']['calls']!=1 or st['U_joint_contrast']['calls']!=1 or st['U_receiver_extrapolation']['calls']!=1
                    or min(obj.smoke_rxe_gradients.values())<=0 or obj.smoke_joint_signature_gradient<=0
                    or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters())):
                    raise ValueError('Actual matched REPL loss/gradient/source exposures differ')
                checks.append(dict(row_id=row['row_id'],actual_seed=a.seed,**shape,U_seen=st['seen'],student_U_views=st['student_views'],L_packets=m['T1_L_packets'],finite_joint_backward=True))
                del net,head,obj,loss,ce,correction,out,x,y,d,loaders,payload,holder
                torch.cuda.empty_cache()
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,actual_checkpoint=True,optimizer_updates=0,target_reads=0,U_truth_reads=0,warm_diagnostic_only=True))

def resources():
    """Exclude every card with a different/unknown user; reserve same-user pending PIDs."""
    from experiments.pamf_repl_completion.runner import processes
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,index,memory.free','--format=csv,noheader,nounits'],text=True)
    devices={};mapping={}
    for line in raw.splitlines():
        uuid,index,free=[v.strip() for v in line.split(',')];gpu=int(index);mapping[uuid]=gpu;devices[gpu]=dict(pids=set(),free=int(free),blocked=False)
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        uuid,pid=[v.strip() for v in line.split(',')];gpu=mapping[uuid];pid=int(pid);devices[gpu]['pids'].add(pid)
        try:uid=Path('/proc',str(pid)).stat().st_uid
        except FileNotFoundError:uid=None
        if uid!=os.getuid():devices[gpu]['blocked']=True
    for pid,gpu in processes().items():devices[gpu]['pids'].add(pid)
    return devices

def eligible(device,own,slot):
    return not device['blocked'] and own<slot and len(device['pids'])<2 and device['free']>=12000

def analyze(rt,jobs,group=None):
    import csv,statistics
    from experiments.cric_dura_comparison.evaluation import metrics
    import numpy as np
    root=Path(rt['run_root']);rows=[]
    for job in jobs:
        if job['kind']!='score':continue
        c=read(job['config'])
        if group is not None and c['view_comparison']['view_count']!=group:continue
        data=read(Path(c['output_root'])/'scores.json')
        if not data['truth_last'] or not data['target_feedback_forbidden']:raise ValueError('Truth-last scoring missing')
        pooled={}
        for view in VIEWS:
            parts=[r for r in data['results'] if r['dimension']=='overall' and r['readout']=='raw' and r['view']==view]
            if len(parts)!=2 or sum(p['count'] for p in parts)!=168000:raise ValueError('Both target populations / 168000 required')
            cm=sum((np.asarray(p['confusion'],np.int64) for p in parts));den=cm.sum(0)+cm.sum(1)
            pooled[view]=dict(accuracy=100.*float(cm.trace()/cm.sum()),macro_f1=100.*float(np.divide(2*cm.diagonal(),den,out=np.zeros(6),where=den!=0).mean()))
            for p in parts:rows.append(dict(view_count=c['view_comparison']['view_count'],seed=c['model_seed'],test=p['test'],view=view,accuracy=100*p['accuracy'],macro_f1=100*p['macro_f1']))
            rows.append(dict(view_count=c['view_comparison']['view_count'],seed=c['model_seed'],test='All target dates (168000)',view=view,**pooled[view]))
        rows.append(dict(view_count=c['view_comparison']['view_count'],seed=c['model_seed'],test='All target dates (168000)',view='six_practical_mean',**{m:statistics.mean(pooled[v][m] for v in VIEWS[1:]) for m in ('accuracy','macro_f1')}))
    out=root/('analysis_three_view' if group==3 else 'analysis');out.mkdir(exist_ok=False)
    with (out/'per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Matched PAMF-REPL temporal shuffle comparison','','Direct recognition; fixed final E50; eight paired scratch seeds; historical scores separate.','','|Signature views|Clean accuracy %|Six practical mean %|','|---|---:|---:|']
    summary={}
    for views in ((3,) if group==3 else (3,1)):
        summary[views]={}
        for view in ('clean','six_practical_mean'):
            vals=[r['accuracy'] for r in rows if r['view_count']==views and r['test']=='All target dates (168000)' and r['view']==view]
            if len(vals)!=8:raise ValueError('Eight complete seeds required')
            summary[views][view]=dict(mean=statistics.mean(vals),sd=statistics.stdev(vals))
        lines.append(f"|{views}|"+'|'.join(f"{summary[views][v]['mean']:.4f} ± {summary[views][v]['sd']:.4f}" for v in ('clean','six_practical_mean'))+'|')
    pairs=[]
    for seed in (() if group==3 else SEEDS):
        for view in ('clean','six_practical_mean'):
            values={r['view_count']:r['accuracy'] for r in rows if r['seed']==seed and r['test']=='All target dates (168000)' and r['view']==view}
            pairs.append(dict(seed=seed,view=view,single_minus_three_pp=values[1]-values[3]))
    write(out/'summary.json',dict(summary=summary,paired_differences=pairs,all_seeds=True,truth_last=True,target_feedback=False))
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(out/'complete.json',dict(status='ANALYZED',models=8 if group==3 else 16,new_F=8 if group==3 else 16,new_G=0))

def worker(c,kind):
    if kind=='score':
        from .evaluation import score
        return score(c)
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from .source import train
        return train(c)
    from .evaluation import predict
    return predict(c)

def dispatch(rt):
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir();jobs=prepare(rt);by={j['id']:j for j in jobs};active={};failed=[]
    state=dict(status='SMOKE_PENDING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,code_commit=rt['code_commit'],jobs={j['id']:dict(status='QUEUED') for j in jobs})
    def launch(j,gpu):
        path=root/'logs'/(j['id']+'.log');cmd=[rt['python'],'-u','-m','experiments.pamf_repl_view_shuffle.runner','worker','--kind',j['kind'],'--config',j['config']]
        env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with path.open('x') as log:p=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return p,dict(status='RUNNING',pid=p.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(path),started_at=time.time())
    atomic(root/'pipeline_state.json',state)
    while True:
        res=resources();gpu=next((g for g in rt['allowed_gpus'] if eligible(res[g],0,1)),None)
        if gpu is not None:break
        time.sleep(15)
    p,item=launch(dict(id='smoke',kind='smoke',config=rt['runtime_file']),gpu);state['smoke']=item;atomic(root/'pipeline_state.json',state)
    status=p.wait();state['smoke'].update(status='PASS' if status==0 else 'FAILED',exit_code=status)
    if status or read(root/'smoke.json')['status']!='PASS':state['status']='FAILED_NO_QUERY_SMOKE';atomic(root/'pipeline_state.json',state);return
    while True:
        for key,(p,item) in list(active.items()):
            if p.poll() is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time())
            if not ok:failed.append(key)
            del active[key]
        tri_scores=[j for j in jobs if j['kind']=='score' and read(j['config'])['view_comparison']['view_count']==3]
        if not (root/'analysis_three_view/complete.json').exists() and all(done(j) for j in tri_scores):
            analyze(rt,jobs,group=3);state['three_view_group']='ANALYZED'
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED'];candidates=[j for j in pending if ready(j)] if not failed else []
        scorer=next((j for j in candidates if j['kind']=='score'),None)
        if scorer and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({p.pid for p,item in active.values()}):
            p,item=launch(scorer,None);active[scorer['id']]=(p,item);state['jobs'][scorer['id']]=item;candidates.remove(scorer)
        res=resources()
        for slot in (1,2):
            for gpu in rt['allowed_gpus']:
                own=sum(item['gpu']==gpu for p,item in active.values())
                if not eligible(res[gpu],own,slot):continue
                available=[j for j in candidates if j['kind']!='score']
                if not available:continue
                job=next((j for j in available if j['kind']=='predict'),available[0])
                p,item=launch(job,gpu);active[job['id']]=(p,item);state['jobs'][job['id']]=item;candidates.remove(job);res[gpu]['pids'].add(p.pid);res[gpu]['free']-=4096
        state.update(status='RUNNING',updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),failed_jobs=failed,
            GPU_workers=sum(item['gpu'] is not None for p,item in active.values()),blocked_foreign_user_gpus=[g for g,d in res.items() if d['blocked']])
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or failed):break
        time.sleep(15)
    if failed:state['status']='PARTIAL_TECHNICAL_FAILURE'
    else:analyze(rt,jobs);state['status']='ANALYZED'
    atomic(root/'pipeline_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
