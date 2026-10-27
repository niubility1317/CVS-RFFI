"""One launch owner; one own GPU worker/card; preserve all unrelated processes."""
import argparse,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from experiments.cric_dura_comparison.io import read,write
from .plan import make_plan

def activate(c):
    from experiments.pamf_repl_completion.runner import activate as native
    native(c)

def worker(c,kind):
    if kind=='score':
        from .evaluation import score
        return score(c)
    activate(c)
    if kind=='smoke':
        from .smoke import smoke
        return smoke(c)
    if kind=='train':
        from .source import train
        return train(c)
    if kind=='predict':
        from .evaluation import predict
        return predict(c)
    raise ValueError('Unknown scientific stage')

def admits(gpu):
    from experiments.pamf_repl_completion.runner import capacity,processes
    cuda,free=capacity(gpu)
    for pid in cuda:
        try:
            if os.stat('/proc/'+str(pid)).st_uid!=os.getuid():return False
        except FileNotFoundError:return False
    pending={p for p,g in processes().items() if g==gpu}
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):mem['available_kib']=int(line.split()[1])
    return len(cuda|pending)<2 and free>=12000 and mem.get('available_kib',0)>=32*1024**2

def launch(rt,job,gpu):
    log=Path(rt['run_root'])/'logs'/(job['id']+'.log')
    cmd=[rt['python'],'-u','-m','experiments.pamf_paper_completion.runner','worker','--kind',job['kind'],'--config',job['config']]
    env=dict(os.environ,PYTHONPATH=rt['release_root'],CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with log.open('x') as f:p=subprocess.Popen(cmd,cwd=rt['release_root'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return p,dict(status='RUNNING',pid=p.pid,gpu=gpu,cwd=rt['release_root'],argv=cmd,log=str(log),started_at=time.time())

def analyze(rt):
    import csv,statistics
    root=Path(rt['run_root']);records=[];groups={};src=read(root/'sources.json')
    for value in src.values():
        row=value['row'];done=read(Path(value['scores'])/'complete.json')
        data=read(Path(value['scores'])/'scores.json')
        if done['status']!='SCORED_COMPLETE' or not data['truth_last'] or not data['target_feedback_forbidden']:raise ValueError('Incomplete independent score')
        for r in data['results']:
            if r['dimension']!='overall':continue
            for metric in ('accuracy','macro_f1'):
                item=dict(data_setting=row['split_name'],family=row['family'],method=row['method'],structure=row['structure'],
                    signature_views=row['signature_views'],permuted=row['permuted'],epochs=row['epochs'],test=r['test'],readout=r['readout'],view=r['view'],metric=metric,
                    model_seed=row['model_seed'],percent=100*float(r[metric]))
                records.append(item)
    # Report S6 per seed before between-seed statistics; never pool seed metrics.
    six={}
    for r in records:
        if r['view']!='clean':
            key=tuple((k,v) for k,v in r.items() if k not in ('view','percent'))
            six.setdefault(key,[]).append(r['percent'])
    for key,values in six.items():
        if len(values)!=6:raise ValueError('Missing declared scene')
        records.append(dict(key,view='six_practical_mean',percent=statistics.mean(values)))
    for r in records:
        key=tuple((k,v) for k,v in r.items() if k not in ('model_seed','percent'))
        groups.setdefault(key,[]).append((r['model_seed'],r['percent']))
    summary=[]
    for key,values in groups.items():
        expected=8 if dict(key)['family']=='architecture' else 3
        if len(values)!=expected or len({s for s,v in values})!=expected:raise ValueError('Missing/duplicate seed in summary')
        vals=[v for s,v in values];summary.append(dict(key,seed_n=expected,mean=statistics.mean(vals),sd=statistics.stdev(vals)))
    out=root/'analysis';out.mkdir(exist_ok=False)
    # Dict key order differs for derived S6, so use the same explicit column set.
    for name,data in (('per_seed.csv',records),('summary.csv',summary)):
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# PAMF paper completion','', '200 logical classifiers: 119 new and 81 reused; no new G.','',
           'CE E200 and SSL E50 remain separate; pure-CE structural history is reused by reference.','',
           '|Setting|Method|Structure|Views|Permuted|Epochs|Test|Readout|Scene|Accuracy %|','|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        if r['metric']=='accuracy' and r['view'] in ('clean','six_practical_mean'):
            lines.append('|'+ '|'.join(str(r[k]) for k in ('data_setting','method','structure','signature_views','permuted','epochs','test','readout','view'))+f"|{r['mean']:.4f} ± {r['sd']:.4f}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'complete.json',dict(status='ANALYZED',logical_F=200,new_F=119,reused_F=81,new_G=0,summary_rows=len(summary)))

def dispatch(rt):
    from .prepare import prepare
    from experiments.practical_full_2x2.cache import atomic
    from experiments.cric_dura_comparison.pipeline import done,ready,block_failed_dependencies
    from experiments.pamf_repl_completion.runner import foreign_CPU_score
    root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False);(root/'logs').mkdir()
    state=dict(status='PREPARING',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,code_commit=rt['code_commit'],owner=rt['launch_owner'],started_at=time.time(),jobs={})
    atomic(root/'pipeline_state.json',state)
    try:jobs=prepare(rt)
    except Exception as error:
        state.update(status='FAILED_PREPARATION',error=repr(error));atomic(root/'pipeline_state.json',state);raise
    by={j['id']:j for j in jobs};state['jobs']={j['id']:dict(status='QUEUED') for j in jobs};active={};failed=[]
    state['status']='SMOKE_WAITING_CAPACITY';atomic(root/'pipeline_state.json',state)
    while True:
        available=[g for g in rt['allowed_gpus'] if admits(g)]
        if available:break
        time.sleep(15)
    p,item=launch(rt,dict(id='smoke',kind='smoke',config=rt['runtime_file']),available[0]);state['smoke']=item;state['status']='SMOKE_RUNNING';atomic(root/'pipeline_state.json',state)
    status=p.wait();ok=(root/'smoke.json').exists() and read(root/'smoke.json')['status']=='PASS'
    state['smoke'].update(exit_code=status,status='PASS' if ok else 'FAILED')
    if not ok:
        state['status']='FAILED_NO_QUERY_SMOKE';atomic(root/'pipeline_state.json',state);return
    state['status']='RUNNING';turn=train_turn=0
    while True:
        for key,(p,item) in list(active.items()):
            if p.poll() is None:continue
            ok=done(by[key]);state['jobs'][key]=dict(item,status='COMPLETE' if ok else 'FAILED',ended_at=time.time(),exit_code=p.returncode)
            if not ok:failed.append(key)
            del active[key]
        block_failed_dependencies(jobs,state,[])
        pending=[j for j in jobs if state['jobs'][j['id']]['status']=='QUEUED']
        candidates=[j for j in pending if ready(j)] if not failed else []
        scorer=next((j for j in candidates if j['kind']=='score'),None)
        if scorer and not any(by[k]['kind']=='score' for k in active) and not foreign_CPU_score({p.pid for p,item in active.values()}):
            p,item=launch(rt,scorer,None);active[scorer['id']]=(p,item);state['jobs'][scorer['id']]=item;candidates.remove(scorer)
        for gpu in rt['allowed_gpus']:
            if any(item['gpu']==gpu for p,item in active.values()) or not admits(gpu):continue
            available=[j for j in candidates if j['kind']!='score']
            if not available:continue
            wanted=('train','train','predict')[turn%3];turn+=1
            pool=[j for j in available if j['kind']==wanted] or available
            if pool[0]['kind']=='train':
                desired=('ssl','architecture','structure','controls')[train_turn%4];train_turn+=1
                def matches(j):
                    return j['family']=='architecture' if desired=='architecture' else j['family']=='structure' if desired=='structure' else j['learning']=='ssl' if desired=='ssl' else j['learning']!='ssl' and j['family']!='architecture'
                pool=[j for j in pool if matches(j)] or pool
            job=pool[0];p,item=launch(rt,job,gpu);active[job['id']]=(p,item);state['jobs'][job['id']]=item;candidates.remove(job)
        state.update(updated_at=time.time(),counts=dict(Counter(v['status'] for v in state['jobs'].values())),failed_jobs=failed,stop_new_dispatch=bool(failed),GPU_workers=sum(i['gpu'] is not None for p,i in active.values()))
        atomic(root/'pipeline_state.json',state)
        if not active and (not pending or failed):break
        time.sleep(15)
    if failed or any(v['status']!='COMPLETE' for v in state['jobs'].values()):state['status']='PARTIAL_TECHNICAL_FAILURE'
    else:
        state['status']='ANALYZING';atomic(root/'pipeline_state.json',state);analyze(rt);state['status']='ANALYZED'
    atomic(root/'pipeline_state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker']);p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
