"""Frozen R42 Group1 completion: all-MIX KN, scene diagnostics, exact reuse."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read,write,require
from .r42_suite_worker import desired_rows,row_key


def tasks(rt):
    require(rt['kernel_method']=='adaptive_residual_ensemble' and rt['arm']=='R42','Only frozen R42')
    require(rt['K']==[1,5,10,20] and rt['N']==[2,5,10,20],'Required full KN matrix')
    require(rt['model_seeds']==[2026092701,2026092702,2026092703,2026092704],'Original four F seeds')
    rx=rt['target_receivers'];require(len(rx)==len(set(rx))==7,'Seven original RX')
    episodes=[f'MIX{i:02d}' for i in range(1,9)]+[f'{p}{i:02d}' for p in ('LS','MU','LU') for i in range(1,6)]
    expected={f'capsules/{p}/{e}/{r}' for p in ('P01','P02','P03') for e in episodes for r in rx}
    require(len(rt['capsules'])==483 and set(rt['capsules'])==expected,'Incomplete Group1 capsules')
    return [dict(key=f's{seed}-c{i:04d}',seed=seed,capsule=cap,
                 points=[[k,n] for k in rt['K'] for n in rt['N']] if '/MIX' in cap else [[5,20]])
            for seed in rt['model_seeds'] for i,cap in enumerate(rt['capsules'])]


def task_runtime(rt,task):
    return dict(rt,run_root=str(Path(rt['run_root'])/'tasks'/task['key']),model_seeds=[task['seed']],capsules=[task['capsule']])


def source_root(rt,seed):return Path(rt['run_root'])/'sources'/f's{seed}'


def source(rt,seed):
    from .features import source_smoke
    from .milkr_adaptive_source import freeze
    require(seed in rt['model_seeds'],'Unexpected F seed')
    parent=rt['source_reuse_by_seed'][str(seed)]
    local=dict(rt,run_root=str(source_root(rt,seed)),model_seeds=[seed],**parent)
    Path(local['run_root']).mkdir(parents=True,exist_ok=False)
    context=source_smoke(local,Path(local['run_root'])/'source_smoke.json')
    freeze(local,context)
    selection=read(Path(local['run_root'])/'source_selection.json')
    require(selection['source_new_optimizer_updates']==0,'Source NN must be reused without fitting')
    write(Path(local['run_root'])/'suite_source_complete.json',dict(status='R42_SOURCE_READY',model_seed=seed,
          source_new_optimizer_updates=0,target_reads=0,query_reads=0,F_updates=0,G_updates=0))


def normal_key(row):
    arm='R42' if row['arm']=='R42' else 'LocalRidge' if row['arm'] in ('local_ridge','default_head','LocalRidge','pamf_tri_sig_time_ssl_milkr') else None
    stage='A0' if row['stage'] in ('A','A0') else row['stage']
    return (arm,stage,row['k'],row['new_count'],row['input_path'])


def reuse_entries(rt,task):
    wanted={row_key(r) for r in desired_rows(task['points'])};found={}
    # Preserve old scored references even when a recovered suite prediction has
    # a different filename. Selection is fixed by identity, never metric values.
    candidates=[]
    if task['seed']==2026092701 and task['capsule']=='capsules/P01/MIX01/1-1':
        candidates.append(dict(prediction=rt['pilot_run_root']+'/predictions/1-1',metrics=rt['pilot_run_root']+'/scores/1-1/metrics.json'))
    if '/MIX' in task['capsule']:
        caps=rt['main_capsules'];i=caps.index(task['capsule']);base=rt['main_run_root']+f'/tasks/s{task["seed"]}-c{i:04d}'
        rx=task['capsule'].split('/')[-1]
        candidates.append(dict(prediction=base+'/predictions/'+rx,metrics=base+'/scores/'+rx+'/metrics.json'))
    refs=read(rt['baseline_refs_file'])
    candidates.extend(r for r in refs if r['seed']==task['seed'] and r['capsule']==task['capsule'])
    for ref in candidates:
        directory=Path(ref['prediction']);index=read(directory/'predictions.json')
        require(index['capsule']==task['capsule'] and index['model_seed']==task['seed'],'Wrong reuse candidate')
        require(Path(ref['metrics']).is_file(),'Completed metrics missing; reconcile reuse source')
        for row in index['rows']:
            key=normal_key(row)
            if key in wanted and key not in found:
                found[key]=dict(key=list(key),prediction_dir=str(directory),row=row,metrics_file=ref['metrics'],
                                metric_prediction=row['prediction'],metric_key=list(key))
    if rt.get('prediction_reuse_refs_file'):
        refs=read(rt['prediction_reuse_refs_file'])
        matches=[r for r in refs if r['seed']==task['seed'] and r['capsule']==task['capsule']]
        require(len(matches)<=1,'Duplicate completed suite prediction reference')
        for ref in matches:
            directory=Path(ref['prediction']);index=read(directory/'predictions.json');done=read(directory/'complete.json')
            require(index['capsule']==task['capsule'] and index['model_seed']==task['seed'],'Wrong recovered prediction identity')
            require(done['status']=='PREDICTIONS_COMPLETE' and done['truth_read'] is False and done['query_fit'] is False,'Recovered prediction not frozen')
            require(done['capsule_id']==index['capsule_id'] and done['rows']==len(index['rows']),'Recovered prediction completion differs')
            seen=set()
            for row in index['rows']:
                key=normal_key(row);require(key in wanted and key not in seen,'Unexpected or duplicate recovered row');seen.add(key)
                previous=found.get(key);metric_file=previous['metrics_file'] if previous else ref.get('metrics')
                entry=dict(key=list(key),prediction_dir=str(directory),row=row,metrics_file=metric_file)
                if previous:
                    require(previous['row']['split_id']==row['split_id'] and previous['row']['registered_classes']==row['registered_classes'],'Recovered/scored split differs')
                    entry.update(metric_prediction=previous['metric_prediction'],metric_key=previous['metric_key'])
                elif metric_file:
                    require(Path(metric_file).is_file(),'Recovered metric reference missing')
                    entry.update(metric_prediction=row['prediction'],metric_key=list(key))
                found[key]=entry
            require(seen==wanted,'Recovered suite prediction matrix incomplete')
    return list(found.values())


def worker(rt,task):
    from .r42_suite_worker import run
    local=task_runtime(rt,task);root=Path(local['run_root']);root.mkdir(parents=True,exist_ok=False)
    src=source_root(rt,task['seed']);done=read(src/'suite_source_complete.json')
    require(done['model_seed']==task['seed'] and done['source_new_optimizer_updates']==0,'Wrong source seed/state')
    for name in ('source_orbit_cca.npz','source_selection.json'):os.link(src/name,root/name)
    entries=reuse_entries(rt,task);reuse=root/'reuse.json';write(reuse,entries);write(root/'runtime.json',local)
    run(local,dict(task,reuse_file=str(reuse)))


def prediction_closed(rt,task):
    folder=Path(task_runtime(rt,task)['run_root'])/'predictions'/task['capsule'].split('/')[-1]
    done=read(folder/'complete.json');index=read(folder/'predictions.json');want=desired_rows(task['points'])
    require(done['status']=='PREDICTIONS_COMPLETE' and done['rows']==len(want) and done['truth_read'] is False and done['query_fit'] is False,'Incomplete suite prediction')
    require(index['model_seed']==task['seed'] and index['capsule']==task['capsule'],'Wrong suite prediction identity')
    require({row_key(r) for r in index['rows']}=={row_key(r) for r in want},'Wrong suite row matrix')
    require(all((folder/r['prediction']).is_file() for r in index['rows']),'Missing prediction file')


def score(rt,task):
    from .episodic_score import run
    require(read(Path(rt['run_root'])/'predictions_complete.json')['status']=='ALL_PREDICTIONS_FIXED','Global prediction closure required before truth')
    local=task_runtime(rt,task);root=Path(local['run_root']);rx=task['capsule'].split('/')[-1]
    prediction=root/'predictions'/rx;index=read(prediction/'predictions.json');done=read(prediction/'complete.json')
    entries=read(root/'reuse.json');cache={};reused={}
    current={row_key(r):r for r in index['rows']}
    for entry in entries:
        path=entry['metrics_file']
        if path is None:continue  # Fixed predictions without scores need only independent scoring.
        if path not in cache:
            parent=read(Path(path).parent/'complete.json');require(parent['status']=='SCORED_COMPLETE','Reused scores not complete')
            cache[path]=read(path)
        payload=cache[path];require(payload['capsule_id']==index['capsule_id'] and payload['model_seed']==task['seed'],'Reused metric binding mismatch')
        key=tuple(entry['key']);original=entry['row'];metric_prediction=entry.get('metric_prediction',original['prediction'])
        matches=[r for r in payload['rows'] if r['prediction']==metric_prediction]
        require(key in current and key not in reused,'Unexpected or duplicate reused score key')
        require(len(matches)==1 and normal_key(matches[0])==key==tuple(entry.get('metric_key',key)),'Reused score row mismatch')
        require(all(matches[0][k]==current[key][k] for k in ('split_id','registered_classes')),'Reused score split/classes mismatch')
        reused[key]=matches[0]
    missing=[r for r in index['rows'] if row_key(r) not in reused];new={}
    if missing:
        subset=root/'new_predictions_for_scoring';subset.mkdir()
        for row in missing:os.link(prediction/row['prediction'],subset/row['prediction'])
        write(subset/'predictions.json',dict(index,rows=missing));write(subset/'complete.json',dict(done,rows=len(missing)))
        output=root/'new_scores';run(local,dict(prediction=str(subset),output=str(output)))
        new={row_key(r):r for r in read(output/'metrics.json')['rows']}
    result=[]
    for row in index['rows']:
        key=row_key(row);value=dict(reused[key] if key in reused else new[key]);value.update(row);result.append(value)
    output=root/'scores'/rx;output.mkdir(parents=True)
    write(output/'metrics.json',dict(capsule_id=index['capsule_id'],model_seed=task['seed'],rows=result,truth_last=True,target_feedback=False,
          reused_metric_rows=len(reused),new_metric_rows=len(missing)))
    if '/MIX' in task['capsule']:common_query_score(local,task,prediction,index,output)
    write(output/'complete.json',dict(status='SCORED_COMPLETE',rows=len(result),truth_last=True,target_feedback=False))


def common_query_score(rt,task,prediction,index,output):
    """Read truth only in the independent scorer; never feed it to any fit."""
    import numpy as np
    from .episodic_io import query_complement
    from .episodic_score import metrics
    cap=Path(rt['data_root'])/task['capsule'];m=read(cap/'manifest.json')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (cap/'splits').glob('*.json'))}
    truthfile=Path(rt['data_root'])/'score_only'/m['physical_pool']/m['episode_id']/(m['receiver']+'.npz')
    with np.load(truthfile) as z:truth=z['labels']
    # Literal intersection protects against an assumption that supports are nested.
    common={}
    for n in [0]+rt['N']:
        reg=np.load(cap/'stage_rows'/f'N{n:02d}.npy')
        sets=[query_complement(reg,np.asarray(specs[k,n]['support_indices'])) for k in rt['K']]
        shared=sets[0]
        for values in sets[1:]:shared=np.intersect1d(shared,values,assume_unique=True)
        require(len(shared)>0,'Empty common-query intersection');common[n]=shared
    values=[]
    for row in index['rows']:
        with np.load(prediction/row['prediction']) as z:q=z['query_indices'];pred=z['predicted_indices']
        mask=np.isin(q,common[row['new_count']]);require(mask.sum()==len(common[row['new_count']]),'Common query alignment failed')
        values.append(dict(row,metrics=metrics(truth[q[mask]],pred[mask],len(row['registered_classes']))))
    write(output/'common_query_metrics.json',dict(scope='intersection of full complements across K1/5/10/20 at the same N; K20 if nested',
          rows=values,truth_last=True,target_feedback=False,common_query_counts={str(k):len(v) for k,v in common.items()}))


def queue(rt,items,action,per_gpu):
    root=Path(rt['run_root']);pending=list(items);active=[];finished=[];failed=[]
    while pending or active:
        for record in list(active):
            item,gpu,p,log=record;code=p.poll()
            if code is None:continue
            log.close();active.remove(record)
            if code:failed.append(dict(key=item['key'],exit_code=code))
            else:
                if action=='worker':prediction_closed(rt,item)
                else:require(read(source_root(rt,item['seed'])/'suite_source_complete.json')['status']=='R42_SOURCE_READY','Missing source closure')
                finished.append(item['key'])
        write(root/f'{action}_state.json',dict(status='FAILED' if failed else 'RUNNING',completed=len(finished),total=len(items),
              active=[dict(key=i['key'],pid=p.pid,gpu=g) for i,g,p,_ in active],failed=failed,pending=len(pending)),exclusive=False)
        if failed:pending.clear()
        for gpu in rt['allowed_gpus']:
            if not pending or failed:break
            if sum(g==gpu for _,g,_,_ in active)>=per_gpu:continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<rt['minimum_free_MiB']:continue
            item=pending.pop(0);cmd=[sys.executable,'-u','-m',__package__+'.milkr_r42_suite',action,'--runtime',rt['runtime_file'],'--key',item['key']]
            log=(root/'logs'/f'{action}-{item["key"]}.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);active.append((item,gpu,p,log))
            write(root/'launches'/f'{action}-{item["key"]}.json',dict(pid=p.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        if pending or active:time.sleep(rt['poll_seconds'])
    require(not failed,'Technical failure; stop admissions and retain outputs: '+str(failed))
    write(root/f'{action}_state.json',dict(status='COMPLETE',completed=len(finished),total=len(items),active=[],failed=[],pending=0),exclusive=False)


def dispatch(rt):
    matrix=tasks(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();(root/'launches').mkdir();write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    queue(rt,[dict(key=f's{s}',seed=s) for s in rt['model_seeds']],'source',1)
    queue(rt,matrix,'worker',2)
    for task in matrix:prediction_closed(rt,task)
    rows=sum(len(desired_rows(t['points'])) for t in matrix)
    write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(matrix),rows=rows,truth_read=False,query_fit=False))
    for i,task in enumerate(matrix):
        cmd=[sys.executable,'-u','-m',__package__+'.milkr_r42_suite','score','--runtime',rt['runtime_file'],'--key',task['key']]
        with (root/'logs'/f'score-{task["key"]}.log').open('x') as log:subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        done=read(Path(task_runtime(rt,task)['run_root'])/'scores'/task['capsule'].split('/')[-1]/'complete.json')
        require(done['status']=='SCORED_COMPLETE','Score did not close');write(root/'score_state.json',dict(completed=i+1,total=len(matrix)),exclusive=False)
    write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(matrix),rows=rows,truth_last=True,target_feedback=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','source','worker','score']);p.add_argument('--runtime',required=True);p.add_argument('--key');a=p.parse_args();rt=read(a.runtime)
    if a.action=='dispatch':dispatch(rt)
    elif a.action=='source':source(rt,int(a.key[1:]))
    else:
        found=[t for t in tasks(rt) if t['key']==a.key];require(len(found)==1,'Unknown suite task')
        (worker if a.action=='worker' else score)(rt,found[0])
