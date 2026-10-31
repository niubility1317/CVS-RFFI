"""Frozen R42 on the existing 672-unit main comparison; no target selection."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .io import read, write, require


def tasks(rt):
    require(rt['kernel_method']=='adaptive_residual_ensemble' and rt['arm']=='R42','Only frozen R42')
    require(rt['K']==[5] and rt['N']==[20],'Main work point must be K5/N20')
    require(rt['model_seeds']==[2026092701,2026092702,2026092703,2026092704],'Four original seeds required')
    caps=rt['capsules']
    expected={f'capsules/{p}/MIX{i:02d}/{rx}' for p in ('P01','P02','P03') for i in range(1,9) for rx in rt['target_receivers']}
    require(len(rt['target_receivers'])==7 and len(set(rt['target_receivers']))==7,'Seven RX required')
    require(len(caps)==168 and set(caps)==expected,'Incomplete or duplicate main capsules')
    return [dict(key=f's{seed}-c{i:04d}',seed=seed,capsule=cap) for seed in rt['model_seeds'] for i,cap in enumerate(caps)]


def task_runtime(rt,task):
    return dict(rt,run_root=str(Path(rt['run_root'])/'tasks'/task['key']),model_seeds=[task['seed']],capsules=[task['capsule']])


def source_root(rt,seed):
    return Path(rt['run_root'])/'sources'/f's{seed}'


def source(rt,seed):
    from .features import source_smoke
    from .milkr.adaptive_residual import STRENGTHS
    require(seed in rt['model_seeds'],'Unexpected source model seed')
    root=source_root(rt,seed);root.mkdir(parents=True,exist_ok=False)
    local=dict(rt,run_root=str(root),model_seeds=[seed])
    if seed==2026092701:
        from .milkr_adaptive_source import freeze
        context=source_smoke(local,root/'source_smoke.json')
        freeze(local,context)
    else:
        # Same R41 source recipe, applied to each matching frozen F seed.
        # Source K1 selection is recorded but is unused at this K5-only work point.
        from .milkr_orbit_source import calibrate
        local['kernel_method']='crossrx_residual_ensemble'
        source_smoke(local,root/'source_smoke.json')
        calibrate(local)
        selected=read(root/'source_selection.json')
        require(selected['status']=='SOURCE_FROZEN' and selected['target_eligible'],'Source encoder not frozen')
        require(len(selected['source_metric_training'])==6,'Missing source NN provenance')
        for row in selected['source_metric_training']:
            for orbit in ('C4','C8'):
                require(row[orbit]['steps']==600,'Wrong source training budget')
                require(all(row[orbit][k]==0 for k in ('target_reads','query_reads','U_truth_reads','F_updates','G_updates')),'Illegal source training input')
        selected['K1_rule'].update(support_residual_strengths=list(STRENGTHS),K1_residual_strength=1.)
        selected.update(source_checkpoint_reused=False,source_stage_method='crossrx_residual_ensemble',
                        feature_method='adaptive_residual_ensemble',source_new_optimizer_updates=7200,
                        source_F_seed=seed,K1_rule_used_in_main=False)
        write(root/'source_selection.json',selected,exclusive=False)
    write(root/'main_source_complete.json',dict(status='R42_SOURCE_READY',model_seed=seed,
          target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0))


def reuse_pilot(rt,task,jobrt):
    if task['seed']!=2026092701 or task['capsule']!='capsules/P01/MIX01/1-1':return False
    old=Path(rt['pilot_run_root'])/'predictions'/'1-1';index=read(old/'predictions.json');done=read(old/'complete.json')
    cap=read(Path(rt['data_root'])/task['capsule']/'manifest.json')
    require(index['capsule']==task['capsule'] and index['capsule_id']==cap['capsule_id'] and index['model_seed']==task['seed'],'Pilot identity mismatch')
    require(done['status']=='PREDICTIONS_COMPLETE' and not done['truth_read'] and not done['query_fit'],'Pilot not fixed')
    rows=[r for r in index['rows'] if r['k']==5 and r['new_count'] in (0,20) and r['arm']=='R42']
    require(len(rows)==12 and {(r['stage'],r['input_path']) for r in rows}=={(stage,f'S{s}Q{q}') for stage in ('B','C') for s in (0,1) for q in (0,1,2)},'Pilot main rows incomplete')
    specs={(s['k'],s['new_count']):s for s in (read(p) for p in (Path(rt['data_root'])/task['capsule']/'splits').glob('*.json'))}
    for r in rows:
        spec=specs[r['k'],r['new_count']]
        require(r['split_id']==spec['split_id'] and r['registered_classes']==spec['registered_classes'],'Pilot split differs')
    root=Path(jobrt['run_root'])/'predictions'/'1-1';root.mkdir(parents=True,exist_ok=False)
    for r in rows:
        os.link(old/r['prediction'],root/r['prediction'])
    for prefix in ('B-K5-N0-','C-K5-N20-'):
        for p in old.glob(prefix+'*.json'):os.link(p,root/p.name)
    write(root/'predictions.json',dict(index,rows=rows))
    write(root/'complete.json',dict(done,rows=12,seconds=0.,reused_from=str(old),reused_existing_predictions=True))
    return True


def worker(rt,task):
    from .milkr_prior_fusion import worker as frozen_worker
    local=task_runtime(rt,task);root=Path(local['run_root']);root.mkdir(parents=True,exist_ok=False)
    src=source_root(rt,task['seed'])
    require(read(src/'main_source_complete.json')['model_seed']==task['seed'],'Wrong source seed')
    for name in ('source_orbit_cca.npz','source_selection.json'):os.link(src/name,root/name)
    write(root/'runtime.json',local)
    if not reuse_pilot(rt,task,local):frozen_worker(local,task['capsule'])


def prediction_closed(rt,task):
    root=Path(task_runtime(rt,task)['run_root'])/'predictions'/task['capsule'].split('/')[-1]
    done=read(root/'complete.json');index=read(root/'predictions.json')
    require(done['status']=='PREDICTIONS_COMPLETE' and done['rows']==12 and not done['truth_read'] and not done['query_fit'],'Incomplete main prediction')
    require(index['model_seed']==task['seed'] and index['capsule']==task['capsule'],'Wrong main task identity')
    require(len(index['rows'])==12 and all((root/r['prediction']).is_file() for r in index['rows']),'Missing prediction files')


def score(rt,task):
    from .episodic_score import run
    local=task_runtime(rt,task);root=Path(local['run_root']);rx=task['capsule'].split('/')[-1]
    run(local,dict(prediction=str(root/'predictions'/rx),output=str(root/'scores'/rx)))


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
                else:require(read(source_root(rt,item['seed'])/'main_source_complete.json')['status']=='R42_SOURCE_READY','Source closure missing')
                finished.append(item['key'])
        write(root/f'{action}_state.json',dict(status='FAILED' if failed else 'RUNNING',completed=len(finished),total=len(items),
              active=[dict(key=i['key'],pid=p.pid,gpu=g) for i,g,p,_ in active],failed=failed,pending=len(pending)),exclusive=False)
        if failed:pending.clear()  # Drain already running own work; do not admit new tasks or retry.
        for gpu in rt['allowed_gpus']:
            if not pending or failed:break
            if sum(g==gpu for _,g,_,_ in active)>=per_gpu:continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<rt['minimum_free_MiB']:continue
            item=pending.pop(0)
            cmd=[sys.executable,'-u','-m',__package__+'.milkr_r42_main',action,'--runtime',rt['runtime_file'],'--key',item['key']]
            log=(root/'logs'/f'{action}-{item["key"]}.log').open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active.append((item,gpu,p,log))
            write(root/'launches'/f'{action}-{item["key"]}.json',dict(pid=p.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        if pending or active:time.sleep(rt['poll_seconds'])
    require(not failed,'Main queue technical failure; no blind retry: '+str(failed))
    write(root/f'{action}_state.json',dict(status='COMPLETE',completed=len(finished),total=len(items),active=[],failed=[],pending=0),exclusive=False)


def dispatch(rt):
    matrix=tasks(rt);root=Path(rt['run_root']);root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir();(root/'launches').mkdir()
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    queue(rt,[dict(key=f's{seed}',seed=seed) for seed in rt['model_seeds']],'source',1)
    queue(rt,matrix,'worker',2)
    for task in matrix:prediction_closed(rt,task)
    write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(matrix),rows=12*len(matrix),truth_read=False,query_fit=False))
    # Truth connection is isolated to subprocesses, after the entire matrix is fixed.
    for i,task in enumerate(matrix):
        cmd=[sys.executable,'-u','-m',__package__+'.milkr_r42_main','score','--runtime',rt['runtime_file'],'--key',task['key']]
        with (root/'logs'/f'score-{task["key"]}.log').open('x') as log:subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        rx=task['capsule'].split('/')[-1];require(read(Path(task_runtime(rt,task)['run_root'])/'scores'/rx/'complete.json')['status']=='SCORED_COMPLETE','Score did not close')
        write(root/'score_state.json',dict(completed=i+1,total=len(matrix)),exclusive=False)
    write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(matrix),rows=12*len(matrix),truth_last=True,target_feedback=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','source','worker','score']);p.add_argument('--runtime',required=True);p.add_argument('--key');a=p.parse_args();rt=read(a.runtime)
    if a.action=='dispatch':dispatch(rt)
    elif a.action=='source':source(rt,int(a.key[1:]))
    else:
        matches=[t for t in tasks(rt) if t['key']==a.key];require(len(matches)==1,'Unknown task key')
        (worker if a.action=='worker' else score)(rt,matches[0])
