"""Single-owner Group2 scheduling and prediction-before-truth barrier."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import read, write, require
from .group2_worker import require_all_rows


def validate_runtime(rt):
    profiles={'protonet_cda','mrior_sda','dadda_sda','twostage',
              'feature_separation_hu2024','poster_ft','radionet_ada_knn'}
    require(len(rt['profiles'])==len(profiles) and set(rt['profiles'])==profiles, 'Incomplete B methods')
    require(len(rt['methods'])==3 and set(rt['methods'])=={'mopc_hr','orthogonal','lora'}, 'Incomplete C methods')
    require(rt['model_seeds']==[2026092701,2026092702,2026092703,2026092704], 'Incomplete source seeds')
    require(rt['K']==[1,5,10,20] and rt['N']==[2,5,10,20], 'Incomplete K/N')
    receivers=rt['target_receivers']
    require(len(receivers)==len(set(receivers))==7, 'Incomplete RX')
    episodes=[f'MIX{i:02d}' for i in range(1,9)]+[f'{scene}{i:02d}' for scene in ('LS','MU','LU') for i in range(1,6)]
    expected={f'capsules/{pool}/{episode}/{receiver}' for pool in ('P01','P02','P03')
              for episode in episodes for receiver in receivers}
    require(len(rt['capsules'])==len(expected)==483 and set(rt['capsules'])==expected, 'Incomplete episode matrix')
    caps={str(i):2 for i in range(8)}
    require(bool(rt['gpu_limits']) and set(rt['gpu_limits']) <= set(caps) and
            all(0 < value <= caps[gpu] for gpu,value in rt['gpu_limits'].items()), 'GPU configuration invalid')
    require(1 <= rt['score_processes'] <= 4 and rt['minimum_free_MiB'] >= 6000, 'Resource configuration invalid')


def tasks(rt):
    profiles, seeds, capsules = rt['profiles'], rt['model_seeds'], rt['capsules']
    require(all(len(values) == len(set(values)) for values in (profiles, seeds, capsules)), 'Duplicate matrix axis')
    require(profiles and seeds and capsules and set(rt['methods']) == {'mopc_hr', 'orthogonal', 'lora'}, 'Invalid comparison matrix')
    result = []
    for profile in profiles:
        for seed in seeds:
            for index, capsule in enumerate(capsules):
                require(capsule.startswith('capsules/') and '..' not in Path(capsule).parts, 'Invalid capsule path')
                points = [(k,n) for k in rt['K'] for n in rt['N']] if '/MIX' in capsule else [(5,20)]
                key = f'{profile}-s{seed}-c{index:04d}'
                result.append(dict(key=key, profile=profile, seed=seed, capsule=capsule,
                    methods=rt['methods'], points=points, output=str(Path(rt['run_root'])/'predictions'/key)))
    return result


def closed_prediction(job, expected):
    output = Path(job['output'])
    done, index = read(output/'complete.json'), read(output/'predictions.json')
    require(done['status']=='PREDICTIONS_COMPLETE' and done['truth_read'] is False and done['query_fit'] is False, 'Prediction not frozen')
    require(done['rows']==expected==len(index['rows']), 'Incomplete prediction row count')
    require(index['capsule']==job['capsule'] and index['model_seed']==job['seed'] and index['profile']==job['profile'], 'Wrong job binding')
    require(done['capsule_id']==index['capsule_id'], 'Wrong capsule binding')
    require_all_rows(index['rows'], job['points'], job['profile'], job['methods'])
    require(all((output/r['prediction']).is_file() for r in index['rows']), 'Missing prediction artifact')


def _gpu_free(gpu):
    value = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    return int(value.strip())


def _smoke_gpu(rt):
    return next((gpu for gpu in rt['gpu_limits'] if _gpu_free(gpu) >= int(rt['minimum_free_MiB'])), None)


def _queue(rt, jobs, action):
    root = Path(rt['run_root']); pending=list(enumerate(jobs)); active=[]; finished=0; failed=[]
    log_dir=root/'logs'; launch_dir=root/'launches'
    while pending or active:
        for record in list(active):
            idx, gpu, proc, log = record
            code=proc.poll()
            if code is None: continue
            log.close(); active.remove(record)
            if code:
                failed.append(dict(index=idx, key=jobs[idx]['key'], exit_code=code))
            else:
                try:
                    if action=='worker': closed_prediction(jobs[idx], len(jobs[idx]['points'])*len(jobs[idx]['methods']))
                    else:
                        score = root/'scores'/jobs[idx]['key']
                        require(read(score/'complete.json')['status']=='SCORED_COMPLETE', 'Scorer did not close')
                        if '/MIX' in jobs[idx]['capsule']: require((score/'common_query_metrics.json').is_file(), 'Common query score missing')
                    finished += 1
                except Exception as exc:
                    failed.append(dict(index=idx, key=jobs[idx]['key'], error=str(exc)))
        write(root/f'{action}_state.json', dict(status='FAILED' if failed else 'RUNNING', completed=finished,
              total=len(jobs), active=[dict(index=i,gpu=g,pid=p.pid) for i,g,p,_ in active],
              pending=len(pending), failed=failed), exclusive=False)
        if failed: pending.clear()
        if action=='worker':
            slots=[(gpu, limit) for gpu,limit in rt['gpu_limits'].items()]
        else:
            slots=[(None, int(rt['score_processes']))]
        for gpu, limit in slots:
            if not pending or failed: break
            if sum(g==gpu for _,g,_,_ in active) >= int(limit): continue
            if gpu is not None and _gpu_free(gpu) < int(rt['minimum_free_MiB']): continue
            idx,_=pending.pop(0)
            cmd=[sys.executable,'-u','-m',__package__+'.group2_schedule',action,'--runtime',rt['runtime_file'],'--index',str(idx)]
            log=(log_dir/f'{action}-{idx:05d}.log').open('x', encoding='utf-8')
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
            proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active.append((idx,gpu,proc,log))
            write(launch_dir/f'{action}-{idx:05d}.json',dict(pid=proc.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        if pending or active: time.sleep(rt['poll_seconds'])
    require(not failed, 'Systemic technical failure; retained output: '+str(failed))
    write(root/f'{action}_state.json',dict(status='COMPLETE',completed=finished,total=len(jobs),active=[],pending=0,failed=[]),exclusive=False)


def _score(rt, job):
    from .episodic_score import run
    from .milkr_r42_suite import common_query_score
    root=Path(rt['run_root'])
    require(read(root/'predictions_complete.json')['status']=='ALL_PREDICTIONS_FIXED', 'Global truth-last barrier not closed')
    output=root/'scores'/job['key']
    run(rt,dict(prediction=job['output'],output=str(output)))
    if '/MIX' in job['capsule']:
        common_query_score(rt,job,Path(job['output']),read(Path(job['output'])/'predictions.json'),output)


def smoke(rt):
    """Real source checkpoints and source L only; no target/query handles used."""
    import numpy as np
    import torch
    from .io import activate
    from .group1 import row, source_scope
    from .group1_source import load_source, source_data, embedding
    from .group2_cross_fit import prepare_b
    from .external_native import _Scores

    activate(rt); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device='cuda:0'; source_rt=source_scope(rt); source_rt['run_root']=rt['source_run_root']
    labeled=source_data(source_rt,'L_s')
    labels=np.asarray(labeled.metadata['y'])
    indices=[int(np.flatnonzero(labels==i)[0]) for i in range(6)]
    x=labeled.x[indices].to(device); y=torch.arange(6,device=device,dtype=torch.long)
    results=[]
    for profile in rt['profiles']:
        bundle=load_source(source_rt,profile,rt['model_seeds'][0],device)
        model=bundle['model'].eval(); extractor=lambda m,v: embedding(m,row(profile)['method'],v)
        with torch.no_grad():
            z=extractor(model,x)
            proto=z.detach().clone() if profile=='protonet_cda' else None
            scores=_Scores('radionet_ada',decision='knn',support_features=z.detach().clone(),
                           support_labels=y.detach().clone(),knn_k=1,count=6) if profile=='radionet_ada_knn' else None
            prepared=prepare_b(model,profile,extractor,x,y,old_count=6,old_prototypes=proto,b_scores=scores)
            require(set(prepared)=={'lora','mopc_hr','orthogonal'},'Unbound B/C bridge: '+profile)
            if profile=='protonet_cda': expected=-torch.cdist(z,proto).square()
            elif profile=='radionet_ada_knn': expected=scores(model,x)
            else: expected=bundle['score_fn'](model,x)
            actual=prepared['mopc_hr'].model(x)['tx_logits']
            require(torch.allclose(actual,expected,rtol=1e-4,atol=1e-4),'B logits changed: '+profile)
            results.append(dict(profile=profile,source_checkpoint=bundle['source_checkpoint'],
                                max_abs_diff=float((actual-expected).abs().max().item())))
        del prepared,bundle,model,z,proto,scores
    torch.cuda.synchronize()
    return dict(status='PASS',profiles=results,source_role='L_s',target_rows=0,query_rows=0,truth_read=False)


def dispatch(rt):
    validate_runtime(rt)
    root=Path(rt['run_root']); root.mkdir(parents=True,exist_ok=False)
    (root/'logs').mkdir(); (root/'launches').mkdir(); (root/'jobs').mkdir()
    jobs=tasks(rt)
    for index,job in enumerate(jobs): write(root/'jobs'/f'{index:05d}.json',job)
    write(root/'owner.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),argv=sys.argv))
    # A real-checkpoint source-only smoke is the first computational step and
    # continues immediately on PASS. Its record is not an authorization token.
    gpu = _smoke_gpu(rt)
    while gpu is None:
        time.sleep(rt['poll_seconds']); gpu = _smoke_gpu(rt)
    cmd=[sys.executable,'-u','-m',__package__+'.group2_schedule','smoke','--runtime',rt['runtime_file']]
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (root/'logs'/'smoke.log').open('x',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(root/'launches'/'smoke.json',dict(pid=proc.pid,gpu=gpu,cwd=str(Path.cwd()),argv=cmd))
        require(proc.wait()==0, 'Real-source no-query smoke failed; see smoke log')
    _queue(rt,jobs,'worker')
    for job in jobs: closed_prediction(job,len(job['points'])*len(job['methods']))
    rows=sum(len(job['points'])*len(job['methods']) for job in jobs)
    write(root/'predictions_complete.json',dict(status='ALL_PREDICTIONS_FIXED',tasks=len(jobs),rows=rows,truth_read=False,query_fit=False))
    _queue(rt,jobs,'score')
    write(root/'complete.json',dict(status='SCORED_COMPLETE',tasks=len(jobs),rows=rows,truth_last=True,target_feedback=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['dispatch','smoke','worker','score'])
    parser.add_argument('--runtime',required=True); parser.add_argument('--index',type=int)
    args=parser.parse_args(); runtime=read(args.runtime)
    if args.action=='dispatch': dispatch(runtime)
    elif args.action=='smoke': write(Path(runtime['run_root'])/'source_smoke.json',smoke(runtime))
    else:
        require(args.index is not None and args.index >= 0, 'Invalid task index')
        job=read(Path(runtime['run_root'])/'jobs'/f'{args.index:05d}.json')
        if args.action=='worker':
            from .group2_worker import predict
            predict(runtime,job)
        else: _score(runtime,job)
