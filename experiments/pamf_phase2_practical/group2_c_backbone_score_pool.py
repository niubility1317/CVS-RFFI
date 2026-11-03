"""Persistent independent CPU scorers after the global prediction barrier."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
from pathlib import Path

from .io import read, require, write


_RUNTIME = None


def _init_scorer(rt):
    global _RUNTIME
    _RUNTIME = rt
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'


def _score_one(job):
    from .group2_c_backbone_schedule import _score
    require(_RUNTIME is not None, 'Scorer runtime not initialized')
    _score(_RUNTIME,job)
    output=Path(_RUNTIME['run_root'])/'scores'/job['key']
    done=read(output/'complete.json')
    require(done['status']=='SCORED_COMPLETE' and done['rows']==len(job.get('points', [None])) and
            done['truth_last'] is True, 'Incomplete score')
    if '/MIX' in job['capsule']:
        require((output/'common_query_metrics.json').is_file(),'Missing common query score')
    return dict(pid=os.getpid(),key=job['key'])


def score_pool(rt, jobs):
    """Score each frozen job once; stop admission on failure, drain in-flight jobs."""
    root=Path(rt['run_root'])
    require(read(root/'predictions_complete.json')['status']=='ALL_PREDICTIONS_FIXED','Truth barrier not closed')
    limit=int(rt['score_processes'])
    require(1 <= limit <= 4,'Invalid scorer concurrency')
    require(len({job['key'] for job in jobs})==len(jobs),'Duplicate scoring job')
    pending=iter(enumerate(jobs)); submitted=0; completed=0; failed=[]; active={}; workers=set()
    def save(status):
        write(root/'score_state.json',dict(status=status,completed=completed,total=len(jobs),
            active=[dict(index=i,key=job['key']) for i,job in active.values()],
            pending=len(jobs)-submitted,failed=failed,worker_pids=sorted(workers),
            implementation='persistent_cpu_workers'),exclusive=False)
    try:
        executor = ProcessPoolExecutor(max_workers=limit,mp_context=multiprocessing.get_context('spawn'),
                                       initializer=_init_scorer,initargs=(rt,))
    except Exception as exc:
        failed.append(dict(index=None,key=None,phase='executor',error=f'{type(exc).__name__}: {exc}'))
        save('FAILED')
        raise ValueError('Independent scoring failure; unable to start scorer pool') from exc
    with executor:
        def admit():
            nonlocal submitted
            while len(active)<limit and submitted<len(jobs) and not failed:
                index,job=next(pending)
                try:
                    future=executor.submit(_score_one,job)
                except Exception as exc:
                    failed.append(dict(index=index,key=job['key'],phase='submit',
                                       error=f'{type(exc).__name__}: {exc}'))
                    break
                active[future]=(index,job);submitted+=1
        admit();save('FAILED' if failed else 'RUNNING')
        while active:
            done,_=wait(active,return_when=FIRST_COMPLETED)
            for future in done:
                index,job=active.pop(future)
                try:
                    result=future.result();require(result['key']==job['key'],'Wrong scorer result')
                    workers.add(result['pid']);completed+=1
                except Exception as exc:
                    failed.append(dict(index=index,key=job['key'],error=f'{type(exc).__name__}: {exc}'))
            admit();save('FAILED' if failed else 'RUNNING')
    require(not failed, 'Independent scoring failure; outputs retained: '+str(failed))
    require(completed==len(jobs),'Independent scoring incomplete')
    save('COMPLETE')
