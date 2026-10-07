"""One owner, exclusive outputs, resource-aware training, then truth-last tests."""
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.cvs_phase1_repair import design as d
from experiments.cvs_phase1_stack.source import require_budget


def competing_owners(processes=None):
    """Conservatively wait for existing launchers that do not share a lock.

    Inspect same-user launch owners, not old status files. A training worker
    may coexist after its owner exits; occupancy still accounts for it.
    """
    from experiments.cvs_phase1_stack.capacity16 import proc
    roots = processes if processes is not None else [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()]
    modules = {'experiments.cvs_phase1_stack.'+x for x in
               ('recover','dispatch','capacity16','sixscene_after','evaluation_recovery','completed_eval','completed_eval_20261007')}
    found=[]
    for pid in roots:
        if pid==os.getpid():continue
        try:
            if Path('/proc',str(pid)).stat().st_uid != os.getuid():continue
            info=proc(pid)
            if not info:continue
            argv=info['argv']
            if not any(x in modules for x in argv):continue
            # Prediction/scoring workers are not producers of new GPU work.
            if '--mode' in argv and argv[argv.index('--mode')+1] in ('predict','score'):continue
            found.append(info)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return found


def wait_for_prior_owners():
    while True:
        owners=competing_owners()
        if not owners:return
        d.write(d.BASE/'queue_state.json',dict(phase='WAITING_PRIOR_OWNERS',active=[],pending=[r['row_id'] for r in d.rows()],
            owners=owners,updated_at=time.time(),reason='Existing launchers do not share an atomic GPU reservation; preserve healthy work'))
        time.sleep(30)


def source_complete(c):
    done = d.read(Path(c['output_root'])/'completion.json')
    require_budget(done['logged_steps'], done['optimizer_steps'])
    if (done['status'] != 'SOURCE_TRAINED' or done['config'] != c or done['epoch'] != 200
            or done['target_access'] or done['target_evaluated']):
        raise ValueError('Source completion mismatch')
    return done


def queue(configs, kind):
    sys.path.insert(0, str(d.ROOT/'experiments/adv3b02_xuc/code'))
    from scripts.dispatch_xuc_full import occupancy
    pending = list(configs); active = {}; receipts = []; failed = []
    while pending or active:
        for rid, job in list(active.items()):
            code = job['process'].poll()
            if code is None: continue
            try:
                if code: raise RuntimeError('Nonzero worker exit '+str(code))
                if kind == 'source': source_complete(job['config'])
                elif d.read(d.BASE/rid/'prediction/complete.json')['status'] != 'PREDICTIONS_COMPLETE':
                    raise ValueError('Missing prediction completion')
            except Exception as error: failed.append(dict(row_id=rid, error=repr(error)))
            del active[rid]
        # Bound this new queue to four workers while old matrix runs coexist.
        # Every training entry is named train_*.py so legacy owners see its
        # CUDA_VISIBLE_DEVICES reservation before CUDA has initialized.
        while pending and len(active) < 4 and not failed:
            if competing_owners():break
            caps = occupancy(active)
            choices = [(len(v['pids']), -v['free_mb'], gpu) for gpu,v in caps.items()
                       if len(v['pids']) < 2 and v['free_mb'] >= 12000]
            if not choices: break
            gpu = min(choices)[2]; c = pending.pop(0); rid = c['row_id']
            if kind == 'source':
                cmd = [sys.executable, '-u', str(d.ROOT/'experiments/cvs_phase1_repair/train_repair.py'),
                       '--config', str(d.BASE/'configs'/(rid+'.json'))]
            else:
                cmd = [sys.executable, '-u', str(d.ROOT/'experiments/cvs_phase1_repair/predict_repair.py'), '--mode', 'predict', '--row', rid]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(d.ROOT)+os.pathsep+str(d.ROOT/'code'),
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            log = Path(d.PROJECT)/'logs'/d.RUN/(kind+'-'+rid+'.log')
            with log.open('x') as f:
                child = subprocess.Popen(cmd,cwd=d.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active[rid] = dict(process=child, gpu=gpu, config=c)
            receipts.append(dict(row_id=rid,pid=child.pid,gpu=gpu,cwd=str(d.ROOT),argv=cmd,log=str(log)))
            d.write(d.BASE/('launch_'+kind+'.json'),dict(rows=receipts))
            print('LAUNCH '+str(receipts[-1]),flush=True)
        d.write(d.BASE/'queue_state.json',dict(phase=kind,active=list(active),pending=[c['row_id'] for c in pending],
                failures=failed,updated_at=time.time(),max_active=4,per_gpu_limit=2))
        if failed and not active: raise RuntimeError('Failed rows preserved; no retry '+repr(failed))
        if pending or active: time.sleep(10)


def dispatch():
    configs = [d.config(r) for r in d.rows()]
    d.BASE.mkdir(parents=True,exist_ok=False)
    (Path(d.PROJECT)/'logs'/d.RUN).mkdir(parents=True,exist_ok=False)
    d.write(d.BASE/'dispatcher.json',dict(run_id=d.RUN,owner=d.OWNER,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
            commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    try:
        for c in configs: d.write(d.BASE/'configs'/(c['row_id']+'.json'),c)
        wait_for_prior_owners()
        queue(configs,'source')
        records = [source_complete(c) for c in configs]
        # Fixed controls: every preregistered arm is evaluated. Source scores
        # describe performance and never change the matrix or final epoch.
        d.write(d.BASE/'source_matrix_frozen.json',dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=configs,
            source_metrics=[r['final_source_metrics'] for r in records],target_access=False,
            selection='ALL_PREREGISTERED_FIXED_CONTROLS',frozen_at=time.time()))
        queue(configs,'predict')
        subprocess.run([sys.executable,'-m','experiments.cvs_phase1_repair.evaluate','--mode','score'],cwd=d.ROOT,check=True)
        if d.read(d.BASE/'scoring_complete.json')['status'] != 'SCORED_COMPLETE': raise ValueError('Scoring incomplete')
        from experiments.cvs_phase1_repair.evaluate import VIEWS
        d.write(d.BASE/'completion.json',dict(status='ANALYZED',rows=len(configs),views=list(VIEWS),
                independent_recount='VERIFIED',target_feedback_forbidden=True))
    except Exception as error:
        d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),no_retry=True)); raise


if __name__ == '__main__': dispatch()
