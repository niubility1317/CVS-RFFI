"""Evaluation-only takeover; preserve source jobs and original frozen weights."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from experiments.cvs_dual_evidence import design as source
from experiments.cvs_dual_evidence import evaluate as evaluation
from experiments.cvs_dual_evidence.dispatch import capacity
from experiments.cvs_receiver_residual_capacity4.control import lock,process

RUN='20261009-phase1-dual-evidence-test-now-manysig-m16-r02'
RELEASE='cvs_dual_evidence_test_now_20261009_r02'
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(source.PROJECT)/'runs'/RUN
LOGS=Path(source.PROJECT)/'logs'/RUN
OLD_RELEASE=Path(source.PROJECT)/'releases'/source.RELEASE
OLD_PID=2074109
OLD_START=76356526
TOTAL_LIMIT=5
MIN_FREE_MB=3000

def configure():
    evaluation.BASE=BASE
    evaluation.ROOT=ROOT
    # The immutable snapshot and checkpoint provenance retain the source run ID.
    # Evaluation output identity is separately recorded, never rewritten into weights.

def verify_idle():
    p=process(OLD_PID)
    expected=['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-m','experiments.cvs_dual_evidence.dispatch']
    if not p or p['start_ticks']!=OLD_START or p['cwd']!=str(OLD_RELEASE) or p['argv']!=expected:
        raise ValueError('Original controller identity changed; no takeover')
    q=source.read(source.BASE/'queue_state.json')
    if q['kind']!='predict' or q['active'] or q['completed'] or q['failures'] or set(q['pending'])!={r['row_id'] for r in source.rows()}:
        raise ValueError('Original test queue no longer entirely idle')
    if (source.BASE/'launch_predict.json').exists() or list(source.BASE.glob('*/prediction')):
        raise ValueError('Existing test launch/artifacts: refuse duplicate evaluation')
    children=Path('/proc')/str(OLD_PID)/'task'/str(OLD_PID)/'children'
    if children.read_text().strip():raise ValueError('Original controller has live children')
    return p

def preflight():
    if BASE.exists() or LOGS.exists():raise FileExistsError('New evaluation output already exists')
    evaluation.preflight()
    return verify_idle()

def takeover():
    with lock(Path(source.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        before=verify_idle()
        os.kill(OLD_PID,signal.SIGTERM)
        for _ in range(50):
            p=process(OLD_PID)
            if p is None or p.get('state') in ('Z','X'):break
            time.sleep(.1)
        else:raise RuntimeError('Original controller did not exit; no replacement workers launched')
        return dict(status='VERIFIED',old_controller=before,old_controller_stopped=True,
                    reason='User requested immediate tests after capacity conflict was explained',
                    healthy_training_mutated=False)

def worker(rid):
    configure()
    if rid not in {r['row_id'] for r in source.rows()}:raise ValueError('Unregistered row')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    with lock(Path(source.PROJECT)/'runs/receiver_residual_capacity4.lock'):
        c=capacity()[gpu]
        if len(c['pids']|{os.getpid()})>TOTAL_LIMIT or c['free_mb']<MIN_FREE_MB:
            raise RuntimeError('Inference capacity changed; no automatic retry')
    evaluation.predict(rid)
    resolved=BASE/rid/'prediction/resolved_config.json'
    value=source.read(resolved);value['run_id']=RUN;value['source_run']=source.RUN
    value['scheduling_override']='one additional inference process per GPU; total at most5'
    source.write(resolved,value)

def dispatch():
    with lock(ROOT/'owner.lock',nonblocking=True):
        preflight()
        BASE.mkdir();LOGS.mkdir()
        source.write(BASE/'takeover.json',takeover())
        source.write(BASE/'source_matrix_frozen.json',source.read(source.BASE/'source_matrix_frozen.json'))
        source.write(BASE/'evaluation_context.json',dict(run_id=RUN,source_run=source.RUN,
            source_commit=(OLD_RELEASE/'release_commit.txt').read_text().strip(),
            evaluation_commit=(ROOT/'release_commit.txt').read_text().strip(),
            source_freeze_ref=str(source.BASE/'source_matrix_frozen.json'),total_gpu_limit=TOTAL_LIMIT,
            per_gpu_test_limit=1,source_training_mutated=False,selection='ALL16_FROZEN_CONTROLS',truth_last=True))
        pending=[r['row_id'] for r in source.rows()];active={};complete=[];launch=[];failures=[]
        try:
            while pending or active:
                for rid,item in list(active.items()):
                    rc=item['p'].poll()
                    if rc is None:continue
                    path=BASE/rid/'prediction/complete.json'
                    if rc or not path.exists() or source.read(path)['status']!='PREDICTIONS_COMPLETE':
                        failures.append(dict(row_id=rid,exit_code=rc))
                    else:complete.append(rid)
                    del active[rid]
                for gpu in range(8):
                    if not pending or failures or any(i['gpu']==gpu for i in active.values()):continue
                    with lock(Path(source.PROJECT)/'runs/receiver_residual_capacity4.lock'):
                        c=capacity()[gpu]
                        if len(c['pids'])>=TOTAL_LIMIT or c['free_mb']<MIN_FREE_MB:continue
                        rid=pending.pop(0)
                        cmd=[sys.executable,'-u',str(ROOT/'experiments/cvs_dual_test_now/train_worker.py'),'--row',rid]
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                        log=LOGS/(rid+'.log')
                        with log.open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                        actual=process(p.pid)
                        if actual is None:raise RuntimeError('New test worker absent')
                        launch.append(dict(row_id=rid,kind='predict',gpu=gpu,pid=p.pid,start_ticks=actual['start_ticks'],cwd=str(ROOT),argv=cmd,log=str(log)))
                        source.write(BASE/'launch_predict.json',dict(rows=launch))
                        active[rid]=dict(p=p,gpu=gpu)
                source.write(BASE/'queue_state.json',dict(kind='predict',active=list(active),pending=pending,completed=complete,failures=failures,total_gpu_limit=TOTAL_LIMIT,updated_at=time.time()))
                if failures and not active:raise RuntimeError('Failed predictions preserved without retry: '+repr(failures))
                if pending or active:time.sleep(5)
            subprocess.run([sys.executable,'-m','experiments.cvs_dual_test_now.driver','--score'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
            scored=source.read(BASE/'scoring_complete.json')
            if scored['metric_records']!=1568 or scored['day_metric_records']!=448:raise ValueError('Incomplete score matrix')
            source.write(BASE/'completion.json',dict(status='ANALYZED',rows=16,views=7,source_run=source.RUN,truth_last=True))
        except Exception as error:
            source.write(BASE/'failure.json',dict(status='FAILED',error=repr(error),no_retry=True));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.score:configure();evaluation.score()
    else:dispatch()
