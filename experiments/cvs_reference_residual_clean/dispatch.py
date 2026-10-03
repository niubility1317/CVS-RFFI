"""One frozen eight-row prediction launch; separate truth-last scorer process."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.cvs_reference_residual_clean.contracts import read,validate_spec
from experiments.cvs_reference_residual_clean.provenance import frozen_source_matrix
from experiments.cvs_clean_design.dispatch import choose_gpu
from experiments.cvs_identity_ce.source import write

def dispatch(path):
    spec=validate_spec(read(path));frozen_source_matrix()
    run,logs=Path(spec['runtime_root']),Path(spec['log_root']);run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='PREDICTING',run_id=spec['run_id'],pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,launch_owner=spec['launch_owner'],
        commit=(ROOT/'release_commit.txt').read_text().strip(),rows={r['row_id']:dict(status='QUEUED',variant=r['variant'],model_seed=r['model_seed']) for r in spec['rows']})
    write(run/'pipeline_state.json',state);pending=list(spec['rows']);active={}
    while pending or active:
        for rid,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[rid];entry=state['rows'][rid]
            entry.update(exit_code=code,finished=time.time(),status='FAILED' if code else 'PREDICTIONS_COMPLETE')
            done=Path(entry['output_root'])/'clean_complete.json'
            if not code and (not done.is_file() or read(done).get('status')!='PREDICTIONS_COMPLETE'):entry['status']='FAILED'
            write(run/'pipeline_state.json',state)
        if pending:
            gpu=choose_gpu(active)
            if gpu is not None:
                row=pending.pop(0);rid=row['row_id'];logpath=logs/(rid+'.log');log=logpath.open('x')
                command=[sys.executable,'-u','-m','experiments.cvs_reference_residual_clean.predict','--config',row['config']]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[rid]=dict(process=child,gpu=gpu,log=log)
                state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=command,log=str(logpath),output_root=row['output_root'],started=time.time())
                write(run/'pipeline_state.json',state)
        if pending or active:time.sleep(5)
    if any(r['status']!='PREDICTIONS_COMPLETE' for r in state['rows'].values()):
        state['status']='FAILED_PARTIAL';write(run/'pipeline_state.json',state);return 1
    state['status']='SCORING';write(run/'pipeline_state.json',state)
    with (logs/'scorer.log').open('x') as log:
        result=subprocess.run([sys.executable,'-u','-m','experiments.cvs_reference_residual_clean.score','--spec',str(path)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    complete=run/'scoring_clean_complete.json'
    okay=result.returncode==0 and complete.is_file() and read(complete).get('status')=='SCORED_COMPLETE'
    if okay:
        state['status']='RECOUNTING';write(run/'pipeline_state.json',state)
        with (logs/'recount.log').open('x') as log:
            check=subprocess.run([sys.executable,'-u','-m','experiments.cvs_reference_residual_clean.recount','--spec',str(path)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        proof=run/'independent_recount.json'
        okay=check.returncode==0 and proof.is_file() and read(proof).get('status')=='VERIFIED'
        state['recount_exit']=check.returncode
    state['status']='SCORED_COMPLETE' if okay else 'SCORING_FAILED';state['scorer_exit']=result.returncode;write(run/'pipeline_state.json',state)
    return 0 if okay else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();raise SystemExit(dispatch(a.spec))
