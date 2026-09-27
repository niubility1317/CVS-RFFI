"""One immutable matrix release; independent rows, two admitted processes per GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
RELEASE=Path(__file__).resolve().parents[2]
SCRIPTS=RELEASE/'code/scripts'
CONFIG=RELEASE/'configs'
TARGET_BASE='phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'

def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def occupancy(active):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    mapping={v[1].strip():(int(v[0]),int(v[2])) for v in (r.split(',') for r in raw.splitlines())}
    result={gpu:{'pids':set(),'free_mb':free} for gpu,free in mapping.values()}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        v=line.split(',')
        if len(v)==2 and v[0].strip() in mapping:result[mapping[v[0].strip()][0]]['pids'].add(int(v[1]))
    # Reserve newly launched children before CUDA initializes; union avoids double counting.
    for job in active.values():
        if job['process'].poll() is None:result[job['gpu']]['pids'].add(job['process'].pid)
    return result

def available_gpu(active):
    candidates=[(len(v['pids']),-v['free_mb'],gpu) for gpu,v in occupancy(active).items() if len(v['pids'])<2 and v['free_mb']>=6500]
    return min(candidates)[2] if candidates else None

def train_command(row,project,run):
    common=['--row',row['id'],'--dataset',str(project/'Dataset_WigSig/ManySig.pkl'),
            '--output',str(run/row['id']),'--source-contract',str(run/'source_contract.json')]
    if row['pipeline']=='core90':
        return [sys.executable,'-u',str(SCRIPTS/'train_xuc.py'),'--matrix',str(CONFIG/'matrix15.json'),
                '--recipe',str(CONFIG/'core90_recipe_reference.json'),*common]
    return [sys.executable,'-u',str(SCRIPTS/'train_xuc_native.py'),'--reference',str(CONFIG/'a1_native_recipe_reference.json'),*common]

def env_for(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(RELEASE/'code')+os.pathsep+str(RELEASE),
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')

def assert_old_run_idle(old):
    prefix=str(old)+'/'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:argv=(proc/'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeError):continue
        associated=any(v==str(old) or v.startswith(prefix) for v in argv)
        if '--run-id' in argv:
            i=argv.index('--run-id');associated|=i+1<len(argv) and argv[i+1]==old.name
        if associated:raise RuntimeError(f'old run process {proc.name} is alive; preserve it before recovery')

def retry_allowed(entry,folder):
    if entry['status'] in ('TRAINING_COMPLETE','PREDICTIONS_FIXED','SCORED'):return False
    complete=folder/'completion.json'
    if complete.is_file() and json.loads(complete.read_text()).get('epochs')==200:return False
    return entry['status'] in ('TRAIN_FAILED','QUEUED','RUNNING')

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--recovery-from',default='')
    p.add_argument('--retry-rows',default='')
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):raise ValueError('unsafe run-id')
    project=a.project;run=project/'runs'/a.run_id;logs=project/'logs'/a.run_id
    if run.exists() or logs.exists():raise FileExistsError('run or log root already exists')
    run.mkdir(parents=True);logs.mkdir(parents=True)
    matrix=json.loads((CONFIG/'matrix15.json').read_text(encoding='utf-8'))
    if len(matrix['runs'])!=15 or {r['id'] for r in matrix['runs']}!={f'M{i:02}' for i in range(15)}:raise ValueError('not frozen matrix15')
    reused={};prediction_refs={};retry=set(a.retry_rows.split(',')) if a.retry_rows else set()
    if a.recovery_from:
        if Path(a.recovery_from).name!=a.recovery_from or a.recovery_from in ('.','..'):raise ValueError('unsafe recovery root')
        old=project/'runs'/a.recovery_from
        prior=json.loads((old/'pipeline_state.json').read_text())
        if not retry<={r['id'] for r in matrix['runs']}:raise ValueError('unknown retry row')
        assert_old_run_idle(old)
        # An old dispatcher/worker must finish before a new owner takes over the matrix.
        for item in [prior,*prior['rows'].values()]:
            pid=item.get('pid')
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
            except FileNotFoundError:continue
            if a.recovery_from in cmd:raise RuntimeError('old owner/worker is still alive; preserve healthy jobs')
        for rid,entry in prior['rows'].items():
            folder=Path(entry.get('artifact_root',str(old/rid)))
            if rid in retry:
                if not retry_allowed(entry,folder):raise ValueError('refusing to retrain healthy E200 row '+rid)
                continue
            complete=json.loads((folder/'completion.json').read_text())
            if complete['epochs']!=200 or not (folder/'final_ssdg.pth').is_file():raise ValueError('nonretry row lacks E200 artifacts')
            reused[rid]=folder
            prediction=Path(entry.get('prediction_path',str(folder/'target_prediction/predictions.json')))
            if (entry.get('predictions_fixed') or entry['status'] in ('PREDICTIONS_FIXED','SCORED')) and prediction.is_file():prediction_refs[rid]=prediction
    elif retry:raise ValueError('retry-rows requires recovery-from')
    state=dict(status='SOURCE_PREPARING',run_id=a.run_id,pid=os.getpid(),release=str(RELEASE),commit=a.commit,
               seed=392005,epochs=200,created=time.time(),rows={r['id']:{'status':'QUEUED'} for r in matrix['runs']})
    write(run/'pipeline_state.json',state);write(run/'effective_matrix.json',matrix)
    state['recovery_from']=a.recovery_from;state['retry_rows']=sorted(retry)
    for rid,folder in reused.items():
        state['rows'][rid]=dict(status='TRAINING_COMPLETE',artifact_root=str(folder),reused_for_inference_only=True)
        if rid in prediction_refs:state['rows'][rid].update(prediction_path=str(prediction_refs[rid]),predictions_fixed=True)
    write(run/'pipeline_state.json',state)
    active={};pending=[r for r in matrix['runs'] if r['id'] not in reused]
    try:
        # Dataset roles are source-only. No model state is inherited.
        with (logs/'source_contract.log').open('x') as handle:
            subprocess.run([sys.executable,str(SCRIPTS/'build_xuc_source_contract.py'),'--matrix',str(CONFIG/'matrix15.json'),
                '--recipe',str(CONFIG/'core90_recipe_reference.json'),'--dataset',str(project/'Dataset_WigSig/ManySig.pkl'),
                '--output',str(run/'source_contract.json')],cwd=RELEASE,env=env_for(''),stdout=handle,stderr=subprocess.STDOUT,check=True)
        target=project/'runs'/TARGET_BASE/'target_inputs'
        if reused:
            expected=json.loads((run/'source_contract.json').read_text())['role_ids']
            for folder in reused.values():
                if json.loads((folder/'source_contract.json').read_text())['role_ids']!=expected:raise ValueError('reused evaluation checkpoint source contract mismatch')
        meta=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
        if meta['record_count']!=168000 or meta.get('contains_labels') is not False:raise ValueError('target package contract mismatch')
        if not (target/'iq.npy').is_file():raise FileNotFoundError('existing validated IQ package missing')
        state['status']='TRAINING';write(run/'pipeline_state.json',state)
        while pending or active:
            for rid,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['log'].close();del active[rid]
                entry=state['rows'][rid];entry.update(exit_code=code,finished=time.time())
                if code:entry['status']='TRAIN_FAILED'
                else:
                    complete=run/rid/'completion.json';ckpt=run/rid/'final_ssdg.pth'
                    if complete.is_file() and ckpt.is_file() and json.loads(complete.read_text())['epochs']==200:entry['status']='TRAINING_COMPLETE'
                    else:entry.update(status='TRAIN_FAILED',error='missing E200 completion/checkpoint')
                write(run/'pipeline_state.json',state)
            if pending:
                gpu=available_gpu(active)
                if gpu is not None:
                    row=pending.pop(0);rid=row['id'];cmd=train_command(row,project,run)
                    log=(logs/(rid+'.train.log')).open('x')
                    child=subprocess.Popen(cmd,cwd=RELEASE,env=env_for(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[rid]=dict(process=child,log=log,gpu=gpu)
                    state['rows'][rid]=dict(status='RUNNING',pid=child.pid,gpu=gpu,argv=cmd,cwd=str(RELEASE),started=time.time())
                    write(logs/(rid+'.launch.json'),state['rows'][rid]);write(run/'pipeline_state.json',state)
                    print(json.dumps(dict(event='LAUNCHED',row=rid,pid=child.pid,gpu=gpu)),flush=True)
            time.sleep(5)
        if any(v['status']!='TRAINING_COMPLETE' for v in state['rows'].values()):
            state['status']='FAILED_PARTIAL';write(run/'pipeline_state.json',state);return 1
        # Every row is frozen before target inference. Truth is not an argument to predictors.
        state['status']='PREDICTING';write(run/'pipeline_state.json',state)
        for row in matrix['runs']:
            rid=row['id'];gpu=available_gpu({})
            folder=Path(state['rows'][rid].get('artifact_root',str(run/rid)))
            state['rows'][rid]['artifact_root']=str(folder)
            if rid in prediction_refs:
                if json.loads(prediction_refs[rid].read_text())['record_count']!=672000:raise ValueError('reused prediction count mismatch')
                state['rows'][rid].update(status='PREDICTIONS_FIXED',prediction_path=str(prediction_refs[rid]),predictions_fixed=True)
                write(run/'pipeline_state.json',state);continue
            while gpu is None:time.sleep(30);gpu=available_gpu({})
            command=[sys.executable,str(SCRIPTS/'predict_phase1_truth_last.py'),'--checkpoint',str(folder/'final_ssdg.pth'),
                '--output-root',str(run/rid/'target_prediction'),'--input-package',str(target),
                '--recipe',str(CONFIG/'core90_recipe_reference.json'),'--run-id',a.run_id,'--row-id',rid,
                '--mode','predict','--batch-size','256','--num-workers','0','--device','cuda:0']
            with (logs/(rid+'.predict.log')).open('x') as h:
                child=subprocess.Popen(command,cwd=RELEASE,env=env_for(gpu),stdout=h,stderr=subprocess.STDOUT)
                state['stage_worker']=dict(pid=child.pid,stage='predict',row=rid,argv=command)
                write(run/'pipeline_state.json',state)
                if child.wait():raise RuntimeError('prediction subprocess failed '+rid)
                state.pop('stage_worker')
            prediction=run/rid/'target_prediction/predictions.json'
            if json.loads(prediction.read_text())['record_count']!=672000:raise ValueError('prediction count mismatch')
            state['rows'][rid].update(status='PREDICTIONS_FIXED',prediction_path=str(prediction),predictions_fixed=True);write(run/'pipeline_state.json',state)
        write(run/'predictions_complete.json',dict(rows=list(state['rows']),record_count=10080000,created=time.time()))
        state['status']='SCORING';write(run/'pipeline_state.json',state)
        # Only the separate scorer now receives the sidecar. No score flows to training.
        for rid in state['rows']:
            with (logs/(rid+'.score.log')).open('x') as h:
                command=[sys.executable,str(SCRIPTS/'score_phase1_truth_last.py'),'--predictions',state['rows'][rid]['prediction_path'],
                    '--truth',str(project/'runs'/TARGET_BASE/'target_truth/truth_sidecar.json'),
                    '--output',str(run/rid/'target_prediction/score.json')]
                child=subprocess.Popen(command,cwd=RELEASE,env=env_for(''),stdout=h,stderr=subprocess.STDOUT)
                state['stage_worker']=dict(pid=child.pid,stage='score',row=rid,argv=command)
                write(run/'pipeline_state.json',state)
                if child.wait():raise RuntimeError('scoring subprocess failed '+rid)
                state.pop('stage_worker')
            state['rows'][rid]['status']='SCORED';write(run/'pipeline_state.json',state)
        state['status']='COMPLETE';state['finished']=time.time();write(run/'pipeline_state.json',state);return 0
    except Exception as error:
        state.update(status='TECHNICAL_FAILURE',error=repr(error),failed=time.time())
        # Children are preserved; recorded PIDs remain authoritative if dispatcher fails.
        write(run/'pipeline_state.json',state);raise

if __name__=='__main__':raise SystemExit(main())
