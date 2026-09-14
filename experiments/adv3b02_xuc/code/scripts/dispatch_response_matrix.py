"""Frozen source-only first wave; no automatic target scoring or expansion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dispatch_xuc_dr import available_gpu,env_for,write,RELEASE,SCRIPTS,CONFIG

def matrix_rows():
    old=json.loads((CONFIG/'native_dr_eg/matrix.json').read_text())
    new=json.loads((CONFIG/'response_games/matrix.json').read_text())
    main={f'{method}_s{seed}' for method in ('SIM','EG','CF_EG','TR_EG','XT_DANN','DRIC') for seed in (392005,392006,392007)}
    selected=[dict(family='response_games',row=x['row']) for x in new['rows'] if x['row']['id'] in main]
    selected += [dict(family='native_dr_eg',row=x) for x in old['rows']]
    if len(selected)!=36 or len({x['row']['id'] for x in selected})!=36:raise ValueError('first wave must contain 36 unique rows')
    # Seed-paired round robin starts every principal mechanism on the first GPUs.
    selected.sort(key=lambda x:(x['row']['joint']['model_seed'],0 if x['family']=='response_games' else 1))
    return selected

def command(item,project,run):
    script='train_response_games.py' if item['family']=='response_games' else 'train_native_dr_eg.py'
    return [sys.executable,'-u',str(SCRIPTS/script),'--config',str(CONFIG/item['family']/(item['row']['id']+'.json')),
        '--dataset',str(project/'Dataset_WigSig/ManySig.pkl'),'--output',str(run/item['row']['id']),
        '--source-contract',str(run/'source_contract.json'),'--device','cuda:0','--execute']

def complete(folder):
    p=folder/'completion.json'
    if not p.is_file() or not (folder/'final_ssdg.pth').is_file():return False
    c=json.loads(p.read_text())
    return c.get('status')=='TRAINING_COMPLETE' and c.get('epochs')==200 and c.get('steps')==44400

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--source-contract',type=Path,required=True)
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):raise ValueError('unsafe run ID')
    rows=matrix_rows();contract=json.loads(a.source_contract.read_text())
    if contract['counts']!={'L_s':6300,'U_s':56700,'V':27000} or not contract['role_ids']:raise ValueError('source role contract mismatch')
    run=a.project/'runs'/a.run_id;logs=a.project/'logs'/a.run_id
    if run.exists() or logs.exists():raise FileExistsError('existing run; reconcile, never relaunch')
    run.mkdir();logs.mkdir()
    state=dict(status='TRAINING',pid=os.getpid(),run_id=a.run_id,commit=a.commit,release=str(RELEASE),created=time.time(),
        initialization='scratch',target_access=False,automatic_matrix_expansion=False,rows={x['row']['id']:dict(status='QUEUED',family=x['family']) for x in rows})
    write(run/'source_contract.json',contract);write(run/'effective_matrix.json',dict(rows=rows,epochs=200,first_wave=36,
        all_prepared_configs=127,remaining_response_rows='SOURCE_SELECTION_PENDING',dependencies='NOT_EXECUTABLE',target_feedback=False))
    write(run/'pipeline_state.json',state)
    pending=list(rows);active={};failed=False
    try:
        while pending or active:
            for rid,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['log'].close();del active[rid]
                ok=code==0 and complete(run/rid)
                state['rows'][rid].update(status='TRAINING_COMPLETE' if ok else 'TRAIN_FAILED',exit_code=code,finished=time.time())
                if not ok:failed=True
                write(run/'pipeline_state.json',state)
            # A technical failure pauses new submissions. Healthy existing workers finish.
            if failed and pending:
                for item in pending:state['rows'][item['row']['id']]['status']='HELD_TECHNICAL_FAILURE'
                pending=[];state['status']='DRAINING_AFTER_TECHNICAL_FAILURE';write(run/'pipeline_state.json',state)
            if pending:
                gpu=available_gpu(active)
                if gpu is not None:
                    item=pending.pop(0);rid=item['row']['id'];cmd=command(item,a.project,run)
                    log=(logs/(rid+'.train.log')).open('x')
                    child=subprocess.Popen(cmd,cwd=RELEASE,env=env_for(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[rid]=dict(process=child,log=log,gpu=gpu)
                    state['rows'][rid].update(status='RUNNING',pid=child.pid,gpu=gpu,argv=cmd,cwd=str(RELEASE),started=time.time())
                    write(logs/(rid+'.launch.json'),state['rows'][rid]);write(run/'pipeline_state.json',state)
                    print(json.dumps(dict(event='LAUNCHED',row=rid,pid=child.pid,gpu=gpu)),flush=True)
            time.sleep(5)
        state.update(status='FAILED_PARTIAL' if failed else 'SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW',finished=time.time())
        write(run/'pipeline_state.json',state);return int(failed)
    except Exception as error:
        state.update(status='TECHNICAL_FAILURE',error=repr(error));write(run/'pipeline_state.json',state)
        # Preserve children and their recorded ownership. No broad termination.
        raise

if __name__=='__main__':raise SystemExit(main())
