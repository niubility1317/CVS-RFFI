"""Launch only the authorized 21 controls, preserving all other jobs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dispatch_xuc_dr import available_gpu,env_for,write,RELEASE,SCRIPTS,CONFIG

def matrix_rows():
    rows=json.loads((CONFIG/'separate_controls/matrix.json').read_text())['rows']
    assert len(rows)==21 and len({r['row']['id'] for r in rows})==21
    return rows

def command(item,project,run):
    script='train_daot_rc4_baseline.py' if item['family']=='native_baseline' else 'train_response_games.py'
    cmd=[sys.executable,'-u',str(SCRIPTS/script),'--config',str(CONFIG/'separate_controls'/(item['row']['id']+'.json')),
        '--dataset',str(project/'Dataset_WigSig/ManySig.pkl'),'--output',str(run/item['row']['id']),
        '--source-contract',str(run/'source_contract.json')]
    return cmd+(['--device','cuda:0','--execute'] if item['family']=='pure_game' else [])

def complete(folder):
    if not (folder/'completion.json').is_file() or not (folder/'final_ssdg.pth').is_file():return False
    c=json.loads((folder/'completion.json').read_text())
    return c.get('status')=='TRAINING_COMPLETE' and c.get('epochs')==200 and (c.get('native') is True or c.get('steps')==44400)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--source-contract',type=Path,required=True);a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):raise ValueError('unsafe run ID')
    rows=matrix_rows();contract=json.loads(a.source_contract.read_text())
    assert contract['counts']==dict(L_s=6300,U_s=56700,V=27000) and contract['role_ids']
    run=a.project/'runs'/a.run_id;logs=a.project/'logs'/a.run_id
    if run.exists() or logs.exists():raise FileExistsError('existing run: reconcile before any launch')
    run.mkdir();logs.mkdir();write(run/'source_contract.json',contract);write(run/'effective_matrix.json',dict(rows=rows))
    state=dict(status='TRAINING',pid=os.getpid(),run_id=a.run_id,commit=a.commit,release=str(RELEASE),created=time.time(),
        initialization='scratch',target_access=False,automatic_matrix_expansion=False,
        rows={r['row']['id']:dict(status='QUEUED',family=r['family']) for r in rows})
    write(run/'pipeline_state.json',state);pending=list(rows);active={};failed=False
    try:
        while pending or active:
            for rid,job in list(active.items()):
                code=job['process'].poll()
                if code is None:continue
                job['log'].close();del active[rid];ok=code==0 and complete(run/rid)
                state['rows'][rid].update(status='TRAINING_COMPLETE' if ok else 'TRAIN_FAILED',exit_code=code,finished=time.time())
                failed=failed or not ok;write(run/'pipeline_state.json',state)
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
        state.update(status='TECHNICAL_FAILURE',error=repr(error));write(run/'pipeline_state.json',state);raise
if __name__=='__main__':raise SystemExit(main())
