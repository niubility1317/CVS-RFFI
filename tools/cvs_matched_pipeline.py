"""Fixed final-checkpoint dependencies: source aggregates, clean/LEO and D92.

No mutation of a training run. Separate subprocesses isolate native and D92
imports, and the scorer is only dispatched after all predictions are complete.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def source_alive(source,launch):
    row=next(r for r in launch['rows'] if r['output']==str(source))
    proc=Path('/proc')/str(row['pid'])/'cmdline'
    try:argv=proc.read_bytes().decode().split('\0')
    except FileNotFoundError:return False
    return argv[:-1]==row['argv']


def worker(spec,row):
    out=Path(row['output_root']);source=Path(row['source_root'])
    out.mkdir(exist_ok=False)
    write(out/'pipeline_startup.json',dict(spec=spec['run_id'],row=row,argv=sys.argv,cwd=str(ROOT),pid=os.getpid(),python=sys.executable))
    launch=read(Path(spec['source_run_root'])/'launch.json')
    write(out/'pipeline_state.json',dict(status='WAITING_SOURCE'))
    while not (source/'completion.json').exists():
        if not source_alive(source,launch):
            raise RuntimeError('Source exited without completion; preserve artifacts, no retry')
        time.sleep(30)
    # Final checkpoint and its contract are checked in the native subprocess.
    def child(label,args):
        write(out/'pipeline_state.json',dict(status=label,argv=args))
        with (out/(label+'.log')).open('x',encoding='utf-8') as log:
            log.write(json.dumps(dict(argv=args,cwd=str(ROOT),time=time.time()))+'\n');log.flush()
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    common=['--source',str(source),'--contract',spec['source_contract'],'--native-code',spec['native_code'],
            '--seed',str(row['seeds']['model']),'--device','cpu']
    native=[sys.executable,str(ROOT/'tools/cvs_native_artifacts.py')]
    d92=[sys.executable,str(ROOT/'tools/cvs_d92_matched.py')]
    child('SOURCE_FEATURES',native+['source-features',*common,'--output',str(out/'source_features')])
    child('GROUND',d92+['ground','--features',str(out/'source_features/source_l_features.npz'),
                      '--source-run',str(source),'--output',str(out/'ground')])
    child('FINAL_PREDICT',native+['final-predict',*common,'--capsule',spec['final_capsule'],'--output',str(out/'final_eval')])
    child('RECEIVED_FEATURES',native+['received-features',*common,'--capsule',spec['phase2_capsule'],'--output',str(out/'received_features')])
    child('D92_PREDICT',d92+['predict','--features',str(out/'received_features/received_features.npz'),
                          '--capsule',spec['phase2_capsule'],'--ground',str(out/'ground'),
                          '--output',str(out),'--seed',str(row['seeds']['model'])])
    if read(out/'predictions_complete.json')['status']!='PREDICTIONS_COMPLETE':
        raise ValueError('Missing D92 predictions')
    write(out/'pipeline_state.json',dict(status='PREDICTIONS_COMPLETE'))


def final_score(spec):
    import numpy as np
    root=Path(spec['execution']['remote_run_root'])
    state=read(root/'state.json')
    if any(s['status']!='PREDICTIONS_COMPLETE' for s in state.values()):
        raise ValueError('Predictions are incomplete')
    for row in spec['rows']:
        if read(Path(row['output_root'])/'final_eval/predictions_complete.json')['status']!='PREDICTIONS_COMPLETE':
            raise ValueError('Final predictions are incomplete')
    capsule=Path(spec['final_capsule'])
    manifest=read(capsule/'manifest.json');index=np.load(capsule/'index.npz',allow_pickle=False)
    truth=read(spec['final_truth']);ids=index['ids'].tolist()
    y=np.asarray([truth[s]['label'] for s in ids]);rx=np.asarray([str(truth[s]['receiver']) for s in ids])
    results=[];classes=len(manifest['classes'])
    for row in spec['rows']:
        pred=np.load(Path(row['output_root'])/'final_eval/predictions.npz',allow_pickle=False)
        if not np.array_equal(pred['ids'],index['ids']):raise ValueError('Prediction identity mismatch')
        for view in ['clean']+manifest['scenes']:
            mask=np.ones(len(ids),dtype=bool) if view=='clean' else index['scenes']==manifest['scenes'].index(view)
            p=pred['clean' if view=='clean' else 'satellite']
            if p.shape!=y.shape or p.min()<0 or p.max()>=classes:raise ValueError('Invalid predictions')
            for receiver in ['ALL']+sorted(set(rx)):
                use=mask if receiver=='ALL' else mask&(rx==receiver)
                cm=np.zeros((classes,classes),dtype='int64');np.add.at(cm,(y[use],p[use]),1)
                den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(classes),where=den!=0)
                results.append(dict(row_id=row['row_id'],view=view,receiver=receiver,count=int(use.sum()),
                                    accuracy=float(cm.diagonal().sum()/use.sum()),macro_f1=float(f1.mean()),confusion=cm.tolist()))
    with (root/'phase1_final_results.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='SCORED',target_feedback_forbidden=True,results=results),f)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','dispatch','worker','final-score'])
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--row');p.add_argument('--commit',default='unspecified')
    a=p.parse_args();spec=read(a.spec);root=Path(spec['execution']['remote_run_root'])
    if a.action=='launch':
        root.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        cmd=[sys.executable,str(Path(__file__).resolve()),'dispatch','--spec',str(a.spec.resolve()),'--commit',a.commit]
        with (root/'dispatcher.log').open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'launch.json',dict(pid=proc.pid,argv=cmd,cwd=str(ROOT),commit=a.commit,owner=spec['execution']['launch_owner']))
        print(json.dumps(dict(pid=proc.pid,run_root=str(root))))
    elif a.action=='worker':
        worker(spec,next(row for row in spec['rows'] if row['row_id']==a.row))
    elif a.action=='final-score':
        final_score(spec)
    else:
        state={r['row_id']:dict(status='WAITING_SOURCE') for r in spec['rows']};jobs={}
        for row in spec['rows']:
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--spec',str(a.spec.resolve()),'--row',row['row_id']]
            with (root/(row['row_id']+'.log')).open('x') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            jobs[row['row_id']]=proc;state[row['row_id']]['pid']=proc.pid
        write(root/'state.json',state)
        while jobs:
            for rowid,proc in list(jobs.items()):
                code=proc.poll()
                if code is not None:
                    state[rowid].update(status='PREDICTIONS_COMPLETE' if code==0 else 'TECHNICAL_FAILURE',exit_code=code)
                    del jobs[rowid]
                    write(root/'state.json',state)
            if jobs:time.sleep(30)
        if any(s['status']!='PREDICTIONS_COMPLETE' for s in state.values()):
            raise RuntimeError('At least one row failed; do not score partial matrix')
        for label,cmd in [('phase1_score',[sys.executable,str(Path(__file__).resolve()),'final-score','--spec',str(a.spec.resolve())]),
                          ('phase2_score',[sys.executable,spec['phase2_scorer'],'--run-root',str(root),'--truth',spec['phase2_truth']])]:
            with (root/(label+'.log')).open('x') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(root/'completion.json',dict(status='SCORED',commit=a.commit,rows=len(state)))


if __name__=='__main__':main()
