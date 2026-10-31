"""Primary ground-trained Tri-SIG-TIME SSL benchmark; no open-world branch."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .cost_probe import gpu_choice,write
from .source_pair import read,validate_source
from .checkpoint import RUNS,load_model
from .cache import load_F0,load_cache,close_cache,tensor

RUN='20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03'
MODULE='experiments.pamf_phase3_restoration.main_split_eval'
G_COMMIT='b33839dd1010dea36b68526043bbc4abe5eb1095'
F_RUN='20261027-phase1-pamf-repl-shuffled-views-m16-r02'
F_ROW='main_split-repl-shuffled_time-3view-s2026092701'
F_COMMIT='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d'
COUNTS={'cross_rx':126000,'cross_both':42000}
STREAMS=('raw','DURA_v1','TF_IQ','TF_F0')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban',
       'practical_high_urban','practical_mid_urban','practical_low_urban')


def official(root,kind='predict'):
    c=read(root/'runs'/F_RUN/'configs'/(kind+'__'+F_ROW+'.json'))
    if (c.get('run_id'),c.get('row_id'),c.get('model_seed'),c.get('classes'))!=(
            F_RUN,F_ROW,2026092701,['14-10','14-7','20-15','20-19','6-15','8-20']):
        raise ValueError('Original ground-trained full Tri-SIG-TIME SSL Main split required')
    if list(c['tests'])!=list(COUNTS) or c['readouts']!=['raw']:
        raise ValueError('Both original Main split populations required')
    return c


def valid_identity(identity):
    expected=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
                  source_run=F_RUN,source_row=F_ROW,source_code_commit=F_COMMIT)
    if any(identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Ground-trained original Tri-SIG-TIME SSL/default head required; no F50/PCD')


def dura(c,device):
    import torch
    release=Path(c['dura_release'])
    if (release/'release_commit.txt').read_text().strip()!='47c264286ca689f060a3f36aafb763fe209a6f06':
        raise ValueError('Fixed default DURA execution required')
    package=importlib.import_module('experiments');modules=release/'experiments'
    package.__path__=[str(modules)]+[str(p) for p in package.__path__ if Path(p)!=modules]
    sys.path[:0]=[str(release),str(release/'code')]
    from experiments.pamf_r65_four_route.runtime import load_checkpoint
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    model,payload=load_checkpoint(c['dura_checkpoint'],device)
    if ((payload.get('arm'),payload.get('epoch'),payload.get('updates'),payload.get('code_commit'))!=
            ('stage2',20,4940,'b2a2ce937a64654e17e84c2aa817b9da4257a48f')
            or sum(p.numel() for p in model.parameters())!=266951):
        raise ValueError('Fixed default DURA v1 R69 required')
    return model.eval().requires_grad_(False),InferenceEngine(model,reuse=True,graph=True,native=False)


def predict(args):
    import torch
    started=time.perf_counter();c=validate_source(read(args.inputs));o=official(args.root)
    if c['reader_config']['source_output']!=o['source_output']:raise ValueError('Different frozen F0 row')
    device='cuda:0';torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    F,identity=load_F0(c,device);valid_identity(identity)
    from experiments.cric_dura_comparison.evaluation import handle
    D,engine=dura(c,device)
    models={'TF_'+arm:load_model(args.root/'runs'/RUNS[arm]/'training/final.pt',device,arm=arm,commit=G_COMMIT) for arm in ('IQ','F0')}
    modules=[F,D,*models.values()]
    if any(p.requires_grad or p.dtype!=torch.float32 for m in modules for p in m.parameters()):
        raise ValueError('All four networks must remain frozen FP32')
    snapshots=[(m,{k:v.detach().clone() for k,v in m.state_dict().items()}) for m in modules]
    out=args.root/'runs'/RUN/'prediction'/args.task;out.mkdir()
    # First NN work is an actual source-only checkpoint smoke; no target input.
    source=load_cache(args.cache,guided=True)
    try:
        if source['manifest']['F0_identity']!=identity:raise ValueError('Actual source F0 cache identity differs')
        x=tensor(source['clean'][:2],device);target=tensor(source['F0_features'][:2],device)
        with torch.no_grad():
            torch.testing.assert_close(F.features(x),target,atol=3e-5,rtol=2e-5)
            probes={'raw':x,'DURA_v1':engine(x)[0],**{name:m(x) for name,m in models.items()}}
            for value in probes.values():
                scores=F.classify_features(F.features(value))
                if scores.shape!=(2,6) or not torch.isfinite(value).all() or not torch.isfinite(scores).all():
                    raise ValueError('Actual source-only TriSSL/default DURA/TF smoke failed')
    finally:close_cache(source)
    write(out/'smoke.json',dict(status='PASS',actual_source_L_s=2,all_four_checkpoint_paths=True,
        original_TriSIGTIME_SSL_cache_match=True,optimizer_updates=0,target_reads=0,TX_labels_read=0))
    print('TRISIGSSL_MAIN_REAL_SOURCE_SMOKE_PASS',args.task,flush=True)
    manifest,ids,positions=handle(o['tests'][args.task]['views_root'],o['classes'])
    if len(ids)!=COUNTS[args.task]:raise ValueError('Original full Main split population required')
    with np.load(Path(o['output_root'])/args.task/'predictions.npz',allow_pickle=False) as f:
        if not np.array_equal(f['ids'],ids):raise ValueError('Official raw prediction physical order differs')
        old={view:f['raw__'+view].copy() for view in VIEWS}
    results={};timings={};mismatches={}
    for view in VIEWS:
        path=manifest['clean_ref'] if view=='clean' else str(Path(manifest['observations_root'])/(view+'.npy'))
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.ndim!=3 or a.shape[1:]!=(2,256) or a.dtype!=np.float32:
            raise ValueError('Original Main split FP32 IQ required')
        if positions is not None and (positions.min()<0 or positions.max()>=len(a)):
            raise ValueError('Original Main split wrapper positions differ')
        decisions={name:[] for name in STREAMS};begin=time.perf_counter()
        try:
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    selected=slice(start,start+128) if positions is None else positions[start:start+128]
                    x=tensor(a[selected],device)
                    values={'raw':x,'DURA_v1':engine(x)[0],**{name:m(x) for name,m in models.items()}}
                    for name,value in values.items():
                        if value.shape!=x.shape or not torch.isfinite(value).all():raise ValueError('Frozen G returned invalid IQ')
                        logits=F.classify_features(F.features(value))
                        if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():raise ValueError('Invalid frozen TriSSL default head')
                        decisions[name].extend(logits.argmax(1).cpu().tolist())
        finally:a._mmap.close()
        torch.cuda.synchronize();timings[view]=time.perf_counter()-begin
        for name,values in decisions.items():results[name+'__'+view]=np.asarray(values,np.int64)
        mismatches[view]=int(np.count_nonzero(results['raw__'+view]!=old[view]))
        print('TRISIGSSL_MAIN_VIEW_COMPLETE',args.task,view,len(ids),'official_raw_disagreements',mismatches[view],flush=True)
    for m,before in snapshots:
        if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen model changed')
    with (out/'predictions.npz').open('xb') as f:np.savez(f,ids=ids,**results)
    write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',run_id=RUN,task=args.task,count=len(ids),
        code_commit=args.commit,F0_identity=identity,G_checkpoint_commit=G_COMMIT,
        readouts=list(STREAMS),views=list(VIEWS),truth_read=False,query_fit=False,model_state_updates=0,
        original_official_raw_decision_disagreements=mismatches,seconds=time.perf_counter()-started,
        per_view_seconds=timings,F0_main_forward_rows=4*len(ids)*7,F0_smoke_feature_rows=10,
        G_forward_rows_each=7*len(ids)+2,DURA_G_rows_are_logical_calls_graph_startup_not_instrumented=True))


def finalize(args):
    out=args.root/'runs'/RUN/'prediction';tests={}
    for task,count in COUNTS.items():
        m=read(out/task/'complete.json');valid_identity(m['F0_identity'])
        required=dict(status='PREDICTIONS_COMPLETE',run_id=RUN,task=task,count=count,
            code_commit=args.commit,G_checkpoint_commit=G_COMMIT,readouts=list(STREAMS),views=list(VIEWS),
            truth_read=False,query_fit=False,model_state_updates=0)
        if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Both complete full Main split tasks required before truth')
        tests[task]=dict(count=count,views=list(VIEWS),readouts=list(STREAMS))
    write(out/'complete.json',dict(status='ALL_PREDICTIONS_COMPLETE',run_id=RUN,row_id='TriSIGTIME_SSL-main-frozen-G',
        tests=tests,views=list(VIEWS),readouts=list(STREAMS),truth_read=False,query_fit=False,
        code_commit=args.commit))


def summary(scores,*,streams=STREAMS):
    cells={}
    for row in scores['results']:
        if row['dimension']=='overall' and row['stratum']=='ALL':
            cells.setdefault(row['test'],{}).setdefault(row['readout'],{})[row['view']]=row
    if set(cells)!=set(COUNTS):raise ValueError('Both Main split score populations required')
    output={}
    for task in [*COUNTS,'combined']:
        table={}
        for name in streams:
            table[name]={}
            for view in VIEWS:
                rows=[cells[t][name][view] for t in COUNTS] if task=='combined' else [cells[task][name][view]]
                cm=np.sum([r['confusion'] for r in rows],axis=0);n=int(cm.sum());correct=int(np.trace(cm))
                table[name][view]=dict(count=n,correct=correct,accuracy=100*correct/n)
        raw=table['raw']['clean'];acceptance={}
        for name in streams:
            for view in VIEWS:
                cell=table[name][view];cell['raw_clean_gap_pp']=raw['accuracy']-cell['accuracy']
                cell['clean_accuracy_retention_percent']=100*cell['correct']/raw['correct']
            acceptance[name]=dict(scenes_passing_3pp=sum(table[name][v]['raw_clean_gap_pp']<=3+1e-12 for v in VIEWS[1:]),
                all_six_passed=all(table[name][v]['raw_clean_gap_pp']<=3+1e-12 for v in VIEWS[1:]),
                clean_preserved=table[name]['clean']['correct']>=raw['correct'])
            acceptance[name]['performance_passed']=acceptance[name]['all_six_passed'] and acceptance[name]['clean_preserved']
        output[task]=dict(metrics=table,acceptance=acceptance)
    return output


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU-only truth scorer required')
    out=args.root/'runs'/RUN;marker=read(out/'prediction/complete.json')
    if marker.get('code_commit')!=args.commit:raise ValueError('Actual prediction code differs')
    for task in COUNTS:valid_identity(read(out/'prediction'/task/'complete.json')['F0_identity'])
    c=official(args.root,'score');c.update(run_id=RUN,row_id='TriSIGTIME_SSL-main-frozen-G',
        readouts=list(STREAMS),prediction_output=str(out/'prediction'),output_root=str(out/'score'))
    from experiments.pamf_repl_view_shuffle.evaluation import score as original_score
    original_score(c)  # Existing full preflight of BOTH tasks/all28 streams before first truth.
    result=read(out/'score/scores.json')
    write(out/'results.json',dict(status='ANALYZED',run_id=RUN,code_commit=args.commit,
        primary_base='Ground-trained Tri-SIG-TIME SSL/3ff8/E50-11100/defaulthead',
        F_source_run=F_RUN,F_source_row=F_ROW,G_checkpoint_commit=G_COMMIT,
        results=summary(result),truth_last=True,training_updates=0,performance_primary=True,
        efficiency_secondary=True,default_DURA_changed=False,goal_complete=False,
        scope='Original Main split cross_rx126000/cross_both42000; all four readouts same IQ/F0 and seven full views. No Phase3 open classifier or oldF50 scoring.'))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty launch owner required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir();(out/'prediction').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),CWD=os.getcwd(),argv=sys.argv,code_commit=args.commit))
    pending=list(COUNTS);active={};failed=[]
    while pending or active:
        for task,(process,gpu) in list(active.items()):
            code=process.poll()
            if code is not None:
                write(out/('terminal-'+task+'.json'),dict(pid=process.pid,GPU=gpu,exit_code=code))
                if code:failed.append(task)
                del active[task]
        choices=[a for a in gpu_choice() if a['GPU'] not in [g for _,g in active.values()]]
        while pending and choices and len(active)<2:
            task=pending.pop(0);admission=choices.pop(0)
            command=[sys.executable,'-u','-m',MODULE,'predict','--root',str(args.root),'--commit',args.commit,
                     '--inputs',str(args.inputs),'--cache',str(args.cache),'--task',task]
            with (out/'logs'/(task+'.log')).open('xb') as log:
                process=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(admission['GPU'])),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[task]=(process,admission['GPU'])
            write(out/('submitted-'+task+'.json'),dict(pid=process.pid,command=command,CWD=os.getcwd(),admission=admission))
        if pending or active:time.sleep(5)
    if failed:raise RuntimeError('Failed full Main split task preserved: '+str(failed))
    finalize(args)
    import fcntl
    with (args.root/'runs'/RUNS['IQ']/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        command=[sys.executable,'-u','-m',MODULE,'score','--root',str(args.root),'--commit',args.commit]
        with (out/'logs/score.log').open('xb') as log:
            subprocess.run(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(out/'complete.json',dict(status='ANALYZED',run_id=RUN,code_commit=args.commit,
        all_main_predictions_before_truth=True,training_updates=0,default_DURA_changed=False,goal_complete=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','predict','score'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--inputs',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--task',choices=tuple(COUNTS))
    a=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Actual execution Git commit required')
    if a.mode in ('run','predict') and (a.inputs is None or a.cache is None):p.error('--inputs/--cache required')
    if a.mode=='predict' and a.task is None:p.error('--task required')
    if a.mode=='run':controller(a)
    elif a.mode=='predict':predict(a)
    else:score(a)


if __name__=='__main__':main()
