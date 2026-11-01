"""Source-only frozen final30 latency; queued after complete parent Main scoring."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from . import covariance_detail_trial as parent
from .source_inverse import R07,panel
from .source_pair import read,validate_source
from .cache import tensor,load_F0,check_F0_state
from .main_split_eval import dura,valid_identity
from .variance_trial import recovery_gpu_choice
from .cost_probe import write

RUN='20261102-diagnostic-Restore-CovarianceDetail-final-latency-s2026092701-r01'
MODULE='experiments.pamf_phase3_restoration.covariance_detail_latency'
G_COMMIT='1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6'
ROUTES=('raw_F0','DURA_G','new_G','DURA_G_F0','new_G_F0')


def schedule():
    rows=[]
    for batch,count,blocks in [(1,100,25),(128,20,4)]:
        for warmup,total in [(True,3),(False,count)]:
            for iteration in range(total):
                start=0 if warmup else (iteration%blocks+iteration//blocks)%5
                routes=ROUTES[start:]+ROUTES[:start]
                for position,route in enumerate(routes):rows.append(dict(sequence=len(rows),batch=batch,warmup=warmup,iteration=iteration,
                    source_block=iteration%blocks,position=position,route=route))
    return rows


def parent_ready(root):
    folder=root/'runs'/parent.RUN
    if any(read(p)['exit_code'] for p in folder.glob('terminal-*.json')):raise RuntimeError('Parent technical failure; preserve/no retry')
    if not (folder/'complete.json').is_file():return False
    if read(folder/'complete.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Complete parent Main required before latency')
    return True


def validate_parent(done):
    expected=dict(status='SOURCE_TRAINED',code_commit=G_COMMIT,arm=parent.ARM,epoch=30,updates=7410,G_parameters=5481223)
    if any(done.get(k)!=v for k,v in expected.items()) or any(done.get(k)!=0 for k in ['G_initialization_checkpoint_loads','TX_labels_read','V_reads','OE_reads','target_training_reads','model_state_F0_updates']):raise ValueError('Exact source-only final30 parent required')
    if done.get('F0_identity',{}).get('source_code_commit')!='3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':raise ValueError('Original ground Tri-SIG-TIME SSL required')


def summarize(rows):
    expected=schedule()
    if len(rows)!=len(expected) or any(any(row.get(k)!=v for k,v in plan.items()) for row,plan in zip(rows,expected)):raise ValueError('Entire fixed warmup/timed schedule required')
    result={}
    for batch in [1,128]:
        result[str(batch)]={}
        for route in ROUTES:
            take=[x for x in rows if x['batch']==batch and x['route']==route and not x['warmup']]
            for x in take:
                if not all(np.isfinite(x[k]) and x[k]>0 for k in ['wall_ms','CUDA_event_ms']):raise ValueError('Finite positive latency required')
                if x['peak_allocated_bytes']-x['baseline_allocated_bytes']!=x['incremental_bytes'] or x['incremental_bytes']<0:raise ValueError('Consistent memory counters required')
            median=float(np.median([x['wall_ms'] for x in take]))
            result[str(batch)][route]=dict(timed_calls=len(take),median_wall_ms=median,p90_wall_ms=float(np.quantile([x['wall_ms'] for x in take],.9)),
                median_CUDA_event_ms=float(np.median([x['CUDA_event_ms'] for x in take])),throughput_records_per_second=1000*batch/median,
                max_resident_allocated_bytes=max(x['peak_allocated_bytes'] for x in take),max_incremental_bytes=max(x['incremental_bytes'] for x in take))
    return result


def worker(args):
    if not parent_ready(args.root):raise ValueError('Latency cannot run before parent full Main completion')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'):raise ValueError('One admitted physical GPU required')
    device='cuda:0';started=time.perf_counter();out=args.root/'runs'/RUN
    folder=args.root/'runs'/parent.RUN/'rows'/parent.ARM/'training';done=read(folder/'complete.json');validate_parent(done)
    p=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if (set(p)!={'config','code_commit','epoch','updates','state'} or p['config']!=parent.configuration(parent.ARM) or (p['code_commit'],p['epoch'],p['updates'])!=(G_COMMIT,30,7410)):raise ValueError('Exact original final G state/configuration required')
    G=parent.factory(parent.ARM);expected=G.state_dict()
    if set(p['state'])!=set(expected) or any(v.shape!=expected[k].shape or v.dtype!=torch.float32 or not torch.isfinite(v).all() for k,v in p['state'].items()):raise ValueError('Exact finite FP32 final G required')
    G.load_state_dict(p['state'],strict=True);G=G.to(device).eval().requires_grad_(False)
    c=validate_source(read(args.inputs));F,identity=load_F0(c,device);valid_identity(identity)
    if identity!=done['F0_identity']:raise ValueError('Same frozen original F0 required')
    D,engine=dura(c,device);models=[G,F,D];snapshots=[{k:v.detach().clone() for k,v in m.state_dict().items()} for m in models]
    old=args.root/'runs'/R07/'cache';m=read(old/'complete.json')
    if (m['status'],m['train_count'],m['epochs'],m['TX_labels_read'],m['target_reads'])!=('ACTION_SOURCE_CACHE_COMPLETE',63000,20,0,0):raise ValueError('Original validated source family required')
    roles=np.load(old/'train/roles.npy',allow_pickle=False);selected=panel(roles)
    a=np.load(old/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)
    try:
        if a.shape!=(63000,2,256) or a.dtype!=np.float32:raise ValueError('Original fixed source draw required')
        source=tensor(a[selected],device)
    finally:a._mmap.close()
    if not torch.isfinite(source).all():raise ValueError('Finite source IQ required')
    logical_counts={r:0 for r in ROUTES}
    def invoke(route,x):
        logical_counts[route]+=len(x)
        if route.startswith('DURA'):value=engine(x)[0]
        elif route.startswith('new'):value=G(x)
        else:value=x
        return F.classify_features(F.features(value)) if route.endswith('F0') else value
    with torch.inference_mode():
        for route in ROUTES:
            value=invoke(route,source[:2]);shape=(2,6) if route.endswith('F0') else (2,2,256)
            if value.shape!=shape or not torch.isfinite(value).all():raise ValueError('Actual source checkpoint latency smoke failed')
        del value
        write(out/'smoke.json',dict(status='PASS',real_source_Ls=2,all_five_routes=True,actual_final30=True,actual_original_F0=True,
            default_DURA_R69=True,DURA_reuse=True,DURA_graph=True,DURA_native=False,optimizer_updates=0,TX_labels_read=0,target_inputs=0))
        print('COVARIANCE_DETAIL_LATENCY_SOURCE_SMOKE_PASS',flush=True)
        records=[];points=np.linspace(0,511,25,dtype=int)
        with (out/'timings.jsonl').open('x',encoding='utf-8',buffering=1) as log:
            for plan in schedule():
                block=plan['source_block'];x=source[int(points[block]):int(points[block])+1] if plan['batch']==1 else source[128*block:128*(block+1)]
                torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                tick=time.perf_counter_ns();begin.record();value=invoke(plan['route'],x);end.record();torch.cuda.synchronize()
                wall=(time.perf_counter_ns()-tick)/1e6;event=begin.elapsed_time(end);peak=torch.cuda.max_memory_allocated()
                shape=(plan['batch'],6) if plan['route'].endswith('F0') else (plan['batch'],2,256)
                if value.shape!=shape or not torch.isfinite(value).all():raise ValueError('Nonfinite timed readout; preserve/no omitted cases')
                row=dict(plan,wall_ms=wall,CUDA_event_ms=event,baseline_allocated_bytes=baseline,peak_allocated_bytes=peak,incremental_bytes=peak-baseline)
                records.append(row);log.write(__import__('json').dumps(row,allow_nan=False)+'\n');del value,begin,end
                if not plan['warmup'] and plan['position']==4 and plan['iteration']%10==9:print('LATENCY_TIMED_PROGRESS',plan['batch'],plan['iteration']+1,flush=True)
    for model,before in zip(models,snapshots):check_F0_state(model,before)
    if logical_counts!={r:3049 for r in ROUTES}:raise ValueError('Fixed logical source visits required')
    write(out/'results.json',dict(status='FINAL_LATENCY_COMPLETE',run_id=RUN,code_commit=args.commit,parent_G_code_commit=G_COMMIT,F0_identity=identity,
        GPU_model=torch.cuda.get_device_name(0),torch_version=torch.__version__,CUDA_version=torch.version.cuda,CPU_threads=torch.get_num_threads(),
        summary=summarize(records),logical_route_rows=logical_counts,new_G_actual_rows=6098,DURA_logical_rows=6098,F0_actual_rows=9147,
        DURA_graph_startup_extra_forwards_not_instrumented=True,source_packets=512,source_draw=1,batches=[1,128],warmups_per_route_per_batch=3,
        timed_calls_B1_per_route=100,timed_calls_B128_per_route=20,wall_includes='G/F0 forward and synchronization;actualinternaldeploymentchecks/graphinputcopy included',
        wall_excludes='model-load,sourceIO/device-transfer,case-selection,graph-first-capture/warmups,post-outputfinitecheck',
        memory_scope='Allthreefrozenmodels+512sourceinputs+allDURAgraphsresident;incrementalcurrentroutepeak. Notstandalonepermodelprocessmemory.',
        G_execution='EagerFP32physical-TF+FP64posterior+FP32detail',DURA_execution='fixed47c264/R69/FP32/InferenceEngine(reuse=True,graph=True,native=False)',
        F0_execution='sameoriginal3ff8fullTriSIGTIME_SSL/FP32/defaultjointhead',TF32=False,shared_GPU=True,model_states_unchanged=True,
        optimizer_updates=0,TX_labels_read=0,V_reads=0,OE_reads=0,novel_reads=0,target_inputs=0,quality_evaluation=False,goal_complete=False,seconds=time.perf_counter()-started))


def controller(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CUDA-empty profile controller required')
    out=args.root/'runs'/RUN;out.mkdir();(out/'logs').mkdir()
    write(out/'launch.json',dict(pid=os.getpid(),command=sys.argv,CWD=os.getcwd(),code_commit=args.commit))
    while not parent_ready(args.root):time.sleep(5)
    choices=recovery_gpu_choice()
    while not choices:time.sleep(5);choices=recovery_gpu_choice()
    admission=choices[0];cmd=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--commit',args.commit,'--inputs',str(args.inputs)]
    with (out/'logs/worker.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=admission['CUDA_ID']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'submitted.json',dict(pid=p.pid,command=cmd,CWD=os.getcwd(),admission=admission));observed=False
    while p.poll() is None:
        if not observed and (out/'smoke.json').is_file():
            proc=Path('/proc')/str(p.pid)
            try:
                command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();cwd=os.readlink(proc/'cwd')
                cuda=[x.decode() for x in (proc/'environ').read_bytes().split(bytes([0])) if x.startswith(b'CUDA_VISIBLE_DEVICES=')]
                query=subprocess.check_output(['nvidia-smi','-i',admission['CUDA_ID'],'--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
                if MODULE+' worker ' in command and args.commit in command and cwd==os.getcwd() and cuda==['CUDA_VISIBLE_DEVICES='+admission['CUDA_ID']] and any(s.split(',')[0].strip()==admission['CUDA_ID'] and int(s.split(',')[1])==p.pid for s in query.splitlines()):
                    write(out/'runtime_worker_observation.json',dict(status='VERIFIED',pid=p.pid,CWD=cwd,command=command,CUDA=cuda,GPU=query,source_smoke=read(out/'smoke.json'),log_bytes=(out/'logs/worker.log').stat().st_size,log_growth_from_zero=True,observer_PID=os.getpid()));observed=True
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        time.sleep(.2)
    code=p.wait();write(out/'terminal.json',dict(pid=p.pid,exit_code=code))
    if code:raise RuntimeError('Frozen latency worker failed; preserve/no retry')
    if read(out/'results.json')['status']!='FINAL_LATENCY_COMPLETE':raise ValueError('Missing complete fixed timing matrix')
    write(out/'complete.json',dict(status='ARTIFACTS_COMPLETE',live_worker_observation=observed,optimizer_updates=0,target_inputs=0,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual code identity required')
    if args.mode=='run':controller(args)
    else:worker(args)
