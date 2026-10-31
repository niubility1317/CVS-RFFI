"""Same-card no-update source step cost: cached head5 versus exact source10."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from experiments.pamf_dura_direct_train import runner as base,f_guided,balanced_tangent,soft_reference
from experiments.pamf_dura_diagnostics.runner import tensor
from .model import direct_objective,supervised_action
from .nonlinear_reference_probe import source as reference_source,smoke as reference_smoke,RUN as QUALIFICATION
from .exact_reference import exact_guide
from .soft_reference_tangent import ALPHA,prototype_tangent_loss

RUN='20261031-diagnostic-DURA-exactReference-stepCost-s2026092701-r01'
MODULE='experiments.pamf_dura_efficiency.nonlinear_profile'
CASES=('cached5','exact10')
VIEWS=('1','20')
WARMUP=3
REPEATS=12


def cost_decision(views):
    if set(views)!=set(VIEWS):raise ValueError('Both fixed source views required')
    rows=[]
    for view,r in views.items():
        if set(r['timings'])!=set(CASES) or any(len(r['timings'][c])!=REPEATS for c in CASES):
            raise ValueError('Full12 paired timings for both cases required')
        old=np.asarray(r['timings']['cached5'],np.float64);new=np.asarray(r['timings']['exact10'],np.float64)
        if not np.isfinite(old).all() or not np.isfinite(new).all() or np.any(old<=0) or np.any(new<=0):raise ValueError('Positive finite source timings required')
        ratio=float(np.median(new)/np.median(old));paired=float(np.median(new/old))
        rows.append(dict(view=view,median_ratio=ratio,paired_median_ratio=paired,passed=bool(ratio<=1.5 and paired<=1.5)))
    return dict(eligible=all(r['passed'] for r in rows),views=rows,maximum_ratio=1.5,automatic_training=False,
        scope='Single shared-card no-update B256 steps;notfullE30time orsame-quality efficiencyproof')


def inputs(args):
    p=base.paths(args.root);a,_,provenance=reference_source(args,p)
    qualification=base.read_json(args.root/'runs'/QUALIFICATION/'results.json')
    if (not qualification['decision']['eligible'] or not qualification['F_unchanged'] or not qualification['G_state_unchanged']
        or qualification['F_teacher']!=f_guided.teacher_reference() or qualification['iterations']!=8 or qualification['half_step']!=.5
        or any(qualification[k]!=0 for k in ['optimizer_updates','TX_labels_read','V_reads','target_reads'])):
        raise ValueError('Fixed nonlinear source reachability qualification required')
    _,roles=base.source_training(p);jac,reuse=balanced_tangent.metric_cache(args.root,p,roles)
    oldp=dict(p,output=args.root/'runs'/soft_reference.RUN);offsets,offsetmeta=soft_reference.load_offsets(oldp,roles)
    if not np.allclose(offsets[a['source_positions']],a['target_phi'][:,:5]-a['clean_phi'][:,:5],atol=3e-6,rtol=2e-6):
        raise ValueError('Exact old cached5 reference must match source10 first five coordinates')
    basis=np.load(args.root/'runs'/balanced_tangent.CACHE_RUN/'scratch_joint/source_clean_J_cache/basis.npy',allow_pickle=False)
    if not np.array_equal(basis,a['basis'][:,:5]):raise ValueError('Same original head5 cache coordinates required')
    return a,p,jac,offsets,dict(provenance=provenance,qualification_run=QUALIFICATION,raw_J_reuse=reuse,old_source_offset=offsetmeta)


def benchmark(args):
    a,p,jac,offsets,reuse=inputs(args)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=base.scratch_model(2026092701).to('cuda:0').train()
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    beforeG={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    beforeF={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    parameters=list(student.parameters());basis=tensor(a['basis'],'cuda:0');take=a['source_positions'][:256]
    if not np.array_equal(a['roles'][:256],np.asarray(['L_s']*128+['U_s']*128)):raise ValueError('Fixed128L128U source batch required')
    base.write_json(args.output/'worker.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],code_commit=args.commit))
    views={};checks={};totalG=totalF=totalVJP=0
    for view in VIEWS:
        folder=p['cache']/'epochs'/f'{int(view):02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ['received','kernel','coefficients','variance']}
        expected=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
        if any(v.shape!=expected[k] or v.dtype!=np.float32 for k,v in arrays.items()):raise ValueError('Original full source cache required')
        physical_reference=None;peak={c:0 for c in CASES};increment={c:0 for c in CASES};pressure={c:0. for c in CASES}
        timings={c:[] for c in CASES};pairs=[];last_metrics={}
        def step(case):
            nonlocal physical_reference,totalG,totalF,totalVJP
            torch.cuda.synchronize()
            for parameter in parameters:parameter.grad=None
            start_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            torch.manual_seed(2026103004);torch.cuda.manual_seed_all(2026103004)
            x=tensor(a['clean'][:256],'cuda:0')
            values={key:tensor(v[take],'cuda:0') for key,v in arrays.items()}
            if case=='cached5':
                j=tensor(jac[take],'cuda:0');d=tensor(offsets[take],'cuda:0')
            else:target=tensor(a['target_phi'][:256],'cuda:0')
            probes=[torch.complex(torch.randn(256,256,device='cuda:0'),torch.randn(256,256,device='cuda:0')) for _ in range(2)]
            _,terms,restored=direct_objective(student,values['received'],x,supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,return_restored=True)
            physical=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']
            if case=='cached5':
                value,info=prototype_tangent_loss(restored,x,j,d);guide=ALPHA*value
            else:
                guide,info=exact_guide(reader,restored,x,target,basis);totalF+=256;totalVJP+=256
            loss=physical+guide
            if not torch.isfinite(loss).all() or float(info['bounded_pressure'].max())>1+2e-5:raise FloatingPointError('Nonfinite/unbounded actual source step')
            loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            metrics=dict(loss=float(loss.detach().mean()),iq_nmse=float(terms['iq_nmse'].detach().mean()),
                prior_nmse=float(terms['prior_nmse'].detach().mean()),spectral_error=float(terms['spectral_error'].detach().mean()),
                action_probe_nmse=float(terms['action_probe_nmse'].detach().mean()),log_variance_mse=float(terms['log_variance_mse'].detach().mean()),
                weighted_guide=float(guide.detach().mean()),mean_weight=float(info['weight'].mean()),limited_fraction=float((info['weight']<1).float().mean()),
                maximum_pressure=float(info['bounded_pressure'].max()),gradient_norm=float(norm))
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            allocated=torch.cuda.max_memory_allocated();peak[case]=max(peak[case],allocated);increment[case]=max(increment[case],allocated-start_allocated)
            pressure[case]=max(pressure[case],metrics['maximum_pressure']);totalG+=256
            physical_values=np.asarray([metrics[k] for k in ['iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse']])
            if physical_reference is None:physical_reference=physical_values
            elif not np.allclose(physical_values,physical_reference,rtol=2e-5,atol=2e-6):raise ValueError('No-update cases changed physical inputs/objective')
            return seconds,metrics
        for repeat in range(WARMUP+REPEATS):
            order=CASES if repeat%2==0 else CASES[::-1];record={}
            for case in order:
                seconds,metrics=step(case);last_metrics[case]=metrics
                if repeat>=WARMUP:timings[case].append(seconds);record[case]=seconds
            if repeat>=WARMUP:pairs.append(dict(repeat=repeat-WARMUP,order=list(order),seconds=record))
        summaries={case:dict(count=len(t),median_seconds=float(np.median(t)),p95_seconds=float(np.quantile(t,.95)),min_seconds=min(t),max_seconds=max(t),
            peak_allocated_bytes=peak[case],peak_increment_bytes=increment[case],maximum_IQ_gradient_pressure=pressure[case]) for case,t in timings.items()}
        views[view]=dict(timings=timings,pairs=pairs,summaries=summaries,last_metrics=last_metrics)
        print(json.dumps(dict(event='SOURCE_EXACT_GUIDE_PROFILE_VIEW',view=view,summaries=summaries)),flush=True)
    for parameter in parameters:parameter.grad=None
    if any(not torch.equal(v.detach().cpu(),beforeG[k]) for k,v in student.state_dict().items()):raise ValueError('No-update benchmark changed G')
    if any(not torch.equal(v.detach().cpu(),beforeF[k]) for k,v in reader.state_dict().items()) or any(p.grad is not None for p in reader.parameters()):raise ValueError('Frozen F changed')
    if (totalG,totalF,totalVJP)!=(15360,7680,7680):raise ValueError('Incomplete fixed source step matrix')
    decision=cost_decision(views)
    base.write_json(args.output/'results.json',dict(status='ANALYZED_SOURCE_ONLY',views=views,decision=decision,
        cases=list(CASES),source_views=list(VIEWS),batch=256,unique_source_records=256,L_s=128,U_s=128,warmup_per_case_view=3,timed_per_case_view=12,
        G_inputs=totalG,F_inputs=totalF,F_input_VJP_inputs=totalVJP,CPU_smoke_F_inputs=4,CPU_smoke_F_VJP_inputs=2,CPU_smoke_G_inputs=2,
        G_unchanged=True,F_unchanged=True,optimizer_updates=0,initialization_checkpoint_loads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        F_teacher=f_guided.teacher_reference(),source_reuse=reuse,FP32=True,TF32=False,CPU_threads=2,physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        scope='Shared-card same-scratchG/source/probeRNG/no-update B256 forward-guide-backward-clip-copy steps;oldcached5 vsnewexact10 differobjectives. Nooptimizer/cacheproduction/epochshuffle/endtoendquality. F residentbothcases,peakmemorynotstandalonedeployment.'))
    base.write_json(args.output/'complete.json',dict(status='ANALYZED_SOURCE_ONLY',decision=decision,optimizer_updates=0,formal_predictions=0,goal_complete=False))


def launch(args):
    import fcntl
    if args.output!=args.root/'runs'/RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable root andCPU-onlycontroller required')
    args.output.mkdir();base.write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,commit=args.commit))
    a,p,_,_,_=inputs(args)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False);before={k:v.clone() for k,v in reader.state_dict().items()}
    reference_smoke(reader,a,p,args.output)
    if any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('CPU smoke changed frozen F')
    del reader,before,a,p;gc.collect()
    while True:
        with (args.root/'runs/receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=base.capacity();used=set();workers=0
            for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines():
                if ' worker ' not in line or not any(token in line for token in ['experiments.pamf_dura_direct_train.','experiments.pamf_dura_efficiency.','experiments.pamf_dura_prior_refine.','experiments.pamf_dura_prior_recovery.']):continue
                pid=line.split(None,1)[0]
                try:environment=(Path('/proc')/pid/'environ').read_bytes().split(bytes([0]))
                except FileNotFoundError:continue
                gpu_values=[item.split(b'=',1)[1] for item in environment if item.startswith(b'CUDA_VISIBLE_DEVICES=')]
                gpu_ids={int(v) for value in gpu_values for v in value.split(b',') if v.isdigit()}
                if gpu_ids:workers+=1;used.update(gpu_ids)
            free=base.eligible_gpus(state,used,3,allowed=[4,6,7]) if workers<2 else []
            if free:
                gpu=free[0];command=[sys.executable,'-u','-m',MODULE,'worker','--root',str(args.root),'--output',str(args.output),'--commit',args.commit]
                with (args.output/'gpu.log').open('xb') as log:process=subprocess.Popen(command,cwd=Path.cwd(),env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                base.write_json(args.output/'submitted.json',dict(pid=process.pid,GPU=gpu,command=command,
                    admission=dict(visible_GPU_compute_PIDs=len(state[gpu]['pids']),free_memory_MB=state[gpu]['free_mb'],maximum_total_after_adding=3)))
                print(json.dumps(dict(event='SUBMITTED',pid=process.pid,GPU=gpu)),flush=True);break
        print(json.dumps(dict(event='WAITING_CAPACITY',goal_workers=workers)),flush=True);time.sleep(30)
    while process.poll() is None:time.sleep(5)
    if process.returncode or not (args.output/'complete.json').exists():raise RuntimeError('Source cost measurement failed;preserve/noautoretry')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('run','worker'))
    for key in ['root','output']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--commit',required=True);args=parser.parse_args();args.root=args.root.resolve();args.output=args.output.resolve()
    if Path('release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':launch(args)
    else:benchmark(args)


if __name__=='__main__':main()
