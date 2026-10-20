import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint,validate_parent_root
from .training import train
from .models import objective,paired_outputs
from .evaluation import predict,score
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_r17_task.runtime import load_original
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,array_tensor,code_commit
from experiments.pamf_restoration.data import IQSource,write_json

PATH_ARGS=('teacher_run','compact_parent','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,arm):
    argv=[sys.executable,'-u','-m','experiments.pamf_r46_complex_fidelity.runner',mode,'--arm',arm]
    for key in PATH_ARGS:argv.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    return argv+['--expected-commit',args.expected_commit]


def smoke(args):
    validate_parent_root(args.compact_parent)
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False);take=np.flatnonzero(roles=='L_s')[:2]
    if len(take)!=2:raise ValueError('Actual two L_s paired examples required')
    x=array_tensor(np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    received=array_tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')
    from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor as fast_tensor
    if not torch.equal(x,fast_tensor(np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')) or not torch.equal(received,fast_tensor(np.load(cache/'epochs/01/received.npy',mmap_mode='r',allow_pickle=False)[take],'cpu')):raise ValueError('Direct buffer must preserve historical control IQ exactly')
    reader=load_reader(args.reader_folder,args.source_cache,'cpu',seed=d.F_SEED).eval().requires_grad_(False)
    before={k:v.detach().clone() for k,v in reader.state_dict().items()};records={}
    for arm in d.ARMS:
        model,teacher,parent=initialize(arm,args.compact_parent,'cpu');teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
        outputs,desired,identity=paired_outputs(model,teacher,received,x)
        loss,_=objective(outputs,desired,x,1,arm,identity);loss.mean().backward()
        groups={'coarse':model.frozen.coarse,'action':model.frozen.estimator,**{'prior'+str(i):v for i,v in enumerate(model.prior)}}
        norms={name:sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None) for name,module in groups.items()}
        if any(not np.isfinite(v) or v<=0 for v in norms.values()):raise ValueError('Every successor component must receive finite nonzero gradients')
        if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('R46 parent must stay unchanged')
        model.eval();path=args.output/('smoke-'+arm+'.pt');save_checkpoint(path,model,None,0,0,parent);fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.equal(model(received),fresh(received)) or not torch.isfinite(reader(fresh(received))).all():raise ValueError('Standalone strict successor reload/default head failed')
        records[arm]=dict(parameters=d.PARAMETERS[arm],component_gradient_norms=norms,checkpoint_output_exact=True,R46_parent_unchanged=True,
            training_F_access=False,F_knowledge_in_loss=False)
    if any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v,before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F/default head changed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_Ls_IQ=2,arms=records,TX_labels_read=0,V_reads=0,target_reads=0,
        teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,optimizer_updates=0))


def predecessor_ready(args,arm):
    gpu=d.GPUS[arm];previous=args.output.parent/d.PREDECESSOR_BY_GPU[gpu];saved=previous/'workers.json'
    if not saved.exists():return False
    entries=[entry for entry in read_json(saved)['workers'] if entry['GPU']==gpu]
    if len(entries)!=1:raise ValueError('Exactly one registered preceding GPU worker required')
    entry=entries[0];row=previous/entry['arm']
    if not any((row/name).exists() for name in ('complete.json','failure.json')):return False
    path=Path('/proc')/str(entry['pid'])
    try:
        state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(path/'cmdline').read_bytes().decode().split(chr(0))
        if state!='Z' and str(previous) in argv and d.PREDECESSOR_MODULE[previous.name] in argv:return False
    except FileNotFoundError:pass
    return True


def score_subprocess(args,role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


@torch.no_grad()
def profile(args):
    values=array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:64],'cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
    models={}
    for name,loader,path in (
        ('original_R17',load_original,args.teacher_run/'unroll_unshared'/d.KIND/'final.pt'),
        ('parent_R46_350521',load_parent,args.compact_parent/d.COMPACT_PARENT_ROW/d.KIND/'final.pt'),
        ('compact',load_checkpoint,args.output/args.arm/d.KIND/'final.pt')):
        model,payload=loader(path,'cpu')
        if (payload['epoch'],payload['updates'])!=((d.EPOCHS,d.UPDATES) if name=='compact' else (20,4940)):raise ValueError('Fixed source final profile checkpoints required')
        del payload;models[name]=model.eval().requires_grad_(False).to('cuda:0')
    results={name:dict(parameters=sum(p.numel() for p in model.parameters()),cases={}) for name,model in models.items()}
    for mode in ('G_only','G_plus_F'):
        for batch,warmup,iterations in ((1,16,64),(32,4,16)):
            samples={name:[] for name in models};rounds={name:[] for name in models};peaks={name:[] for name in models};resident={name:[] for name in models}
            for repeat in range(3):
                names=list(models) if repeat%2==0 else list(reversed(models))
                for name in names:
                    model=models[name]
                    def forward(x):
                        y=model(x)
                        return reader(y) if mode=='G_plus_F' else y
                    for _ in range(warmup):forward(values[:batch])
                    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();base_memory=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
                    for index in range(iterations):
                        start=(index*batch)%(len(values)-batch+1);x=values[start:start+batch]
                        torch.cuda.synchronize();started=time.perf_counter();y=forward(x);torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
                        if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite matched source benchmark output')
                        del y
                    samples[name].extend(times);rounds[name].append(float(np.median(times)));peaks[name].append(torch.cuda.max_memory_allocated());resident[name].append(base_memory)
            for name in models:
                times=np.asarray(samples[name]);key=mode+'_B'+str(batch)
                results[name]['cases'][key]=dict(batch=batch,warmup_per_round=warmup,rounds=3,iterations_per_round=iterations,source_V_unique_IQ=64,
                    median_ms=float(np.median(times)),p90_ms=float(np.quantile(times,.9)),p95_ms=float(np.quantile(times,.95)),
                    round_median_ms=rounds[name],throughput_samples_per_second=float(batch*1000/times.mean()),
                    resident_before_forward_bytes=max(resident[name]),peak_allocated_bytes=max(peaks[name]),
                    incremental_peak_bytes=max(p-r for p,r in zip(peaks[name],resident[name])))
    for name,model in models.items():
        hooks=[];macs=[0]
        def count(module,inputs,output):
            if isinstance(module,torch.nn.Conv1d):macs[0]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
            elif isinstance(module,torch.nn.Linear):macs[0]+=output.numel()*module.in_features
        for module in model.modules():
            if isinstance(module,(torch.nn.Conv1d,torch.nn.Linear)):hooks.append(module.register_forward_hook(count))
        model(values[:1]);torch.cuda.synchronize()
        for hook in hooks:hook.remove()
        results[name]['ConvLinear_MAC_per_IQ']=macs[0]
    write_json(args.output/args.arm/'matched_inference_profile.json',dict(status='PROFILE_COMPLETE',physical_GPU=d.GPUS[args.arm],results=results,
        checkpoint_payload_on_GPU=False,optimizer_on_GPU=False,source_V_IQ=64,TX_labels_read=0,state_updates=0,
        includes_full_context_3prior_6CG=True,reader_weights_read_for_evaluation_only=True,training_F_access=False,
        memory_scope='All3frozenGand1Fresident in every case;incrementalpeak perforward is comparable;resident is shared suite,not standalone deploymentmemory',
        MAC_scope='Conv1d+Linear only;FFT/complexCG/norm/activation/memory excluded;F MAC excluded',
        timing_scope='Synchronized preloaded-device FP32 eval;excludes transfer,preprocess,loader;alternating forward/reverse candidate order perround',
        whole_pipeline='G_plus_F includes original frozen F default head and internal FFT/PA/time/stats;no new head'))
    del models,reader,values;gc.collect();torch.cuda.empty_cache()

def worker(args):
    row=args.output/args.arm
    if row.exists():raise ValueError('New own row required;existing outputs stay unchanged')
    if read_json(args.output/'launch.json')['commit']!=args.expected_commit or read_json(args.output/'comparison_plan.json')!=d.configuration():raise ValueError('Matching launched run required before row mutation')
    try:
        train(args);torch.cuda.empty_cache();predict(args,'source');profile(args);score_subprocess(args,'source')
        write_json(args.output/args.arm/'source_complete.json',dict(status='ANALYZED',arm=args.arm,role='source',diagnostic_only=True,
            source_selection=False,all_predictions_before_truth=True,target_feedback=False))
        torch.cuda.empty_cache();predict(args,'target');score_subprocess(args,'target')
        write_json(args.output/args.arm/'complete.json',dict(status='ANALYZED',arm=args.arm,role='target',all_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        row=args.output/args.arm;row.mkdir(exist_ok=True)
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Immutable new root and CPU controller required')
    validate_parent_root(args.compact_parent);args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/R46-complex-fidelity',
        compact_parent_run=d.COMPACT_PARENT_RUN,compact_parent_commit=d.COMPACT_PARENT_COMMIT))
    write_json(args.output/'resolved_config.json',d.configuration());write_json(args.output/'comparison_plan.json',d.configuration())
    try:
        smoke(args);print(json.dumps(dict(event='R46_COMPLEX_FIDELITY_SMOKE_PASS',arms=list(d.ARMS))),flush=True)
        waiting=list(d.ARMS);active={};launches=[];failed=[];last_wait=0.
        write_status(args.output/'workers.json',dict(workers=launches))
        while waiting or active:
            for arm in waiting[:]:
                if not predecessor_ready(args,arm):continue
                argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
                with log.open('xb') as stream:
                    process=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=process;waiting.remove(arm);launches.append(dict(arm=arm,pid=process.pid,GPU=d.GPUS[arm],cwd=str(ROOT),argv=argv,log=str(log)))
                write_status(args.output/'workers.json',dict(workers=launches))
            for arm,process in list(active.items()):
                if process.poll() is None:continue
                if process.returncode or not (args.output/arm/'complete.json').exists():failed.append(dict(arm=arm,exit_code=process.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(stage='wait_actual_predecessor_then_fixed_source_target',waiting=waiting,active={a:p.pid for a,p in active.items()},failed=failed))
            if waiting and time.monotonic()-last_wait>=30:
                print(json.dumps(dict(event='WAIT_ACTUAL_PRECEDING_GPU_HANDLES',waiting=waiting,source_score_reads=0,target_score_reads=0)),flush=True);last_wait=time.monotonic()
            if waiting or active:time.sleep(2)
        if failed:raise RuntimeError('Technical failures preserved after healthy rows completed:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),no_source_score_selection=True,target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def main():
    parser=argparse.ArgumentParser(description='Source complex-frequency fidelity and clean identity on fixed 350521-parent compression')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned successor code/root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Fixed successor arm required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()
