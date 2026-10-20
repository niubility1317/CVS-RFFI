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
from .graph import FrozenPipeline, GraphCache
from .runtime import ROOT, load_model, source_checkpoint, validate_sources
from .evaluation import predict, score
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json, code_commit
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_iq_joint_compact.models import convolution_linear_macs

PATH_ARGS = ('teacher_run','compact_parent','factor_run','parent_action','source_cache',
             'reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args, mode, arm):
    value = [sys.executable,'-u','-m','experiments.pamf_r46_cuda_graph.runner',mode,'--arm',arm]
    for key in PATH_ARGS:
        value += ['--'+key.replace('_','-'),str(getattr(args,key))]
    return value+['--expected-commit',args.expected_commit]


def predecessor_ready(args, gpu):
    previous = args.output.parent/d.PREDECESSOR
    workers = read_json(previous/'workers.json')['workers']
    selected = [w for w in workers if w['GPU'] == gpu]
    if len(selected) != 1:
        raise ValueError('One actual preceding owned GPU handle required')
    entry = selected[0]
    if not any((previous/entry['arm']/name).exists() for name in ('complete.json','failure.json')):
        return False
    proc = Path('/proc')/str(entry['pid'])
    try:
        state = (proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        argv = (proc/'cmdline').read_bytes().decode().split(chr(0))
        return not (state != 'Z' and str(previous) in argv and d.PREDECESSOR_MODULE in argv)
    except FileNotFoundError:
        return True


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state = capacity()[d.GPUS[args.arm]]
            if state['free_mb'] >= 18000 and len(state['pids']) < 2:
                value = torch.zeros(1,device='cuda:0')
                torch.cuda.synchronize()
                return value
        time.sleep(5)


@torch.no_grad()
def smoke(args, model, reader, pipeline):
    cache = args.parent_action/'cache/train'
    roles = np.load(cache/'roles.npy',allow_pickle=False)
    take = np.flatnonzero(roles == 'L_s')[:2]
    if len(take) != 2:
        raise ValueError('Two actual legal Ls examples required')
    values = array_tensor(np.load(cache/'clean.npy',mmap_mode='r',allow_pickle=False)[take],'cuda:0')
    before = [{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in (model,reader)]
    direct = FrozenPipeline(model,reader)
    records = {}
    for batch in (1,8,32,64,256):
        x = values.repeat((batch+1)//2,1,1)[:batch]
        other = values.flip(0).repeat((batch+1)//2,1,1)[:batch]
        native = direct(x)
        first = pipeline(x)
        held = tuple(v.clone() for v in first)
        second = pipeline(other)
        desired = direct(other)
        for old,stable in zip(first,held):
            if not torch.equal(old,stable):
                raise ValueError('Returned output overwritten by a later replay')
        for a,b in (*zip(native,first),*zip(desired,second)):
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Nonfinite actual-source smoke output')
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        records[str(batch)] = dict(max_abs=[float((a-b).abs().max()) for a,b in zip(native,first)],
                                  bitwise=[torch.equal(a,b) for a,b in zip(native,first)],input_replaced=True,outputs_owned=True)
    for module,snapshot in zip((model,reader),before):
        if any(p.grad is not None or p.requires_grad for p in module.parameters()):
            raise ValueError('Frozen inference only')
        if any(not torch.equal(v.cpu(),snapshot[k]) for k,v in module.state_dict().items()):
            raise ValueError('Source checkpoint or default head changed')
    write_json(args.output/args.arm/'smoke.json',dict(status='PASS',real_Ls_IQ=2,batches=records,
        TX_labels_read=0,target_reads=0,optimizer_updates=0,F_parameter_updates=0,head_fit_updates=0,
        source_checkpoint_weights_unchanged=True,native_fallback=False))
    print(json.dumps(dict(event='REAL_LS_GRAPH_SMOKE_PASS',arm=args.arm,batches=list(records))),flush=True)


@torch.no_grad()
def profile(args, current, reader, current_graph):
    values = array_tensor(np.load(args.parent_action/'cache/validation/practical_low_urban.npy',mmap_mode='r',allow_pickle=False)[:64],'cuda:0')
    parent = current if args.arm == d.ARMS[0] else load_model(args,d.ARMS[0],'cuda:0')[0]
    models = {'parent_R46_350521':parent}
    if args.arm != d.ARMS[0]:
        models['candidate'] = current
    calls,graphs,results = {},{},{}
    for name,model in models.items():
        results[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
            ConvLinear_MAC_per_IQ=convolution_linear_macs(model,values[:1]),cases={})
        for mode in ('G_only','G_plus_F'):
            fn = FrozenPipeline(model,reader if mode == 'G_plus_F' else None)
            graph = current_graph if model is current and mode == 'G_plus_F' else GraphCache(FrozenPipeline(model,reader if mode == 'G_plus_F' else None,graph_safe=True))
            calls[name,mode],graphs[name,mode] = fn,graph
            for batch in (1,32):
                native,got = fn(values[:batch]),graph(values[:batch])
                for a,b in zip(native,got):
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    for mode in ('G_only','G_plus_F'):
        for batch,warm,iterations in ((1,16,64),(32,4,16)):
            keys = [(name,execution) for name in models for execution in ('native','graph')]
            samples,rounds,peaks,resident = ({k:[] for k in keys} for _ in range(4))
            for repeat in range(3):
                for name,execution in (keys if repeat%2 == 0 else list(reversed(keys))):
                    fn = calls[name,mode] if execution == 'native' else graphs[name,mode]
                    for _ in range(warm):
                        fn(values[:batch])
                    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
                    before = torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
                    for index in range(iterations):
                        x = values[(index*batch)%(65-batch):][:batch]
                        torch.cuda.synchronize();started=time.perf_counter();y=fn(x);torch.cuda.synchronize()
                        times.append((time.perf_counter()-started)*1000)
                        if any(not torch.isfinite(v).all() for v in y):
                            raise ValueError('Nonfinite source timing output')
                        del y
                    key = name,execution
                    samples[key].extend(times);rounds[key].append(float(np.median(times)))
                    peaks[key].append(torch.cuda.max_memory_allocated());resident[key].append(before)
            for name,execution in keys:
                key = name,execution;times=np.asarray(samples[key])
                results[name]['cases'][mode+'_B'+str(batch)+'_'+execution] = dict(batch=batch,warmup_per_round=warm,
                    iterations_per_round=iterations,rounds=3,source_V_unique_IQ=64,median_ms=float(np.median(times)),
                    p90_ms=float(np.quantile(times,.9)),p95_ms=float(np.quantile(times,.95)),round_median_ms=rounds[key],
                    throughput_samples_per_second=float(batch*1000/times.mean()),resident_before_forward_bytes=max(resident[key]),
                    peak_allocated_bytes=max(peaks[key]),incremental_peak_bytes=max(p-r for p,r in zip(peaks[key],resident[key])))
    write_json(args.output/args.arm/'matched_inference_profile.json',dict(status='PROFILE_COMPLETE',arm=args.arm,
        physical_GPU=d.GPUS[args.arm],results=results,setup={name+'_'+mode:g.setup() for (name,mode),g in graphs.items()},
        TX_labels_read=0,target_reads=0,state_updates=0,training_F_access=False,
        memory_scope='Parent(if different)+candidate+one F and all capture pools resident for every timed case;standalone deployed graph memory recorded separately',
        timing_scope='Synchronized preloaded-device FP32;graph includes input copy and owned output clones;setup excluded and reported',
        MAC_scope='Conv1d+Linear only;FFT/complexCG/norm/activation/memory/F excluded',includes_full_context_3prior_6CG=True))


def score_subprocess(args, role):
    import fcntl
    with (args.output.parent/d.SCORING_RUN/'score.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        subprocess.run(command(args,'score_'+role,args.arm),cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)


def worker(args):
    row = args.output/args.arm
    try:
        acquire(args)
        row.mkdir();(row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
        write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
        model,payload = load_model(args,args.arm,'cuda:0')
        reader = load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
        before = torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        pipeline = GraphCache(FrozenPipeline(model,reader,graph_safe=True))
        smoke(args,model,reader,pipeline)
        gc.collect();torch.cuda.synchronize()
        write_json(row/'standalone_graph_memory.json',dict(scope='One G+F plus captured batches1/8/32/64/256;no optimizer;no benchmark control model',
            before_capture_allocated_bytes=before,after_capture_allocated_bytes=torch.cuda.memory_allocated(),
            after_capture_reserved_bytes=torch.cuda.memory_reserved(),capture_peak_allocated_bytes=torch.cuda.max_memory_allocated(),setup=pipeline.setup()))
        write_json(row/'runtime_ready.json',dict(status='FROZEN_RUNTIME_READY',arm=args.arm,parameters=d.PARAMETERS[args.arm],
            new_training_updates=0,F_parameter_updates=0,head_fit_updates=0))
        write_json(row/'checkpoint_source.json',dict(source_checkpoint=str(source_checkpoint(args,args.arm)),
            source_arm=payload['arm'],epoch=payload['epoch'],updates=payload['updates'],parameters=d.PARAMETERS[args.arm],
            checkpoint_bytes=source_checkpoint(args,args.arm).stat().st_size,new_training_updates=0,training_F_access=False))
        del payload
        predict(args,'source',pipeline);profile(args,model,reader,pipeline);gc.collect();torch.cuda.empty_cache()
        score_subprocess(args,'source')
        write_json(row/'source_complete.json',dict(status='ANALYZED',arm=args.arm,diagnostic_only=True,source_selection=False,target_feedback=False))
        predict(args,'target',pipeline);score_subprocess(args,'target')
        write_json(row/'complete.json',dict(status='ANALYZED',arm=args.arm,all_predictions_before_truth=True,target_feedback=False))
    except BaseException as error:
        row.mkdir(exist_ok=True)
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def write_status(path,value):
    tmp = path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(path)


def run(args):
    if args.output.name != d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Immutable new root and CPU controller required')
    validate_sources(args)
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/R46-CUDAgraph'))
    write_json(args.output/'comparison_plan.json',d.configuration())
    write_json(args.output/'resolved_config.json',d.configuration())
    waiting=list(d.ARMS);active={};launches=[];failed=[]
    try:
        write_status(args.output/'workers.json',dict(workers=[]))
        while waiting or active:
            for arm in waiting[:]:
                gpu=d.GPUS[arm]
                if len(active)>=2 or any(d.GPUS[a]==gpu for a in active) or not predecessor_ready(args,gpu):
                    continue
                argv=command(args,'worker',arm);log=args.output/'logs'/('worker-'+arm+'.log')
                with log.open('xb') as stream:
                    p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=p;waiting.remove(arm)
                launches.append(dict(arm=arm,pid=p.pid,GPU=gpu,cwd=str(ROOT),argv=argv,log=str(log)))
                write_status(args.output/'workers.json',dict(workers=launches))
            for arm,p in list(active.items()):
                if p.poll() is None:
                    continue
                if p.returncode or not (args.output/arm/'complete.json').exists():
                    failed.append(dict(arm=arm,exit_code=p.returncode))
                del active[arm]
            write_status(args.output/'stages.json',dict(waiting=waiting,active={a:p.pid for a,p in active.items()},failed=failed))
            if waiting or active:
                time.sleep(2)
        if failed:
            raise RuntimeError('Technical failures preserved; healthy rows completed:'+str(failed))
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_rows=list(d.ARMS),target_rows=list(d.ARMS),target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise


def main():
    parser=argparse.ArgumentParser(description='Fixed-function CUDA graph execution for original R46 and both R18 restorers')
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'))
    parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:
        parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    for key in PATH_ARGS:
        setattr(args,key,getattr(args,key).resolve())
    if args.output.name != d.RUN or code_commit() != args.expected_commit:
        raise ValueError('Pinned code and immutable graph run required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode == 'run':
        run(args)
    elif args.arm is None:
        parser.error('Registered row required')
    elif args.mode.startswith('score_'):
        score(args,args.mode[6:])
    else:
        worker(args)


if __name__ == '__main__':
    main()
