import argparse,gc,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .runtime import ROOT,convert_source,save_checkpoint,load_checkpoint,validate_sources,source_checkpoint
from .evaluation import predict,score
from experiments.pamf_r46_cuda_graph.graph import FrozenPipeline
from experiments.pamf_r46_complex_fidelity.runtime import load_checkpoint as load_source
from experiments.pamf_r44_pruned_refine.runtime import load_checkpoint as load_parent
from experiments.pamf_r17_task.runtime import load_original
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.runner import read_json,code_commit
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity

PATH_ARGS=('teacher_run','compact_parent','complex_run','parent_action','source_cache','reader_folder','formal_reference','formal_f50','target_truth','output')

def command(args,mode,arm):
    argv=[sys.executable,'-u','-m','experiments.pamf_r46_prior_collapse.runner',mode,'--arm',arm]
    for key in PATH_ARGS:argv.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    return argv+['--expected-commit',args.expected_commit]


def predecessor_ready(args,arm):
    gpu=d.GPUS[arm];previous=args.output.parent/d.PREDECESSOR_BY_GPU[gpu];saved=previous/'workers.json'
    if not saved.exists():return False
    entries=[entry for entry in read_json(saved)['workers'] if entry['GPU']==gpu]
    if not entries:return False
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
        ('lowrank_R22',load_source,args.complex_run/d.SOURCE_ROWS[args.arm]/d.KIND/'final.pt'),
        ('compact',load_checkpoint,args.output/args.arm/d.KIND/'final.pt')):
        model,payload=loader(path,'cpu')
        if (payload['epoch'],payload['updates'])!=((d.EPOCHS,d.UPDATES) if name in ('compact','lowrank_R22') else (20,4940)):raise ValueError('Fixed source final profile checkpoints required')
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
        memory_scope='All4frozenGand1Fresident in every case;incrementalpeak perforward is comparable;resident is shared suite,not standalone deploymentmemory',
        MAC_scope='Conv1d+Linear only;FFT/complexCG/norm/activation/memory excluded;F MAC excluded',
        timing_scope='Synchronized preloaded-device FP32 eval;excludes transfer,preprocess,loader;alternating forward/reverse candidate order perround',
        whole_pipeline='G_plus_F includes original frozen F default head and internal FFT/PA/time/stats;no new head'))
    del models,reader,values;gc.collect();torch.cuda.empty_cache()

def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=18000 and len(state['pids'])<2:
                value=torch.zeros(1,device='cuda:0');torch.cuda.synchronize();return value
        time.sleep(5)


def prepare(args):
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    shutil.copy2(args.output/'exports'/(args.arm+'.pt'),row/d.KIND/'final.pt')
    model,value=load_checkpoint(row/d.KIND/'final.pt','cpu');del model
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'runtime_ready.json',dict(status='FROZEN_RUNTIME_READY',arm=args.arm,parameters=d.PARAMETERS[args.arm],new_training_updates=0,F_parameter_updates=0,head_fit_updates=0))
    write_json(row/'checkpoint_source.json',dict(source_checkpoint=str(source_checkpoint(args,args.arm)),
        source_run=d.COMPLEX_RUN,source_row=d.SOURCE_ROWS[args.arm],source_commit=d.COMPLEX_COMMIT,
        inherited_epoch=20,inherited_updates=19700,new_training_updates=0,source_parameters=278180,parameters=299297))
    write_json(row/'collapse.json',value['collapse'])


def prediction_pipeline(args):
    model,_=load_checkpoint(args.output/args.arm/d.KIND/'final.pt','cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
    return FrozenPipeline(model,reader)


@torch.no_grad()
def smoke(args):
    cache=args.parent_action/'cache';roles=np.load(cache/'train/roles.npy',allow_pickle=False);take=np.flatnonzero(roles=='L_s')[:2]
    if len(take)!=2:raise ValueError('Actual two legal source L_s required')
    panel={view:array_tensor(np.load(cache/path,mmap_mode='r',allow_pickle=False)[take],'cpu')
        for view,path in (('clean','train/clean.npy'),('received01','epochs/01/received.npy'))}
    reader=load_reader(args.reader_folder,args.source_cache,'cpu',seed=d.F_SEED).eval().requires_grad_(False)
    reader_before={k:v.clone() for k,v in reader.state_dict().items()};records={}
    for arm in d.ARMS:
        model,source,payload,audit=convert_source(args,arm);before={k:v.clone() for k,v in source.state_dict().items()}
        save_checkpoint(args.output/'exports'/(arm+'.pt'),model,arm,payload,audit)
        fresh,_=load_checkpoint(args.output/'exports'/(arm+'.pt'),'cpu');views={}
        for view,x in panel.items():
            native=FrozenPipeline(source,reader)(x);converted=FrozenPipeline(model,reader)(x);reloaded=FrozenPipeline(fresh,reader)(x)
            from .verification import verify_outputs
            views[view]=verify_outputs(native,converted,reloaded)
        if any(not torch.equal(v,before[k]) for k,v in source.state_dict().items()):raise ValueError('Original frozen source weights changed')
        records[arm]=dict(parameters=299297,source_parameters=278180,conversion=audit,views=views,
            strict_reload_exact=True,source_unchanged=True,new_training_updates=0,training_F_access=False)
    if any(not torch.equal(v,reader_before[k]) for k,v in reader.state_dict().items()):raise ValueError('Original F50/default head changed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_L_s_IQ=2,source_views=list(panel),arms=records,
        TX_labels_read=0,V_reads=0,target_reads=0,new_training_updates=0,F_parameter_updates=0,head_fit_updates=0))


def worker(args):
    row=args.output/args.arm
    if row.exists():raise ValueError('Existing row remains unchanged')
    if read_json(args.output/'launch.json')['commit']!=args.expected_commit or read_json(args.output/'comparison_plan.json')!=d.configuration():raise ValueError('Fixed launched collapse matrix required')
    try:
        reservation=acquire(args);prepare(args);pipeline=prediction_pipeline(args)
        predict(args,'source',pipeline);del pipeline;gc.collect();torch.cuda.empty_cache();profile(args)
        score_subprocess(args,'source')
        write_json(row/'source_complete.json',dict(status='ANALYZED',arm=args.arm,diagnostic_only=True,source_selection=False,target_feedback=False))
        pipeline=prediction_pipeline(args);predict(args,'target',pipeline);del pipeline;gc.collect();torch.cuda.empty_cache()
        score_subprocess(args,'target')
        write_json(row/'complete.json',dict(status='ANALYZED',arm=args.arm,all_predictions_before_truth=True,target_feedback=False,new_training_updates=0))
    except BaseException as error:
        row.mkdir(exist_ok=True);write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise


def write_status(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(path)


def run(args):
    if args.output.name!=d.RUN or args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Immutable new root and CPU controller required')
    validate_sources(args);args.output.mkdir();(args.output/'logs').mkdir();(args.output/'exports').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/R46-prior-collapse',
        compact_parent_run=d.COMPACT_PARENT_RUN,compact_parent_commit=d.COMPACT_PARENT_COMMIT))
    write_json(args.output/'resolved_config.json',d.configuration());write_json(args.output/'comparison_plan.json',d.configuration())
    try:
        smoke(args);print(json.dumps(dict(event='R46_PRIOR_COLLAPSE_SMOKE_PASS',arms=list(d.ARMS))),flush=True)
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
    parser=argparse.ArgumentParser(description='Frozen prior factor-pair collapse with full seven-scene evaluation',allow_abbrev=False)
    parser.add_argument('mode',choices=('run','worker','score_source','score_target'));parser.add_argument('--arm',choices=d.ARMS)
    for key in PATH_ARGS:parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);args=parser.parse_args()
    for key in PATH_ARGS:setattr(args,key,getattr(args,key).resolve())
    if args.output.name!=d.RUN or code_commit()!=args.expected_commit:raise ValueError('Pinned frozen collapse run/root required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='run':run(args)
    elif args.arm is None:parser.error('Fixed collapse arm required')
    elif args.mode.startswith('score_'):score(args,args.mode[6:])
    else:worker(args)


if __name__=='__main__':main()
