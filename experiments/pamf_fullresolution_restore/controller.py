import os
import subprocess
import sys
import time
import traceback
import torch
from . import design as d
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint
from .evaluation import qualify_source
from experiments.pamf_native_formal.runner import parent_contract
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress

PATH_ARGS=('source_cache','parent_run','parent_action','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,row):
    value=[sys.executable,'-u','-m','experiments.pamf_fullresolution_restore.runner',mode,'--row',row]
    for key in (*PATH_ARGS,'expected_commit'):value+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return value


def smoke(args):
    if args.parent_run.name!=d.REFERENCE or not parent_contract(args.parent_run)['qualified']:raise ValueError('Fixed source-only native reference required')
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');r=x+torch.linspace(-.01,.01,256)[None,None,:]
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval();outputs=[]
    for candidate in d.ROWS:
        model,parent=initialize(candidate,args.parent_action,'cpu');model.eval();output=model(r);outputs.append(output.detach())
        (output-x).square().mean().backward();grad=[p.grad for p in model.prior.parameters() if p.grad is not None]
        if not grad or any(not torch.isfinite(g).all() for g in grad) or sum(float(g.norm()) for g in grad)<=0:raise ValueError('Paired gradient must reach new prior')
        if any(p.grad is not None for p in model.frozen.parameters()):raise ValueError('Frozen parents received gradients')
        path=args.output/('smoke-'+candidate+'.pt');save_checkpoint(path,model,None,0,0,parent);fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.allclose(output,fresh(r),atol=1e-6,rtol=1e-5) or not torch.isfinite(reader(output.detach())).all():raise ValueError('Actual checkpoint/defaulthead source smoke failed')
    if not torch.equal(*outputs):raise ValueError('Zero-output heads must yield same initial full restoration')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,rows=list(d.ROWS),initial_full_outputs_equal=True,
        scratch_prior_gradient=True,frozen_context_gradient=False,TX_labels_read=0,target_reads=0,optimizer_updates=0,F_training=False,head_fit_updates=0))


def queue(args,names,mode,launches):
    import fcntl
    pending=list(names);active={};failed=[];completed=[]
    while pending or active:
        for name,(process,record) in list(active.items()):
            if process.poll() is None:continue
            required=args.output/name/('target_complete.json' if mode=='target' else 'complete.json')
            if process.returncode or not required.is_file():failed.append(dict(row=name,exit_code=process.returncode))
            else:completed.append(name)
            del active[name]
        if pending and not failed and len(active)<2:
            with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity();reserved={item['GPU'] for _,item in active.values()}
                cards=[gpu for gpu in d.GPUS if gpu not in reserved and state[gpu]['free_mb']>=12000 and len(state[gpu]['pids'])<2]
                cards.sort(key=lambda g:(len(state[g]['pids']),-state[g]['free_mb'],list(d.GPUS).index(g)))
                if cards:
                    gpu=cards[0];name=pending.pop(0);cmd=command(args,mode,name);log=args.output/'logs'/(mode+'-'+name+'.log')
                    with log.open('x',encoding='utf-8') as stream:
                        process=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),start_new_session=True)
                    record=dict(row=name,mode=mode,pid=process.pid,GPU=gpu,cwd=str(ROOT),argv=cmd,log=str(log),started=time.time())
                    launches.append(record);active[name]=(process,record);progress(args.output/'workers.json',dict(rows=launches));deadline=time.monotonic()+45
                    while process.poll() is None and time.monotonic()<deadline:
                        if process.pid in capacity()[gpu]['pids']:break
                        time.sleep(.5)
        progress(args.output/'queue_state.json',dict(mode=mode,active=[r for _,r in active.values()],pending=pending,completed=completed,failed=failed,time=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Technical failure;healthy rows preserved,pending held,noautomaticretry:'+str(failed))


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable output and CPUcontroller required')
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/fullresolution-restoration',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);launches=[];queue(args,d.ROWS,'train',launches)
        result=qualify_source(read_json(args.output/'unet_control'/d.KIND/'summary.json'),read_json(args.output/'tcn_prior'/d.KIND/'summary.json'),read_json(args.parent_run/d.KIND/'summary.json'))
        write_json(args.output/'source_decision.json',result);queue(args,result['target_rows'],'target',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=result['qualified'],source_rows=list(d.ROWS),target_rows=result['target_rows'],
            training_TX_F_access=False,source_only=not result['qualified'],target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise
