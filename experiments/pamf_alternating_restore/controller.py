import os
import subprocess
import sys
import time
import traceback
import torch
from . import design as d,models
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint
from .evaluation import select_source,validate_references
from experiments.pamf_unrolled_restore.runtime import load_checkpoint as load_reference
from unittest.mock import patch
from experiments.pamf_native_formal.runner import parent_contract
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress

PATH_ARGS=('unrolled_run','single_run','fft_run','source_cache','parent_run','parent_action','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,row):
    value=[sys.executable,'-u','-m','experiments.pamf_alternating_restore.runner',mode,'--row',row]
    for key in (*PATH_ARGS,'expected_commit'):value+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return value


def smoke(args):
    if args.parent_run.name!=d.REFERENCE or not parent_contract(args.parent_run)['qualified']:
        raise ValueError('Fixed source-only native reference required')
    validate_references(args.single_run,args.fft_run,args.unrolled_run)
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    r=x+torch.linspace(-.01,.01,256)[None,None,:]
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    control,payload=load_reference(args.unrolled_run/d.REFERENCE_CANDIDATE/d.KIND/'final.pt','cpu')
    if payload['candidate']!=d.REFERENCE_CANDIDATE or payload['epoch']!=20 or payload['updates']!=4940:
        raise ValueError('Fixed original R17 source checkpoint required')
    with torch.no_grad():
        if not torch.isfinite(reader(control(r))).all():raise ValueError('Original R17/defaulthead smoke failed')
    del control
    initial,meta=load_reference(args.unrolled_run/('smoke-'+d.REFERENCE_CANDIDATE+'.pt'),'cpu')
    if meta['epoch']!=0 or meta['updates']!=0:raise ValueError('Fixed R17 epoch0 comparison required')
    with torch.no_grad():initial_output=initial(r)
    initial_state=initial.state_dict();del initial
    new_initial=None;gradients={};counts={};parameters={}
    for candidate in d.ROWS:
        model,parent=initialize(candidate,args.parent_action,'cpu');model.eval();state=model.state_dict()
        for key,value in initial_state.items():
            parts=key.split('.')
            target='.'.join(parts[:2]+['temporal']+parts[2:]) if parts[0]=='prior' else key
            if not torch.equal(value,state[target]):raise ValueError('Original R17 temporal/frozen initial weights differ')
        if new_initial is None:new_initial={key:value.clone() for key,value in state.items()}
        elif set(state)!=set(new_initial) or any(not torch.equal(value,new_initial[key]) for key,value in state.items()):
            raise ValueError('Matched full initial weights required between new rows')
        calls=[];core_calls=[];solve=models.weighted_map
        def wrapped(*a,**kw):calls.append(kw['steps']);return solve(*a,**kw)
        handles=[core.register_forward_hook(lambda module,inputs,output:core_calls.append(1)) for stage in model.prior for core in (stage.temporal,stage.auxiliary)]
        try:
            with patch.object(models,'weighted_map',side_effect=wrapped):stages,_,_,_=model.stage_outputs(r)
        finally:
            for handle in handles:handle.remove()
        if calls!=[2,2,2] or len(core_calls)!=6 or len(stages)!=3:raise ValueError('Actual Core/CG calls differ')
        output=stages[-1]
        if not torch.allclose(output,initial_output,atol=1e-6,rtol=1e-5):raise ValueError('Full initial output must match R17 epoch0')
        (output-x).square().mean().backward();norms=[]
        for stage in model.prior:
            item={}
            for label,core in (('temporal',stage.temporal),('auxiliary',stage.auxiliary)):
                grad=[p.grad for p in core.parameters() if p.grad is not None];norm=sum(float(g.norm()) for g in grad)
                if not grad or any(not torch.isfinite(g).all() for g in grad) or norm<=0:raise ValueError('Every temporal and auxiliary Core requires finite nonzero gradient')
                item[label]=norm
            norms.append(item)
        if any(p.grad is not None for p in model.frozen.parameters()):raise ValueError('Frozen context received gradients')
        path=args.output/('smoke-'+candidate+'.pt');save_checkpoint(path,model,None,0,0,parent);fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.allclose(output,fresh(r),atol=1e-6,rtol=1e-5) or not torch.isfinite(reader(output.detach())).all():
                raise ValueError('Actual alternating/defaulthead checkpoint roundtrip failed')
        gradients[candidate]=norms;counts[candidate]=dict(Core_calls=len(core_calls),CG_calls=calls)
        parameters[candidate]=sum(p.numel() for p in model.prior.parameters())
    if len(set(parameters.values()))!=1:raise ValueError('Same trainable parameter capacity required')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,rows=list(d.ROWS),steps=3,
        actual_calls=counts,Core_calls_per_row=6,full_initial_temporal_state_matches_R17_epoch0=True,
        full_new_initial_states_equal=True,full_initial_outputs_match_R17=True,all_step_gradients=gradients,
        trainable_parameters=parameters,frozen_context_gradient=False,TX_labels_read=0,target_reads=0,
        optimizer_updates=0,F_training=False,head_fit_updates=0))


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
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/alternating-restoration',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);launches=[];queue(args,d.ROWS,'train',launches)
        result=select_source({name:read_json(args.output/name/d.KIND/'summary.json') for name in d.ROWS},read_json(args.single_run/'unet_control'/d.KIND/'summary.json'),read_json(args.parent_run/d.KIND/'summary.json'),read_json(args.fft_run/'unet_control'/d.KIND/'summary.json'),read_json(args.unrolled_run/d.REFERENCE_CANDIDATE/d.KIND/'summary.json'))
        write_json(args.output/'source_decision.json',result);queue(args,result['target_rows'],'target',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=result['qualified'],source_rows=list(d.ROWS),target_rows=result['target_rows'],
            training_TX_F_access=False,source_only=not result['qualified'],target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise
