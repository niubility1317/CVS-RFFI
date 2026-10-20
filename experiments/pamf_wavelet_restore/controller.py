import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from . import design as d
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint
from .models import WaveletRestorer
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from .evaluation import select_source,validate_references,validate_control
from .training import iq_loss
from experiments.pamf_native_formal.runner import parent_contract
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress

PATH_ARGS=('preceding_run','alternating_run','unrolled_run','single_run','fft_run','source_cache','parent_run','parent_action','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,row):
    value=[sys.executable,'-u','-m','experiments.pamf_wavelet_restore.runner',mode,'--row',row]
    for key in (*PATH_ARGS,'expected_commit'):value+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return value


def require_preceding_finished(args,proc_root=Path('/proc')):
    root=args.preceding_run
    validate_control(root)
    records=[read_json(root/'launch.json'),*read_json(root/'workers.json')['rows']]
    for record in records:
        process=proc_root/str(record['pid'])
        if not process.is_dir():continue
        status=(process/'status').read_text()
        if any(line.startswith('State:') and 'Z (' in line for line in status.splitlines()):continue
        cmd=(process/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
        if str(root) in cmd and 'experiments.pamf_joint_coarse_restore.runner' in cmd:
            raise ValueError('Original R23 process still live;keep R29 unscheduled')


def smoke(args):
    if args.parent_run.name!=d.REFERENCE or not parent_contract(args.parent_run)['qualified']:
        raise ValueError('Fixed source-only native reference required')
    validate_references(args.single_run,args.fft_run,args.unrolled_run,args.alternating_run)
    validate_control(args.preceding_run)
    from experiments.pamf_joint_coarse_restore.runtime import load_checkpoint as load_control
    control,payload=load_control(args.preceding_run/d.CONTROL_CANDIDATE/d.KIND/'final.pt','cpu')
    if payload['candidate']!=d.CONTROL_CANDIDATE or (payload['epoch'],payload['updates'])!=(20,4940):
        raise ValueError('Actual whole source control checkpoint contract differs')
    del control
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    r=x+torch.linspace(-.01,.01,256)[None,None,:]
    with torch.no_grad():reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    initial_context=None;initial_output=None;initial_full=None;gradients={};parameters={}
    for candidate in d.ROWS:
        model,parent=initialize(candidate,args.parent_action,'cpu');model.train();state=model.frozen.state_dict()
        full=model.state_dict()
        if initial_full is None:initial_full={key:value.clone() for key,value in full.items()}
        elif set(full)!=set(initial_full) or any(not torch.equal(value,initial_full[key]) for key,value in full.items()):
            raise ValueError('Matched complete wavelet initial states required')
        if initial_context is None:
            initial_context={key:value.clone() for key,value in state.items()}
            with torch.no_grad():initial_output=UnrolledRestorer('unroll_unshared',model.frozen).eval()(r).clone()
            model.train()
        elif set(state)!=set(initial_context) or any(not torch.equal(value,initial_context[key]) for key,value in state.items()):
            raise ValueError('Matched original frozen R07 context required')
        stages,*_=model.stage_outputs(r);output=stages[-1]
        if not torch.equal(output.detach(),initial_output):raise ValueError('All zero-head complete initial outputs must match')
        for stage in stages:stage.retain_grad()
        loss,_=iq_loss(output,x);loss.mean().backward();norms=[]
        for core in model.prior:
            gs=[p.grad for p in core.parameters() if p.grad is not None];norm=sum(float(g.norm()) for g in gs)
            if not gs or any(not torch.isfinite(g).all() for g in gs) or norm<=0:raise ValueError('Every prior step needs finite gradient')
            norms.append(norm)
        if len(stages)!=3 or any(s.grad is None or not torch.isfinite(s.grad).all() or s.grad.norm()<=0 for s in stages):
            raise ValueError('All three deployment stages need finite gradient')
        if any(p.grad is not None or p.requires_grad for p in model.frozen.parameters()):raise ValueError('All source context must remain frozen')
        prior=list(model.prior.parameters())
        if {id(p) for p in prior}!={id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Only scratch priors may train')
        parameters[candidate]=dict(prior=sum(p.numel() for p in prior),total=sum(p.numel() for p in model.parameters()),
            position_buffer_elements=sum(v.numel() for key,v in model.named_buffers() if key.endswith('position')))
        path=args.output/('smoke-'+candidate+'.pt');save_checkpoint(path,model,None,0,0,parent);fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.allclose(output,fresh(r),atol=1e-6,rtol=1e-5) or not torch.isfinite(reader(output.detach())).all():
                raise ValueError('Actual checkpoint and original defaulthead smoke failed')
        gradients[candidate]=norms
    if len({value['prior'] for value in parameters.values()})!=1:raise ValueError('Matched wavelet prior capacity required')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,rows=list(d.ROWS),
        frozen_context_states_equal=True,full_initial_outputs_equal=True,hidden_states_equal=True,steps=3,CG_per_step=2,
        all_step_gradients=gradients,parameters=parameters,reused_control_checkpoint='strict loaded;not initialization',
        frozen_context_gradient=False,TX_labels_read=0,target_reads=0,optimizer_updates=0,F_training=False,head_fit_updates=0))


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
                    if process.poll() is None and process.pid not in capacity()[gpu]['pids']:
                        raise RuntimeError('Tracked worker not visible in CUDA within45s;holdnewlaunches,preserveprocess')
        progress(args.output/'queue_state.json',dict(mode=mode,active=[r for _,r in active.values()],pending=pending,completed=completed,failed=failed,time=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Technical failure;healthy rows preserved,pending held,noautomaticretry:'+str(failed))


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable output and CPUcontroller required')
    require_preceding_finished(args)
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,owner='codex/root/wavelet-restoration',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);launches=[];queue(args,d.ROWS,'train',launches)
        result=select_source({name:read_json(args.output/name/d.KIND/'summary.json') for name in d.ROWS},
            read_json(args.single_run/'unet_control'/d.KIND/'summary.json'),read_json(args.parent_run/d.KIND/'summary.json'),
            read_json(args.fft_run/'unet_control'/d.KIND/'summary.json'),read_json(args.unrolled_run/d.REFERENCE_CANDIDATE/d.KIND/'summary.json'),
            read_json(args.alternating_run/d.ALTERNATING_CANDIDATE/d.KIND/'summary.json'),validate_control(args.preceding_run))
        write_json(args.output/'source_decision.json',result);queue(args,result['target_rows'],'target',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=result['qualified'],source_rows=list(d.ROWS),target_rows=result['target_rows'],
            training_TX_F_access=False,source_only=not result['qualified'],target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False));raise
