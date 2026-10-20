import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from . import design as d
from .runtime import ROOT,initialize,save_checkpoint,load_checkpoint
from .models import teacher_response,training_delta,paired_objective,response_objective
from .evaluation import select_source,validate_reference
from experiments.pamf_restoration.models import pair
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_native_formal.runner import parent_contract
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,read_json
from experiments.pamf_restoration.dispatcher import capacity,progress

PATH_ARGS=('unrolled_run','source_cache','parent_run','parent_action','reader_folder','formal_reference','formal_f50','target_truth','output')


def command(args,mode,row):
    value=[sys.executable,'-u','-m','experiments.pamf_response_restore.runner',mode,'--row',row]
    for key in (*PATH_ARGS,'expected_commit'):value+=['--'+key.replace('_','-'),str(getattr(args,key))]
    return value


def smoke(args):
    if args.parent_run.name!=d.REFERENCE or not parent_contract(args.parent_run)['qualified']:
        raise ValueError('Fixed original source-only native reference required')
    validate_reference(args.unrolled_run)
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu');r=x+torch.linspace(-.01,.01,256)[None,None,:]
    ones=torch.ones((2,256),dtype=torch.complex64)
    known=Action(torch.ones((2,1),dtype=torch.complex64),ones,torch.zeros_like(ones));variance=torch.full((2,),1e-4)
    delta,valid,codes=training_delta(x,[0,1],1)
    change,target,eligible,_=teacher_response(known,delta,variance,valid)
    reader=load_reader(args.reader_folder,args.source_cache,'cpu').requires_grad_(False).eval()
    initial=None;first_output=None;gradients={};counts={}
    for name in d.ROWS:
        model,parent=initialize(name,args.parent_action,'cpu');model.train();state=model.state_dict()
        if initial is None:initial={key:value.clone() for key,value in state.items()}
        elif set(state)!=set(initial) or any(not torch.equal(value,initial[key]) for key,value in state.items()):
            raise ValueError('Identical complete initial states required across matched rows')
        output=model(r);plus=model(r+pair(change));minus=model(r-pair(change))
        if first_output is None:first_output=output.detach().clone()
        elif not torch.equal(output.detach(),first_output):raise ValueError('Matched initial deployed outputs differ')
        paired=paired_objective(output,plus,minus,x,delta)
        response,_=response_objective(plus,minus,target,delta,eligible)
        loss=(paired+(d.RESPONSE_WEIGHT*response if name==d.CANDIDATE else 0.)).mean();loss.backward()
        norms=[]
        for core in model.prior:
            gs=[p.grad for p in core.parameters() if p.grad is not None]
            norm=sum(float(g.norm()) for g in gs)
            if not gs or any(not torch.isfinite(g).all() for g in gs) or norm<=0:
                raise ValueError('Each of three priors requires finite nonzero gradient')
            norms.append(norm)
        if any(p.grad is not None or p.requires_grad for p in model.frozen.parameters()):
            raise ValueError('Frozen source context received training gradient')
        parameters=list(model.prior.parameters())
        if {id(p) for p in parameters}!={id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Only scratch prior parameters may train')
        path=args.output/('smoke-'+name+'.pt');save_checkpoint(path,model,None,0,0,parent)
        fresh,_=load_checkpoint(path,'cpu')
        with torch.no_grad():
            if not torch.allclose(output,fresh(r),atol=1e-6,rtol=1e-5) or not torch.isfinite(reader(output.detach())).all():
                raise ValueError('Actual full checkpoint/default-head no-query smoke failed')
        gradients[name]=norms;counts[name]=dict(prior=sum(p.numel() for p in parameters),total=sum(p.numel() for p in model.parameters()))
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,rows=list(d.ROWS),
        complete_initial_states_equal=True,initial_outputs_equal=True,all_step_gradients=gradients,
        parameters=counts,response_teacher_eligible=int(eligible.sum()),response_codes=codes.tolist(),
        TX_labels_read=0,target_reads=0,F_training=False,optimizer_updates=0,head_fit_updates=0))


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
                        raise RuntimeError('Tracked worker not CUDA-visible within45s;preserve process;noautomaticretry')
        progress(args.output/'queue_state.json',dict(mode=mode,active=[r for _,r in active.values()],pending=pending,completed=completed,failed=failed,time=time.time()))
        if failed and not active:break
        if pending or active:time.sleep(2)
    if failed:raise RuntimeError('Technical failure;healthy rows preserved,pending held,noautomaticretry:'+str(failed))


def run(args):
    if args.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('New immutable output and CPU controller required')
    args.output.mkdir();(args.output/'logs').mkdir()
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/observable-response-restoration',started=time.time()))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);launches=[];queue(args,d.ROWS,'train',launches)
        result=select_source({name:read_json(args.output/name/d.KIND/'summary.json') for name in d.ROWS},
            {name:read_json(args.output/name/'source_response/summary.json') for name in d.ROWS},validate_reference(args.unrolled_run))
        write_json(args.output/'source_decision.json',result);queue(args,result['target_rows'],'target',launches)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_qualified=result['qualified'],source_rows=list(d.ROWS),target_rows=result['target_rows'],
            training_TX_F_access=False,source_only=not result['qualified'],target_feedback=False,goal_complete=False))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True,automatic_retry=False))
        raise
