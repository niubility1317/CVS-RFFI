"""Single-owner capacity queue for an authorized, isolated source matrix.

Linux/N607 only. Existing GPU processes are counted and never stopped.
The queue never overwrites an existing run, retries, or selects by performance.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def scratch_checkpoint_smoke(row,release,output,gpu):
    """Real source input + fresh lineage checkpoint reload; never V/query."""
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    sys.path.insert(0,str(Path(release)/'code'))
    import torch
    from cvsrffi.game_tracking.config import parse_args
    from cvsrffi.game_tracking.runtime import build_model,json_write,make_grad_scaler
    from cvsrffi.game_tracking.solvers import GameSolver
    from cvsrffi.game_tracking.step_context import prepare_context,Core90Objective
    from cvsrffi.game_tracking.legacy.losses import PrototypeMemoryBank
    from cvsrffi.game_tracking.legacy.options import _loss_weights
    from cvsrffi.schedule import build_stage_state
    from cvsrffi.game_tracking.data import build_source
    args=parse_args(['--game_config_json',row['config_path'],'--output_dir',str(output),'--device','cuda:0'])
    source=build_source(args)
    model=build_model(args,len(source.domains),torch.device('cuda:0')).eval()
    x,y,d,meta=next(iter(source.loader('train',2)))
    output.mkdir(parents=True,exist_ok=False)
    path=output/'scratch_smoke.pth'
    torch.save(dict(model=model.state_dict(),initialization='scratch_only',target_contact=False,source_info=source.info),path)
    loaded=torch.load(path,map_location='cuda:0',weights_only=False)
    model.load_state_dict(loaded['model'])
    with torch.no_grad(): logits=model(x.cuda(),return_aux=True)['tx_logits']
    if logits.shape!=(2,args.num_classes) or not torch.isfinite(logits).all(): raise ValueError('Source checkpoint smoke failed')
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    scaler=make_grad_scaler(torch.device('cuda:0'),args.amp)
    proto=PrototypeMemoryBank(args.num_classes,len(source.domains));proto._lazy_init(160,torch.device('cuda:0'),torch.float32)
    ctx=prepare_context((x,y,d,meta),None,model,None,args,1,1,_loss_weights(args,build_stage_state(1,args)),torch.Generator(device='cuda:0').manual_seed(args.seed))
    solver=GameSolver(model,optimizer,'simultaneous',scaler=scaler,max_grad_norm=args.game_max_grad_norm)
    result=solver.step(lambda:Core90Objective(model,args,proto)(ctx))
    if not result.accepted: raise RuntimeError('Source optimizer smoke did not commit a finite update')
    json_write(output/'result.json',dict(status='PASS',source_role='L_s',query_access=False,checkpoint=str(path),initialization=loaded['initialization'],target_contact=loaded['target_contact'],logit_shape=list(logits.shape),torch_version=torch.__version__,optimizer_step_accepted=result.accepted))
    del model,source,loaded,x,logits
    torch.cuda.empty_cache()


def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={uuid.strip():int(index) for index,uuid in csv.reader(io.StringIO(rows))}
    counts={index:0 for index in mapping.values()}
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    seen=set()
    for uuid,pid in csv.reader(io.StringIO(processes)):
        key=(uuid.strip(),int(pid.strip()))
        if key not in seen and key[0] in mapping:
            counts[mapping[key[0]]]+=1;seen.add(key)
    return counts,seen,mapping


def free_memory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(index):int(memory) for index,memory in csv.reader(io.StringIO(rows))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix',required=True)
    p.add_argument('--release',required=True)
    p.add_argument('--status',required=True)
    p.add_argument('--poll-seconds',type=int,default=20)
    p.add_argument('--snapshot',action='store_true')
    p.add_argument('--smoke-gpu')
    p.add_argument('--max-processes-per-gpu',type=int,default=2)
    p.add_argument('--min-free-memory-mib',type=int,default=4096)
    p.add_argument('--startup-reserve-mib',type=int,default=3072)
    a=p.parse_args()
    if a.max_processes_per_gpu<1 or min(a.min_free_memory_mib,a.startup_reserve_mib)<0: raise ValueError('Invalid GPU capacity')
    if a.smoke_gpu is not None:
        row=json.loads(Path(a.matrix).read_text())['runs'][0]
        scratch_checkpoint_smoke(row,a.release,Path(a.status),a.smoke_gpu)
        return
    if a.snapshot:
        counts,_,_=inventory();print(json.dumps(counts));return
    import fcntl
    status=Path(a.status)
    status.parent.mkdir(parents=True,exist_ok=True)
    lock=status.with_suffix('.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if status.exists(): raise FileExistsError('Queue status exists; reconcile before restart')
    manifest=json.loads(Path(a.matrix).read_text())
    if not manifest.get('source_development_only') or manifest.get('target_evaluation'):
        raise ValueError('Only authorized source matrix is accepted')
    pending=list(manifest['runs']); active={}; completed=[]; failed=[]; blocked=False
    for row in pending:
        if Path(row['config']['output_dir']).exists():
            raise FileExistsError(row['config']['output_dir'])
    initial_counts,_,initial_mapping=inventory()
    memory=free_memory()
    available=next((gpu for gpu,count in sorted(initial_counts.items()) if count<a.max_processes_per_gpu and memory[gpu]>=a.min_free_memory_mib),None)
    if available is None: raise RuntimeError('No capacity for initial source checkpoint smoke')
    # Isolate CUDA context so the scheduler itself consumes no training slot.
    smoke_uuid=next(uuid for uuid,index in initial_mapping.items() if index==available)
    smoke=subprocess.run([sys.executable,__file__,'--matrix',a.matrix,'--release',a.release,'--status',str(status.parent/'scratch_smoke'),'--smoke-gpu',smoke_uuid],cwd=a.release)
    if smoke.returncode: raise RuntimeError('Initial source checkpoint smoke failed')
    def save():
        state=dict(owner_pid=os.getpid(),release=str(Path(a.release).resolve()),pending=[r['run_id'] for r in pending],
                   max_processes_per_gpu=a.max_processes_per_gpu,min_free_memory_mib=a.min_free_memory_mib,
                   active=[v['record'] for v in active.values()],completed=completed,failed=failed,
                   state='TECHNICAL_FAILURE_PENDING_INSPECTION' if blocked else ('RUNNING' if pending or active else 'COMPLETE'),updated_unix=time.time())
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(status)
    while pending or active:
        for pid,entry in list(active.items()):
            code=entry['process'].poll()
            if code is None: continue
            completion=Path(entry['record']['output_dir'])/'completion.json'
            verified=code==0 and completion.exists() and json.loads(completion.read_text()).get('status')=='SOURCE_ARTIFACTS_COMPLETE'
            record=dict(entry['record'],exit_code=code,artifact_verified=verified)
            (completed if verified else failed).append(record)
            entry['stream'].close();del active[pid]
        # Two failed pre-prediction rows are the registered technical stop
        # condition. Preserve pending rows and all artifacts; never kill peers.
        blocked=blocked or sum(not (Path(r['output_dir'])/'source_final_eval/source_prediction_manifest.json').exists() for r in failed)>=2
        if blocked:
            save()
            if not active: break
            time.sleep(a.poll_seconds)
            continue
        counts,seen,mapping=inventory()
        memory=free_memory()
        seen_pids={pid for _,pid in seen}
        # Reserve slots during Python startup before CUDA context appears.
        for pid,entry in active.items():
            if pid not in seen_pids:
                counts[entry['record']['gpu']]+=1
                memory[entry['record']['gpu']]-=a.startup_reserve_mib
        for gpu in sorted(counts):
            while counts[gpu]<a.max_processes_per_gpu and memory[gpu]>=a.min_free_memory_mib and pending:
                row=pending.pop(0)
                out=Path(row['config']['output_dir'])
                if out.exists(): raise FileExistsError('Run appeared during dispatch: '+str(out))
                log=status.parent/(row['run_id']+'.stdout.log')
                stream=log.open('x')
                uuid=next(uuid for uuid,index in mapping.items() if index==gpu)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                argv=[sys.executable,*row['argv'],'--device','cuda:0']
                process=subprocess.Popen(argv,cwd=a.release,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record=dict(run_id=row['run_id'],pid=process.pid,gpu=gpu,gpu_uuid=uuid,argv=argv,output_dir=str(out),log=str(log),started_unix=time.time())
                active[process.pid]=dict(process=process,record=record,stream=stream)
                counts[gpu]+=1;memory[gpu]-=a.startup_reserve_mib;save()
        save()
        if pending or active: time.sleep(a.poll_seconds)
    save()


if __name__=='__main__': main()
