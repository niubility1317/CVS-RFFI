import argparse,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import numpy as np
import torch
from . import design as d
from .models import initialize,feature_distance,posterior_distance,save_checkpoint,load_checkpoint
from .guidance import load_teachers,guidance
from .gradients import output_gradient
from experiments.pamf_change_preserve.runner import load_restorer
from .evaluation import predict,score
from experiments.pamf_quality_preserve.runner import validate_cache
from experiments.pamf_formal_restore.runner import source_residual_parent
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_reception_restore.runner import pair_loss
from experiments.pamf_reception_restore.data import route_array
from experiments.pamf_restoration.data import IQSource,write_json
from experiments.pamf_restoration.runner import array_tensor,tensor_array,code_commit
from experiments.pamf_restoration.dispatcher import capacity


def training_batch(clean,arrays,quantile,selected):
    target=np.asarray(clean[selected],np.float32);observed=target.copy()
    route=route_array('unet_mixed',quantile[selected])
    for key in ('residual','full'):
        mask=route==key;observed[mask]=arrays[key][selected[mask]]
    return np.concatenate((observed,target)),np.concatenate((target,target)),route


def learning_rate(update):
    if not 0<=update<d.UPDATES:raise ValueError('Registered update range required')
    if update<247:return 2e-4*(update+1)/247
    return 2e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/(d.UPDATES-248)))/2)


def empty_gradient_state():
    return dict(calls=0,IQ_examples=0,conflict=0,scaled=0,observed_scaled=0,clean_scaled=0,
        scale_sum=0.,scale_min=1.,scale_max=0.,observed_scale_sum=0.,clean_scale_sum=0.,
        dot_before_sum=0.,dot_after_sum=0.,wave_norm2_sum=0.,identity_norm2_sum=0.,identity_after_norm2_sum=0.)


def accumulate_gradients(state,stats,n):
    state['calls']+=1;state['IQ_examples']+=2*n
    for key in ('conflict','scaled'):state[key]+=int(stats[key].sum())
    state['observed_scaled']+=int(stats['scaled'][:n].sum());state['clean_scaled']+=int(stats['scaled'][n:].sum())
    state['scale_sum']+=float(stats['scale'].sum());state['scale_min']=min(state['scale_min'],float(stats['scale'].min()));state['scale_max']=max(state['scale_max'],float(stats['scale'].max()))
    state['observed_scale_sum']+=float(stats['scale'][:n].sum());state['clean_scale_sum']+=float(stats['scale'][n:].sum())
    for key in ('dot_before','dot_after','wave_norm2','identity_norm2','identity_after_norm2'):state[key+'_sum']+=float(stats[key].sum())


def source_smoke(args):
    teachers=load_teachers(args,'cpu');source=IQSource(args.source_cache,'L_s')
    indices=np.concatenate([np.flatnonzero(source.rx==rx)[:2] for rx in (1,3)])
    x=array_tensor(source.iq[indices],'cpu');y=torch.cat((.95*x,x))
    with torch.no_grad():references=torch.stack([reader.features(x) for reader in teachers.values()],1).repeat(2,1,1)
    before={seed:{k:v.detach().clone() for k,v in reader.state_dict().items()} for seed,reader in teachers.items()}
    records={};initial=None
    for arm in d.ARMS:
        torch.manual_seed(d.MODEL_SEED);model=initialize('cpu',arm)
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if initial is None:initial=state
        elif any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Matched random initialization required')
        with torch.no_grad():
            if not torch.equal(model(x),x):raise ValueError('Original zero-output scratch model must start at received IQ')
        model.train();estimate=model(y)
        if arm==d.ARMS[1]:
            combined,_,decision=guidance(teachers,estimate,references,arm)
            direct=torch.autograd.grad(decision.mean(),estimate,retain_graph=True)[0]
            if not torch.isfinite(direct).all() or float(direct.abs().sum())<=0:raise ValueError('Native defaulthead posterior IQ gradient required')
        else:combined=torch.zeros(len(y))
        paired,_=pair_loss(estimate,x.repeat(2,1,1));gradient,stats=output_gradient(estimate,paired,combined,arm);estimate.backward(gradient)
        values=[p.grad for p in model.temporal.out.parameters()]
        if not all(g is not None and torch.isfinite(g).all() for g in values) or sum(float(g.abs().sum()) for g in values)<=0:raise ValueError('Scratch zero-output temporal head must receive gradients')
        if any(not torch.equal(v,state[k]) for k,v in model.state_dict().items()):raise ValueError('No smoke parameter/statistic updates allowed')
        if any(p.grad is not None for reader in teachers.values() for p in reader.parameters()):raise ValueError('F must remain gradient-free')
        actual=empty_gradient_state();accumulate_gradients(actual,stats,len(x))
        path=args.output/(arm+'_smoke_checkpoint.pt');save_checkpoint(path,model,None,arm,0,0)
        fresh,_=load_checkpoint(path,arm,'cpu');model.eval()
        with torch.no_grad():
            if not torch.equal(model(y),fresh(y)):raise ValueError('Scratch checkpoint roundtrip failed')
        records[arm]=dict(initial_equals_received=True,matching_initial_state=True,zero_head_gradient=True,
            gradient_state=actual,trainable_parameters=sum(p.numel() for p in model.parameters()))
    if any(not torch.equal(v,before[seed][k]) for seed,reader in teachers.items() for k,v in reader.state_dict().items()):raise ValueError('Original F complete state changed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=len(x),arms=records,actual_F50=True,
        initialization='scratch',restoration_weights_loaded=0,frozen_F_no_grad=True,checkpoint_roundtrip=True,
        native_posterior_input_gradient=True,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration(args.arm)['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=18000 and len(state['pids'])<2:
                model=initialize('cuda:0',args.arm)
                torch.cuda.synchronize();return model
        time.sleep(5)


def train(args):
    worker_started=time.perf_counter()
    row=args.output/args.arm;roles=validate_cache(args.parent_cache)
    clean=np.load(args.parent_cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256):raise ValueError('Original source shape required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model=acquire(args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    initial_parameters={k:v.detach().clone() for k,v in model.named_parameters()}
    reader=reference_features=reader_before=None
    if args.arm==d.ARMS[1]:
        cache_started=time.perf_counter()
        reader=load_teachers(args,'cuda:0')
        reader_before={seed:{k:v.detach().clone() for k,v in model.state_dict().items()} for seed,model in reader.items()}
        reference_features=np.lib.format.open_memmap(row/'source_reference_features.npy',mode='w+',dtype=np.float32,shape=(63000,3,160))
        with torch.no_grad():
            for offset in range(0,len(clean),128):
                batch=array_tensor(clean[offset:offset+128],'cuda:0')
                features=torch.stack([model.features(batch) for model in reader.values()],1)
                if features.shape!=(min(128,len(clean)-offset),3,160) or not torch.isfinite(features).all():
                    raise ValueError('Finite source-only reference embeddings required')
                reference_features[offset:offset+len(features)]=tensor_array(features)
        reference_features.flush()
        write_json(row/'feature_cache_complete.json',dict(status='SOURCE_FEATURES_FIXED',count=63000,
            roles=['L_s','U_s'],F_updates=0,loss_uses_logits=True,F_seeds=list(d.TEACHER_SEEDS),TX_labels_read=0,V_reads=0,target_reads=0,
            seconds=time.perf_counter()-cache_started))
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    permutation=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;gradient_state=empty_gradient_state();started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            folder=args.parent_cache/'epochs'/f'{d.cache_epoch(epoch):02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('residual','full')}
            quantile=np.load(folder/'quantile.npy',allow_pickle=False)
            if quantile.shape!=(63000,) or any(a.shape!=clean.shape for a in arrays.values()):raise ValueError('Original epoch cache layout differs')
            order=permutation.permutation(len(clean));sums={k:0. for k in ('total','iq_nmse','log_power','feature','posterior','clean_iq_nmse','observed_iq_nmse')}
            routes={k:0 for k in ('clean','residual','full')};clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];n=len(selected)
                values,targets,route=training_batch(clean,arrays,quantile,selected)
                y,x=array_tensor(values,'cuda:0'),array_tensor(targets,'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                estimate=model(y);paired,terms=pair_loss(estimate,x)
                if reader is None:combined=perceptual=posterior=torch.zeros_like(paired)
                else:
                    reference=array_tensor(reference_features[selected],'cuda:0').repeat(2,1,1)
                    combined,perceptual,posterior=guidance(reader,estimate,reference,args.arm)
                loss=(paired+combined).mean();lr=learning_rate(updates)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement loss')
                for group in optimizer.param_groups:group['lr']=lr
                merged,stats=output_gradient(estimate,paired,combined,args.arm);estimate.backward(merged)
                accumulate_gradients(gradient_state,stats,n)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                values=dict(total=loss,iq_nmse=terms['iq_nmse'].mean(),log_power=terms['log_power'].mean(),
                    feature=perceptual.mean(),posterior=posterior.mean(),clean_iq_nmse=terms['iq_nmse'][n:].mean(),observed_iq_nmse=terms['iq_nmse'][:n].mean())
                for key,value in values.items():sums[key]+=float(value)*n
                for key in routes:routes[key]+=int((route==key).sum())
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,cache_epoch=d.cache_epoch(epoch),updates=updates,losses={k:v/len(clean) for k,v in sums.items()},route_counts=routes,
                L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
                learning_rate_last=lr,clip_fraction=clipped/247,
                gradient_state=dict(gradient_state),seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='REFINE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(d.UPDATES,6300*d.EPOCHS,56700*d.EPOCHS):raise ValueError('Full matched training budget required')
    for part in ('spectral.','temporal.'):
        if all(torch.equal(v,initial_parameters[k]) for k,v in model.named_parameters() if k.startswith(part)):raise ValueError('Joint trainable branch did not update')
    if reader is not None and (any(p.grad is not None for model in reader.values() for p in model.parameters()) or
        any(not torch.equal(v,reader_before[seed][k]) for seed,model in reader.items() for k,v in model.state_dict().items())):raise ValueError('Frozen F changed')
    if gradient_state['calls']!=d.UPDATES or gradient_state['IQ_examples']!=5040000:raise ValueError('Full merged-gradient exposure required')
    if not 0<=gradient_state['scale_min']<=gradient_state['scale_max']<=1:raise ValueError('Identity scale must stay bounded0..1')
    if gradient_state['scaled']!=gradient_state['observed_scaled']+gradient_state['clean_scaled']:raise ValueError('Actual paired cap counts required')
    save_checkpoint(row/'final.pt',model,optimizer,args.arm,d.EPOCHS,updates,gradient_state)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',epochs=d.EPOCHS,updates=updates,
        L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds_including_preparation_and_capacity_wait=time.perf_counter()-worker_started,
        gradient_state=gradient_state,gradient_epochs=d.EPOCHS,gradient_updates=updates,initialization='scratch',restoration_weights_loaded=0,restorer_epochs=d.EPOCHS,cache_epochs=[d.cache_epoch(e) for e in range(1,d.EPOCHS+1)],
        F_optimizer_updates=0,F_seeds=list(d.TEACHER_SEEDS),F_commit=d.F_COMMIT,guidance_enabled=args.arm==d.ARMS[1],teacher_training_calls=3*updates if reader is not None else 0,teacher_training_IQ_examples=3*gradient_state['IQ_examples'] if reader is not None else 0,head_fit_updates=0,TX_labels_read=0,target_reads=0))
    del model,optimizer,reader,initial_parameters,reader_before;torch.cuda.empty_cache()


def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPUS[args.arm]):raise ValueError('Registered per-arm GPU required')
    row=args.output/args.arm;row.mkdir();write_json(row/'resolved_config.json',d.configuration(args.arm))
    try:
        train(args);predict(args)
        import fcntl
        preceding=args.output.parent/d.PRECEDING_RUN
        while not any((preceding/name).exists() for name in ('complete.json','failure.json')):time.sleep(5)
        with (args.output/'score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            command=[sys.executable,'-X','utf8','-m','experiments.pamf_scratch_guidance.runner','score']
            for key in ('output','parent_cache','source_cache','arm'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'source_frozen.json',dict(status='SOURCE_FROZEN',arm=args.arm,epochs=d.EPOCHS,updates=d.UPDATES,
            source_only=True,target_reads=0,checkpoint_selection='fixed_final_E40',target_policy='FIXED_COMPARISON'))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',stage='source',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise

def launch_target(args,arm,logs):
    command=[sys.executable,'-u','-m','experiments.pamf_scratch_guidance.target','run',
        '--source-run',str(args.output),'--output',str(args.output/arm/'target'),'--arm',arm]
    for key in ('source_cache','formal_residual','reader_root','target_truth','expected_commit'):
        command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    with (logs/(arm+'_target.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(args.output/arm/'target_launch.json',dict(pid=process.pid,argv=command,GPU=d.GPUS[arm],arm=arm))
    return process


def write_stage_status(output,value):
    temporary=output/'stages.json.tmp'
    temporary.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(output/'stages.json')


def previous_GPU_rows_live(previous,proc_root=Path('/proc')):
    saved=previous/'workers.json';workers=json.loads(saved.read_text())['workers'] if saved.exists() else [];live=[]
    for worker in workers:
        entries=[worker];target=previous/worker['arm']/'target_launch.json'
        if target.exists():entries.append(json.loads(target.read_text()))
        for entry in entries:
            p=proc_root/str(entry['pid'])
            try:
                state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
                command=(p/'cmdline').read_bytes().decode().replace(chr(0),' ')
                if state!='Z' and str(previous) in command and 'experiments.pamf_teacher_agreement.' in command:live.append(entry['pid'])
            except FileNotFoundError:pass
    return live


def wait_for_GPU_predecessor(args):
    previous=args.output.parent/d.GPU_PRECEDING_RUN;last=-30.
    while True:
        terminal=any((previous/name).exists() for name in ('complete.json','failure.json'))
        live=previous_GPU_rows_live(previous)
        if terminal and not live:
            write_json(args.output/'GPU_predecessor_released.json',dict(status='PREVIOUS_OWNED_GPU_HANDLES_EXITED',run=d.GPU_PRECEDING_RUN,live_owned_PIDs=[],scores_read=0))
            return
        if time.monotonic()-last>=30:
            print(json.dumps(dict(event='WAIT_PREVIOUS_GPU_ROWS',run=d.GPU_PRECEDING_RUN,live_owned_PIDs=live,scores_read=0)),flush=True);last=time.monotonic()
        time.sleep(5)


def run(args):
    if args.output.exists():raise FileExistsError('Immutable new fixed-matrix run required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code/config commit required')
    metadata=source_residual_parent(args.parent_restoration)
    args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/scratch-guidance',initialization='scratch',restoration_weights_loaded=0,baseline_R03=metadata,fixed_target_arms=list(d.ARMS)))
    write_json(args.output/'matrix.json',d.target_matrix())
    try:source_smoke(args)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',stage='smoke',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
    wait_for_GPU_predecessor(args)
    logs=args.output/'logs';logs.mkdir();active={};launches=[];failed={};completed=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_scratch_guidance.runner','worker','--arm',arm]
        for key in ('output','parent_cache','source_cache','parent_restoration','reader_folder','reader_root'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/(arm+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[arm]=('source',process);launches.append(dict(arm=arm,pid=process.pid,argv=command,GPU=d.GPUS[arm]))
    write_json(args.output/'workers.json',dict(workers=launches))
    while active:
        for arm,(stage,process) in list(active.items()):
            code=process.poll()
            if code is None:continue
            if code!=0:
                failed[arm]=dict(stage=stage,exit_code=code);del active[arm]
            elif stage=='source':
                # poll() proves this source process has exited before the card is reused.
                active[arm]=('target',launch_target(args,arm,logs))
            else:
                write_json(args.output/arm/'complete.json',dict(status='ANALYZED',source_and_target_complete=True,arm=arm))
                completed.append(arm);del active[arm]
        write_stage_status(args.output,dict(active={arm:dict(stage=stage,pid=process.pid) for arm,(stage,process) in active.items()},
            completed=completed,failed=failed,no_target_feedback=True))
        if active:time.sleep(2)
    if failed:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',rows=failed,completed_rows=completed,healthy_rows_preserved=True,automatic_retry=False));return
    write_json(args.output/'complete.json',dict(status='ANALYZED',source_and_target_complete=True,arms=list(d.ARMS),
        target_policy='FIXED_COMPARISON',source_selection=False,target_feedback=False,epochs_per_arm=d.EPOCHS))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker','score'))
    for name in ('output','parent-cache','source-cache'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('parent-restoration','reader-folder'):p.add_argument('--'+name,type=Path)

    for key in ('formal-residual','reader-root','target-truth'):p.add_argument('--'+key,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for name in ('output','parent_cache','source_cache','parent_restoration','reader_folder','formal_residual','reader_root','target_truth'):
        if getattr(args,name,None) is not None:setattr(args,name,getattr(args,name).resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':
        if args.arm is None:p.error('One declared arm required')
        score(args)
    elif args.mode=='worker':
        if args.arm is None or args.parent_restoration is None or args.reader_folder is None:p.error('Actual frozen parent/reader and arm required')
        worker(args)
    else:
        if args.expected_commit is None or args.parent_restoration is None or args.reader_folder is None:p.error('Actual parents and code commit required')
        if any(getattr(args,k) is None for k in ('formal_residual','reader_root','target_truth')):p.error('Registered automatic target inputs required')
        run(args)


if __name__=='__main__':main()
