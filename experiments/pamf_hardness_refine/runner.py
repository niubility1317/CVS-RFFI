import argparse,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import numpy as np
import torch
from . import design as d
from .models import initialize,feature_distance,posterior_distance,save_checkpoint,load_checkpoint
from .guidance import load_teachers,guidance
from .weighting import ObservedFocus
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
    if update<247:return 3e-5*(update+1)/247
    return 3e-5*(.1+.9*(1+math.cos(math.pi*(update-247)/(d.UPDATES-248)))/2)


def source_smoke(args):
    model=initialize(args.parent_restoration/'residual_deterministic/final.pt','cpu')
    teachers=load_teachers(args,'cpu')
    source=IQSource(args.source_cache,'L_s');x=array_tensor(source.iq[:2],'cpu')
    rx=torch.tensor(source.rx[:2].tolist(),dtype=torch.long)
    with torch.no_grad():references=torch.stack([reader.features(x) for reader in teachers.values()],1)
    estimate=model(x);combined,distance,decision=guidance(teachers,estimate,references,d.ARM)
    direct=torch.autograd.grad(decision.mean(),estimate,retain_graph=True)[0]
    if not torch.isfinite(direct).all() or float(direct.abs().sum())<=0:raise ValueError('Native posterior input gradient required')
    paired,_=pair_loss(estimate,x)
    focused=torch.stack([ObservedFocus(arm)(combined,rx) for arm in d.ARMS]).mean()
    loss=(paired+combined).mean()+.5*focused;loss.backward()
    gradients=[p.grad for p in model.correction.parameters() if p.grad is not None]
    coarse_gradients=[p.grad for p in model.coarse.parameters() if p.grad is not None]
    if (not gradients or not all(torch.isfinite(g).all() for g in gradients) or
        sum(float(g.abs().sum()) for g in gradients)<=0 or any(p.grad is not None for reader in teachers.values() for p in reader.parameters()) or
        not coarse_gradients or not all(torch.isfinite(g).all() for g in coarse_gradients) or
        sum(float(g.abs().sum()) for g in coarse_gradients)<=0):
        raise ValueError('Real frozen F input gradient/correction ownership smoke failed')
    path=args.output/'smoke_checkpoint.pt';save_checkpoint(path,model,None,d.ARM,0,0)
    fresh,_=load_checkpoint(path,d.ARM,'cpu');model.eval()
    with torch.no_grad():
        if not torch.equal(model(x),fresh(x)):raise ValueError('Real-source new checkpoint roundtrip failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,actual_R03_F50=True,
        finite_nonzero_correction_gradient=True,frozen_F_no_grad=True,coarse_and_correction_gradients=True,checkpoint_roundtrip=True,
        native_posterior_input_gradient=True,actual_focus_arms=list(d.ARMS),RX_metadata_only=True,F_seeds=list(d.TEACHER_SEEDS),TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration(args.arm)['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=18000 and len(state['pids'])<2:
                model=initialize(args.parent_restoration/'residual_deterministic/final.pt','cuda:0')
                torch.cuda.synchronize();return model
        time.sleep(5)


def train(args):
    worker_started=time.perf_counter()
    row=args.output/args.arm;roles=validate_cache(args.parent_cache)
    clean=np.load(args.parent_cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256):raise ValueError('Original source shape required')
    rx=np.load(args.parent_cache/'train/rx.npy',allow_pickle=False)
    if rx.shape!=(63000,) or rx.dtype.kind not in 'iu' or set(rx.tolist())!=set(d.RX):raise ValueError('Matched original train RX metadata required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model=acquire(args);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
    focus=ObservedFocus(args.arm,'cuda:0')
    initial_parameters={k:v.detach().clone() for k,v in model.named_parameters()}
    reader=reference_features=reader_before=None
    if args.arm in d.ARMS:
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
    permutation=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            folder=args.parent_cache/'epochs'/f'{epoch:02d}'
            arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('residual','full')}
            quantile=np.load(folder/'quantile.npy',allow_pickle=False)
            if quantile.shape!=(63000,) or any(a.shape!=clean.shape for a in arrays.values()):raise ValueError('Original epoch cache layout differs')
            order=permutation.permutation(len(clean));sums={k:0. for k in ('total','iq_nmse','log_power','feature','posterior','observed_focus','clean_iq_nmse','observed_iq_nmse')}
            routes={k:0 for k in ('clean','residual','full')};clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];n=len(selected)
                values,targets,route=training_batch(clean,arrays,quantile,selected)
                y,x=array_tensor(values,'cuda:0'),array_tensor(targets,'cuda:0')
                optimizer.zero_grad(set_to_none=True);estimate=model(y);paired,terms=pair_loss(estimate,x)
                if reader is None:combined=perceptual=posterior=torch.zeros_like(paired)
                else:
                    reference=array_tensor(reference_features[selected],'cuda:0').repeat(2,1,1)
                    combined,perceptual,posterior=guidance(reader,estimate,reference,args.arm)
                observed_rx=torch.tensor(rx[selected].tolist(),dtype=torch.long,device='cuda:0')
                focused=focus(combined[:n],observed_rx)
                loss=(paired+combined).mean()+.5*focused;lr=learning_rate(updates)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement loss')
                for group in optimizer.param_groups:group['lr']=lr
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                values=dict(total=loss,iq_nmse=terms['iq_nmse'].mean(),log_power=terms['log_power'].mean(),
                    feature=perceptual.mean(),posterior=posterior.mean(),observed_focus=focused,clean_iq_nmse=terms['iq_nmse'][n:].mean(),observed_iq_nmse=terms['iq_nmse'][:n].mean())
                for key,value in values.items():sums[key]+=float(value)*n
                for key in routes:routes[key]+=int((route==key).sum())
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/len(clean) for k,v in sums.items()},route_counts=routes,
                L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
                learning_rate_last=lr,clip_fraction=clipped/247,focus_state=focus.state(),seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='REFINE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(d.UPDATES,6300*d.EPOCHS,56700*d.EPOCHS):raise ValueError('Full matched training budget required')
    for part in ('coarse.','correction.'):
        if all(torch.equal(v,initial_parameters[k]) for k,v in model.named_parameters() if k.startswith(part)):raise ValueError('Joint trainable branch did not update')
    if reader is not None and (any(p.grad is not None for model in reader.values() for p in model.parameters()) or
        any(not torch.equal(v,reader_before[seed][k]) for seed,model in reader.items() for k,v in model.state_dict().items())):raise ValueError('Frozen F changed')
    if focus.seen!=630000 or focus.selected!=(157500 if args.arm==d.ARMS[1] else 630000) or focus.updates!=(2470 if args.arm==d.ARMS[2] else 0):raise ValueError('Full registered observed focus exposure required')
    save_checkpoint(row/'final.pt',model,optimizer,args.arm,d.EPOCHS,updates,focus.state())
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',epochs=d.EPOCHS,updates=updates,
        L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds_including_preparation_and_capacity_wait=time.perf_counter()-worker_started,
        observed_focus=focus.state(),serial_inherited_passes=50,original_correction_epochs=20,total_correction_epochs=20+d.EPOCHS,coarse_epochs=20+d.EPOCHS,component_epochs=40+2*d.EPOCHS,
        F_optimizer_updates=0,F_seeds=list(d.TEACHER_SEEDS),teacher_aggregation=d.configuration(args.arm)['teacher_aggregation'],
        coarse_updates=updates,head_fit_updates=0,TX_labels_read=0,target_reads=0))
    del model,optimizer,reader,initial_parameters,reader_before,focus;torch.cuda.empty_cache()


def worker(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPUS[args.arm]):raise ValueError('Registered per-arm GPU required')
    row=args.output/args.arm;row.mkdir()
    write_json(row/'resolved_config.json',d.configuration(args.arm))
    try:
        train(args);predict(args)
        import fcntl
        # Serialize this conversation's CPU scoring behind the preceding live run.
        preceding=args.output.parent/d.PRECEDING_RUN
        while not any((preceding/name).exists() for name in ('complete.json','failure.json')):time.sleep(5)
        with (args.output/'source_score.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            command=[sys.executable,'-X','utf8','-m','experiments.pamf_hardness_refine.runner','score']
            for key in ('output','parent_cache','source_cache','arm'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'complete.json',dict(status='ANALYZED',source_only=True,target_reads=0,epochs=d.EPOCHS,updates=d.UPDATES))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise


def source_decision(args):
    current={arm:json.loads((args.output/arm/'summary.json').read_text()) for arm in d.ARMS}
    control=json.loads((args.control_source/'posterior03/summary.json').read_text())
    decision=d.source_rule(current,control)
    write_json(args.output/'source_decision.json',dict(status='SOURCE_COMPARISON_FROZEN',**decision,
        checkpoint_rule='All three hardness arms fixed finalE10 from original R03; source-only choice frozen',target_reads=0,no_target_feedback=True))
    return decision['qualified']


def validate_control(root):
    done=json.loads((root/'source_complete.json').read_text())
    trained=json.loads((root/'posterior03/training_complete.json').read_text())
    summary=json.loads((root/'posterior03/summary.json').read_text())
    if (root.name!=d.CONTROL_RUN or done.get('status')!='ANALYZED' or done.get('target_reads')!=0 or
        not done.get('source_only') or (trained.get('epochs'),trained.get('updates'),trained.get('coarse_updates'))!=(10,2470,2470) or
        any(trained.get(k)!=0 for k in ('TX_labels_read','target_reads','F_optimizer_updates','head_fit_updates')) or
        not summary.get('source_only') or summary.get('target_reads')!=0 or not summary.get('truth_last') or summary.get('count')!=1800):
        raise ValueError('Completed source-only posterior03 reference required')


def run(args):
    if args.output.exists():raise FileExistsError('Immutable new source run required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code/config commit required')
    metadata=source_residual_parent(args.parent_restoration)
    validate_control(args.control_source);args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/hardness-refine',source_parent=metadata,source_only=True))
    source_smoke(args)
    logs=args.output/'logs';logs.mkdir();children={};launches=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_hardness_refine.runner','worker','--arm',arm]
        for key in ('output','parent_cache','source_cache','parent_restoration','reader_folder','reader_root'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/(arm+'.log')).open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[arm])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[arm]=process;launches.append(dict(arm=arm,pid=process.pid,argv=command,GPU=d.GPUS[arm]))
    write_json(args.output/'workers.json',dict(workers=launches))
    codes={arm:process.wait() for arm,process in children.items()}
    if any(code!=0 for code in codes.values()):
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',exit_codes=codes,healthy_rows_preserved=True,automatic_retry=False));return
    qualified=source_decision(args)
    write_json(args.output/'source_complete.json',dict(status='ANALYZED',source_only=True,arms=list(d.ARMS),target_reads=0,epochs_per_arm=d.EPOCHS))
    if qualified:
        command=[sys.executable,'-u','-m','experiments.pamf_hardness_refine.target','run',
            '--source-run',str(args.output),'--output',str(args.output/'target'),
            '--arm',json.loads((args.output/'source_decision.json').read_text())['selected_arm']]
        for key in ('source_cache','formal_residual','reader_root','target_truth','expected_commit'):
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        with (logs/'target.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(d.GPUS[json.loads((args.output/'source_decision.json').read_text())['selected_arm']])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(args.output/'target_launch.json',dict(pid=process.pid,argv=command,GPU=d.GPUS[json.loads((args.output/'source_decision.json').read_text())['selected_arm']]))
        if process.wait()!=0:
            write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',stage='target',automatic_retry=False));return
    write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=not qualified,arms=list(d.ARMS),
        source_qualified=qualified,target_evaluation='COMPLETE' if qualified else 'NOT_QUALIFIED_NOT_ACCESSED',epochs_per_arm=d.EPOCHS))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','worker','score'))
    for name in ('output','parent-cache','source-cache'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('parent-restoration','reader-folder'):p.add_argument('--'+name,type=Path)

    for key in ('control-source','formal-residual','reader-root','target-truth'):p.add_argument('--'+key,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for name in ('output','parent_cache','source_cache','parent_restoration','reader_folder','control_source','formal_residual','reader_root','target_truth'):
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
        if any(getattr(args,k) is None for k in ('control_source','formal_residual','reader_root','target_truth')):p.error('Registered control and automatictarget inputs required')
        run(args)


if __name__=='__main__':main()
