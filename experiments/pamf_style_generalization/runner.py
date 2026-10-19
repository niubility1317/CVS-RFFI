import argparse,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:sys.path.append(str(ROOT/'code'))
import numpy as np
import torch
from . import design as d
from .models import initialize,feature_distance,posterior_distance,save_checkpoint,load_checkpoint
from .guidance import load_teachers,guidance
from .mixing import make_plan
from .pseudo import load_pseudo
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
    if update<247:return 3e-5*(update+1)/247
    return 3e-5*(.1+.9*(1+math.cos(math.pi*(update-247)/(d.UPDATES-248)))/2)


def source_smoke(args):
    teachers=load_teachers(args,'cpu');source=IQSource(args.source_cache,'L_s')
    indices=np.concatenate([np.flatnonzero(source.rx==rx)[:32] for rx in d.SOURCE_RX])
    all_x=array_tensor(source.iq[indices],'cpu')
    with torch.no_grad():
        all_refs=torch.stack([reader.features(all_x) for reader in teachers.values()],1)
        probs=torch.stack([reader.classify_features(all_refs[:,i]).softmax(-1) for i,reader in enumerate(teachers.values())],1)
    confidence,classes=probs.max(-1);accepted=((classes==classes[:,:1]).all(1)&(confidence>=.9).all(1)).tolist()
    pseudo=np.asarray(classes[:,0].tolist(),np.int64);rx=source.rx[indices];accepted=np.asarray(accepted,np.bool_)
    selected=None
    for label in np.unique(pseudo[accepted]):
        groups=[np.flatnonzero(accepted&(pseudo==label)&(rx==r))[:2] for r in d.SOURCE_RX]
        groups=[g for g in groups if len(g)==2]
        if len(groups)>=2:selected=np.concatenate(groups[:2]);break
    if selected is None:raise ValueError('Real source same-pseudo two-RX smoke quartet required')
    x=all_x[selected];references=all_refs[selected].repeat(2,1,1);y=x.repeat(2,1,1)
    parent=load_restorer(args.parent_restoration/'residual_deterministic/final.pt')
    with torch.no_grad():expected=parent(x)
    records={}
    for arm in d.ARMS:
        model=initialize(args.parent_restoration/'residual_deterministic/final.pt','cpu')
        with torch.no_grad():
            if not torch.equal(model(x),expected):raise ValueError('Inference augmentation-off must exactly equal R03')
        plan=make_plan(rx[selected],pseudo[selected],accepted[selected],np.random.default_rng(1),arm,p=1.)
        if not plan.physical_active.any():raise ValueError('Actual paired source smoke mixture required')
        model.train();estimate=model(y,plan);combined,_,decision=guidance(teachers,estimate,references,arm)
        direct=torch.autograd.grad(decision.mean(),estimate,retain_graph=True)[0]
        if not torch.isfinite(direct).all() or float(direct.abs().sum())<=0:raise ValueError('Native posterior input gradient required')
        paired,_=pair_loss(estimate,y);(paired+combined).mean().backward()
        for module in (model.coarse,model.correction):
            gradients=[p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients) or sum(float(g.abs().sum()) for g in gradients)<=0:
                raise ValueError('Actual coarse/correction source gradients required')
        if any(p.grad is not None for reader in teachers.values() for p in reader.parameters()):raise ValueError('F must remain gradient-free')
        if any(model.correction.style_applications[str(i)]!=2*int(plan.physical_active.sum()) for i in d.LAYERS):raise ValueError('Actual shallow/middle mixing required')
        path=args.output/(arm+'_smoke_checkpoint.pt');save_checkpoint(path,model,None,arm,0,0)
        fresh,_=load_checkpoint(path,arm,'cpu');model.eval()
        with torch.no_grad():
            if not torch.equal(model(x),fresh(x)):raise ValueError('Actual checkpoint roundtrip failed')
        records[arm]=dict(augmentation_off_equals_R03=True,coarse_correction_gradients=True,
            layer_IQ_applications=dict(model.correction.style_applications),trainable_parameters=sum(p.numel() for p in model.parameters()),paired_plan=plan.counts())
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=4,source_Ls_reference_count=len(indices),arms=records,
        actual_R03_F50=True,frozen_F_no_grad=True,checkpoint_roundtrip=True,native_posterior_input_gradient=True,
        TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))

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
    if rx.shape!=(63000,) or rx.dtype.kind not in 'iu' or set(rx.tolist())!=set(d.SOURCE_RX):raise ValueError('Matched original train RX metadata required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model=acquire(args);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4)
    pseudo,accepted=load_pseudo(args.pseudo_source,args.parent_cache,args.source_cache)
    mix_rng=np.random.default_rng(d.MIX_SEED)
    mixing=dict(physical_records=0,eligible=0,attempted=0,active=0,L_active=0,U_active=0)
    pair_counts=np.zeros((5,5),np.int64)
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
            order=permutation.permutation(len(clean));sums={k:0. for k in ('total','iq_nmse','log_power','feature','posterior','clean_iq_nmse','observed_iq_nmse')}
            routes={k:0 for k in ('clean','residual','full')};clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];n=len(selected)
                values,targets,route=training_batch(clean,arrays,quantile,selected)
                y,x=array_tensor(values,'cuda:0'),array_tensor(targets,'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                plan=make_plan(rx[selected],pseudo[selected],accepted[selected],mix_rng,args.arm,'cuda:0')
                estimate=model(y,plan);paired,terms=pair_loss(estimate,x)
                if reader is None:combined=perceptual=posterior=torch.zeros_like(paired)
                else:
                    reference=array_tensor(reference_features[selected],'cuda:0').repeat(2,1,1)
                    combined,perceptual,posterior=guidance(reader,estimate,reference,args.arm)
                loss=(paired+combined).mean();lr=learning_rate(updates)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite refinement loss')
                for group in optimizer.param_groups:group['lr']=lr
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                values=dict(total=loss,iq_nmse=terms['iq_nmse'].mean(),log_power=terms['log_power'].mean(),
                    feature=perceptual.mean(),posterior=posterior.mean(),clean_iq_nmse=terms['iq_nmse'][n:].mean(),observed_iq_nmse=terms['iq_nmse'][:n].mean())
                for key,value in values.items():sums[key]+=float(value)*n
                for key in routes:routes[key]+=int((route==key).sum())
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
                counts=plan.counts()
                for key in ('physical_records','eligible','attempted','active'):mixing[key]+=counts[key]
                active=plan.physical_active;donor=np.asarray(plan.donor[:n].tolist(),np.int64)
                if (pseudo[selected][active]!=pseudo[selected][donor[active]]).any():raise ValueError('Only same pseudo identity style donors allowed')
                same=rx[selected][active]==rx[selected][donor[active]]
                if (not same.all() if args.arm==d.ARMS[0] else same.any()):raise ValueError('Registered RX donor regime required')
                mixing['L_active']+=int((active&(roles[selected]=='L_s')).sum());mixing['U_active']+=int((active&(roles[selected]=='U_s')).sum())
                source_rx=np.searchsorted(d.SOURCE_RX,rx[selected][active]);donor_rx=np.searchsorted(d.SOURCE_RX,rx[selected][donor[active]])
                np.add.at(pair_counts,(source_rx,donor_rx),1)
            record=dict(epoch=epoch,updates=updates,losses={k:v/len(clean) for k,v in sums.items()},route_counts=routes,
                L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
                learning_rate_last=lr,clip_fraction=clipped/247,augmentation_state=dict(mixing,per_layer_IQ_applications=dict(model.correction.style_applications),layer_forward_calls=dict(model.correction.style_calls),RX_pair_counts=pair_counts.tolist()),seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='REFINE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(d.UPDATES,6300*d.EPOCHS,56700*d.EPOCHS):raise ValueError('Full matched training budget required')
    for part in ('coarse.','correction.'):
        if all(torch.equal(v,initial_parameters[k]) for k,v in model.named_parameters() if k.startswith(part)):raise ValueError('Joint trainable branch did not update')
    if reader is not None and (any(p.grad is not None for model in reader.values() for p in model.parameters()) or
        any(not torch.equal(v,reader_before[seed][k]) for seed,model in reader.items() for k,v in model.state_dict().items())):raise ValueError('Frozen F changed')
    if mixing['physical_records']!=630000 or mixing['active']!=mixing['L_active']+mixing['U_active'] or int(pair_counts.sum())!=mixing['active']:
        raise ValueError('Complete actual mixing exposure required')
    if any(model.correction.style_applications[str(i)]!=2*mixing['active'] or model.correction.style_calls[str(i)]!=d.UPDATES for i in d.LAYERS):
        raise ValueError('Actual both registered mixing layers required')
    augmentation_state=dict(mixing,per_layer_IQ_applications=dict(model.correction.style_applications),layer_forward_calls=dict(model.correction.style_calls),RX_pair_counts=pair_counts.tolist(),pseudo_source=d.PSEUDO_RUN,TX_labels_read=0,target_reads=0)
    save_checkpoint(row/'final.pt',model,optimizer,args.arm,d.EPOCHS,updates,augmentation_state)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',epochs=d.EPOCHS,updates=updates,
        L_physical_exposures=seen_l,U_physical_exposures=seen_u,IQ_examples=2*(seen_l+seen_u),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds_including_preparation_and_capacity_wait=time.perf_counter()-worker_started,
        augmentation_state=augmentation_state,serial_inherited_passes=50,original_correction_epochs=20,total_correction_epochs=20+d.EPOCHS,coarse_epochs=20+d.EPOCHS,component_epochs=40+2*d.EPOCHS,
        F_optimizer_updates=0,F_seeds=list(d.TEACHER_SEEDS),teacher_aggregation=d.configuration(args.arm)['teacher_aggregation'],
        coarse_updates=updates,head_fit_updates=0,TX_labels_read=0,target_reads=0))
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
            command=[sys.executable,'-X','utf8','-m','experiments.pamf_style_generalization.runner','score']
            for key in ('output','parent_cache','source_cache','arm'):command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(row/'source_frozen.json',dict(status='SOURCE_FROZEN',arm=args.arm,epochs=d.EPOCHS,updates=d.UPDATES,
            source_only=True,target_reads=0,checkpoint_selection='fixed_final_E10',target_policy='FIXED_COMPARISON'))
    except BaseException as error:
        write_json(row/'failure.json',dict(status='FAILED_TECHNICAL',stage='source',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise

def launch_target(args,arm,logs):
    command=[sys.executable,'-u','-m','experiments.pamf_style_generalization.target','run',
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


def run(args):
    if args.output.exists():raise FileExistsError('Immutable new fixed-matrix run required')
    if code_commit()!=args.expected_commit:raise ValueError('Pinned code/config commit required')
    metadata=source_residual_parent(args.parent_restoration)
    load_pseudo(args.pseudo_source,args.parent_cache,args.source_cache)
    args.output.mkdir(parents=True)
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,commit=args.expected_commit,
        owner='codex/root/style-generalization',source_parent=metadata,fixed_target_arms=list(d.ARMS)))
    write_json(args.output/'matrix.json',d.target_matrix())
    try:source_smoke(args)
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',stage='smoke',error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
    logs=args.output/'logs';logs.mkdir();active={};launches=[];failed={};completed=[]
    for arm in d.ARMS:
        command=[sys.executable,'-u','-m','experiments.pamf_style_generalization.runner','worker','--arm',arm]
        for key in ('output','parent_cache','source_cache','parent_restoration','reader_folder','reader_root','pseudo_source'):
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

    for key in ('pseudo-source','formal-residual','reader-root','target-truth'):p.add_argument('--'+key,type=Path)
    p.add_argument('--arm',choices=d.ARMS);p.add_argument('--expected-commit');args=p.parse_args()
    for name in ('output','parent_cache','source_cache','parent_restoration','reader_folder','pseudo_source','formal_residual','reader_root','target_truth'):
        if getattr(args,name,None) is not None:setattr(args,name,getattr(args,name).resolve())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='score':
        if args.arm is None:p.error('One declared arm required')
        score(args)
    elif args.mode=='worker':
        if args.arm is None or args.parent_restoration is None or args.reader_folder is None or args.pseudo_source is None:p.error('Actual frozen parent/reader/pseudo and arm required')
        worker(args)
    else:
        if args.expected_commit is None or args.parent_restoration is None or args.reader_folder is None:p.error('Actual parents and code commit required')
        if any(getattr(args,k) is None for k in ('pseudo_source','formal_residual','reader_root','target_truth')):p.error('Registered pseudo source and automatic target inputs required')
        run(args)


if __name__=='__main__':main()
