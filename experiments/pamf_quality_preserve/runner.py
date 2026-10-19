"""One N607 GPU1 owner; fixed training, freeze, seven-view prediction, CPU score."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT/'code') not in sys.path:
    sys.path.append(str(ROOT/'code'))

import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .model import QualityGate, select_iq
from experiments.pamf_restoration.data import IQSource, write_json
from experiments.pamf_restoration.runner import array_tensor, tensor_array, code_commit
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_change_preserve.model import relative_change
from experiments.pamf_change_preserve.runner import load_restorer, score as independent_score, READOUTS
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from experiments.pamf_fcrossrx_readout.source_diagnostic import source_reference, VIEWS


def observation_epochs(available, epoch_index):
    available=np.asarray(available,bool)
    if available.ndim!=2 or not 0<=epoch_index<len(available):
        raise ValueError('Invalid cached observation availability')
    plan=np.full(available.shape[1],-1,dtype=np.int16)
    for offset in range(len(available)):
        candidate=(epoch_index+offset)%len(available)
        use=(plan<0)&available[candidate]
        plan[use]=candidate
    if (plan<0).any():
        raise ValueError('Every physical sample needs a valid damaged observation')
    return plan


def pair_batch(clean, observations, epoch_plan, routes, indices):
    x=np.asarray(clean[indices],np.float32)
    route=np.asarray(routes[indices],bool)
    cached_epochs=np.asarray(epoch_plan[indices])
    damaged=np.empty_like(x)
    for cached_epoch in np.unique(cached_epochs):
        arrays=observations[int(cached_epoch)]
        for residual,key in ((True,'residual'),(False,'full')):
            selected=(cached_epochs==cached_epoch)&(route==residual)
            damaged[selected]=arrays[key][indices[selected]]
    batch=np.concatenate((x,damaged))
    labels=np.concatenate((np.zeros(len(x),np.float32),np.ones(len(x),np.float32)))
    return batch,labels,route


def validate_cache(cache):
    complete=json.loads((cache/'complete.json').read_text(encoding='utf-8'))
    expected=dict(status='RECEPTION_SOURCE_CACHE_COMPLETE',train_count=63000,L_count=6300,
                  U_count=56700,TX_labels_read=0,target_reads=0)
    if any(complete.get(k)!=v for k,v in expected.items()):
        raise ValueError('Existing full validated source cache required')
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if len(roles)!=63000 or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original L/U source pool differs')
    return roles


def save_checkpoint(path,model,optimizer,epoch,updates):
    torch.save(dict(schema='source_quality_gate_checkpoint_v1',config=d.configuration(),
        model=model.state_dict(),optimizer=None if optimizer is None else optimizer.state_dict(),
        epoch=epoch,updates=updates,initialization='scratch',ancestors=[],
        TX_labels_read=0,F_training_access=False,target_training=False),path)


def load_checkpoint(path,device):
    item=torch.load(path,map_location='cpu',weights_only=False)
    if (item['schema']!='source_quality_gate_checkpoint_v1' or item['config']!=d.configuration()
            or item['initialization']!='scratch' or item['ancestors'] or item['TX_labels_read']!=0
            or item['F_training_access'] or item['target_training']
            or not 0<=item['updates']<=d.UPDATES or not 0<=item['epoch']<=d.EPOCHS):
        raise ValueError('Quality gate checkpoint provenance/budget mismatch')
    model=QualityGate().to(device)
    model.load_state_dict(item['model'],strict=True)
    return model.eval(),item


def smoke(args):
    x=array_tensor(IQSource(args.source_cache,'L_s').iq[:2],'cpu')
    parent=load_restorer(args.restorer_checkpoint)
    gate=QualityGate().eval()
    with torch.no_grad():
        logits=gate(x);restored=parent(x);value,_=select_iq(x,restored,logits)
    path=args.output/'source_smoke.pt'
    save_checkpoint(path,gate,None,0,0)
    fresh,_=load_checkpoint(path,'cpu')
    with torch.no_grad():
        check,_=select_iq(x,parent(x),fresh(x))
    if not torch.equal(value,check) or not torch.isfinite(value).all():
        raise ValueError('Real frozen-R03 no-query smoke failed')
    write_json(args.output/'smoke.json',dict(status='PASS',real_source_samples=2,
               actual_R03_checkpoint=True,quality_roundtrip=True,TX_labels_read=0,
               source_V_reads=0,F_training_access=False,target_reads=0,optimizer_updates=0))


def acquire(args):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=capacity()[d.GPU]
            if state['free_mb']>=12000 and len(state['pids'])<2:
                model=QualityGate().to('cuda:0')
                with torch.no_grad():
                    if not torch.isfinite(model(torch.zeros(2,2,256,device='cuda:0'))).all():
                        raise FloatingPointError('Quality model GPU initialization failed')
                torch.cuda.synchronize()
                return model
        time.sleep(5)


def learning_rate(update):
    if update<247:return 3e-4*(update+1)/247
    return 3e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/(d.UPDATES-248)))/2)


def train(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPU):
        raise ValueError('Registered physical GPU1 only')
    roles=validate_cache(args.parent_cache)
    clean=np.load(args.parent_cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    if clean.shape!=(63000,2,256):raise ValueError('Original source IQ shape differs')
    observations=[];availability=[]
    for cached_epoch in range(1,d.EPOCHS+1):
        folder=args.parent_cache/'epochs'/f'{cached_epoch:02d}'
        arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                for key in ('residual','full')}
        quantile=np.load(folder/'quantile.npy',allow_pickle=False)
        if (any(value.shape!=clean.shape for value in arrays.values()) or
                quantile.shape!=(len(roles),) or not np.isfinite(quantile).all()):
            raise ValueError('Original cached observation shape/quantile differs')
        observations.append(arrays);availability.append(quantile>=.15)
    availability=np.stack(availability)
    observation_epochs(availability,0)
    torch.manual_seed(d.MODEL_SEED)
    model=acquire(args)
    torch.cuda.manual_seed_all(d.MODEL_SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    order_rng=np.random.default_rng(d.DATA_SEED)
    updates=seen_l=seen_u=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    write_json(args.output/'gpu_ready.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
               physical_GPU=d.GPU,CPU_threads=2))
    with (args.output/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            plan=observation_epochs(availability,epoch-1)
            route_rng=np.random.default_rng(np.random.SeedSequence([d.DATA_SEED,epoch,7525]))
            routes=route_rng.random(len(roles))<.75
            order=order_rng.permutation(len(roles));sums=dict(loss=0.,clean_loss=0.,damaged_loss=0.)
            route_counts=dict(residual=0,full=0);clipped=0;model.train()
            for offset in range(0,len(order),d.BATCH_SIZE):
                selected=order[offset:offset+d.BATCH_SIZE]
                values,targets,route=pair_batch(clean,observations,plan,routes,selected)
                x=array_tensor(values,'cuda:0');y=array_tensor(targets,'cuda:0')
                logits=model(x);per=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
                loss=per.mean();lr=learning_rate(updates)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite synthetic quality loss')
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                n=len(selected);sums['loss']+=float(loss)*n
                sums['clean_loss']+=float(per[:n].mean())*n
                sums['damaged_loss']+=float(per[n:].mean())*n
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
                route_counts['residual']+=int(route.sum());route_counts['full']+=int((~route).sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/len(roles) for k,v in sums.items()},
                L_physical_exposures=seen_l,U_physical_exposures=seen_u,
                quality_examples=2*(seen_l+seen_u),route_counts=route_counts,
                cached_epoch_reused=int((plan!=epoch-1).sum()),
                learning_rate_last=lr,gradient_clip_fraction=clipped/247,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='QUALITY_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):
        raise ValueError('Full registered L/U budget required')
    save_checkpoint(args.output/'final.pt',model,optimizer,20,updates)
    write_json(args.output/'training_complete.json',dict(status='SOURCE_TRAINED',epochs=20,updates=updates,
        L_physical_exposures=seen_l,U_physical_exposures=seen_u,quality_examples=2520000,
        TX_labels_read=0,F_training_access=False,target_reads=0,restorer_updates=0,head_fit_updates=0,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del model,optimizer;torch.cuda.empty_cache()


@torch.no_grad()
def predict(args,gate_checkpoint=None,selector=select_iq):
    reference=args.parent_cache/'validation'
    ids,_,_=source_reference(reference,args.source_cache)
    gate,item=load_checkpoint(args.output/'final.pt' if gate_checkpoint is None else gate_checkpoint,'cuda:0')
    if item['epoch']!=20 or item['updates']!=4940:raise ValueError('Fixed final quality gate required')
    gate.requires_grad_(False)
    parent=load_restorer(args.restorer_checkpoint).to('cuda:0')
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0')
    saved=args.output/'prediction';saved.mkdir()
    snapshots={k:v.clone() for k,v in gate.state_dict().items()}
    route_records=[]
    for view in VIEWS:
        started=time.perf_counter()
        observed=np.load(reference/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if observed.shape!=(len(ids),2,256):raise ValueError('Fixed source observation shape differs')
        restored_iq=np.lib.format.open_memmap(saved/(view+'_restored.npy'),mode='w+',dtype=np.float32,shape=observed.shape)
        decisions={k:[] for k in READOUTS};preserved=[];changes=[];scores=[]
        for offset in range(0,len(ids),64):
            y=array_tensor(observed[offset:offset+64],'cuda:0')
            logits=gate(y);restored=parent(y);output,keep=selector(y,restored,logits)
            if not torch.isfinite(restored).all():raise FloatingPointError('Nonfinite R03 output')
            for name,value in zip(READOUTS,(y,restored,output)):
                identity=reader(value)
                if identity.shape!=(len(y),6) or not torch.isfinite(identity).all():
                    raise ValueError('Invalid frozen original default-head prediction')
                decisions[name].extend(identity.argmax(1).tolist())
            restored_iq[offset:offset+len(y)]=tensor_array(restored)
            preserved.extend(keep.tolist());scores.extend(logits.tolist())
            changes.extend(relative_change(y,restored).tolist())
        restored_iq.flush()
        with (saved/(view+'.npz')).open('xb') as stream:
            np.savez_compressed(stream,ids=ids,**{k:np.asarray(v,np.uint8) for k,v in decisions.items()},
                preserve=np.asarray(preserved,bool),change=np.asarray(changes,np.float64))
        np.savez_compressed(saved/(view+'_quality.npz'),ids=ids,damage_logit=np.asarray(scores,np.float32))
        record=dict(view=view,count=len(ids),preserved=int(sum(preserved)),seconds=time.perf_counter()-started)
        route_records.append(record);print(json.dumps(dict(event='QUALITY_VIEW_COMPLETE',**record)),flush=True)
    if any(not torch.equal(v,snapshots[k]) for k,v in gate.state_dict().items()):
        raise ValueError('Frozen quality network changed')
    write_json(args.output/'gate_diagnostic.json',dict(records=route_records,decision_F_or_TX_access=False,
               threshold_logit=0.,model_frozen_before_V=True))
    write_json(saved/'complete.json',dict(status='SOURCE_PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),
        readouts=list(READOUTS),role='source_V',target_reads=0,TX_label_reads=0,state_updates=0,threshold_frozen_before_V=True))
    del gate,parent,reader;torch.cuda.empty_cache()


def run(args):
    for name in ('source_cache','parent_cache','reader_folder','restorer_checkpoint','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.output.exists():raise FileExistsError('Preserve immutable run output')
    if code_commit()!=args.expected_commit:raise ValueError('Use the declared Git code/config commit')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(d.MODEL_SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_json(args.output/'launch.json',dict(pid=os.getpid(),cwd=str(ROOT),argv=sys.argv,
        commit=args.expected_commit,owner='codex/root/quality-preserve',physical_GPU=d.GPU,source_only=True))
    write_json(args.output/'resolved_config.json',d.configuration())
    try:
        smoke(args);train(args);predict(args)
        command=[sys.executable,'-X','utf8','-m','experiments.pamf_quality_preserve.runner','score',
                 '--parent-cache',str(args.parent_cache),'--source-cache',str(args.source_cache),'--output',str(args.output)]
        subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        write_json(args.output/'complete.json',dict(status='ANALYZED',source_only=True,target_reads=0,
                   epochs=20,updates=4940,views=list(VIEWS),readouts=list(READOUTS)))
    except BaseException as error:
        write_json(args.output/'failure.json',dict(status='FAILED_TECHNICAL',error=repr(error),
                   traceback=traceback.format_exc(),automatic_retry=False,outputs_preserved=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('run','score'))
    for name in ('source-cache','parent-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('reader-folder','restorer-checkpoint'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--expected-commit')
    args=p.parse_args()
    if args.mode=='score':
        independent_score(args.parent_cache/'validation',args.source_cache,args.output)
    else:
        if args.reader_folder is None or args.restorer_checkpoint is None or args.expected_commit is None:
            p.error('Actual frozen R03/F50 and Git commit required')
        run(args)


if __name__=='__main__':main()
