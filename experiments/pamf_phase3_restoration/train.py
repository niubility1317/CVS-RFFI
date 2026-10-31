"""Scratch joint30; source-only inputs. Evaluators run in fresh processes."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from .model import scratch_model
from .objective import PairedObjective,feature_guide
from .checkpoint import training_config,save_model,load_model,RUNS
from .cache import load_cache,close_cache,tensor,load_F0,check_F0_state
from .source_pair import read,validate_source
from .cost_probe import write


def schedule(n=63000):
    """Independent data generator: teacher/bootstrap random draws cannot alter it."""
    generator=np.random.default_rng(2026110101)
    order=np.empty((30,n),np.int32);clean=np.empty((30,n),bool)
    for epoch in range(30):
        order[epoch]=generator.permutation(n)
        clean[epoch]=generator.random(n)<.15
    return order,clean


def learning_rate(update):
    if not 1<=update<=7410:raise ValueError('Fixed single30 update index required')
    if update<=247:return .001*update/247
    return 2e-5+.5*(.001-2e-5)*(1+math.cos(math.pi*(update-247)/(7410-247)))


def smoke(G,F0,bundle,arm,commit,out,device):
    started=time.perf_counter()
    x=tensor(bundle['received'][:2],device);y=tensor(bundle['clean'][:2],device)
    previous={k:v.detach().clone() for k,v in G.state_dict().items()}
    fprevious={k:v.detach().clone() for k,v in F0.state_dict().items()}
    with torch.no_grad():target=F0.features(y).detach()
    if arm=='F0':torch.testing.assert_close(target,tensor(bundle['F0_features'][:2],device),rtol=2e-5,atol=3e-5)
    restored=G(x);torch.testing.assert_close(restored,x,atol=0,rtol=0)
    physical,_=PairedObjective().to(device)(restored,y)
    guide,info=feature_guide(F0,restored,y,target)
    actual=torch.autograd.grad(guide.sum(),restored,retain_graph=True)[0].flatten(1).norm(dim=1)
    if (actual>.5*info['IQ_gradient_norm']*(1+2e-5)).any() or not actual.sum()>0:
        raise ValueError('Actual F0 input VJP/cap failed')
    (physical+(guide if arm=='F0' else 0)).mean().backward()
    if not G.output.weight.grad.abs().sum()>0:raise ValueError('Actual paired source G gradient missing')
    G.zero_grad(set_to_none=True)
    if any(not torch.equal(v,G.state_dict()[k]) for k,v in previous.items()):raise ValueError('Smoke changed G state')
    check_F0_state(F0,fprevious)
    save_model(out/'smoke_E0.pt',G,arm,commit,0,0)
    loaded=load_model(out/'smoke_E0.pt',device,arm=arm,commit=commit,epoch=0)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x),restored,atol=0,rtol=0)
        logits=F0.classify_features(F0.features(loaded(x)))
    if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual F0/default head invalid')
    write(out/'smoke.json',dict(status='PASS',real_L_s=2,strict_E0_reload=True,actual_F0_input_VJP=True,
        cached_F0_alignment_checked=arm=='F0',G_initial_identity=True,output_gradient_nonzero=True,
        models_unchanged=True,optimizer_updates=0,TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,
        F0_feature_rows=6,F0_head_rows=2,G_rows=6,seconds=time.perf_counter()-started))
    print('P3_RESTORE_TF_REAL_SOURCE_SMOKE_PASS',arm,flush=True)


def train(args):
    started=time.perf_counter();config=training_config(args.arm)
    if (Path.cwd()/'release_commit.txt').read_text().strip()!=args.commit:raise ValueError('Actual release code differs')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('One admitted physical GPU required')
    out=args.root/'runs'/RUNS[args.arm]/'training';out.mkdir()
    write(out/'config.json',dict(config,code_commit=args.commit))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';guided=args.arm=='F0'
    bundle=load_cache(args.cache,guided=guided);c=validate_source(read(args.inputs))
    try:
        F0,identity=load_F0(c,device)
        if identity!=bundle['manifest']['F0_identity']:raise ValueError('Cached and actual original F0 differ')
        G=scratch_model(2026092701).to(device)
        smoke(G,F0,bundle,args.arm,args.commit,out,device)
        if guided:fbefore={k:v.detach().clone() for k,v in F0.state_dict().items()}
        else:del F0
        load_started=time.perf_counter()
        x,y=tensor(bundle['received'],device),tensor(bundle['clean'],device)
        target=tensor(bundle['F0_features'],device) if guided else None
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (guided and not torch.isfinite(target).all()):
            raise ValueError('Nonfinite original source cache')
        torch.cuda.synchronize();resident_seconds=time.perf_counter()-load_started
    finally:close_cache(bundle)
    order,clean_masks=schedule()
    with (out/'data_schedule.npz').open('xb') as stream:np.savez(stream,order=order,clean_masks=clean_masks)
    G.train();objective=PairedObjective().to(device)
    optimizer=torch.optim.AdamW(G.parameters(),lr=.001,weight_decay=1e-4)
    updates=0;online_F0_rows=0;total_identity=0;training_started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (out/'training.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for epoch in range(30):
            torch.cuda.synchronize();epoch_started=time.perf_counter()
            sums=dict(loss=0.,iq_nmse=0.,spectral_error=0.,feature_error=0.,guide_weight=0.)
            identity_count=0
            for offset in range(0,63000,256):
                indices=torch.tensor(order[epoch,offset:offset+256].tolist(),device=device,dtype=torch.long)
                mask=torch.tensor(clean_masks[epoch,order[epoch,offset:offset+256]].tolist(),device=device,dtype=torch.bool)
                received=torch.where(mask[:,None,None],y[indices],x[indices]);clean=y[indices]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates+1)
                restored=G(received);loss,terms=objective(restored,clean)
                if guided:
                    guide,info=feature_guide(F0,restored,clean,target[indices]);loss=loss+guide
                    sums['feature_error']+=float(info['feature_error'].detach().sum())
                    sums['guide_weight']+=float(info['weight'].sum());online_F0_rows+=len(indices)
                if not torch.isfinite(loss).all():raise ValueError('Nonfinite source objective; preserve failed run')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(G.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                for key,values in [('loss',loss),*terms.items()]:sums[key]+=float(values.detach().sum())
                identity_count+=int(mask.sum())
            torch.cuda.synchronize();seconds=time.perf_counter()-epoch_started;total_identity+=identity_count
            if guided:check_F0_state(F0,fbefore)
            row=dict(epoch=epoch+1,updates=updates,count=63000,identity_clean_inputs=identity_count,
                G10_P0_inputs=63000-identity_count,seconds=seconds,steps=247,
                last_learning_rate=learning_rate(updates),metrics={k:v/63000 for k,v in sums.items()},
                training_F_access=guided,online_F0_rows=online_F0_rows)
            log.write(json.dumps(row,allow_nan=False)+'\n');print('P3_RESTORE_TF_EPOCH',json.dumps(row),flush=True)
    torch.cuda.synchronize();training_seconds=time.perf_counter()-training_started
    if updates!=7410 or online_F0_rows!=(1890000 if guided else 0):raise ValueError('Fixed full30 exposure differs')
    save_model(out/'final.pt',G,args.arm,args.commit,30,7410)
    write(out/'training_complete.json',dict(status='SOURCE_TRAINED',run_id=RUNS[args.arm],arm=args.arm,
        code_commit=args.commit,epoch=30,updates=updates,training_F_access=guided,
        target_reads=0,TX_labels_read=0,OE_reads=0,V_reads=0,G_initialization_checkpoint_loads=0,
        source_Ls_exposures=189000,source_Us_exposures=1701000,
        identity_clean_inputs=total_identity,G10_P0_inputs=1890000-total_identity,
        online_F0_feature_rows=online_F0_rows,F0_cache_production_shared_rows=63000,
        source_cache_resident_load_seconds=resident_seconds,training_seconds=training_seconds,
        worker_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        original_F0_guidance_identity=identity if guided else None,
        source_cache_capsule=config['capsule_id'],source_cache_split=config['split_id'],
        all_epochs_fixed_before_evaluation=True,checkpoint_selection='final30_only'))
    print('P3_RESTORE_TF_SOURCE_TRAINING_COMPLETE',args.arm,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--arm',choices=tuple(RUNS),required=True)
    p.add_argument('--commit',required=True);a=p.parse_args()
    try:train(a)
    except Exception as e:
        folder=a.root/'runs'/RUNS[a.arm]
        if folder.is_dir():write(folder/'training_failure.json',dict(status='FAILED_PRESERVED',error=str(e),type=type(e).__name__))
        raise


if __name__=='__main__':main()
