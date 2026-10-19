import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .runtime import initialize,save_checkpoint,load_checkpoint,update_counts
from experiments.pamf_fullresolution_restore.training import learning_rate
from experiments.pamf_restoration.runner import array_tensor,tensor_array
from experiments.pamf_restoration.data import write_json


def iq_loss(estimate,clean):
    target=clean.detach()
    value=(estimate-target).square().mean((1,2))/target.square().mean((1,2)).clamp_min(1e-4)
    return value,dict(iq_nmse=value)


def build_optimizer(model):
    groups=[dict(name='prior',params=list(model.prior.parameters()),lr=2e-4,lr_multiplier=1.)]
    if model.candidate=='joint_coarse':
        groups.append(dict(name='coarse',params=list(model.frozen.coarse.parameters()),lr=2e-5,lr_multiplier=.1))
    actual=[p for group in groups for p in group['params']]
    if (len({id(p) for p in actual})!=len(actual)
            or {id(p) for p in actual}!={id(p) for p in model.parameters() if p.requires_grad}):
        raise ValueError('Optimizer must cover exactly authorized prior/coarse parameters')
    return torch.optim.AdamW(groups,weight_decay=1e-4)


def set_learning_rate(optimizer,update):
    rate=learning_rate(update)
    for group in optimizer.param_groups:group['lr']=rate*group['lr_multiplier']
    return {group['name']:group['lr'] for group in optimizer.param_groups}


def clip_groups(optimizer):
    return {group['name']:float(torch.nn.utils.clip_grad_norm_(group['params'],1.,error_if_nonfinite=True))
            for group in optimizer.param_groups}


def train(args):
    if args.row not in d.ROWS or args.parent_action.name!=d.ACTION:raise ValueError('Original fullLU coarse-update row required')
    model,parent=initialize(args.row,args.parent_action,'cuda:0');optimizer=build_optimizer(model)
    root=args.output/args.row;root.mkdir();row=root/d.KIND;row.mkdir()
    (root/'reference_controls').symlink_to(args.parent_run/'reference_controls',target_is_directory=True)
    write_json(root/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    groups={g['name']:sum(p.numel() for p in g['params']) for g in optimizer.param_groups}
    write_json(root/'resolved_config.json',dict(d.configuration(),candidate=args.row,frozen_parent=parent,parameter_groups=groups))
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Exactly complete original L/U without V required')
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (root/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Full original cached received epoch required')
            with np.load(cache/'epochs'/f'{epoch:02d}'/'groups.npz',allow_pickle=False) as archive:
                strata_labels={key:archive[key] for key in archive.files}
            order=rng.permutation(63000);model.train();clipped={key:0 for key in groups};strata={};totals=dict(loss=0.,iq_nmse=0.)
            for begin in range(0,63000,256):
                index=order[begin:begin+256];x=array_tensor(clean[index],'cuda:0');r=array_tensor(received[index],'cuda:0')
                output=model(r);paired,metrics=iq_loss(output,x);loss=paired.mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired IQ objective')
                rates=set_learning_rate(optimizer,updates);optimizer.zero_grad(set_to_none=True);loss.backward()
                norms=clip_groups(optimizer)
                for key,norm in norms.items():clipped[key]+=int(norm>1.)
                optimizer.step();updates+=1;seen_l+=int((roles[index]=='L_s').sum());seen_u+=int((roles[index]=='U_s').sum())
                totals['loss']+=float(loss)*len(index);totals['iq_nmse']+=float(metrics['iq_nmse'].sum())
                values=tensor_array(metrics['iq_nmse']);masks={key:roles[index]==key for key in ('L_s','U_s')}
                for key,labels in strata_labels.items():masks.update({key+':'+str(label):labels[index]==label for label in np.unique(labels)})
                for key,mask in masks.items():
                    item=strata.setdefault(key,dict(count=0,sum=0.));item['count']+=int(mask.sum());item['sum']+=float(values[mask].sum())
                if updates==1:
                    if (any(p.grad is not None for p in model.frozen.estimator.parameters())
                            or (args.row=='joint_coarse' and not any(p.grad is not None and p.grad.norm()>0 for p in model.frozen.coarse.parameters()))):
                        raise ValueError('Actual coarse/estimator gradient permissions failed')
                    save_checkpoint(row/'initial_source_smoke.pt',model,optimizer,epoch,updates,parent)
                    fresh,_=load_checkpoint(row/'initial_source_smoke.pt','cuda:0');model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(r[:2]),fresh(r[:2]),atol=1e-6,rtol=1e-5):raise ValueError('Complete checkpoint roundtrip differs')
                    del fresh;model.train()
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                learning_rate_last=rates['prior'],group_learning_rates=rates,gradient_clip_fraction=clipped['prior']/247,
                group_gradient_clip_fractions={key:value/247 for key,value in clipped.items()},seconds=time.perf_counter()-started,
                component_updates=update_counts(args.row,updates),strata={key:dict(count=v['count'],mean_iq_nmse=v['sum']/v['count']) for key,v in strata.items()})
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='JOINT_COARSE_EPOCH_COMPLETE',candidate=args.row,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Full fixed budget/exposure required')
    if any(p.grad is not None or p.requires_grad for p in model.frozen.estimator.parameters()):raise ValueError('Estimator must remain frozen')
    if args.row=='prior_only' and any(p.grad is not None or p.requires_grad for p in model.frozen.coarse.parameters()):raise ValueError('Control coarse must remain frozen')
    save_checkpoint(row/'final.pt',model,optimizer,20,updates,parent)
    write_json(root/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=args.row,epochs=20,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,**update_counts(args.row,updates),
        prior_initialization='scratch',inherited_prior_epochs=0,coarse_initial_epochs=20,coarse_cumulative_epochs=40 if args.row=='joint_coarse' else 20,
        parameter_groups=groups,trainable_parameters=sum(groups.values()),parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
