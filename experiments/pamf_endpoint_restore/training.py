import json
import math
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .runtime import initialize,save_checkpoint,load_checkpoint
from experiments.pamf_restoration.runner import array_tensor,tensor_array
from experiments.pamf_restoration.data import write_json


def learning_rate(update):
    if not 0<=update<4940:raise ValueError('Update outside fixed budget')
    return 2e-4*(update+1)/247 if update<247 else 2e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/4692))/2)


def train(args):
    cfg=d.configuration();config=d.configuration;build=initialize;save=save_checkpoint;load=load_checkpoint
    if args.row not in cfg['rows'] or args.parent_action.name!=cfg['action_parent']:raise ValueError('Registered scratch row/original source cache required')
    model,parent=build(args.row,args.parent_action,'cuda:0');parameters=list(model.prior.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    root=args.output/args.row;root.mkdir();row=root/d.KIND;row.mkdir()
    (root/'reference_controls').symlink_to(args.parent_run/'reference_controls',target_is_directory=True)
    write_json(root/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write_json(root/'resolved_config.json',dict(config(),candidate=args.row,frozen_parent=parent,prior_parameters=sum(p.numel() for p in parameters)))
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:raise ValueError('Exactly complete original L/U without V required')
    rng=np.random.default_rng(cfg['data_seed']);noise_generator=torch.Generator(device='cuda:0').manual_seed(d.TRAINING_NOISE_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (root/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Full original cached epoch required')
            with np.load(cache/'epochs'/f'{epoch:02d}'/'groups.npz',allow_pickle=False) as archive:groups={key:archive[key] for key in archive.files}
            order=rng.permutation(63000);model.train();clipped=0;strata={};totals=dict(loss=0.)
            for begin in range(0,63000,256):
                index=order[begin:begin+256];x=array_tensor(clean[index],'cuda:0');r=array_tensor(received[index],'cuda:0')
                paired,metrics=model.training_loss(x,r,noise_generator);loss=paired.mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired objective')
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[index]=='L_s').sum());seen_u+=int((roles[index]=='U_s').sum())
                totals['loss']+=float(loss)*len(index)
                for key,metric in metrics.items():totals[key]=totals.get(key,0.)+float(metric.sum())
                values=tensor_array(metrics['objective_nmse']);masks={key:roles[index]==key for key in ('L_s','U_s')}
                for key,labels in groups.items():masks.update({key+':'+str(label):labels[index]==label for label in np.unique(labels)})
                for key,mask in masks.items():
                    item=strata.setdefault(key,dict(count=0,sum=0.));item['count']+=int(mask.sum());item['sum']+=float(values[mask].sum())
                if updates==1:
                    save(row/'initial_source_smoke.pt',model,optimizer,epoch,updates,parent)
                    fresh,_=load(row/'initial_source_smoke.pt','cuda:0');model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(r[:2],torch.ones_like(r[:2])),fresh(r[:2],torch.ones_like(r[:2])),atol=1e-6,rtol=1e-5):raise ValueError('Actual complete checkpoint roundtrip differs')
                    del fresh;model.train()
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                learning_rate_last=lr,gradient_clip_fraction=clipped/247,seconds=time.perf_counter()-started,
                strata={key:dict(count=value['count'],mean_objective_nmse=value['sum']/max(value['count'],1)) for key,value in strata.items()})
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(dict(event='ENDPOINT_EPOCH_COMPLETE',candidate=args.row,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Full new prior budget/exposure required')
    save(row/'final.pt',model,optimizer,20,updates,parent)
    if any(p.grad is not None for p in model.frozen.parameters()):raise ValueError('Frozen context received parameter gradients')
    write_json(root/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=args.row,epochs=20,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,prior_initialization='scratch',inherited_prior_epochs=0,
        trainable_parameters=sum(p.numel() for p in parameters),parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))
