import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d,runtime
from .models import training_delta,teacher_response,paired_objective,response_objective
from experiments.pamf_fullresolution_restore.training import learning_rate
from experiments.pamf_iqonly_restore.training import iq_loss
from experiments.pamf_action_restore.evaluation import teacher_action
from experiments.pamf_restoration.models import pair
from experiments.pamf_restoration.runner import array_tensor,tensor_array
from experiments.pamf_restoration.data import write_json


def train(args):
    if args.row not in d.ROWS or args.parent_action.name!=d.ACTION:raise ValueError('Fixed source response experiment required')
    model,parent=runtime.initialize(args.row,args.parent_action,'cuda:0');parameters=list(model.prior.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    root=args.output/args.row;root.mkdir();row=root/d.KIND;row.mkdir()
    (root/'reference_controls').symlink_to(args.parent_run/'reference_controls',target_is_directory=True)
    write_json(root/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    write_json(root/'resolved_config.json',dict(d.configuration(),candidate=args.row,frozen_parent=parent,
        prior_parameters=sum(p.numel() for p in parameters)))
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Complete original L6300/U56700 without V required')
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=eligible_total=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (root/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            folder=cache/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
                    for key in ('received','kernel','coefficients','variance')}
            if any(arrays[key].shape!=shape for key,shape in [('received',clean.shape),('kernel',(63000,2,64)),
                    ('coefficients',(63000,4,256)),('variance',(63000,))]):raise ValueError('Original matched source action epoch required')
            with np.load(folder/'groups.npz',allow_pickle=False) as archive:groups={key:archive[key] for key in archive.files}
            order=rng.permutation(63000);model.train();clipped=eligible_epoch=0;strata={}
            totals=dict(pair_loss=0.,base_iq_nmse=0.,response_error_all=0.,response_error_eligible_sum=0.,loss=0.)
            direction_counts={name:0 for name in d.DIRECTIONS};amplitude_counts={'1%':0,'2%':0}
            for begin in range(0,63000,256):
                batch=order[begin:begin+256];size=len(batch);optimizer.zero_grad(set_to_none=True)
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                for offset in range(0,size,64):
                    index=batch[offset:offset+64];x=array_tensor(clean[index],'cuda:0');r=array_tensor(arrays['received'][index],'cuda:0')
                    delta,valid,codes=training_delta(x,index,epoch)
                    known=teacher_action(arrays,index,'cuda:0');variance=array_tensor(arrays['variance'][index],'cuda:0')
                    change,target,eligible,teacher=teacher_response(known,delta,variance,valid)
                    output=model(r);plus=model(r+pair(change));minus=model(r-pair(change))
                    paired=paired_objective(output,plus,minus,x,delta)
                    response,error=response_objective(plus,minus,target,delta,eligible)
                    objective=paired+(d.RESPONSE_WEIGHT*response if args.row==d.CANDIDATE else 0.)
                    if not torch.isfinite(objective).all():raise FloatingPointError('Nonfinite paired/response objective')
                    (objective.sum()/size).backward()
                    base_nmse=iq_loss(output.detach(),x)[0];count=int(eligible.sum());eligible_epoch+=count;eligible_total+=count
                    totals['pair_loss']+=float(paired.detach().sum());totals['loss']+=float(objective.detach().sum())
                    totals['base_iq_nmse']+=float(base_nmse.sum());totals['response_error_all']+=float(error.detach().sum())
                    totals['response_error_eligible_sum']+=float(response.detach().sum())
                    for k,name in enumerate(d.DIRECTIONS):direction_counts[name]+=int((codes[:,0]==k).sum())
                    for k,name in enumerate(('1%','2%')):amplitude_counts[name]+=int((codes[:,1]==k).sum())
                    values=tensor_array(base_nmse);selected=tensor_array(eligible).astype(bool)
                    masks={key:roles[index]==key for key in ('L_s','U_s')}
                    for key,labels in groups.items():masks.update({key+':'+str(label):labels[index]==label for label in np.unique(labels)})
                    for key,mask in masks.items():
                        item=strata.setdefault(key,dict(count=0,sum=0.,observable=0));item['count']+=int(mask.sum())
                        item['sum']+=float(values[mask].sum());item['observable']+=int((mask&selected).sum())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[batch]=='L_s').sum());seen_u+=int((roles[batch]=='U_s').sum())
                if updates==1:
                    runtime.save_checkpoint(row/'initial_source_smoke.pt',model,optimizer,epoch,updates,parent)
                    fresh,_=runtime.load_checkpoint(row/'initial_source_smoke.pt','cuda:0');model.eval()
                    with torch.no_grad():
                        if not torch.allclose(model(r[:2]),fresh(r[:2]),atol=1e-6,rtol=1e-5):raise ValueError('Actual response checkpoint roundtrip differs')
                    del fresh;model.train()
            record=dict(epoch=epoch,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
                L_waveform_forwards=3*seen_l,U_waveform_forwards=3*seen_u,eligible_epoch=eligible_epoch,
                eligible_total=eligible_total,eligible_fraction=eligible_epoch/63000,
                direction_counts=direction_counts,amplitude_counts=amplitude_counts,
                losses={key:value/63000 for key,value in totals.items()},
                response_error_eligible_mean=totals['response_error_eligible_sum']/max(eligible_epoch,1),
                learning_rate_last=lr,gradient_clip_fraction=clipped/247,seconds=time.perf_counter()-started,
                strata={key:dict(count=v['count'],mean_iq_nmse=v['sum']/max(v['count'],1),observable_count=v['observable']) for key,v in strata.items()})
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='RESPONSE_EPOCH_COMPLETE',candidate=args.row,**record),allow_nan=False),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Complete fixed new E20/fullLU budget required')
    if any(p.grad is not None or p.requires_grad for p in model.frozen.parameters()):raise ValueError('Original context received training gradients')
    runtime.save_checkpoint(row/'final.pt',model,optimizer,20,updates,parent)
    write_json(root/'training_complete.json',dict(status='SOURCE_TRAINED',candidate=args.row,epochs=20,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,L_waveform_forwards=3*seen_l,U_waveform_forwards=3*seen_u,
        eligible_response_visits=eligible_total,response_eligible_fraction=eligible_total/(seen_l+seen_u),
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,target_reads=0,frozen_context_updates=0,
        prior_initialization='scratch',inherited_prior_epochs=0,prior_updates=updates,coarse_updates=0,action_estimator_updates=0,
        response_constraint_active=args.row==d.CANDIDATE,trainable_parameters=sum(p.numel() for p in parameters),
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
