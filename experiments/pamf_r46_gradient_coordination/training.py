import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import objective,paired_outputs
from .runtime import initialize,save_checkpoint
from .gradients import assign_gradients,parameter_groups,empty_epoch_stats,add_epoch_stats,finish_epoch_stats
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity


def acquire(args,training=True):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=(18000 if training else 12000) and len(state['pids'])<2:
                value=initialize(args.arm,args.compact_parent,'cuda:0') if training else torch.zeros(1,device='cuda:0')
                torch.cuda.synchronize();return value
        time.sleep(5)


def train(args):
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN or (d.BATCH,d.MICRO)!=(64,64):raise ValueError('Fixed source cache and arm required before IO')
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:raise ValueError('Original TX-hidden full source roles required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args);teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    context_before={k:v.detach().clone() for k,v in model.frozen.state_dict().items()};parameters,gradient_groups=parameter_groups(model)
    if not all(p.requires_grad for p in parameters):raise ValueError('Whole successor must remain trainable')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report);write_json(row/'resolved_config.json',dict(d.configuration(),arm=args.arm))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Original fixed received epoch required')
            names=['loss','iq_nmse','spectral_error','teacher_stage_nmse','balanced_complex_nmse','clean_identity_nmse',
                'base_loss','clean_moment_mse','teacher_moment_mse','moment_objective','weighted_moment_objective']
            totals=dict.fromkeys(names,0.);clipped=0;gradient_totals=empty_epoch_stats();model.train();order=rng.permutation(63000)
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];optimizer.zero_grad(set_to_none=True);lr=d.learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                x=array_tensor(clean[selected],'cuda:0');r=array_tensor(received[selected],'cuda:0')
                outputs,desired,identity=paired_outputs(model,teacher,r,x)
                loss,terms=objective(outputs,desired,x,epoch,args.arm,identity)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source gradient-coordination objective')
                gradient_stats=assign_gradients(model,terms,d.GRADIENT_POLICY[args.arm],(parameters,gradient_groups))
                add_epoch_stats(gradient_totals,gradient_stats)
                ordered=dict(loss=loss,**terms)
                values=torch.stack([value.detach().sum() for value in ordered.values()]).cpu().tolist()
                for key,value in zip(ordered,values):totals[key]+=value
                del outputs,desired,identity,terms,loss,x,r,ordered
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                teacher_IQ_examples=seen_l+seen_u,source_IQ_examples=seen_l+seen_u,extra_clean_student_inputs=(seen_l+seen_u)*int(d.CLEAN_WEIGHT[args.arm]>0),F_inputs=0,teacher_weight=d.teacher_weight(epoch),
                moment_weight=d.MOMENT_WEIGHT,moment_reference=d.MOMENT_REFERENCE[args.arm],
                gradient_policy=d.GRADIENT_POLICY[args.arm],gradient_evaluations=2*updates,gradient_statistics=finish_epoch_stats(gradient_totals),
                extra_moment_G_inputs=0,moment_descriptor_inputs=3*(seen_l+seen_u),
                learning_rate_last=lr,clip_fraction=clipped/985,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(dict(event='R46_GRADIENT_COORDINATION_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(19700,126000,1134000):raise ValueError('Exactly twenty new epochs required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('Fixed R46 teacher changed')
    for prefix in ('coarse.','estimator.'):
        if not any(not torch.equal(v,context_before[k]) for k,v in model.frozen.state_dict().items() if k.startswith(prefix)):raise ValueError('A joint context module did not update')
    save_checkpoint(row/d.KIND/'final.pt',model,None,20,updates,parent)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epochs=20,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,teacher_parameter_updates=0,
        parameters=d.PARAMETERS[args.arm],trainable_parameters=sum(p.numel() for p in parameters),context_parameters=d.CONTEXT_PARAMETERS[args.arm],
        context_updates=updates,coarse_updates=updates,action_updates=updates,prior_updates=updates,context_trainable=True,paired_clean_optimized=True,balanced_complex_weight=d.COMPLEX_WEIGHT,clean_identity_weight=d.CLEAN_WEIGHT[args.arm],extra_clean_student_inputs=1260000,student_total_IQ_examples=2520000,
        moment_weight=d.MOMENT_WEIGHT,moment_reference=d.MOMENT_REFERENCE[args.arm],
        extra_moment_G_inputs=0,moment_descriptor_inputs=3780000,conditional_teacher_prior_examples=0,
        gradient_policy=d.GRADIENT_POLICY[args.arm],gradient_evaluations=39400,gradient_groups=list(d.GRADIENT_GROUPS),gradient_norm_floor=d.NORM_FLOOR,
        inherited_component_epochs=80,cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,
        cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,
        F_parameter_updates=0,head_fit_updates=0,R17_training_forward_examples=0,F_training_reference=None,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
