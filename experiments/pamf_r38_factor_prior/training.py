import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import objective,paired_stages
from .runtime import initialize,save_checkpoint
from experiments.pamf_restoration.runner import array_tensor
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
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN:raise ValueError('Fixed source cache and arm required before IO')
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:raise ValueError('Original TX-hidden full source roles required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args);teacher_before={k:v.detach().clone() for k,v in teacher.state_dict().items()}
    context_before={k:v.detach().clone() for k,v in model.frozen.state_dict().items()};parameters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in parameters)!=d.PRIOR_PARAMETERS[args.arm]:raise ValueError('Only complete shared prior is trainable')
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report);write_json(row/'resolved_config.json',dict(d.configuration(),arm=args.arm))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Original fixed received epoch required')
            names=['loss','iq_nmse','teacher_stage_nmse']
            totals=dict.fromkeys(names,0.);clipped=0;model.train();order=rng.permutation(63000)
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];optimizer.zero_grad(set_to_none=True);lr=d.learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                for micro in range(0,len(selected),d.MICRO):
                    take=selected[micro:micro+d.MICRO];x=array_tensor(clean[take],'cuda:0');r=array_tensor(received[take],'cuda:0')
                    outputs,desired=paired_stages(model,teacher,r);loss,terms=objective(outputs,desired,x)
                    if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source successor objective')
                    (loss.sum()/len(selected)).backward()
                    for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                teacher_IQ_examples=seen_l+seen_u,source_IQ_examples=seen_l+seen_u,F_inputs=2*(seen_l+seen_u)*int(d.TASK[args.arm]),
                learning_rate_last=lr,clip_fraction=clipped/247,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(dict(event='R38_FACTOR_PRIOR_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(2470,63000,567000):raise ValueError('Exactly ten new epochs required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('R38 teacher changed')
    if any(p.grad is not None for p in model.frozen.parameters()) or any(not torch.equal(v,context_before[k]) for k,v in model.frozen.state_dict().items()):raise ValueError('Original R38 context changed')
    save_checkpoint(row/d.KIND/'final.pt',model,None,10,updates,parent)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epochs=10,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
        source_IQ_examples=630000,teacher_IQ_examples=630000,student_context_examples=0,shared_context_examples=630000,teacher_parameter_updates=0,
        parameters=d.PARAMETERS[args.arm],trainable_parameters=sum(p.numel() for p in parameters),context_parameters=d.CONTEXT_PARAMETERS[args.arm],
        context_updates=0,coarse_updates=0,action_updates=0,prior_updates=updates,context_trainable=False,paired_clean_optimized=True,
        inherited_component_epochs=40,cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_prior_epochs=50,
        cumulative_component_epoch_sum=130,cumulative_serial_training_passes=90,initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=d.TASK[args.arm],F_weights_read=int(d.TASK[args.arm]),F_outputs_read=1260000*int(d.TASK[args.arm]),
        F_parameter_updates=0,head_fit_updates=0,R17_training_forward_examples=0,F_training_reference='source_paired_clean' if d.TASK[args.arm] else None,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
