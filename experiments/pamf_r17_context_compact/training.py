import json
import os
import sys
import time
import numpy as np
import torch

from . import design as d
from .models import stages,distillation_loss
from .teacher import teacher_and_student_context
from tools.r17_task_fidelity import frozen_task_fidelity
from experiments.pamf_fcrossrx_readout.frozen import load_reader
from .runtime import initialize,save_checkpoint
from experiments.pamf_fullresolution_restore.training import learning_rate
from experiments.pamf_restoration.runner import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity


def acquire(args,training=True):
    import fcntl
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=(18000 if training else 12000) and len(state['pids'])<2:
                if training:
                    value=initialize(args.arm,args.teacher_run,'cuda:0')
                else:
                    value=torch.zeros(1,device='cuda:0')
                torch.cuda.synchronize()
                return value
        time.sleep(5)


def train(args):
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN:
        raise ValueError('Registered compact arm and original R07 cache required before IO/GPU access')
    cache=args.parent_action/'cache'
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Complete original L/U only required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args)
    reader=load_reader(args.reader_folder,args.source_cache,'cuda:0',seed=d.F_SEED).eval().requires_grad_(False)
    reader_before={key:value.detach().clone() for key,value in reader.state_dict().items()}
    before={key:value.detach().clone() for key,value in teacher.state_dict().items()}
    frozen_before={key:value.detach().clone() for key,value in model.frozen.state_dict().items()}
    parameters=list(model.prior.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report)
    write_json(row/'resolved_config.json',dict(d.configuration(),arm=args.arm))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,d.EPOCHS+1):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Original full cached epoch required')
            totals=dict(loss=0.,iq_nmse=0.,teacher_stage_nmse=0.,task_KL=0.,task_feature_distance=0.);clipped=0;model.train()
            order=rng.permutation(len(clean))
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];x=array_tensor(clean[selected],'cuda:0');r=array_tensor(received[selected],'cuda:0')
                with torch.no_grad():
                    desired,context=teacher_and_student_context(teacher,model,r)
                output=model.stage_outputs(r,context)[0]
                paired,terms=distillation_loss(output,desired,x)
                task,task_terms=frozen_task_fidelity(output[-1],desired[-1],reader)
                terms.update(task_terms);loss=(paired+task).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite compact distillation loss')
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
                for key,value in dict(loss=loss,**{k:v.mean() for k,v in terms.items()}).items():totals[key]+=float(value)*len(selected)
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},
                        L_exposures=seen_l,U_exposures=seen_u,teacher_IQ_examples=seen_l+seen_u,
                        learning_rate_last=lr,clip_fraction=clipped/247,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='TASK_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(d.UPDATES,126000,1134000):raise ValueError('Fixed full compact exposure required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('R17 teacher must remain unchanged')
    if any(p.grad is not None for p in model.frozen.parameters()) or any(not torch.equal(v,frozen_before[k]) for k,v in model.frozen.state_dict().items()):
        raise ValueError('Frozen coarse/action must remain unchanged')
    if any(p.grad is not None for p in reader.parameters()) or any(not torch.equal(v,reader_before[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Original F/default head must stay unchanged')
    save_checkpoint(row/d.KIND/'final.pt',model,None,d.EPOCHS,updates,parent)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epochs=d.EPOCHS,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,teacher_IQ_examples=seen_l+seen_u,
        inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_serial_training_passes=80,
        initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],
        total_teacher_IQ_examples=seen_l+seen_u,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
        teacher_parameter_updates=0,frozen_context_updates=0,context_parameters=d.CONTEXT_PARAMETERS,
        teacher_context_forward_examples=seen_l+seen_u,student_context_forward_examples=seen_l+seen_u,TX_labels_read=0,F_outputs_read=2*(seen_l+seen_u),F_weights_read=1,training_F_access=True,
        F_teacher_forward_examples=seen_l+seen_u,F_student_forward_examples=seen_l+seen_u,F_parameter_updates=0,head_fit_updates=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
