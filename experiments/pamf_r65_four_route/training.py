import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .runtime import initialize,save_checkpoint
from experiments.pamf_r44_pruned_refine.models import spectral_error
from experiments.pamf_r46_multipoint_fast.tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity


def objective(student,teacher,clean,epoch,arm):
    targets=[teacher[i] for i in d.STAGE_TARGETS[arm]]
    if len(student)!=len(targets) or any(x.shape!=clean.shape for x in (*student,*targets)):
        raise ValueError('Complete aligned source IQ trajectory required')
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    paired=(student[-1]-clean.detach()).square().mean((1,2))/energy
    fidelity=torch.stack([(a-b.detach()).square().mean((1,2))/energy for a,b in zip(student,targets)]).mean(0)
    spectral=spectral_error(student[-1],clean)
    return paired+.05*spectral+d.teacher_weight(epoch)*fidelity,dict(iq_nmse=paired,spectral_error=spectral,teacher_stage_nmse=fidelity)


def acquire(args):
    import fcntl
    last=0.
    while True:
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=18000 and not state['pids']:
                value=initialize(args.arm,args.parent_run,'cuda:0')
                torch.cuda.synchronize()
                return value
        if time.monotonic()-last>30:
            print(json.dumps(dict(event='WAIT_FREE_GPU',arm=args.arm,GPU=d.GPUS[args.arm],free_mb=state['free_mb'],compute_pids=len(state['pids']))),flush=True)
            last=time.monotonic()
        time.sleep(5)


def train(args):
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN:
        raise ValueError('Fixed route and original source cache required before IO')
    cache=args.parent_action/'cache'
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original TX-hidden L6300/U56700 source roles required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher=acquire(args)
    teacher_before={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
    before={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    parameters=list(model.parameters())
    if not parameters or not all(p.requires_grad for p in parameters):
        raise ValueError('All retained/new student components must jointly optimize')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report)
    write_json(row/'resolved_config.json',dict(d.configuration(),arm=args.arm,
        parameters=sum(p.numel() for p in parameters),stage_count=len(d.STAGE_TARGETS[args.arm])))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,21):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:
                raise ValueError('Original fixed source received epoch required')
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','teacher_stage_nmse'),0.)
            model.train();order=rng.permutation(63000);clipped=0
            for offset in range(0,63000,256):
                selected=order[offset:offset+256]
                optimizer.zero_grad(set_to_none=True);lr=d.learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                x=array_tensor(clean[selected],'cuda:0');r=array_tensor(received[selected],'cuda:0')
                with torch.no_grad():desired=teacher.stage_outputs(r)[0]
                outputs=model.stage_outputs(r)[0]
                loss,terms=objective(outputs,desired,x,epoch,args.arm)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source objective')
                loss.mean().backward()
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,teacher_IQ_examples=seen_l+seen_u,
                source_IQ_examples=seen_l+seen_u,F_inputs=0,teacher_weight=d.teacher_weight(epoch),
                learning_rate_last=lr,clip_fraction=clipped/247,seconds=time.perf_counter()-started)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='R65_ROUTE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):
        raise ValueError('Complete fixed twenty source epochs required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v.detach().cpu(),teacher_before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('Frozen original R65 teacher changed')
    changed=[name for name,p in model.named_parameters() if not torch.equal(p.detach().cpu(),before[name])]
    if not changed:
        raise ValueError('No student parameter updated')
    torch.cuda.synchronize()
    save_checkpoint(row/d.KIND/'final.pt',model,20,updates,args.expected_commit)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epochs=20,updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,source_IQ_examples=1260000,teacher_IQ_examples=1260000,
        parameters=sum(p.numel() for p in parameters),trainable_parameters=sum(p.numel() for p in parameters),
        changed_parameter_tensors=len(changed),teacher_parameter_updates=0,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,
        training_F_access=False,V_reads=0,target_reads=0,F_parameter_updates=0,head_fit_updates=0,
        source_tensor_backend='direct_float32_buffer',seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),initialization_IQ_examples=0,new_calibration_IQ=0,
        parent_run=d.PARENT_RUN,parent_row=d.PARENT_ROW,all_components_joint=True))
