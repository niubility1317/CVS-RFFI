import json
import os
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import objective,make_probes,action_from_arrays
from experiments.pamf_r17_compact.models import stages
from experiments.pamf_restoration.runner import read_json
from .runtime import initialize,save_checkpoint
from experiments.pamf_fullresolution_restore.training import learning_rate
from experiments.pamf_restoration.runner import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity


def acquire(args,training=True):
    import fcntl
    while True:
        with (args.output.parent/'receiver_residual_capacity4.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
            if state['free_mb']>=(18000 if training else 12000) and len(state['pids'])<2:
                value=initialize(args.arm,args.teacher_run,'cuda:0') if training else torch.zeros(1,device='cuda:0')
                torch.cuda.synchronize();return value
        time.sleep(5)


def train(args):
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN:raise ValueError('Fixed source-only row/cache required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(d.GPUS[args.arm]):raise ValueError('Registered physical GPU required')
    validate_cache(args.parent_action)
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False)
    roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Full original TX-hidden L6300/U56700 required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args)
    before={key:value.detach().clone() for key,value in teacher.state_dict().items()}
    parameters=list(model.parameters())
    if not all(p.requires_grad for p in parameters):raise ValueError('Joint coarse/action/prior optimization required')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir()
    (row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report)
    write_json(row/'resolved_config.json',dict(d.configuration(),arm=args.arm))
    generator=torch.Generator(device='cuda:0').manual_seed(d.PROBE_SEED)
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            arrays=epoch_arrays(cache,epoch)
            received=arrays['received']
            if received.shape!=clean.shape:raise ValueError('Original full received source epoch required')
            totals=dict(loss=0.,iq_nmse=0.,teacher_stage_nmse=0.,**{key:0. for key in d.WEIGHTS});clipped=0;model.train();order=rng.permutation(63000)
            for offset in range(0,len(order),d.BATCH):
                selected=order[offset:offset+d.BATCH];optimizer.zero_grad(set_to_none=True)
                lr=learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                for micro in range(0,len(selected),d.MICRO):
                    take=selected[micro:micro+d.MICRO];x=array_tensor(clean[take],'cuda:0');r=array_tensor(received[take],'cuda:0')
                    with torch.no_grad():desired=teacher.stage_outputs(r)[0]
                    context=model.estimate_context(r);outputs=stages(model.prior,r,context)[0]
                    target=action_from_arrays(array_tensor(arrays['kernel'][take],'cuda:0'),array_tensor(arrays['coefficients'][take],'cuda:0'))
                    native_target=array_tensor(arrays['variance'][take],'cuda:0')
                    loss,terms=objective(outputs,desired,x,epoch,context,target,native_target,make_probes(len(take),'cuda:0',generator))
                    if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite joint paired source objective')
                    (loss.sum()/len(selected)).backward()
                    for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},teacher_weight=d.teacher_weight(epoch),physics_weights=d.WEIGHTS,physics_target_examples=seen_l+seen_u,action_probe_vectors=2*(seen_l+seen_u),
                L_exposures=seen_l,U_exposures=seen_u,teacher_IQ_examples=seen_l+seen_u,student_context_examples=seen_l+seen_u,
                learning_rate_last=lr,clip_fraction=clipped/247,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='IQ_PHYSICS_COMPACT_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Complete fixed joint compression budget required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('Original source IQ teacher must stay unchanged')
    save_checkpoint(row/d.KIND/'final.pt',model,None,20,updates,parent)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epochs=20,updates=updates,L_exposures=seen_l,U_exposures=seen_u,
        teacher_IQ_examples=seen_l+seen_u,student_context_examples=seen_l+seen_u,teacher_parameter_updates=0,
        context_updates=updates,prior_updates=updates,paired_clean_optimized=True,simulated_physics_anchors=True,physics_target_examples=seen_l+seen_u,action_probe_vectors=2*(seen_l+seen_u),
        inherited_prior_epochs=20,cumulative_prior_epochs=40,cumulative_coarse_epochs=40,cumulative_action_epochs=40,
        original_teacher_component_epochs=60,new_student_component_epochs=60,
        initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters),
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,training_F_access=False,head_fit_updates=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def validate_cache(root):
    if root.name!=d.ACTION_RUN or read_json(root/'launch.json').get('commit')!=d.ACTION_COMMIT:
        raise ValueError('Original R07 source-only physics cache required')
    header=read_json(root/'cache/complete.json')
    expected=dict(status='ACTION_SOURCE_CACHE_COMPLETE',train_count=63000,source_V_count=1800,TX_labels_read=0,target_reads=0)
    if any(header.get(k)!=v for k,v in expected.items()):raise ValueError('Existing original simulated source cache required')


def epoch_arrays(cache,epoch):
    if type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Fixed original20 source epochs required')
    folder=cache/'epochs'/f'{epoch:02d}'
    shapes=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
    values={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in shapes}
    if any(values[key].shape!=shape for key,shape in shapes.items()):raise ValueError('Matched full original source physics arrays required')
    return values
