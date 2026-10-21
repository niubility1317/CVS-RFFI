import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from . import design as d
from .models import objective
from .runtime import initialize,save_checkpoint
from .averaging import SourceParameterMeans
from .tensor_io import array_tensor
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity
from experiments.pamf_restoration.runner import read_json


def predecessor_exited(args,gpu):
    root=args.output.parent/d.configuration()['predecessor_run'];arm=d.configuration()['predecessor_by_GPU'][str(gpu)]
    row=root/arm
    if not (row/'complete.json').is_file():return False
    if read_json(row/'complete.json').get('status')!='ANALYZED' or (row/'failure.json').exists():raise ValueError('Preceding fixed row did not complete successfully')
    matches=[w for w in read_json(root/'workers.json')['workers'] if w['arm']==arm and w['GPU']==gpu]
    if len(matches)!=1:raise ValueError('One exact same-card predecessor handle required')
    proc=Path('/proc')/str(matches[0]['pid'])
    try:
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(proc/'cmdline').read_bytes().decode().split(chr(0))
        return state=='Z' or not ('experiments.pamf_r46_delayed_prune.runner' in argv and str(root) in argv)
    except FileNotFoundError:return True


def acquire(args,*,source):
    import fcntl
    gpu=d.SOURCE_GPU if source else d.GPUS[args.arm]
    while True:
        if not predecessor_exited(args,gpu):time.sleep(5);continue
        with (args.output.parent/d.configuration()['capacity_lock']).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[gpu]
            if state['free_mb']>=(18000 if source else 12000) and len(state['pids'])<2:
                value=initialize(args.compact_parent,'cuda:0') if source else torch.zeros(1,device='cuda:0')
                torch.cuda.synchronize();return value
        time.sleep(5)


def train_source(args):
    if args.arm!=d.SOURCE_ARM or args.parent_action.name!=d.ACTION_RUN:raise ValueError('Only one registered source path before IO')
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Original full TX-hidden L/U source data required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args,source=True);teacher_before={n:p.detach().cpu().clone() for n,p in teacher.state_dict().items()}
    before={n:p.detach().cpu().clone() for n,p in model.state_dict().items()};parameters=list(model.parameters())
    if not all(p.requires_grad for p in parameters):raise ValueError('All context and prior parameters must jointly train')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4);means=SourceParameterMeans(model)
    row=args.output/d.SOURCE_ARM;row.mkdir();write_json(row/'initialization.json',model.initialization_report)
    write_json(row/'resolved_config.json',dict(d.configuration(),arm=d.SOURCE_ARM))
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.SOURCE_GPU))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=0;mean_host_seconds=0.;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Original fixed received epoch required')
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','teacher_stage_nmse'),0.);clipped=0;model.train();order=rng.permutation(63000)
            for offset in range(0,len(order),256):
                selected=order[offset:offset+256];optimizer.zero_grad(set_to_none=True);lr=d.learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                x=array_tensor(clean[selected],'cuda:0');r=array_tensor(received[selected],'cuda:0')
                with torch.no_grad():desired=teacher.stage_outputs(r)[0]
                outputs=model.stage_outputs(r)[0];loss,terms=objective(outputs,desired,x,epoch)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source objective')
                loss.mean().backward()
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1
                tick=time.perf_counter();means.after_step(model,updates);mean_host_seconds+=time.perf_counter()-tick
                seen_l+=int((roles[selected]=='L_s').sum());seen_u+=int((roles[selected]=='U_s').sum())
            tick=time.perf_counter();means.after_epoch(model,epoch);mean_host_seconds+=time.perf_counter()-tick
            torch.cuda.synchronize()
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                teacher_IQ_examples=seen_l+seen_u,source_IQ_examples=seen_l+seen_u,F_inputs=0,teacher_weight=d.teacher_weight(epoch),
                learning_rate_last=lr,clip_fraction=clipped/247,microbatch=256,source_tensor_backend='direct_float32_buffer',
                EMA_updates=means.ema_count,late_mean_snapshots=means.mean_count,averaging_host_enqueue_seconds=mean_host_seconds,
                shared_source_training_paths=1,seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            print(json.dumps(dict(event='R46_WEIGHT_MEAN_SOURCE_EPOCH_COMPLETE',arm=d.SOURCE_ARM,**record)),flush=True)
    if (updates,seen_l,seen_u,means.ema_count,means.mean_count)!=(4940,126000,1134000,2470,5):raise ValueError('One full twenty-epoch source path and all averages required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(p.detach().cpu(),teacher_before[n]) for n,p in teacher.state_dict().items()):raise ValueError('Original M35 teacher changed')
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        if not any(not torch.equal(p.detach().cpu(),before[n]) for n,p in model.state_dict().items() if n.startswith(prefix)):
            raise ValueError('A jointly trained source component did not update')
    torch.cuda.synchronize();training_seconds=time.perf_counter()-started
    for arm in d.ARMS:
        readout=args.output/arm;readout.mkdir();(readout/d.KIND).mkdir()
        (readout/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
        write_json(readout/'initialization.json',model.initialization_report)
        write_json(readout/'resolved_config.json',dict(d.configuration(),arm=arm,source_training_shared=True))
        info=means.metadata(arm);write_json(readout/'parameter_readout.json',info)
        state=means.state_for(model,arm);save_checkpoint(readout/d.KIND/'final.pt',state,arm,20,4940,parent,model.initialization_report,info)
        del state
    complete=dict(status='SOURCE_TRAINED',arm=d.SOURCE_ARM,epochs=20,updates=4940,L_exposures=seen_l,U_exposures=seen_u,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,shared_source_training_paths=1,
        training_counted_once_across_readouts=True,readouts=list(d.ARMS),parameters=273765,trainable_parameters=273765,context_parameters=185615,
        context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,paired_clean_optimized=True,
        prior_parameters_changed=True,EMA_updates=2470,late_mean_snapshots=5,averaging_host_enqueue_seconds=mean_host_seconds,
        averaging_time_scope='Host enqueue walltime only;GPU averaging kernels included in actual full training seconds,not isolated added GPU cost',
        inherited_component_epochs=80,cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,
        cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],
        new_calibration_IQ=0,new_trainable_parameters=0,extra_training_neural_forwards=0,external_student_checkpoint_reads=0,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,teacher_parameter_updates=0,
        F_parameter_updates=0,head_fit_updates=0,R17_training_forward_examples=0,F_training_reference=None,
        seconds=training_seconds,seconds_including_all_readout_serialization=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(row/'training_complete.json',complete)
    write_json(row/'complete.json',dict(status='SOURCE_TRAINED',readouts=list(d.ARMS),all_three_readouts_frozen=True,training_counted_once=True,goal_complete=False))
