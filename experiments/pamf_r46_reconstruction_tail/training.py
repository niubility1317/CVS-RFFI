import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .losses import objective
from .runtime import initialize,save_checkpoint
from .tensor_io import array_tensor
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.dispatcher import capacity


def predecessor_exited(args,gpu):
    cfg=d.runtime_configuration();root=args.output.parent/cfg['predecessor_run']
    if (root/'failure.json').exists() or (root/'source_path/failure.json').exists():raise ValueError('Predecessor failed;preserve outputs,no GPU takeover')
    if read_json(root/'launch.json').get('commit')!=cfg['predecessor_commit']:raise ValueError('Exact R66 predecessor method required')
    source=root/'source_worker.json'
    if not (root/'shared_source_complete.json').is_file():return False
    if read_json(root/'shared_source_complete.json').get('source_worker_exited') is not True:raise ValueError('Actual R66 source exit required')
    required=cfg['predecessor_by_GPU'][str(gpu)]
    for arm in required:
        row=root/arm
        if (row/'failure.json').exists():raise ValueError('R66 predecessor technical failure;no GPU takeover')
        if not (row/'complete.json').is_file():return False
        if read_json(row/'complete.json').get('status')!='ANALYZED':raise ValueError('R66 fixed readout completion required')
    handles=read_json(root/'workers.json')['workers']
    for arm in required:
        matches=[v for v in handles if v['arm']==arm and v['GPU']==gpu]
        if len(matches)!=1:raise ValueError('One exact preceding readout handle required')
        proc=Path('/proc')/str(matches[0]['pid'])
        try:
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(proc/'cmdline').read_bytes().decode().split(chr(0))
            if state!='Z' and 'experiments.pamf_r46_weight_mean.runner' in argv and str(root) in argv:return False
        except FileNotFoundError:pass
    return True


def acquire(args,training=True):
    import fcntl
    last=0.
    while True:
        ready=predecessor_exited(args,d.GPUS[args.arm])
        if ready:
            with (args.output.parent/d.runtime_configuration()['capacity_lock']).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);state=capacity()[d.GPUS[args.arm]]
                if state['free_mb']>=(18000 if training else 12000) and len(state['pids'])<2:
                    value=initialize(args.arm,args.compact_parent,'cuda:0') if training else torch.zeros(1,device='cuda:0')
                    torch.cuda.synchronize();return value
        if time.monotonic()-last>=30:
            print(json.dumps(dict(event='CPU_WAIT_R66_ACTUAL_SAME_CARD_EXIT_OR_CAPACITY',arm=args.arm,GPU=d.GPUS[args.arm],predecessor_ready=ready,optimizer_updates=0,F_inputs=0,target_reads=0)),flush=True)
            last=time.monotonic()
        time.sleep(5)


def train(args):
    if args.arm not in d.ARMS or args.parent_action.name!=d.ACTION_RUN or d.BATCH!=256 or d.MICRO!=256:raise ValueError('Fixed full source/global batch required before IO')
    cache=args.parent_action/'cache';clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:raise ValueError('All original TX-hidden L/U required')
    torch.manual_seed(d.MODEL_SEED);torch.cuda.manual_seed_all(d.MODEL_SEED)
    model,teacher,parent=acquire(args);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    teacher_before={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()};parameters=list(model.parameters())
    if not all(p.requires_grad for p in parameters):raise ValueError('All context and prior parameters must jointly train')
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=1e-4)
    row=args.output/args.arm;row.mkdir();(row/d.KIND).mkdir();(row/'reference_controls').symlink_to(args.teacher_run/'unroll_unshared/reference_controls',target_is_directory=True)
    write_json(row/'gpu_ready.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,physical_GPU=d.GPUS[args.arm]))
    write_json(row/'initialization.json',model.initialization_report);write_json(row/'resolved_config.json',dict(d.runtime_configuration(),arm=args.arm))
    rng=np.random.default_rng(d.DATA_SEED);updates=seen_l=seen_u=hard_l=hard_u=0;enqueue=0.;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,21):
            received=np.load(cache/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape!=clean.shape:raise ValueError('Original fixed received epoch required')
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','teacher_stage_nmse','reconstruction_mean','reconstruction_tail','reconstruction_risk'),0.)
            clipped=0;model.train();order=rng.permutation(63000)
            for offset in range(0,len(order),256):
                take=order[offset:offset+256];optimizer.zero_grad(set_to_none=True);lr=d.learning_rate(updates)
                for group in optimizer.param_groups:group['lr']=lr
                x=array_tensor(clean[take],'cuda:0');r=array_tensor(received[take],'cuda:0')
                with torch.no_grad():desired=teacher.stage_outputs(r)[0]
                outputs=model.stage_outputs(r)[0];loss,terms,focus=objective(outputs,desired,x,epoch,args.arm)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite reconstruction risk')
                loss.mean().backward()
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
                mask=np.asarray(focus['selected'].detach().cpu().tolist(),dtype=bool)
                hard_l+=int((roles[take][mask]=='L_s').sum());hard_u+=int((roles[take][mask]=='U_s').sum());enqueue+=focus['host_enqueue_seconds']
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);clipped+=int(norm>1)
                optimizer.step();updates+=1;seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
            torch.cuda.synchronize()
            record=dict(epoch=epoch,updates=updates,losses={k:v/63000 for k,v in totals.items()},L_exposures=seen_l,U_exposures=seen_u,
                hard_L_exposures=hard_l,hard_U_exposures=hard_u,hard_exposures=hard_l+hard_u,
                effective_L_weight_sum=.5*seen_l+.5/d.FRACTION[args.arm]*hard_l,effective_U_weight_sum=.5*seen_u+.5/d.FRACTION[args.arm]*hard_u,
                teacher_IQ_examples=seen_l+seen_u,source_IQ_examples=seen_l+seen_u,F_inputs=0,teacher_weight=d.teacher_weight(epoch),
                focus_fraction=d.FRACTION[args.arm],focus_mix=d.MIX,focus_host_enqueue_seconds=enqueue,
                learning_rate_last=lr,clip_fraction=clipped/247,microbatch=256,source_tensor_backend='direct_float32_buffer',seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(dict(event='R46_RECONSTRUCTION_TAIL_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000) or hard_l+hard_u!=int(1260000*d.FRACTION[args.arm]):raise ValueError('Full source and actual tail exposure required')
    if any(p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v.detach().cpu(),teacher_before[k]) for k,v in teacher.state_dict().items()):raise ValueError('Original M35 teacher changed')
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        if not any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items() if k.startswith(prefix)):raise ValueError('A joint component did not update')
    save_checkpoint(row/d.KIND/'final.pt',model,None,20,4940,parent)
    complete=dict(status='SOURCE_TRAINED',arm=args.arm,epochs=20,updates=4940,L_exposures=seen_l,U_exposures=seen_u,
        source_IQ_examples=1260000,teacher_IQ_examples=1260000,student_context_examples=1260000,parameters=273765,trainable_parameters=273765,context_parameters=185615,
        context_updates=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,context_trainable=True,paired_clean_optimized=True,prior_parameters_changed=True,
        inherited_component_epochs=80,cumulative_coarse_epochs=100,cumulative_action_epochs=100,cumulative_prior_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,
        initialization_IQ_examples=0,initialization_seconds=model.initialization_report['seconds'],new_calibration_IQ=0,new_trainable_parameters=0,extra_training_G_forwards=0,external_student_checkpoint_reads=0,
        TX_labels_read=0,V_reads=0,target_reads=0,training_F_access=False,F_weights_read=0,F_outputs_read=0,teacher_parameter_updates=0,F_parameter_updates=0,head_fit_updates=0,
        R17_training_forward_examples=0,F_training_reference=None,focus_fraction=d.FRACTION[args.arm],focus_mix=d.MIX,
        hard_L_exposures=hard_l,hard_U_exposures=hard_u,hard_exposures=hard_l+hard_u,
        effective_L_weight_sum=.5*seen_l+.5/d.FRACTION[args.arm]*hard_l,effective_U_weight_sum=.5*seen_u+.5/d.FRACTION[args.arm]*hard_u,
        focus_host_enqueue_seconds=enqueue,focus_time_scope='Only weighting host enqueue;GPU weighting kernels included in full training walltime,not isolated addedGPUcost',
        persistent_hardness_state=False,teacher_term_reweighted=False,seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(row/'training_complete.json',complete)
