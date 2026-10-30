"""One joint E30 schedule with early frozen source-R17 coarse-IQ hints."""
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from . import no_variance
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.r17_teacher_paths import (
    TEACHER_RUN,TEACHER_COMMIT,validate_teacher)
base=no_variance.base
RUN='20261031-phase1-DURA-coarse-IQhint-E30-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.coarse_hint'


def hint_weight(epoch):
    if not 1<=epoch<=30:raise ValueError('Fixed E30 source schedule required')
    return 1.-.9*(epoch-1)/9 if epoch<=10 else 0.


def learning_rate(update):
    if not 0<=update<7410:raise ValueError('Fixed E30 source update budget required')
    if update<247:return .001*(update+1)/247
    return .00002+(.001-.00002)*.5*(1+math.cos(math.pi*(update-247)/(7410-247-1)))


def load_teacher(root,device):
    from experiments.pamf_unrolled_restore.runtime import load_checkpoint
    folder=root/'runs'/TEACHER_RUN;row=folder/'unroll_unshared'
    teacher,payload=load_checkpoint(row/'native_map_prior/final.pt',device)
    validate_teacher(payload,base.read_json(row/'training_complete.json'),
        base.read_json(folder/'launch.json'),base.read_json(folder/'source_decision.json'))
    return teacher.eval().requires_grad_(False)


def objective(student,received,clean,action,variance,probes,reference=None,weight=0.):
    _,terms,restored,coarse=source_objective(student,received,clean,action,variance,probes,return_coarse=True)
    energy=clean.detach().square().mean((1,2)).clamp_min(1e-4)
    if weight:
        if reference is None:raise ValueError('Active IQ hint requires the matched frozen source reference')
        hint=(coarse-reference.detach()).square().mean((1,2))/energy
    else:hint=torch.zeros_like(terms['iq_nmse'])
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+weight*hint
    terms=dict(terms,coarse_hint_nmse=hint,coarse_clean_iq_nmse=(coarse.detach()-clean.detach()).square().mean((1,2))/energy)
    return loss,terms


@torch.no_grad()
def prepare_teacher_cache(args,p,row):
    started=time.perf_counter();folder=row/'coarse_teacher_cache';folder.mkdir()
    teacher=load_teacher(args.root,'cpu');coarse=teacher.frozen.coarse
    before={k:v.detach().cpu().clone() for k,v in coarse.state_dict().items()}
    coarse.to('cuda:0');inputs=0
    for epoch in range(1,11):
        received=np.load(p['cache']/'epochs'/f'{epoch:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
        if received.shape!=(63000,2,256) or received.dtype!=np.float32:raise ValueError('Original source corruption view required')
        target=np.lib.format.open_memmap(folder/f'epoch{epoch:02d}.npy',mode='w+',dtype=np.float32,shape=received.shape)
        for offset in range(0,63000,256):
            end=min(offset+256,63000);iq=coarse(base.tensor(received[offset:end],'cuda:0'))
            if iq.shape!=(end-offset,2,256) or not torch.isfinite(iq).all():raise FloatingPointError('Nonfinite frozen coarse IQ cache')
            target[offset:end]=base.numpy_values(iq);inputs+=end-offset
        target.flush()
        print(json.dumps(dict(event='FROZEN_SOURCE_COARSE_CACHE_EPOCH',epoch=epoch,count=63000,teacher_inputs=inputs)),flush=True)
    if inputs!=630000 or any(v.grad is not None for v in teacher.parameters()) or any(not torch.equal(v.detach().cpu(),before[k]) for k,v in coarse.state_dict().items()):
        raise ValueError('Frozen source IQ teacher state or input budget changed')
    ids=np.load(p['cache']/'train/physical_ids.npy',allow_pickle=False);np.save(folder/'physical_ids.npy',ids,allow_pickle=False)
    record=dict(status='SOURCE_COARSE_IQ_CACHE_COMPLETE',teacher_run=TEACHER_RUN,teacher_commit=TEACHER_COMMIT,
        teacher_component='frozen.coarse',teacher_used_parameters=sum(v.numel() for v in coarse.parameters()),
        teacher_loaded_parameters=sum(v.numel() for v in teacher.parameters()),teacher_component_epochs=20,
        teacher_bundle_component_epochs=60,epochs=list(range(1,11)),count_per_epoch=63000,teacher_inputs=inputs,
        numeric_bytes=630000*2*256*4,teacher_updates=0,teacher_unchanged=True,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    base.write_json(folder/'complete.json',record)
    del teacher,coarse;gc.collect();torch.cuda.empty_cache()
    return folder,record


def smoke(args,p):
    clean,roles=base.source_training(p);take=np.flatnonzero(roles=='L_s')[:2]
    folder=p['cache']/'epochs/01'
    values={k:base.tensor(np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu') for k in ('received','kernel','coefficients','variance')}
    x=base.tensor(clean[take],'cpu');teacher=load_teacher(args.root,'cpu')
    before={k:v.clone() for k,v in teacher.state_dict().items()}
    with torch.no_grad():reference=teacher.frozen.coarse(values['received']).detach()
    student=base.scratch_model(2026092701);generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms=objective(student,values['received'],x,base.supervised_action(values['kernel'],values['coefficients']),
        values['variance'],probes,reference,1.)
    loss.mean().backward();gradients=[v.grad for v in student.parameters() if v.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):
        raise ValueError('Finite nonzero source student gradients required')
    if any(v.grad is not None for v in teacher.parameters()) or any(not torch.equal(v,before[k]) for k,v in teacher.state_dict().items()):
        raise ValueError('IQ hint changed frozen R17')
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';base.save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    fresh=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(fresh(values['received']),expected,atol=0,rtol=0)
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():logits=reader(expected)
    if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Original F/default-head source smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,student_scratch=True,initialization_checkpoint_loads=0,
        IQ_teacher_loaded=True,IQ_teacher_unchanged=True,training_F_access=False,F_used_only_in_noquery_smoke=True,
        TX_labels_read=0,target_reads=0,source_student_gradient_nonzero=True,strict_reload=True))
    print(json.dumps(dict(event='REAL_SOURCE_COARSE_HINT_NOQUERY_SMOKE_PASS')),flush=True)


def configure():
    no_variance.configure();previous_configuration=base.configuration;previous_command=base.command
    def configuration():
        value=previous_configuration()
        value.update(run=RUN,recipe='scratch_E30_early_coarse_IQ_hint',epochs=30,updates=7410,new_cumulative_epochs=30,
            L_exposures=189000,U_exposures=1701000,corruption_epochs=20,
            source_freeze='fixedfinalE30/7410,noearlystop/source-target-ranking',
            objective='noVar4base + earlyweight*NMSE(studentcoarse,frozenR17coarse);allGjointfromfirststep',
            teacher_model=dict(run=TEACHER_RUN,commit=TEACHER_COMMIT,component='frozen.coarse',
                component_pretrained_epochs=20,bundle_component_epochs=60,student_copied_weights=0),
            coarse_hint_weights=[hint_weight(epoch) for epoch in range(1,31)],teacher_cache_inputs=630000,
            teacher_training_visits=630000,teacher_cache_numeric_bytes=1290240000,
            source_supervision='Sourcepairedclean/action andfrozenoriginalR17coarseIQ,noTX/F/V/targettraining',
            training_stages=1,parameters=266951,training_F_access=False,F_guided_training=False,
            extra_loss_metrics=['coarse_hint_nmse','coarse_clean_iq_nmse'],
            optimizer_control='FixedE30physicalreadout ofIQenhance run;physicalparametershadnoFgradient,butwholeprocessF-guided/costnotindependentF-free',
            allowed_GPUs=[4,6,7],max_total_visible_GPU_processes=3,host_numpy_conversion='DLPack_owned_FP32')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if v==no_variance.MODULE else v for v in previous_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command;base.learning_rate=learning_rate
    base.smoke=smoke;base.train=train


# Reuse the original source/cache/checkpoint contracts. The configuration is
# resolved after configure(), so source-final E30 metadata stays exact.
source_training=base.source_training
tensor=base.tensor
supervised_action=base.supervised_action
training_pair=base.training_pair
scratch_model=base.scratch_model
write_json=base.write_json
save_model=base.save_model
def configuration():return base.configuration()


def train(args,p):
    started_total=time.perf_counter()
    clean,roles=source_training(p)
    config=configuration()
    row=p['output']/args.arm;row.mkdir()
    write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit))
    write_json(row/'resolved_config.json',config)
    teacher_cache,cache_record=prepare_teacher_cache(args,p,row)
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=scratch_model(2026092701).to('cuda:0')
    before={key:value.detach().cpu().clone() for key,value in student.state_dict().items()}
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3,weight_decay=1e-4)
    write_json(row/'initialization.json',student.initialization_report)
    rng=np.random.default_rng(2026103004)
    updates=seen_l=seen_u=teacher_visits=0
    preparation_seconds=time.perf_counter()-started_total
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,config['epochs']+1):
            cache_epoch=(epoch-1)%config.get('corruption_epochs',20)+1
            folder=p['cache']/'epochs'/f'{cache_epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
            expected=dict(received=(63000,2,256),kernel=(63000,2,64),coefficients=(63000,4,256),variance=(63000,))
            if any(value.shape!=expected[key] or value.dtype!=np.float32 for key,value in arrays.items()):
                raise ValueError('Fixed matching full sourceIQ/action epoch required')
            totals=dict.fromkeys(('loss','iq_nmse','spectral_error','prior_nmse','action_probe_nmse','log_variance_mse',
                                  *config.get('extra_loss_metrics',[])),0.)
            weight=hint_weight(epoch)
            targets=np.load(teacher_cache/f'epoch{epoch:02d}.npy',mmap_mode='r',allow_pickle=False) if weight else None
            if targets is not None and (targets.shape!=(63000,2,256) or targets.dtype!=np.float32):
                raise ValueError('Matched source coarse IQ cache required')
            clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256]
                x=tensor(clean[take],'cuda:0')
                values={key:tensor(value[take],'cuda:0') for key,value in arrays.items()}
                action=supervised_action(values['kernel'],values['coefficients'])
                values['received'],x=training_pair(values['received'],x,action,take)
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=learning_rate(updates)
                reference=tensor(targets[take],'cuda:0') if targets is not None else None
                loss,terms=objective(student,values['received'],x,action,values['variance'],probes,reference,weight)
                if reference is not None:teacher_visits+=len(take)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source direct objective')
                loss.mean().backward()
                norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,coarse_hint_weight=weight,seconds=time.perf_counter()-started,
                F_inputs=0,TX_labels_read=0,teacher_reference_visits=teacher_visits,live_teacher_training_inputs=0,
                learning_rate_last=optimizer.param_groups[0]['lr'],initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='SOURCE_EPOCH_COMPLETE',arm=args.arm,**record)),flush=True)
    if (updates,seen_l,seen_u)!=(config['updates'],config['L_exposures'],config['U_exposures']):raise ValueError('Incomplete fixed source budget')
    if teacher_visits!=630000:raise ValueError('Full first10 IQ hint exposure required')
    changed=sum(not torch.equal(value.detach().cpu(),before[key]) for key,value in student.state_dict().items())
    if not changed:raise ValueError('Random initialized model did not train')
    save_model(row/'final.pt',student,args.arm,config['epochs'],updates,args.commit)
    write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',arm=args.arm,epoch=config['epochs'],updates=updates,
        L_exposures=seen_l,U_exposures=seen_u,training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,
        teacher_updates=0,teacher_reference_visits=teacher_visits,live_teacher_training_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,
        cumulative_epochs=config['epochs'],head_fit_updates=0,parameters=266951,changed_state_tensors=changed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-started,
        source_preparation_seconds=preparation_seconds,total_G_training_pipeline_seconds=time.perf_counter()-started_total,
        teacher_cache=cache_record,teacher_cache_forward_inputs=630000,teacher_cache_numeric_bytes=1290240000,
        teacher_component_epochs=20,teacher_bundle_component_epochs=60,student_copied_teacher_parameters=0))


if __name__=='__main__':configure();base.main()
