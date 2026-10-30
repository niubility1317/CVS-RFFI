"""Authorized source-only frozen-F guidance; one G training schedule, no TX reads."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from . import runner as base
from . import no_variance
from experiments.pamf_dura_efficiency.model import direct_objective as source_objective
from experiments.pamf_dura_efficiency.proposed_feature_loss import frozen_feature_loss

RUN='20261030-phase1-DURA-source-Fguide-cos010-s2026092701-r01'
MODULE='experiments.pamf_dura_direct_train.f_guided'
F_SOURCE_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
GUIDANCE_METRIC='F_feature_cosine_error'


def teacher_reference():return dict(run=base.F_RUN,seed=2026092701,source_commit=F_SOURCE_COMMIT,epoch=50,updates=11100)


def source_ids(p,roles):
    ids=np.load(p['cache']/'train/physical_ids.npy',allow_pickle=False)
    if ids.shape!=(63000,) or len(set(ids.tolist()))!=63000:raise ValueError('Original unique source physical IDs required')
    for role in ('L_s','U_s'):
        reference=np.load(p['source_cache']/role/'physical_ids.npy',allow_pickle=False)
        if set(ids[roles==role].tolist())!=set(reference.tolist()):
            raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: G source physical role versus F source')
    return ids


def save_model(path,model,arm,epoch,updates,commit):
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    validate_aliases(model)
    payload=dict(schema='dura_source_F_guided_v1',arm=arm,epoch=epoch,updates=updates,code_commit=commit,
        config=base.configuration(),source_contract='original7/63/30',parent_run=None,parent_arm=None,
        initialization='scratch_no_checkpoint',training_TX_labels_read=0,training_F_access=True,
        V_training=False,target_training=False,F_teacher=teacher_reference(),F_updates=0,model=model.state_dict())
    with path.open('xb') as stream:torch.save(payload,stream)


def load_model(path,device,*,arm,commit,epoch=20):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    expected=dict(schema='dura_source_F_guided_v1',arm=arm,epoch=epoch,updates=epoch*247,code_commit=commit,
        config=base.configuration(),source_contract='original7/63/30',parent_run=None,parent_arm=None,
        initialization='scratch_no_checkpoint',training_TX_labels_read=0,training_F_access=True,
        V_training=False,target_training=False,F_teacher=teacher_reference(),F_updates=0)
    if any(payload.get(key)!=value for key,value in expected.items()):
        raise ValueError('Authorized source F-guided checkpoint metadata required')
    from experiments.pamf_r65_four_route.runtime import validate_aliases
    model=base.scratch_model(2026092701)
    validate_aliases(model,payload['model']);model.load_state_dict(payload['model'],strict=True)
    return model.to(device).eval().requires_grad_(False)


def objective(student,reader,received,clean,action,variance,probes,reference):
    _,terms,restored=source_objective(student,received,clean,action,variance,probes,return_restored=True)
    guide=frozen_feature_loss(reader,restored,reference)
    loss=terms['iq_nmse']+.1*terms['prior_nmse']+.05*terms['spectral_error']+.05*terms['action_probe_nmse']+.1*guide
    return loss,dict(terms,**{GUIDANCE_METRIC:guide}),restored


def smoke(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean,roles=base.source_training(p);source_ids(p,roles)
    take=np.flatnonzero(roles=='L_s')[:2];folder=p['cache']/'epochs/01'
    values={key:base.tensor(np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)[take],'cpu')
            for key in ('received','kernel','coefficients','variance')}
    x=base.tensor(clean[take],'cpu');student=base.scratch_model(2026092701)
    reader=load_reader(p['reader'],p['source_cache'],'cpu').eval().requires_grad_(False)
    frozen_state={key:value.detach().clone() for key,value in reader.state_dict().items()}
    with torch.no_grad():reference=reader.features(x).detach()
    generator=torch.Generator().manual_seed(2026103004)
    probes=[torch.complex(torch.randn(2,256,generator=generator),torch.randn(2,256,generator=generator)) for _ in range(2)]
    loss,terms,restored=objective(student,reader,values['received'],x,
        base.supervised_action(values['kernel'],values['coefficients']),values['variance'],probes,reference)
    feature_input_gradient=torch.autograd.grad(terms[GUIDANCE_METRIC].mean(),restored,retain_graph=True)[0]
    if not torch.isfinite(feature_input_gradient).all() or not feature_input_gradient.abs().sum()>0:
        raise ValueError('Actual frozen F feature loss must provide a finite nonzero IQ gradient')
    loss.mean().backward()
    gradients=[value.grad for value in student.parameters() if value.grad is not None]
    if not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum()>0 for v in gradients):
        raise ValueError('Finite nonzero source G gradients required')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v,frozen_state[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Source guidance changed frozen F')
    with torch.no_grad():expected=student(values['received']).clone()
    checkpoint=p['output']/'smoke-scratch_joint.pt';save_model(checkpoint,student,'scratch_joint',0,0,args.commit)
    loaded=load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():
        if not torch.equal(loaded(values['received']),expected):raise ValueError('Guided G strict reload changed IQ')
        logits=reader(expected)
        if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise ValueError('Actual original F default-head smoke failed')
    base.write_json(p['output']/'smoke.json',dict(status='PASS',real_Ls=2,training_regime='authorized_source_F_guided',
        TX_labels_read=0,target_reads=0,F_updates=0,F_unchanged=True,head_fit_updates=0,
        nonzero_feature_to_IQ_gradient=True,source_physical_roles_match_F=True,initialization_checkpoint_loads=0))
    print(json.dumps(dict(event='REAL_SOURCE_F_GUIDANCE_NOQUERY_SMOKE_PASS')),flush=True)


def train(args,p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    started_total=time.perf_counter();row=p['output']/args.arm;row.mkdir()
    base.write_json(row/'launch.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],commit=args.commit,training_regime='source_F_guided'))
    clean,roles=base.source_training(p);ids=source_ids(p,roles)
    reader=load_reader(p['reader'],p['source_cache'],'cuda:0').eval().requires_grad_(False)
    frozen_state={key:value.detach().cpu().clone() for key,value in reader.state_dict().items()}
    cache=row/'source_feature_cache';cache.mkdir();cache_started=time.perf_counter()
    targets=np.lib.format.open_memmap(cache/'features.npy',mode='w+',dtype=np.float32,shape=(63000,160))
    with torch.no_grad():
        for offset in range(0,63000,256):
            end=min(offset+256,63000);features=reader.features(base.tensor(clean[offset:end],'cuda:0'))
            if features.shape!=(end-offset,160) or not torch.isfinite(features).all():raise ValueError('Invalid source F feature cache')
            targets[offset:end]=base.numpy_values(features)
    targets.flush();np.save(cache/'physical_ids.npy',ids,allow_pickle=False)
    cache_seconds=time.perf_counter()-cache_started
    base.write_json(cache/'complete.json',dict(status='SOURCE_F_FEATURES_COMPLETE',count=63000,dimension=160,
        numeric_bytes=63000*160*4,TX_labels_read=0,V_reads=0,target_reads=0,F_updates=0,F_teacher=teacher_reference(),seconds=cache_seconds))
    target_features=base.tensor(targets,'cuda:0');del targets
    # F loading/cache construction must not change the matched G/probe RNG.
    torch.manual_seed(2026092701);torch.cuda.manual_seed_all(2026092701)
    student=base.scratch_model(2026092701).to('cuda:0')
    before={key:value.detach().cpu().clone() for key,value in student.state_dict().items()}
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3,weight_decay=1e-4)
    base.write_json(row/'resolved_config.json',base.configuration())
    base.write_json(row/'initialization.json',student.initialization_report)
    preparation_seconds=time.perf_counter()-started_total
    rng=np.random.default_rng(2026103004);updates=seen_l=seen_u=0;started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,21):
            folder=p['cache']/'epochs'/f'{epoch:02d}'
            arrays={key:np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False) for key in ('received','kernel','coefficients','variance')}
            totals=dict.fromkeys(('loss','iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse',GUIDANCE_METRIC),0.)
            clipped=0;order=rng.permutation(63000)
            for offset in range(0,63000,256):
                take=order[offset:offset+256];x=base.tensor(clean[take],'cuda:0')
                values={key:base.tensor(value[take],'cuda:0') for key,value in arrays.items()}
                action=base.supervised_action(values['kernel'],values['coefficients'])
                probes=[torch.complex(torch.randn(len(take),256,device='cuda:0'),torch.randn(len(take),256,device='cuda:0')) for _ in range(2)]
                reference=target_features[torch.tensor(take.tolist(),device='cuda:0',dtype=torch.long)]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=base.learning_rate(updates)
                loss,terms,_=objective(student,reader,values['received'],x,action,values['variance'],probes,reference)
                if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite source guided objective')
                loss.mean().backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.,error_if_nonfinite=True)
                clipped+=int(norm>1);optimizer.step();updates+=1
                seen_l+=int((roles[take]=='L_s').sum());seen_u+=int((roles[take]=='U_s').sum())
                for key,value in dict(loss=loss,**terms).items():totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,updates=updates,losses={key:value/63000 for key,value in totals.items()},
                L_exposures=seen_l,U_exposures=seen_u,clip_fraction=clipped/247,seconds=time.perf_counter()-started,
                F_inputs=seen_l+seen_u,F_clean_cache_inputs=63000,F_updates=0,TX_labels_read=0,initialization_checkpoint_loads=0)
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            print(json.dumps(dict(event='SOURCE_GUIDED_EPOCH_COMPLETE',**record)),flush=True)
    if (updates,seen_l,seen_u)!=(4940,126000,1134000):raise ValueError('Incomplete guided source budget')
    if any(v.grad is not None for v in reader.parameters()) or any(not torch.equal(v.detach().cpu(),frozen_state[k]) for k,v in reader.state_dict().items()):
        raise ValueError('Training changed frozen F')
    changed=sum(not torch.equal(v.detach().cpu(),before[k]) for k,v in student.state_dict().items())
    if not changed:raise ValueError('Source G did not train')
    save_model(row/'final.pt',student,args.arm,20,updates,args.commit)
    base.write_json(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=20,updates=4940,arm=args.arm,
        L_exposures=seen_l,U_exposures=seen_u,training_TX_labels_read=0,training_F_access=True,V_reads=0,target_reads=0,
        F_updates=0,F_unchanged=True,F_clean_teacher_inputs=63000,F_restored_training_inputs=1260000,F_teacher=teacher_reference(),
        initialization_checkpoint_loads=0,G_ancestor_epochs=0,cumulative_G_epochs=20,F_reused_teacher_epochs=50,
        head_fit_updates=0,parameters=266951,changed_state_tensors=changed,seconds=time.perf_counter()-started,
        source_preparation_seconds=preparation_seconds,feature_cache_seconds=cache_seconds,
        total_G_training_pipeline_seconds=time.perf_counter()-started_total,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_regime='authorized_source_F_guided_no_direct_TX_labels'))


def configure():
    no_variance.configure();original_configuration=base.configuration;original_command=base.command
    def configuration():
        value=original_configuration()
        value.update(run=RUN,recipe='source_F_guided_cos010',training_F_access=True,F_guided_training=True,
            teacher_model=teacher_reference(),
            source_supervision='Paired source IQ/action and frozen F clean features;inherits source-labeled F knowledge;direct TX labels0',
            F_teacher=teacher_reference(),F_feature_weight=.1,F_clean_cache_count=63000,F_feature_dimension=160,
            objective='noVarAuxIQ+prior+spectral+action base+.1 frozenF clean-feature cosine',
            authorization='2026-10-30 user explicitly allowed independent source F feature guidance;direct TX/V/target hidden',
            claim_scope='F-guided/inherits source labeled F knowledge;not F-free or fully TX-unsupervised')
        return value
    def command(args,mode,arm=None,role=None):
        return [MODULE if item==no_variance.MODULE else item for item in original_command(args,mode,arm,role)]
    base.RUN=RUN;base.configuration=configuration;base.command=command
    base.save_model=save_model;base.load_model=load_model;base.smoke=smoke;base.train=train


if __name__=='__main__':
    configure();base.main()
