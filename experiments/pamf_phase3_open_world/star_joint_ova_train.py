"""Original-F0 P1 feature teachers and matched source joint-OVA increments.

Only cached source L/OE P1 enters either stage. The identity head remains the
original six-class head; the independent 966-parameter auxiliary layer reads
unit fused160. Eval mode disables dropout/BN updates while preserving training
gradients. No native augmentation, U loop, EMA, V, novel or target is called.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import runner
from . import source_open_embedding_train as native
from . import star_representation_train as previous
from . import star_consensus_run as frozen
from . import star_joint_ova_loss as losses

RUN_ID='20261029-phase3-star-joint-ova-s2026092701-r02'
SCHEMA='pamf_phase3_star_joint_ova_train_v1'
STATUS='STAR_JOINT_OVA_SOURCE_FROZEN'
TEACHER_SCHEMA='pamf_phase3_star_joint_ova_teacher_v1'
TEACHER_STATUS='STAR_JOINT_OVA_TEACHER_FROZEN'
ARMS=('detach_control','joint')
STEPS=2220
BATCH=128
MODEL_PARAMETERS=232993
AUX_PARAMETERS=966
COUNTS=dict(L=6300,OE=6134)
NATIVE_FIELDS=runner.COMMON|{'source_cache'}
EXPORT_FIELDS={'run_id','code_commit','native_config','L_root','OE_root','output_root','batch_size'}
FIELDS={'run_id','code_commit','arm','native_config','L_root','OE_root','teacher_root','output_root','steps','lr','L_seed','OE_seed'}
load_L,load_OE,close_role=previous.load_L,previous.load_OE,previous.close_role
forward,_tensor,_optimizer,_clip=previous.forward,previous._tensor,previous._optimizer,previous._clip
_smoke=previous._smoke
_teacher_backend,_teacher_smoke=frozen._backend,frozen._smoke
_snapshot,_check_models=frozen._snapshot,frozen._check_models
_write,_npz=runner._write,frozen._npz


def _native_config(config):
    # Native graph/strict original-E50 loader only; never call its trainer.
    return dict(config['native_config'],arm='supervised_only',steps=config.get('steps',STEPS),lr=config.get('lr',.0001),
        L_seed=config.get('L_seed',2026092701),OE_seed=config.get('OE_seed',2026102907))


def _common(config,fields,row):
    if (not isinstance(config,dict) or set(config)!=fields or config.get('run_id')!=RUN_ID
            or not isinstance(config.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('native_config'),dict) or set(config['native_config'])!=NATIVE_FIELDS):
        raise ValueError('Only this fixed source-only joint OVA schema is allowed')
    n=config['native_config']
    if any(not isinstance(config[k],str) or not config[k].strip() for k in ('L_root','OE_root','output_root')):
        raise ValueError('Separate L/OE caches and a new output root are required')
    if (n.get('run_id')!=RUN_ID or n.get('row_id')!=row or n.get('code_commit')!=config['code_commit']
            or n.get('output_root')!=config['output_root']):
        raise ValueError('Native constructor run/row/code/output must match this stage')
    roots=[config['L_root'],config['OE_root'],n['source_output'],n['source_cache']]
    if Path(config['L_root']).resolve()==Path(config['OE_root']).resolve():raise ValueError('L/OE roles must differ')
    if fields==FIELDS:
        if not isinstance(config['teacher_root'],str) or not config['teacher_root'].strip():raise ValueError('Frozen original-F0 feature teacher required')
        roots.append(config['teacher_root'])
    if any(Path(config['output_root']).resolve().is_relative_to(Path(p).resolve()) for p in roots):
        raise ValueError('Output cannot replace an input cache or original F0 state')
    native.validate_config(_native_config(config))
    return config


def validate_export_config(config):
    _common(config,EXPORT_FIELDS,'teacher-s2026092701')
    if type(config['batch_size']) is not int or config['batch_size']!=128:raise ValueError('Teacher batch size must be 128')
    return config


def validate_config(config):
    if not isinstance(config,dict) or config.get('arm') not in ARMS:raise ValueError('Only detach_control and joint are allowed')
    _common(config,FIELDS,config['arm']+'-s2026092701')
    fixed=dict(steps=STEPS,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
    if any(type(config[k]) is not type(v) or config[k]!=v for k,v in fixed.items()):
        raise ValueError('Fixed 2220 updates, constant LR and independent source sampling seeds required')
    return config


def _role(bundle,role,config):
    previous._role(bundle,role,config)
    if len(bundle['ids'])!=COUNTS[role]:raise ValueError('Fixed source role counts required')
    return bundle


def _same_roles(L,OE):
    if (set(L['ids'].tolist())&set(OE['ids'].tolist())
            or L['manifest']['parent_F0_identity']!=OE['manifest']['parent_F0_identity']
            or L['manifest']['dura_identity']!=OE['manifest']['dura_identity']):
        raise ValueError('Source L/OE require disjoint records and the same original F0/frozen DURA')


def _teacher_encode(reader,values,device):
    with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
        z=reader.features(_tensor(values,device));unit=F.normalize(z,p=2,dim=1)
    result=np.asarray(unit.detach().cpu().tolist(),np.float32)
    if (result.shape!=(len(values),160) or not np.isfinite(result).all()
            or not np.allclose(np.linalg.norm(result,axis=1),1.,rtol=1e-5,atol=1e-5)):
        raise ValueError('Finite original-F0 unit fused160 teacher features required')
    return result


def export(config,device='cuda:0'):
    validate_export_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable teacher output root is required')
    started=time.perf_counter();backend,contract=_teacher_backend(config['native_config'],device);snapshots=_snapshot(backend)
    smoke=_teacher_smoke(backend,device);_check_models(snapshots)
    if smoke.get('status')!='PASS':raise ValueError('Actual frozen original-F0 synthetic G10 smoke must pass first')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke);L=OE=None
    try:
        L=load_L(config['L_root']);_role(L,'L',config)
        OE=load_OE(config['OE_root']);_role(OE,'OE',config);_same_roles(L,OE)
        parent=L['manifest']['parent_F0_identity']
        if parent!=backend['reader_identity']:raise ValueError('Actual frozen teacher F0 and cache F0 differ')
        for role,bundle in (('L',L),('OE',OE)):
            rows=[]
            for start in range(0,len(bundle['ids']),config['batch_size']):
                stop=min(start+config['batch_size'],len(bundle['ids']))
                rows.append(_teacher_encode(backend['reader'],bundle['restored'][start:stop],device))
                print('STAR_JOINT_OVA_TEACHER_PROGRESS',role,stop,len(bundle['ids']),flush=True)
            _npz(output/(role+'.npz'),dict(ids=bundle['ids'],unit_features=np.concatenate(rows)))
        _check_models(snapshots)
        done=dict(schema=TEACHER_SCHEMA,status=TEACHER_STATUS,run_id=RUN_ID,code_commit=config['code_commit'],config=config,
            source_only=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0,
            parent_F0_identity=parent,dura_identity=L['manifest']['dura_identity'],cache_handles=dict(L=L['manifest'],OE=OE['manifest']),
            source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],input_path='cached_P1_only',feature='unit_fused160',
            model_parameters=232993,model_buffers_unchanged=True,encoder_updates=0,identity_head_updates=0,G_updates=0,G_forward_rows=0,
            F0_feature_rows=sum(COUNTS.values()),synthetic_F0_forward_rows=2,synthetic_G10_generated_rows=2,
            received_cache_values_read=0,SourceDataset_calls=0,legacy_clean_IQ_reads=0,
            refs=dict(L='L.npz',OE='OE.npz'),seconds=time.perf_counter()-started,final_phase3_complete=False)
        _write(output/'complete.json',done);print(TEACHER_STATUS,flush=True);return done
    finally:
        if OE is not None:close_role(OE)
        if L is not None:close_role(L)


def load_teacher(root,L,OE,config):
    root=Path(root);marker=runner._read(root/'complete.json')
    if (marker.get('schema')!=TEACHER_SCHEMA or marker.get('status')!=TEACHER_STATUS
            or marker.get('source_only') is not True or marker.get('source_L_count')!=COUNTS['L'] or marker.get('source_OE_count')!=COUNTS['OE']
            or any(marker.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads'))
            or marker.get('parent_F0_identity')!=L['manifest']['parent_F0_identity']
            or marker.get('dura_identity')!=L['manifest']['dura_identity']):
        raise ValueError('A complete matching source-only original-F0 P1 feature teacher is required')
    n=marker.get('config',{}).get('native_config',{})
    teacher_config=marker.get('config',{})
    if (teacher_config.get('run_id')!=config['run_id'] or teacher_config.get('code_commit')!=config['code_commit']
            or any(n.get(k)!=config['native_config'].get(k) for k in ('source_output','source_contract','source_cache','base_release','frozen_sources','registered_classes','model_seed'))):
        raise ValueError('Feature teacher and student original-F0 source contracts differ')
    result={'manifest':marker}
    for role,bundle in (('L',L),('OE',OE)):
        if (marker.get('cache_handles',{}).get(role)!=bundle['manifest']
                or bundle['manifest']['parent_F0_identity']!=marker['parent_F0_identity']):
            raise ValueError('Feature teacher source role identity differs')
        with np.load(root/(role+'.npz'),allow_pickle=False) as z:
            if set(z.files)!={'ids','unit_features'}:raise ValueError('Only aligned IDs and unit features may be in the teacher')
            ids=z['ids'].copy();unit=z['unit_features'].copy()
        if (ids.dtype.kind!='U' or not np.array_equal(ids,bundle['ids']) or unit.shape!=(COUNTS[role],160)
                or unit.dtype!=np.float32 or not np.isfinite(unit).all()
                or not np.allclose(np.linalg.norm(unit,axis=1),1.,rtol=1e-5,atol=1e-5)):
            raise ValueError('Teacher unit features must match each source role ID in order')
        result[role]=unit
    return result


def _auxiliary(device):
    # A local RNG scope gives both independent arms identical auxiliary weights.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026092701)
        aux=nn.Linear(160,6,device='cpu',dtype=torch.float32)
        nn.init.normal_(aux.weight,mean=0.,std=.01);nn.init.zeros_(aux.bias)
    return aux.to(device).eval().requires_grad_(True)


def _model(model):
    # The native training container also retains unused domain/adversarial heads.
    # The deployment identity count belongs to the complete encoder, as in runner.
    if sum(p.numel() for p in model.id_backbone.encoder.parameters())!=MODEL_PARAMETERS:
        raise ValueError('Only the current complete original-F0 identity encoder is allowed')
    return model.eval().requires_grad_(True)


def _loop(model,aux,L,OE,teacher,config,device,F_optimizer,aux_optimizer,*,steps=None,on_step=None):
    steps=config['steps'] if steps is None else steps;params=list(model.parameters());aux_params=list(aux.parameters())
    named=dict(model.named_parameters());sinc=[named[native.SINC_PREFIX+k] for k in ('low_coordinate','band_coordinate')];head=named[native.HEAD_NAME]
    L_rng=torch.Generator().manual_seed(config['L_seed']);sampler=native.DomainSampler(OE['metadata']['domain'],config['OE_seed']);trace=[]
    model.eval().requires_grad_(True);aux.eval().requires_grad_(True)
    for step in range(1,steps+1):
        started=time.perf_counter();ix=torch.randint(len(L['ids']),(BATCH,),generator=L_rng).tolist();oi=sampler.sample(BATCH)
        x=_tensor(np.concatenate((L['restored'][ix],OE['restored'][oi])),device)
        y=_tensor(L['metadata']['y'][ix],device,torch.long);logits_teacher=_tensor(L['teacher_logits'][ix],device)
        unit_L=_tensor(teacher['L'][ix],device);unit_OE=_tensor(teacher['OE'][oi],device)
        before_sinc=[p.detach().clone() for p in sinc];before_head=head.detach().clone()
        F_optimizer.zero_grad(set_to_none=True);aux_optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            z,logits=forward(model,x)
            values=losses.compute(z[:BATCH],z[BATCH:],logits[:BATCH],logits_teacher,y,unit_L,unit_OE,
                aux.weight,aux.bias,joint=config['arm']=='joint')
            values['total'].backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params+aux_params):
            raise FloatingPointError('Nonfinite native/auxiliary joint OVA gradients')
        sn=sum(float(p.grad.norm()) for p in sinc if p.grad is not None);hn=float(head.grad.norm()) if head.grad is not None else 0.
        fg=_clip(F_optimizer,params);ag=nn.utils.clip_grad_norm_(aux_params,5.)
        if not bool(torch.isfinite(torch.as_tensor(fg))) or not bool(torch.isfinite(torch.as_tensor(ag))):
            raise FloatingPointError('Nonfinite native physical/auxiliary global gradient')
        F_optimizer.step();aux_optimizer.step()
        row=dict(step=step,cumulative_updates=11100+step,lr=float(F_optimizer.param_groups[0]['lr']),aux_lr=float(aux_optimizer.param_groups[0]['lr']),
            loss=float(values['total'].detach()),**{k:float(v.detach()) for k,v in values.items() if k!='total'},
            sinc_coordinate_gradient_norm=sn,head_gradient_norm=hn,physical_global_gradient_norm=float(fg),auxiliary_gradient_norm=float(ag),
            sinc_change_hz_norm=sum(float((p.detach()-q).norm())*25000. for p,q in zip(sinc,before_sinc)),
            head_change_norm=float((head.detach()-before_head).norm()),L_sampled_records=BATCH,OE_sampled_records=BATCH,
            L_unique_in_batch=len(set(ix)),OE_unique_in_batch=len(set(oi.tolist())),L_forward_samples=BATCH,OE_forward_samples=BATCH,
            F_forward_calls=1,F_optimizer_steps=1,aux_optimizer_steps=1,seconds=time.perf_counter()-started)
        trace.append(row)
        if on_step is not None:on_step(row)
    return trace


def _train_identity(config,parent,native_parent,checkpoint=None):
    return dict(run_id=RUN_ID,arm=config['arm'],row_id=config['arm']+'-s2026092701',code_commit=config['code_commit'],
        checkpoint=None if checkpoint is None else str(checkpoint),updates=config['steps'],cumulative_updates=11100+config['steps'],
        encoder_updates=config['steps'],identity_head_updates=config['steps'],auxiliary_updates=config['steps'],
        model_parameters=MODEL_PARAMETERS,aux_parameters=AUX_PARAMETERS,original_parent_identity=parent,native_parent_identity=native_parent,
        identity_head_replaced=False,input_path='cached_P1_only')


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable joint OVA training output root is required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(native.make_trainable_model(_native_config(config),device));smoke=_smoke(model,device)
    if smoke.get('status')!='PASS':raise ValueError('Actual native original-F0 gradient smoke must pass before source cache reads')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke);L=OE=None;started=time.perf_counter()
    try:
        L=load_L(config['L_root']);_role(L,'L',config);OE=load_OE(config['OE_root']);_role(OE,'OE',config);_same_roles(L,OE)
        teacher=load_teacher(config['teacher_root'],L,OE,config);aux=_auxiliary(device)
        buffers={k:v.detach().clone() for k,v in model.named_buffers()};params=list(model.parameters())
        F_optimizer=_optimizer(model,params,lr=config['lr'],weight_decay=float(model._phase3_parent_args['weight_decay']),betas=(.9,.999),eps=1e-8,foreach=False)
        aux_optimizer=torch.optim.AdamW(aux.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)
        if F_optimizer.state or aux_optimizer.state:raise ValueError('Both optimizers must cold-start independently')
        _write(output/'resolved_config.json',config)
        with (output/'steps.jsonl').open('x',encoding='utf-8') as stream:
            def progress(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if row['step']==1 or row['step']%100==0 or row['step']==config['steps']:
                    print('STAR_JOINT_OVA_PROGRESS',config['arm'],row['step'],config['steps'],'loss',row['loss'],flush=True)
            trace=_loop(model,aux,L,OE,teacher,config,device,F_optimizer,aux_optimizer,on_step=progress)
        updates=config['steps']
        if (len(trace)!=updates or F_optimizer._sinc_unit_steps!=updates or F_optimizer._sinc_physical_clip_calls!=updates
                or sum(r['F_optimizer_steps'] for r in trace)!=updates or sum(r['aux_optimizer_steps'] for r in trace)!=updates
                or len(aux_optimizer.state)!=2 or any(float(s.get('step',0))!=updates for s in aux_optimizer.state.values())
                or any(sum(r[k] for r in trace)!=BATCH*updates for k in ('L_sampled_records','OE_sampled_records','L_forward_samples','OE_forward_samples'))):
            raise ValueError('Exactly the declared F/auxiliary updates and source exposures must complete')
        if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Eval-mode native model buffers changed')
        model.zero_grad(set_to_none=True);aux.zero_grad(set_to_none=True);model.eval().requires_grad_(False);aux.eval().requires_grad_(False)
        handles=dict(L=L['manifest'],OE=OE['manifest']);parent=handles['L']['parent_F0_identity'];native_parent=model._phase3_parent_identity
        identity=_train_identity(config,parent,native_parent)
        result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],config=config,identity=identity,
            original_parent_identity=parent,native_parent_identity=native_parent,cache_handles=handles,teacher_identity=teacher['manifest'],
            source_data_handle=model._phase3_source_handle,
            updates=updates,parent_updates=11100,cumulative_updates=11100+updates,parent_epoch=50,final_epoch=50+updates//222,
            encoder_updates=updates,identity_head_updates=updates,auxiliary_updates=updates,optimizer_step_calls=2*updates,
            optimizer_reset=True,aux_optimizer_reset=True,F_optimizer='AdamW_CENTER_UNITS',aux_optimizer='AdamW',
            lr=config['lr'],aux_lr=.001,aux_weight_decay=.0001,physical_global_clip=5.,aux_global_clip=5.,lr_schedule='constant',
            source_only=True,target_reads=0,U_reads=0,validation_reads=0,novel_validation_reads=0,OE_truth_reads=0,
            legacy_clean_IQ_reads=0,SourceDataset_calls=0,received_cache_values_read=0,G_forward_rows=0,G_updates=0,
            model_classes=6,parameters=MODEL_PARAMETERS,identity_parameters=sum(p.numel() for p in model.id_backbone.encoder.parameters()),
            training_container_parameters=sum(p.numel() for p in model.parameters()),aux_parameters=AUX_PARAMETERS,default_head_joint=True,identity_head_replaced=False,
            input_path='cached_P1_only',model_mode='eval_with_grad',dropout_enabled=False,BN_updates=0,model_buffers_unchanged=True,
            precision='FP32',TF32=False,source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],
            L_sampled_exposures=BATCH*updates,OE_sampled_exposures=BATCH*updates,L_forward_samples=BATCH*updates,OE_forward_samples=BATCH*updates,
            F_forward_calls=updates,synthetic_F0_forward_samples=2,synthetic_G10_generated_rows=2,
            auxiliary_input='unit fused160',auxiliary_initialization=dict(seed=2026092701,weight_normal_std=.01,bias=0),
            CE_label_smoothing=0.,KD_temperature=2.,KD_weight=1.,feature_preservation_weight=5.,feature_preservation_role_prior=dict(L=.5,OE=.5),
            OVA_weight=1.,OVA_feature_gradient=config['arm']=='joint',teacher_logits='original_F0_same_P1_from_L_cache',
            OE_transmitter_labels_used=False,ordinary_augmentation=False,As_generated=0,CRIC_evaluations=0,PL_evaluations=0,EMA_updates=0,VICReg_evaluations=0,
            budget_scope='fixed iid source updates, not natural data epochs',checkpoint='final.pt',seconds=time.perf_counter()-started,final_phase3_complete=False)
        payload=dict(schema=SCHEMA,config=config,identity=identity,original_parent_identity=parent,native_parent_identity=native_parent,
            cache_handles=handles,source_data_handle=model._phase3_source_handle,updates=updates,cumulative_updates=11100+updates,optimizer_reset=True,aux_optimizer_reset=True,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},auxiliary={k:v.detach().cpu() for k,v in aux.state_dict().items()},
            F_optimizer=F_optimizer.state_dict(),aux_optimizer=aux_optimizer.state_dict(),training_summary=result)
        with (output/'final.pt').open('xb') as stream:torch.save(payload,stream)
        _write(output/'training.json',result);_write(output/'complete.json',result);print(STATUS,config['arm'],flush=True);return result
    finally:
        if OE is not None:close_role(OE)
        if L is not None:close_role(L)


def load_trained_model(config,checkpoint,device='cuda:0'):
    """Strict-load own F and separate auxiliary state after original-F0 verification."""
    validate_config(config);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (payload.get('schema')!=SCHEMA or payload.get('config')!=config or payload.get('updates')!=config['steps']
            or payload.get('cumulative_updates')!=11100+config['steps'] or payload.get('optimizer_reset') is not True
            or payload.get('aux_optimizer_reset') is not True):
        raise ValueError('Only this independent source joint OVA checkpoint may load')
    summary=payload.get('training_summary',{})
    if (summary.get('status')!=STATUS or summary.get('source_only') is not True or summary.get('input_path')!='cached_P1_only'
            or summary.get('default_head_joint') is not True or summary.get('identity_head_replaced') is not False
            or summary.get('parameters')!=MODEL_PARAMETERS or summary.get('aux_parameters')!=AUX_PARAMETERS
            or summary.get('encoder_updates')!=config['steps'] or summary.get('identity_head_updates')!=config['steps']
            or summary.get('auxiliary_updates')!=config['steps'] or summary.get('optimizer_step_calls')!=2*config['steps']
            or any(summary.get(k)!=0 for k in ('target_reads','U_reads','validation_reads','novel_validation_reads','OE_truth_reads'))):
        raise ValueError('Complete source-only F/default-head/auxiliary training boundaries required')
    expected=_train_identity(config,payload.get('original_parent_identity'),payload.get('native_parent_identity'))
    if payload.get('identity')!=expected:raise ValueError('Own joint OVA run/arm/source/checkpoint identity differs')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(native.make_trainable_model(_native_config(config),device))
    if (payload.get('native_parent_identity')!=model._phase3_parent_identity
            or payload.get('source_data_handle')!=model._phase3_source_handle):raise ValueError('Original native F0 ancestor/source role handle differs')
    parent=payload.get('original_parent_identity',{})
    if (parent!=payload.get('cache_handles',{}).get('L',{}).get('parent_F0_identity')
            or any(parent.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or parent.get('source_output')!=config['native_config']['source_output']):
        raise ValueError('Only same-input original-F0 teacher/source cache identities may load')
    model.load_state_dict(payload['model'],strict=True);aux=_auxiliary(device);aux.load_state_dict(payload['auxiliary'],strict=True)
    model.eval().requires_grad_(False);aux.eval().requires_grad_(False)
    identity=_train_identity(config,parent,model._phase3_parent_identity,checkpoint)
    model._star_joint_ova_identity=identity
    return model,aux,identity


def main(argv=None):
    print(json.dumps(dict(event='STAR_JOINT_OVA_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for action in ('export','train'):
        command=sub.add_parser(action);command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);config=runner._read(args.config);done=(export if args.action=='export' else run)(config,args.device)
    print(json.dumps(dict(status=done['status'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
