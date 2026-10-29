"""Two native F0 increments with matched OE Gram and optional default-head OE."""
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
from . import star_joint_ova_train as joint
from . import star_relation_oe_loss as losses

RUN_ID='20261030-phase3-star-relation-oe-s2026092701-r01'
SCHEMA='pamf_phase3_star_relation_oe_train_v1'
STATUS='STAR_RELATION_OE_SOURCE_FROZEN'
ARMS=('relation_control','relation_oe')
STEPS=2220;BATCH=128;MODEL_PARAMETERS=232993
COUNTS=dict(L=6300,OE=6134)
TEACHER_RUN='20261029-phase3-star-joint-ova-s2026092701-r02'
TEACHER_COMMIT='5aa041122396d48c49b8cb3d5912649bdc3217e1'
TEACHER_SCHEMA=joint.TEACHER_SCHEMA;TEACHER_STATUS=joint.TEACHER_STATUS
FIELDS={'run_id','code_commit','arm','native_config','L_root','OE_root','teacher_root','output_root','steps','lr','L_seed','OE_seed'}
NATIVE_FIELDS=joint.NATIVE_FIELDS
runner,native,previous=joint.runner,joint.native,joint.previous
load_L,load_OE,close_role=joint.load_L,joint.load_OE,joint.close_role
forward,_tensor,_optimizer,_clip,_smoke=joint.forward,joint._tensor,joint._optimizer,joint._clip,joint._smoke
_write=joint._write
_same_roles=joint._same_roles
_native_config=joint._native_config


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or not isinstance(c.get('native_config'),dict) or set(c['native_config'])!=NATIVE_FIELDS):
        raise ValueError('Only the fixed source relation/OE training schema is allowed')
    fixed=dict(steps=STEPS,lr=.0001,L_seed=2026092701,OE_seed=2026102907)
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):raise ValueError('Fixed updates, LR and source sampling seeds required')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('L_root','OE_root','teacher_root','output_root')):
        raise ValueError('Only separate source role/teacher paths and a new output root are allowed')
    n=c['native_config']
    if (n.get('run_id')!=RUN_ID or n.get('row_id')!=c['arm']+'-s2026092701' or n.get('code_commit')!=c['code_commit'] or n.get('output_root')!=c['output_root']):
        raise ValueError('Native run, arm row, code and output must match')
    roots=(c['L_root'],c['OE_root'],c['teacher_root'],n['source_output'],n['source_cache'])
    if Path(c['L_root']).resolve()==Path(c['OE_root']).resolve() or any(Path(c['output_root']).resolve().is_relative_to(Path(p).resolve()) for p in roots):
        raise ValueError('Output cannot overwrite source roles, teacher or original model')
    native.validate_config(_native_config(c));return c


def _role(bundle,role,config):
    previous._role(bundle,role,config)
    if len(bundle['ids'])!=COUNTS[role]:raise ValueError('Fixed source role count required')
    return bundle


def _model(model):
    if sum(p.numel() for p in model.id_backbone.encoder.parameters())!=MODEL_PARAMETERS:
        raise ValueError('Only the current original-F0 complete identity encoder may train')
    return model.eval().requires_grad_(True)


def load_teacher(root,L,OE,config):
    root=Path(root);m=runner._read(root/'complete.json');tc=m.get('config',{});tn=tc.get('native_config',{});n=config['native_config']
    if (m.get('schema')!=TEACHER_SCHEMA or m.get('status')!=TEACHER_STATUS or m.get('run_id')!=TEACHER_RUN or m.get('code_commit')!=TEACHER_COMMIT
            or tc.get('run_id')!=TEACHER_RUN or tc.get('code_commit')!=TEACHER_COMMIT or m.get('source_only') is not True
            or m.get('parent_F0_identity')!=L['manifest']['parent_F0_identity'] or m.get('dura_identity')!=L['manifest']['dura_identity']
            or m.get('model_parameters')!=232993 or m.get('model_buffers_unchanged') is not True
            or m.get('input_path')!='cached_P1_only' or m.get('feature')!='unit_fused160'
            or any(m.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','OE_truth_reads','encoder_updates','identity_head_updates','G_updates','G_forward_rows'))
            or any(tn.get(k)!=n.get(k) for k in ('source_output','source_contract','source_cache','base_release','frozen_sources','registered_classes','model_seed'))
            or any(tc.get(k)!=config[k] for k in ('L_root','OE_root'))
            or any(m.get('source_'+r+'_count')!=COUNTS[r] or m.get('cache_handles',{}).get(r)!=b['manifest']
                or b['manifest']['parent_F0_identity']!=m['parent_F0_identity'] or b['manifest']['dura_identity']!=m['dura_identity'] for r,b in (('L',L),('OE',OE)))):
        raise ValueError('Exact completed r02 original-F0 teacher and matching source-role contracts required')
    with np.load(root/'OE.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','unit_features'}:raise ValueError('Only public OE teacher IDs and unit features allowed')
        ids=z['ids'].copy();unit=z['unit_features'].copy()
    if (ids.dtype.kind!='U' or not np.array_equal(ids,OE['ids']) or unit.shape!=(COUNTS['OE'],160) or unit.dtype!=np.float32
            or not np.isfinite(unit).all() or not np.allclose(np.linalg.norm(unit.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
        raise ValueError('Original F0 OE unit teacher must match the exact source role ID order')
    return dict(manifest=m,OE=unit)


def _identity(c,parent,native_parent):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],row_id=c['arm']+'-s2026092701',checkpoint=str(Path(c['output_root'])/'final.pt'),
        updates=c['steps'],cumulative_updates=11100+c['steps'],encoder_updates=c['steps'],identity_head_updates=c['steps'],
        model_parameters=MODEL_PARAMETERS,aux_parameters=0,original_parent_identity=parent,native_parent_identity=native_parent,
        identity_head_replaced=False,input_path='cached_P1_only')


def _loop(model,L,OE,teacher,c,device,optimizer,on_step=None):
    params=list(model.parameters());named=dict(model.named_parameters());sinc=[named[native.SINC_PREFIX+k] for k in ('low_coordinate','band_coordinate')];head=named[native.HEAD_NAME]
    rng=torch.Generator().manual_seed(c['L_seed']);sampler=native.DomainSampler(OE['metadata']['domain'],c['OE_seed']);rows=[]
    model.eval().requires_grad_(True)
    for step in range(1,c['steps']+1):
        started=time.perf_counter();ix=torch.randint(len(L['ids']),(BATCH,),generator=rng).tolist();oi=sampler.sample(BATCH)
        x=_tensor(np.concatenate((L['restored'][ix],OE['restored'][oi])),device);y=_tensor(L['metadata']['y'][ix],device,torch.int64)
        tlogits=_tensor(L['teacher_logits'][ix],device);tunit=_tensor(teacher['OE'][oi],device)
        before_sinc=[p.detach().clone() for p in sinc];before_head=head.detach().clone();optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            z,logits=forward(model,x);v=losses.compute(z[BATCH:],logits[:BATCH],logits[BATCH:],tlogits,y,tunit,c['arm']);v['total'].backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('Nonfinite native relation/OE gradient')
        sn=sum(float(p.grad.norm()) for p in sinc if p.grad is not None);hn=float(head.grad.norm()) if head.grad is not None else 0.;gn=_clip(optimizer,params)
        if not bool(torch.isfinite(torch.as_tensor(gn))):raise FloatingPointError('Nonfinite physical native gradient norm')
        optimizer.step()
        row=dict(step=step,cumulative_updates=11100+step,lr=float(optimizer.param_groups[0]['lr']),loss=float(v['total'].detach()),
            **{k:float(w.detach()) for k,w in v.items() if k!='total'},sinc_coordinate_gradient_norm=sn,head_gradient_norm=hn,physical_global_gradient_norm=float(gn),
            sinc_change_hz_norm=sum(float((p.detach()-q).norm())*25000 for p,q in zip(sinc,before_sinc)),head_change_norm=float((head.detach()-before_head).norm()),
            L_sampled_records=BATCH,OE_sampled_records=BATCH,L_forward_samples=BATCH,OE_forward_samples=BATCH,F_forward_calls=1,F_optimizer_steps=1,
            L_unique_in_batch=len(set(ix)),OE_unique_in_batch=len(set(oi.tolist())),L_indices=ix,OE_indices=oi.tolist(),seconds=time.perf_counter()-started)
        rows.append(row)
        if on_step is not None:on_step(row)
    return rows


def train(config,device='cuda:0'):
    validate_config(config);out=Path(config['output_root'])
    if out.exists():raise FileExistsError('A new immutable relation/OE source output is required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(native.make_trainable_model(_native_config(config),device));smoke=_smoke(model,device)
    if smoke.get('status')!='PASS':raise ValueError('Actual native F0 synthetic G10 gradient smoke must pass before source roles')
    out.mkdir(parents=True,exist_ok=False);_write(out/'smoke.json',smoke);L=OE=None;started=time.perf_counter()
    try:
        L=load_L(config['L_root']);_role(L,'L',config);OE=load_OE(config['OE_root']);_role(OE,'OE',config);_same_roles(L,OE)
        teacher=load_teacher(config['teacher_root'],L,OE,config);buffers={k:v.detach().clone() for k,v in model.named_buffers()};params=list(model.parameters())
        optimizer=_optimizer(model,params,lr=config['lr'],weight_decay=float(model._phase3_parent_args['weight_decay']),betas=(.9,.999),eps=1e-8,foreach=False)
        if optimizer.state:raise ValueError('Native physical AdamW must cold-start')
        _write(out/'resolved_config.json',config)
        with (out/'steps.jsonl').open('x',encoding='utf-8') as stream:
            def progress(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if row['step']==1 or row['step']%100==0 or row['step']==config['steps']:print('STAR_RELATION_OE_PROGRESS',config['arm'],row['step'],config['steps'],row['loss'],flush=True)
            rows=_loop(model,L,OE,teacher,config,device,optimizer,progress)
        updates=config['steps']
        if (len(rows)!=updates or optimizer._sinc_unit_steps!=updates or optimizer._sinc_physical_clip_calls!=updates
                or sum(r['F_optimizer_steps'] for r in rows)!=updates
                or any(sum(r[k] for r in rows)!=BATCH*updates for k in ('L_sampled_records','OE_sampled_records','L_forward_samples','OE_forward_samples'))):
            raise ValueError('Exactly the declared single-optimizer native updates and matched source exposures required')
        if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Frozen eval-mode native buffers changed')
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite trained native parameters')
        model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
        handles=dict(L=L['manifest'],OE=OE['manifest']);parent=handles['L']['parent_F0_identity'];native_parent=model._phase3_parent_identity;identity=_identity(config,parent,native_parent)
        summary=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],config=config,identity=identity,
            original_parent_identity=parent,native_parent_identity=native_parent,cache_handles=handles,teacher_identity=teacher['manifest'],source_data_handle=model._phase3_source_handle,
            teacher_run_id=TEACHER_RUN,teacher_code_commit=TEACHER_COMMIT,updates=updates,parent_updates=11100,cumulative_updates=11100+updates,
            parent_epoch=50,final_epoch=50+updates//222,budget_scope='fixed IID source updates; epoch equivalent is annotation, not natural data traversal',
            encoder_updates=updates,identity_head_updates=updates,optimizer_step_calls=updates,optimizer_reset=True,F_optimizer='AdamW_CENTER_UNITS',
            lr=config['lr'],weight_decay=float(model._phase3_parent_args['weight_decay']),physical_global_clip=5.,lr_schedule='constant',parameters=MODEL_PARAMETERS,aux_parameters=0,
            training_container_parameters=sum(p.numel() for p in model.parameters()),default_head_joint=True,identity_head_replaced=False,input_path='cached_P1_only',
            source_only=True,target_reads=0,U_reads=0,validation_reads=0,novel_validation_reads=0,OE_truth_reads=0,legacy_clean_IQ_reads=0,SourceDataset_calls=0,received_cache_values_read=0,
            G_forward_rows=0,G_updates=0,source_L_count=COUNTS['L'],source_OE_count=COUNTS['OE'],L_sampled_exposures=BATCH*updates,OE_sampled_exposures=BATCH*updates,
            L_forward_samples=BATCH*updates,OE_forward_samples=BATCH*updates,F_forward_calls=updates,synthetic_F0_forward_samples=2,synthetic_G10_generated_rows=2,
            model_mode='eval_with_grad',dropout_enabled=False,BN_updates=0,model_buffers_unchanged=True,precision='FP32',TF32=False,
            KD_temperature=2.,KD_weight=1.,Gram_OE_weight=1.,OE_uniform_weight=float(config['arm']=='relation_oe'),OE_uniform_direction='KL(U6||softmax(default_logits_OE))',
            pointwise_feature_preservation=False,OE_transmitter_labels_used=False,ordinary_augmentation=False,As_generated=0,CRIC_evaluations=0,PL_evaluations=0,EMA_updates=0,VICReg_evaluations=0,
            checkpoint='final.pt',seconds=time.perf_counter()-started,final_phase3_complete=False)
        payload=dict(schema=SCHEMA,config=config,identity=identity,original_parent_identity=parent,native_parent_identity=native_parent,cache_handles=handles,
            teacher_identity=teacher['manifest'],source_data_handle=model._phase3_source_handle,updates=updates,cumulative_updates=11100+updates,optimizer_reset=True,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},F_optimizer=optimizer.state_dict(),training_summary=summary)
        with (out/'final.pt').open('xb') as stream:torch.save(payload,stream)
        _write(out/'training_summary.json',summary);_write(out/'complete.json',summary);print(STATUS,config['arm'],flush=True);return summary
    finally:
        if OE is not None:close_role(OE)
        if L is not None:close_role(L)


run=train


def load_candidate(root,device='cuda:0'):
    root=Path(root);config=runner._read(root/'resolved_config.json');validate_config(config);payload=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if (not isinstance(payload,dict) or payload.get('schema')!=SCHEMA or payload.get('config')!=config or payload.get('updates')!=config['steps']
            or payload.get('cumulative_updates')!=11100+config['steps'] or payload.get('optimizer_reset') is not True
            or 'auxiliary' in payload or 'aux_optimizer' in payload):raise ValueError('Only a complete own-run relation/OE checkpoint may load')
    summary=payload.get('training_summary',{});identity=_identity(config,payload.get('original_parent_identity'),payload.get('native_parent_identity'))
    if (payload.get('identity')!=identity or summary.get('identity')!=identity or summary.get('status')!=STATUS
            or any(summary.get(k)!=v for k,v in dict(run_id=RUN_ID,code_commit=config['code_commit'],arm=config['arm'],parameters=MODEL_PARAMETERS,
                encoder_updates=config['steps'],identity_head_updates=config['steps'],optimizer_step_calls=config['steps'],aux_parameters=0,
                target_reads=0,U_reads=0,validation_reads=0,novel_validation_reads=0,OE_truth_reads=0,G_updates=0).items())
            or summary.get('source_only') is not True or summary.get('default_head_joint') is not True or summary.get('identity_head_replaced') is not False
            or summary.get('input_path')!='cached_P1_only' or summary.get('model_buffers_unchanged') is not True
            or summary.get('teacher_run_id')!=TEACHER_RUN or summary.get('teacher_code_commit')!=TEACHER_COMMIT
            or payload.get('teacher_identity')!=summary.get('teacher_identity')
            or payload.get('teacher_identity',{}).get('run_id')!=TEACHER_RUN or payload.get('teacher_identity',{}).get('code_commit')!=TEACHER_COMMIT):
        raise ValueError('Own source-only native/default-head budget and provenance required')
    if runner._read(root/'complete.json')!=summary:raise ValueError('Completed source marker and checkpoint summary differ')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(native.make_trainable_model(_native_config(config),device));parent=payload['original_parent_identity']
    if (payload.get('native_parent_identity')!=model._phase3_parent_identity or payload.get('source_data_handle')!=model._phase3_source_handle
            or parent!=payload.get('cache_handles',{}).get('L',{}).get('parent_F0_identity')
            or any(parent.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or parent.get('source_output')!=config['native_config']['source_output']):raise ValueError('Original F0/source ancestry must match before candidate load')
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False);model._star_relation_oe_identity=identity
    return model,config,payload


def main():
    print(json.dumps(dict(event='STAR_RELATION_OE_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train']);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();train(runner._read(a.config),a.device)


if __name__=='__main__':main()
