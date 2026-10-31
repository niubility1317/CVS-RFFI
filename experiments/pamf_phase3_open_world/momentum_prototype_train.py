"""Matched fixed/EMA teachers from pure r03, reusing only fresh star U views."""
from __future__ import annotations
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from torch.nn import functional as F
from . import momentum_prototype_core as core
from . import fresh_pcd_train as fresh
from . import fresh_world_run as reference

body=fresh.body
RUN_ID='20261031-phase3-momentum-prototype-s2026092701-r01'
SCHEMA='pamf_phase3_momentum_prototype_train_v1'
STATUS='MOMENTUM_PROTOTYPE_TRAIN_COMPLETE'
ROW_STATUS='MOMENTUM_PROTOTYPE_STUDENT_FROZEN'
ARMS=('FIXED_TEACHER','EMA_TEACHER')
STEPS=444
U_COUNT=576
U_BATCH=64
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
TRAIN_FIELDS=FIELDS
ZEROS=body.ZEROS
_read,_write,_snapshot=body._read,body._write,body._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
        or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or type(c.get('steps')) is not int or c['steps']!=STEPS
        or type(c.get('lr')) is not float or c['lr']!=.0001 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field matched momentum configuration required; no truth/K/Q inputs')
    r=reference.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed fresh-world reference required')
    t=r['train_config'];flat=fresh._flat(t);p=flat['reference_config'];source=p['train_config'];out=Path(c['output_root']).resolve()
    roots=[r['output_root'],t['output_root'],t['U_public_root'],r['source_control_config']['output_root'],p['output_root'],p['reference_prediction_root'],p['dura_checkpoint'],p['dura_release'],source['output_root'],source['IQ_root'],source['feature_root'],source['control_root']]+[v['public_root'] for v in r['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):
        raise ValueError('New immutable output overlaps existing source/control/public/model inputs')
    return c


def _fresh_metadata(c,pm,control):
    r=c['reference_config'];t=r['train_config'];m=_read(Path(r['output_root'])/'complete.json')
    expected=dict(schema=reference.SCHEMA,status=reference.STATUS,run_id=reference.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        all_three_DURA_conditions_complete=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
        classification_unchanged=True,source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,private_truth_reads=0)
    if (any(m.get(k)!=v for k,v in expected.items()) or m.get('model_identities',{}).get('E0')!=pm['model_identity']
        or m.get('model_identities',{}).get('SOURCE')!=control['identity']):raise ValueError('Complete fresh metadata and pure source-control identities required')
    done=_read(Path(t['output_root'])/'complete.json')
    required=dict(schema=fresh.SCHEMA,status=fresh.STATUS,run_id=fresh.RUN_ID,code_commit=REFERENCE_CODE,config=t,arm='TARGET_PCD',
        updates=STEPS,total_optimizer_updates=STEPS,source_control_source_only=True,source_control_frozen_before_training=True,
        source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_bodies_changed=True,all_student_projections_changed=True,source_only=False,target_U_used=True,target_U_reads=U_COUNT)
    if (any(done.get(k)!=v for k,v in required.items()) or done.get('parent_signal_identity')!=pm['model_identity']
        or done.get('source_control_identity')!=control['identity'] or done.get('identity')!=m['model_identities'].get('TARGET_PCD')):
        raise ValueError('Original completed fresh-U cache metadata required; no target checkpoint initialization')
    manifest=done.get('U_manifest',{})
    if any(manifest.get(k)!=v for k,v in dict(count=U_COUNT,capsule_id=t['U_capsule_id'],split_id=t['U_split_id']).items()):
        raise ValueError('Fresh public U576 capsule/split metadata differs')
    return done,manifest


def _load_U(c,expected,source_ids):
    t=c['reference_config']['train_config'];root=Path(t['output_root']);public=Path(t['U_public_root'])
    manifest=_read(root/'U_manifest.json');header=_read(public/'manifest.json')
    if (manifest!=expected or header!=manifest or any(header.get(k)!=v for k,v in dict(schema='phase3_min_v1',data_status='VALIDATED_ONCE',
        capsule_id=t['U_capsule_id'],split_id=t['U_split_id'],count=U_COUNT,registered_classes=body.base.native.CLASSES,
        processing_route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000,single_observation=True,iq_shape=[U_COUNT,2,256]).items())):
        raise ValueError('Existing star-U cache and public headers differ')
    public_ids=np.load(public/'ids.npy',allow_pickle=False)
    with np.load(root/'U_features.npz',allow_pickle=False) as z:
        if set(z.files)!={'ids','views','parent_z','neighbors'}:raise ValueError('Original fresh U cache fields differ')
        # The neighbors member is never indexed or materialized by this method.
        U={k:z[k].copy() for k in ('ids','views','parent_z')}
    ids=U['ids']
    if (ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT
        or any(re.fullmatch('p3fresh-[0-9a-f]{24}',str(x)) is None for x in ids) or not np.array_equal(ids,public_ids)
        or set(ids.tolist())&source_ids or U['views'].shape!=(3,U_COUNT,2,256) or U['views'].dtype!=np.float32
        or U['parent_z'].shape!=(U_COUNT,160) or U['parent_z'].dtype!=np.float32
        or not np.isfinite(U['views']).all() or not np.isfinite(U['parent_z']).all()
        or not np.allclose(np.linalg.norm(U['parent_z'].astype(np.float64),axis=1),1.,rtol=0,atol=2e-6)):
        raise ValueError('Aligned finite unchanged three-view P1 U cache and public opaque IDs required')
    return U


def _gradient_smoke(student,device):
    before=_snapshot(student)
    try:
        with torch.random.fork_rng(devices=body.base.native._cuda_devices(student)):
            torch.manual_seed(2026103115);iq=torch.randn(32,2,256,device=device)
            teacher=deepcopy(student).eval().requires_grad_(False)
            with torch.no_grad():weak=teacher(iq)
            centers,_=body.core.initialize_prototypes(weak);head=torch.nn.Parameter(centers);center=weak.new_zeros(16)
            q=core.teacher_targets(core.cosine_logits(weak,centers),center)
            generator=torch.Generator(device='cpu').manual_seed(body.AUG_SEED)
            views=[body.views.phase_gain_view(iq[:8].detach().cpu(),generator).to(device) for _ in range(2)]
            z=[student(x) for x in views]
            ce=.5*sum(core.soft_target_ce(core.cosine_logits(value,head),q[:8]) for value in z)
            probe=(z[0]*torch.linspace(.1,1.,160,device=device)).sum(1).mean()
            (probe+.1*ce).backward();g=body._gradients(student,head)
            if any(not np.isfinite(v) for v in g.values()) or any(g[k]<=0 for k in ('body','middle','hidden','last','prototype')):
                raise ValueError('Synthetic same-record teacher/student body/projection/head gradients must be live')
            if any(not torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step gradient smoke changed state')
            if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher must remain stop-gradient')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,Sinc_gradients_none=True,teacher_gradients_none=True,
                parameters_buffers_unchanged=True,synthetic_student_signal_rows=16,synthetic_teacher_signal_rows=32)
    finally:student.zero_grad(set_to_none=True)


def _identity(c,arm,parent,m,U_manifest,control):
    result=body._identity(fresh._flat(c['reference_config']['train_config']),'TARGET_PCD',parent,m,U_manifest)
    result.update(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),
        updates=STEPS,target_U_reads=U_COUNT,source_only=False,target_U_used=True,temporary_head_parameters=2560,
        auxiliary_prototype_parameters=2560,teacher_update_momentum=.996 if arm=='EMA_TEACHER' else None,
        teacher_source_initialization='original_source_only_r03',source_control_identity=control['identity'],source_control_source_only=True,
        prototype_deployed=False,teacher_deployed=False,center_deployed=False,U_cache_reused=True,neighbors_consumed=False)
    return result


def _loop(student,teacher,head,teacher_head,center,banks,U,c,arm,device,opt,folder,control_path):
    p=fresh._flat(c['reference_config']['train_config'])['reference_config']['train_config']
    sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed']);data={k:body._runtime(v,device) for k,v in banks.items()}
    target=body._runtime(U,device);rng=np.random.default_rng(body.U_SEED);usage={k:set() for k in ('L','OE','U')};trace=[]
    with control_path.open(encoding='utf-8') as control,(folder/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);old=json.loads(control.readline())
            if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):
                raise ValueError('Matched source sampling differs from frozen SOURCE444 control')
            opt.zero_grad(set_to_none=True);source,metrics=body._source_objective(student,data['L'],data['OE'],sample['L'],sample['OE'],device)
            ui=rng.choice(U_COUNT,U_BATCH,replace=False);indices=body.base._tensor(ui,device,torch.long)
            old_center=center.detach().clone()
            with torch.no_grad():
                logits=core.cosine_logits(teacher(target['views'][0]),teacher_head)
                q=core.teacher_targets(logits,center)
            strong=[student(target['views'][j,indices]) for j in (1,2)]
            losses=[core.soft_target_ce(core.cosine_logits(z,head),q[indices]) for z in strong]
            ce=.5*(losses[0]+losses[1]);total=source+.1*ce
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source/same-record self-distillation objective')
            # q and the loss above always use the pre-update center.
            core.update_center_(center,logits)
            total.backward();gradients=body._gradients(student,head)
            norm=float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],5.))
            if not np.isfinite(norm) or any(not np.isfinite(v) for v in gradients.values()):raise FloatingPointError('Nonfinite student/head gradient')
            opt.step()
            with torch.no_grad():head.copy_(F.normalize(head,dim=1))
            updated=0
            if arm=='EMA_TEACHER':
                updated=core.ema_update_(teacher,student,.996);core.ema_tensor_(teacher_head,head,.996)
            if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients')
            for k in ('L','OE'):usage[k].update(map(int,sample[k]))
            usage['U'].update(map(int,ui))
            row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),
                loss_total=float(total.detach().cpu()),**metrics,U_CE=float(ce.detach().cpu()),U_CE_view1=float(losses[0].detach().cpu()),U_CE_view2=float(losses[1].detach().cpu()),
                source_anchor_weight=10.,U_CE_weight=.1,student_temperature=.1,teacher_temperature=.04,center_momentum=.9,
                center_before=old_center.cpu().tolist(),center_after=center.detach().cpu().tolist(),center_updates=step,
                teacher_entropy=float((-(q*q.clamp_min(1e-12).log()).sum(1)).mean().cpu()),teacher_slot_mass=q.mean(0).cpu().tolist(),
                gradient_norms=gradients,gradient_norm_before_clip=norm,teacher_gradients_none=True,teacher_parameter_updates=updated,
                teacher_EMA_updates=step if arm=='EMA_TEACHER' else 0,student_optimizer_updates=step,head_normalizations=step,
                source_L_exposures=192*step,source_OE_exposures=64*step,teacher_forward_rows=U_COUNT*step,
                student_forward_rows=(256+2*U_BATCH)*step,U_strong_record_exposures=2*U_BATCH*step,
                ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True)
            trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%100==0 or step==STEPS:print('MOMENTUM_PROTOTYPE_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if control.readline():raise ValueError('SOURCE control trace contains extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable momentum training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    t=c['reference_config']['train_config'];flat=fresh._flat(t);b=body._backend(flat,device)
    snapshots=[(m,_snapshot(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def check():
        if any(not torch.equal(v,m.state_dict()[k].detach().cpu()) for m,state in snapshots for k,v in state.items()):raise ValueError('Original F0/r03/DURA changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source/U')
    pm,_,_,_=fresh._reference(t);control,control_steps=fresh._source_control(t,pm);fresh_done,U_manifest=_fresh_metadata(c,pm,control)
    parent=pm['model_identity']
    if b['model_identity']!=parent or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Only matching pure source r03/F0/G initialization allowed')
    probe=body._student(b['model']);gradient=_gradient_smoke(probe,device);del probe;check()
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));roles={}
    try:
        banks,m,roles=body._source_cache(flat,b,out,check)
        if m!=control['source_feature_manifest']:raise ValueError('Source replay handles differ from SOURCE control')
        U=_load_U(c,U_manifest,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()))
        _write(out/'U_manifest.json',U_manifest)
        _write(out/'U_cache_reference.json',dict(path=str(Path(t['output_root'])/'U_features.npz'),reused=True,count=U_COUNT,
            capsule_id=U_manifest['capsule_id'],split_id=U_manifest['split_id'],new_physical_observations=0,neighbors_consumed=False,new_DURA_forward_rows=0))
        centers,initialization=body.core.initialize_prototypes(body.base._tensor(U['parent_z'],device));_write(out/'prototype_initialization.json',initialization)
        initial=_snapshot(b['model']);rows={};matched=None;oe_teacher=banks['OE']['teacher_u'].copy()
        for arm in ARMS:
            student=body._student(b['model']);teacher=deepcopy(b['model']).eval().requires_grad_(False)
            if any(not torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in initial.items()):raise ValueError('Independent student initialization differs from pure r03')
            head=torch.nn.Parameter(centers.detach().clone());teacher_head=centers.detach().clone();center=centers.new_zeros(16)
            folder=out/arm;folder.mkdir();buffers={k:v.detach().clone() for k,v in student.named_buffers()};teacher_buffers={k:v.detach().clone() for k,v in teacher.named_buffers()}
            opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad]+[head],lr=c['lr'],weight_decay=.0001,eps=1e-8)
            trace,unique=_loop(student,teacher,head,teacher_head,center,banks,U,c,arm,device,opt,folder,control_steps)
            trajectory=[(x['L_indices'],x['OE_indices'],x['U_indices']) for x in trace]
            if matched is not None and trajectory!=matched:raise ValueError('Both arms require identical source/U streams')
            matched=trajectory
            if any(not torch.equal(v,dict(student.named_buffers())[k]) for k,v in buffers.items()) or any(not torch.equal(v,dict(teacher.named_buffers())[k]) for k,v in teacher_buffers.items()):raise ValueError('Student/teacher buffers changed')
            state=_snapshot(student);teacher_state=_snapshot(teacher)
            if any(not torch.equal(v,state[k]) or not torch.equal(v,teacher_state[k]) for k,v in initial.items() if k.startswith(body.signalcore.SINC_PREFIX)):
                raise ValueError('Frozen student/teacher Sinc changed')
            changed={k:not torch.equal(v,state[k]) for k,v in initial.items()}
            if not any(v for k,v in changed.items() if k.startswith('signal.') and not k.startswith(body.signalcore.SINC_PREFIX)) or not any(v for k,v in changed.items() if k.startswith('projection.')):
                raise ValueError('Body and projection must actually update')
            if arm=='FIXED_TEACHER' and (any(not torch.equal(v,teacher_state[k]) for k,v in initial.items()) or not torch.equal(teacher_head,centers)):
                raise ValueError('Fixed teacher/model head changed')
            if not np.array_equal(oe_teacher,banks['OE']['teacher_u']):raise ValueError('Original OE topology teacher changed')
            if any(not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Finite full SignalOpen state required')
            check();identity=_identity(c,arm,parent,m,U_manifest,control)
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=state,parent_signal_identity=parent,source_feature_manifest=m),folder/'final.pt')
            torch.save(dict(schema=SCHEMA,arm=arm,network=teacher_state,head=teacher_head.detach().cpu(),student_head=head.detach().cpu(),center=center.detach().cpu(),training_audit_only=True,deployed=False),folder/'aux_teacher.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=identity,updates=STEPS,
                source_only=False,target_U_used=True,target_U_reads=U_COUNT,U_manifest=U_manifest,source_feature_manifest=m,
                initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,
                model_buffers_unchanged=True,teacher_buffers_unchanged=True,teacher_Sinc_unchanged=True,original_models_unchanged=True,
                teacher_gradients_none=True,teacher_forward_rows=U_COUNT*STEPS,student_forward_rows=(256+2*U_BATCH)*STEPS,
                source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],
                prototype_deployed=False,teacher_deployed=False,center_deployed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('MOMENTUM_PROTOTYPE_ARM_COMPLETE',arm,flush=True)
            del student,teacher,head,teacher_head,opt
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={a:v['identity'] for a,v in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],
            source_feature_manifest=m,U_manifest=U_manifest,source_control_identity=control['identity'],source_control_source_only=True,source_control_frozen_before_training=True,
            source_sampling_matched=True,U_sampling_matched=True,source_only=False,target_U_used=True,target_U_reads=U_COUNT,U_cache_reused=True,neighbors_consumed=False,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,
            teacher_forward_rows_per_arm=U_COUNT*STEPS,student_forward_rows_per_arm=(256+2*U_BATCH)*STEPS,
            real_parent_signal_forward_rows=sum(COUNTS.values()),real_student_signal_forward_rows=2*(256+2*U_BATCH)*STEPS,real_teacher_signal_forward_rows=2*U_COUNT*STEPS,
            real_F_forward_rows=0,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,
            synthetic_student_signal_rows=gradient['synthetic_student_signal_rows'],synthetic_teacher_signal_rows=gradient['synthetic_teacher_signal_rows'],
            prototype_deployed=False,teacher_deployed=False,center_deployed=False,per_record_U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.),**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():body.base.close_role(value)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only the fixed two momentum-teacher arms may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both momentum arms must complete before prediction') from exc
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_control_source_only=True,source_control_frozen_before_training=True,source_sampling_matched=True,U_sampling_matched=True,source_only=False,target_U_reads=U_COUNT,
        original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,
        final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in required.items()) or done.get('rows')!=rows:raise ValueError('Complete matched two-arm source/U training required')
    t=c['reference_config']['train_config'];pm,_,_,_=fresh._reference(t);control,_=fresh._source_control(t,pm);_,manifest=_fresh_metadata(c,pm,control)
    p=fresh._flat(t)['reference_config']['train_config'];model,parent=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    if parent!=pm['model_identity'] or done.get('parent_signal_identity')!=parent or done.get('source_control_identity')!=control['identity'] or done.get('U_manifest')!=manifest:
        raise ValueError('Strict pure source-r03/SOURCE444/cache ancestry required')
    for a,row in rows.items():
        identity=_identity(c,a,parent,done['source_feature_manifest'],manifest,control)
        expected=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=a,identity=identity,updates=STEPS,source_only=False,target_U_used=True,target_U_reads=U_COUNT,
            initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,teacher_buffers_unchanged=True,
            teacher_Sinc_unchanged=True,original_models_unchanged=True,teacher_gradients_none=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in expected.items()) or done.get('identities',{}).get(a)!=identity:raise ValueError('Complete strict per-arm identity required')
    identity=done['identities'][arm];value=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if (value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity or value.get('parent_signal_identity')!=parent
        or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict full SignalOpen identity/config/source handles required')
    state=value.get('network')
    if not isinstance(state,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in state.values()):raise ValueError('Finite full student state required')
    original=model.state_dict();fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(body.signalcore.SINC_PREFIX)}
    if any(k not in state or state[k].dtype!=original[k].dtype or not torch.equal(original[k].detach().cpu(),state[k]) for k in fixed):raise ValueError('Frozen Sinc/persistent buffers cannot change')
    if set(state)!=set(original) or any(state[k].shape!=original[k].shape or state[k].dtype!=original[k].dtype for k in original):raise ValueError('Strict complete student tensor shapes/dtypes required')
    try:model.load_state_dict(state,strict=True)
    except RuntimeError as exc:raise ValueError('Strict complete SignalOpen network required') from exc
    model.eval().requires_grad_(False);model._momentum_prototype_identity=identity;return model,identity


def main():
    print(json.dumps(dict(event='MOMENTUM_PROTOTYPE_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
