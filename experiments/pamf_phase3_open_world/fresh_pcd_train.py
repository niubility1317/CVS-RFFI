"""Fixed PCD on a new public discovery capsule; no private data configuration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from torch.nn import functional as F
from . import star_body_prototype_train as body
from . import star_body_prototype_run as reference
from .data import load_public

RUN_ID='20261031-phase3-fresh-development-rx3-2-s2026092701-r01'
SCHEMA='pamf_phase3_fresh_pcd_train_v1'
STATUS='FRESH_PCD_TRAINING_COMPLETE'
ARM='TARGET_PCD'
ARMS=(ARM,)
STEPS=444
U_COUNT=576
U_BATCH=64
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='4273e88392abe0717c1bd94fca1370d19c67cbb3'
FIELDS=TRAIN_FIELDS=set(body.FIELDS)|{'U_expected_count'}
ZEROS=body.ZEROS
_read,_write,_snapshot=body._read,body._write,body._snapshot


def _flat(c):return {k:(body.RUN_ID if k=='run_id' else c['reference_config']['reference_config'] if k=='reference_config' else c[k]) for k in body.FIELDS}


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or type(c.get('U_expected_count')) is not int or c['U_expected_count']!=U_COUNT
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('U_public_root','U_capsule_id','U_split_id','output_root'))):
        raise ValueError('Exact ten-field public-only fresh PCD configuration required; no TX/K/truth/Q')
    r=reference.validate_config(c['reference_config']);u=Path(c['U_public_root'])
    if r['code_commit']!=REFERENCE_CODE or u.name!='public' or u.parent.name!='discovery' or not c['U_capsule_id'].startswith('p3fresh-'):raise ValueError('Fixed completed SOURCE control and fresh star discovery public capsule required')
    out=Path(c['output_root']).resolve();p=r['reference_config'];t=p['train_config']
    roots=[r['output_root'],r['train_config']['output_root'],c['U_public_root'],p['output_root'],p['reference_prediction_root'],p['dura_checkpoint'],p['dura_release'],t['output_root'],t['IQ_root'],t['feature_root'],t['control_root']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New immutable PCD output overlaps source/control/public/model input')
    return c


def _reference(c):return body._reference(_flat(c))


def _source_control(c,pm):
    r=c['reference_config'];t=r['train_config'];root=Path(t['output_root']);row=_read(root/'SOURCE/complete.json')
    expected=dict(schema=body.SCHEMA,status=body.ROW_STATUS,run_id=body.RUN_ID,code_commit=reference.TRAIN_CODE,arm='SOURCE',config=t,updates=STEPS,
        source_only=True,target_U_used=False,target_U_reads=0,U_manifest=None,initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,original_models_unchanged=True)
    i=row.get('identity',{})
    if (any(row.get(k)!=v for k,v in expected.items()) or any(i.get(k)!=v for k,v in dict(run_id=body.RUN_ID,code_commit=reference.TRAIN_CODE,arm='SOURCE',updates=STEPS,source_only=True,target_U_used=False,target_U_reads=0).items())
            or i.get('parent_signal_identity')!=pm['model_identity'] or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):raise ValueError('Frozen source-only SOURCE control ancestry required; no old target student')
    return row,root/'SOURCE/steps.jsonl'


def _public(c):
    m,ids,iq=load_public(c['U_public_root'])
    if (any(m.get(k)!=v for k,v in dict(schema='phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id=c['U_capsule_id'],split_id=c['U_split_id'],count=U_COUNT,registered_classes=body.base.native.CLASSES).items())
            or ids.shape!=(U_COUNT,) or any(re.fullmatch('p3fresh-[0-9a-f]{24}',str(x)) is None for x in ids)):
        iq._mmap.close();raise ValueError('Fresh star U public capsule/split/opaque IDs differ')
    return m,ids,iq


def _target_cache(c,b,out,device,check,source_ids):
    iq=None
    try:
        manifest,ids,iq=_public(c)
        if set(ids.tolist())&source_ids:raise ValueError('Source and new U physical IDs overlap')
        blocks=[]
        for start in range(0,U_COUNT,128):
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                x=body.base._tensor(iq[start:start+128],device);value=b['dura'][1](x)[0].detach().clone()
            if value.shape!=x.shape or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):raise ValueError('One finite fixed DURA output per public U required')
            blocks.append(np.asarray(value.cpu().tolist(),np.float32));check()
        restored=np.concatenate(blocks);raw=body.base._tensor(restored,'cpu');generator=torch.Generator(device='cpu').manual_seed(body.AUG_SEED)
        values=[restored]+[np.asarray(body.views.phase_gain_view(raw,generator).tolist(),np.float32) for _ in range(2)]
        weak=body._parent_z(b['model'],restored,device);check();neighbors=body.core.nearest_neighbors(body.base._tensor(weak,'cpu'),ids)
        cache=dict(ids=ids.copy(),views=np.stack(values),parent_z=weak,neighbors=np.asarray(neighbors.cpu().tolist(),np.int64));body._npz(out/'U_features.npz',cache);_write(out/'U_manifest.json',manifest)
        return cache,manifest
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()


def _identity(c,parent_identity,m,U_manifest,control):
    value=body._identity(_flat(c),ARM,parent_identity,m,U_manifest)
    value.update(run_id=RUN_ID,arm=ARM,checkpoint=str(Path(c['output_root'])/'final.pt'),updates=STEPS,projection_updates=STEPS,new_projection_updates=STEPS,new_body_updates=STEPS,
        open_signal_updates=STEPS,signal_encoder_updates=STEPS,projection_cumulative_updates=parent_identity['projection_updates']+STEPS,signal_feature_cumulative_updates=parent_identity['signal_feature_cumulative_updates']+STEPS,
        target_U_reads=U_COUNT,U_manifest=U_manifest,source_control_identity=control['identity'],source_control_source_only=True,CCD_updates=0)
    return value


def _loop(model,banks,c,device,opt,out,U,prototypes,control_steps):
    p=c['reference_config']['reference_config']['train_config'];sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed']);data={k:body._runtime(v,device) for k,v in banks.items()}
    target=body._runtime(U,device);neighbors=body.base._tensor(U['neighbors'],device,torch.long);rng=np.random.default_rng(body.U_SEED);trace=[];usage={k:set() for k in ('L','OE','U')}
    with control_steps.open(encoding='utf-8') as control,(out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);old=json.loads(control.readline())
            if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched source sampling differs from frozen SOURCE control')
            opt.zero_grad(set_to_none=True);total,metrics=body._source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device)
            ui=rng.choice(U_COUNT,U_BATCH,replace=False);indices=body.base._tensor(ui,device,torch.long)
            with torch.no_grad():weak=model(target['views'][0])
            strong=[model(target['views'][j,indices]) for j in (1,2)];pcd=body.core.consistency(weak,*strong,prototypes,indices,neighbors);total=total+.1*pcd['loss']
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite fixed source/PCD objective')
            total.backward();gradients=body._gradients(model,prototypes) if step<=2 else None;params=[p for g in opt.param_groups for p in g['params']];norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite ordinary body/projection gradient norm')
            opt.step()
            with torch.no_grad():prototypes.copy_(F.normalize(prototypes,dim=1))
            for k in ('L','OE'):usage[k].update(map(int,sample[k]))
            usage['U'].update(map(int,ui));row=dict(step=step,arm=ARM,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),loss_total=float(total.detach().cpu()),**metrics,
                U_global=float(pcd['global'].detach().cpu()),U_local=float(pcd['local'].detach().cpu()),source_anchor_weight=10.,U_PCD_weight=.1,CCD_weight=0.,gradient_norms=gradients,gradient_norm_before_clip=norm,
                ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True)
            trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%100==0 or step==STEPS:print('FRESH_PCD_PROGRESS',step,STEPS,row['loss_total'],flush=True)
        if control.readline():raise ValueError('Source control trace contains extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable fresh PCD training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;flat=_flat(c);b=body._backend(flat,device);snapshots=[(m,_snapshot(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def check():
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for model,before in snapshots for k,v in before.items()):raise ValueError('Original F0/r03/DURA changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint no-query smoke must precede source/U')
    pm,_,_,_=_reference(c);control,control_steps=_source_control(c,pm);parent_identity=pm['model_identity']
    if b['model_identity']!=parent_identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original source F0/r03/G identity differs')
    student=body._student(b['model']);gradient=body._gradient_smoke(student,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));roles={}
    try:
        banks,m,roles=body._source_cache(flat,b,out,check)
        if m!=control['source_feature_manifest']:raise ValueError('Source replay handle differs from frozen SOURCE control')
        teacher=banks['OE']['teacher_u'].copy();U,U_manifest=_target_cache(c,b,out,device,check,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()))
        centers,initialization=body.core.initialize_prototypes(body.base._tensor(U['parent_z'],device));prototypes=torch.nn.Parameter(centers)
        before={n:p.detach().cpu().clone() for n,p in student.named_parameters()};buffers={n:p.detach().clone() for n,p in student.named_buffers()}
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad]+[prototypes],lr=c['lr'],weight_decay=.0001,eps=1e-8);trace,unique=_loop(student,banks,c,device,opt,out,U,prototypes,control_steps)
        if any(not torch.equal(v,dict(student.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Student buffers changed')
        changed={n:not torch.equal(v,dict(student.named_parameters())[n].detach().cpu()) for n,v in before.items()}
        if any(v for n,v in changed.items() if n.startswith(body.signalcore.SINC_PREFIX)):raise ValueError('Frozen Sinc changed')
        if not any(v for n,v in changed.items() if n.startswith('signal.') and not n.startswith(body.signalcore.SINC_PREFIX)) or not any(v for n,v in changed.items() if n.startswith('projection.')):raise ValueError('Body/projection must actually update')
        if not np.array_equal(teacher,banks['OE']['teacher_u']):raise ValueError('Original OE topology teacher changed')
        check();identity=_identity(c,parent_identity,m,U_manifest,control);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite full SignalOpen state required')
        torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent_identity,source_feature_manifest=m),out/'final.pt')
        body._npz(out/'prototype_state.npz',dict(prototypes=np.asarray(prototypes.detach().cpu().tolist(),np.float32)));_write(out/'prototype_initialization.json',initialization)
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,identity=identity,updates=STEPS,total_optimizer_updates=STEPS,
            parent_signal_identity=parent_identity,parent_F0_identity=parent_identity['parent_F0_identity'],dura_identity=parent_identity['dura_identity'],source_feature_manifest=m,
            source_control_identity=control['identity'],source_control_training_run_id=body.RUN_ID,source_control_training_code_commit=reference.TRAIN_CODE,source_control_source_only=True,source_control_frozen_before_training=True,source_sampling_matched=True,
            source_only=False,target_U_used=True,target_U_reads=U_COUNT,U_manifest=U_manifest,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
            all_student_bodies_changed=True,all_student_projections_changed=True,initial_state_equal_to_parent=True,original_OE_teacher_unchanged=True,
            student_encoder_unchanged=False,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,
            source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,real_parent_signal_forward_rows=sum(COUNTS.values())+U_COUNT,real_student_signal_forward_rows=(256+U_COUNT+2*U_BATCH)*STEPS,
            G_forward_rows=U_COUNT,real_F_forward_rows=0,synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_signal_rows=gradient['synthetic_student_signal_rows'],
            first_step=trace[0],last_step=trace[-1],unique_training_rows=unique,CCD_updates=0,source_density_fit_updates=0,threshold_fit_updates=0,prototype_deployed=False,per_record_U_bank_deployed=False,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=STEPS),formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():body.base.close_role(value)


def load_trained_model(c,device='cuda:0'):
    validate_config(c);root=Path(c['output_root'])
    try:done=_read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Complete fresh PCD training required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,updates=STEPS,total_optimizer_updates=STEPS,source_control_source_only=True,
        source_control_frozen_before_training=True,source_sampling_matched=True,source_only=False,target_U_used=True,target_U_reads=U_COUNT,original_models_unchanged=True,all_student_buffers_unchanged=True,
        all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,CCD_updates=0,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen source-matched fresh PCD completion required')
    pm,_,_,_=_reference(c);control,_=_source_control(c,pm);p=c['reference_config']['reference_config']['train_config'];model,parent_identity=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    identity=_identity(c,parent_identity,done['source_feature_manifest'],done['U_manifest'],control);value=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if (parent_identity!=pm['model_identity'] or done['identity']!=identity or done['source_control_identity']!=control['identity'] or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity
            or value.get('parent_signal_identity')!=parent_identity or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict original r03/SOURCE/fresh-U ancestry required')
    network=value.get('network')
    if not isinstance(network,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite SignalOpen state required')
    original=model.state_dict();fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(body.signalcore.SINC_PREFIX)}
    if any(k not in network or not torch.equal(original[k].detach().cpu(),network[k]) for k in fixed):raise ValueError('Frozen Sinc/persistent buffers cannot be replaced')
    try:model.load_state_dict(network,strict=True)
    except RuntimeError as exc:raise ValueError('Strict full body/projection state required') from exc
    model.eval().requires_grad_(False);model._fresh_pcd_identity=identity
    return model,identity


def main():
    print(json.dumps(dict(event='FRESH_PCD_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
