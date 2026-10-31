"""Matched source/native-identity and target-PCD joint signal students."""
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
from . import fresh_pcd_train as fresh
from . import joint_identity_model as model_api

body=fresh.body
RUN_ID='20261031-phase3-joint-identity-s2026092701-r01'
SCHEMA='pamf_phase3_joint_identity_train_v1'
STATUS='JOINT_IDENTITY_TRAINING_COMPLETE'
ROW_STATUS='JOINT_IDENTITY_STUDENT_FROZEN'
ARMS=('SOURCE_JOINT','TARGET_JOINT')
STEPS=444
U_COUNT=576
U_BATCH=64
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
ZEROS=tuple(k for k in body.ZEROS if k!='identity_head_updates')
_read,_write,_snapshot=body._read,body._write,body._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):raise ValueError('Exact six-field joint identity configuration required; no truth/Q/K')
    r=fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed fresh source/public data configuration required')
    old=r['reference_config'];p=old['reference_config'];t=p['train_config'];out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['U_public_root'],old['output_root'],old['train_config']['output_root'],p['output_root'],p['reference_prediction_root'],p['dura_checkpoint'],p['dura_release'],t['output_root'],t['IQ_root'],t['feature_root'],t['control_root']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('New joint training output overlaps an input')
    return c


def _reference(c):return fresh._reference(c['reference_config'])


def _backend(c,device):return body._backend(fresh._flat(c['reference_config']),device)


def _student(b):return model_api.JointIdentity(b['model'],b['reader'])


def _gradients(model,prototypes=None):
    g=body._gradients(model,prototypes)
    def norm(items):return sum(float(p.grad.detach().double().square().sum().cpu()) for p in items if p.grad is not None)**.5
    g['head']=norm([model.signal.encoder.core.id_backbone.cls_head.weight])
    g['body']=norm(p for n,p in model.named_parameters() if n.startswith('signal.') and n!=model_api.HEAD_WEIGHT and not n.startswith(model_api.SINC_PREFIX))
    return g


def _gradient_smoke(model,device):
    before=_snapshot(model)
    try:
        with torch.random.fork_rng(devices=body.base.native._cuda_devices(model)):
            torch.manual_seed(2026103115);iq=torch.randn(32,2,256,device=device)
            with torch.no_grad():weak=model(iq)
            centers,_=body.core.initialize_prototypes(weak);prototypes=torch.nn.Parameter(centers);neighbors=body.core.nearest_neighbors(weak)
            first=model.joint(iq[:8]);second=model(iq[8:16]);pcd=body.core.consistency(weak,first['z'],second,prototypes,torch.arange(8,device=device),neighbors)
            ce=F.cross_entropy(first['logits'],torch.arange(8,device=device)%6);projection_probe=(first['z']*torch.linspace(.1,1.,160,device=device)).sum(1).mean()
            (ce+projection_probe+.1*pcd['loss']).backward();g=_gradients(model,prototypes)
            if any(not np.isfinite(v) for v in g.values()) or any(g[k]<=0 for k in ('head','body','middle','hidden','last','prototype')):raise ValueError('Live native head/body/projection/prototype gradient smoke failed')
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step joint smoke changed parameters/buffers')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,Sinc_gradients_none=True,parameters_buffers_unchanged=True,synthetic_student_signal_rows=48,
                native_CE=float(ce.detach().cpu()),native_head_scale=30.,native_head_eps=.0001,single_encoder_pass_per_input=True)
    finally:model.zero_grad(set_to_none=True)


def _source_objective(model,L,OE,ix,oi,device):
    i=body.base._tensor(ix,device,torch.long);o=body.base._tensor(oi,device,torch.long);lm,om=L['metadata'],OE['metadata']
    own=model.joint(L['iq'][i]);z=own['z'];oz=model(OE['iq'][o])
    known=body.base.loss.crossset_metric(z,oz,lm['y'][i],lm['rx_i'][i],lm['day_i'][i],om['rx_i'][o],om['day_i'][o])
    other=body.base.loss.oe_structure(oz,OE['teacher_u'][o].detach(),om['rx_i'][o],om['day_i'][o]);anchor=.5*(z-L['parent_z'][i].detach()).square().sum(1).mean()+.5*(oz-OE['parent_z'][o].detach()).square().sum(1).mean()
    ce=F.cross_entropy(own['logits'],lm['y'][i],label_smoothing=0.);total=known['loss']+other['loss']+10*anchor+ce
    return total,dict(source_loss=float(total.detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),OE_KL=float(other['loss'].detach().cpu()),source_anchor=float(anchor.detach().cpu()),L_CE=float(ce.detach().cpu()),identity_CE_weight=1.)


def _identity(c,arm,parent_identity,m,U_manifest):
    target=arm=='TARGET_JOINT';flat=fresh._flat(c['reference_config']);i=body._identity(flat,'TARGET_PCD' if target else 'SOURCE',parent_identity,m,U_manifest)
    i.update(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,
        projection_updates=STEPS,new_projection_updates=STEPS,new_body_updates=STEPS,open_signal_updates=STEPS,signal_encoder_updates=STEPS,identity_head_updates=STEPS,
        projection_cumulative_updates=parent_identity['projection_updates']+STEPS,signal_feature_cumulative_updates=parent_identity['signal_feature_cumulative_updates']+STEPS,
        signal_parameters=335681,signal_feature_parameters=232033,metric_parameters=102688,identity_head_parameters=960,trainable_parameters=335633,
        body_trainable_parameters=231985,trainable_projection_parameters=102688,identity_head_jointly_trained=True,identity_head_replaced=False,native_head_structure_unchanged=True,
        identity_head_initialization='same_source_original_F0_native_cosine_weight',identity_head_scale=30.,identity_head_eps=.0001,identity_path='student_native_raw_fused160',
        source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None)
    return i


def _loop(model,banks,c,arm,device,opt,out,control_steps,U=None,prototypes=None):
    p=c['reference_config']['reference_config']['reference_config']['train_config'];sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed']);data={k:body._runtime(v,device) for k,v in banks.items()}
    target=arm=='TARGET_JOINT';target_data=body._runtime(U,device) if target else None;neighbors=body.base._tensor(U['neighbors'],device,torch.long) if target else None
    rng=np.random.default_rng(body.U_SEED) if target else None;matched=(Path(c['output_root'])/'SOURCE_JOINT/steps.jsonl').open(encoding='utf-8') if target else None;usage={k:set() for k in ('L','OE','U')};trace=[]
    try:
        with control_steps.open(encoding='utf-8') as old_source,(out/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step);old=json.loads(old_source.readline())
                if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Frozen source sampling differs')
                if target:
                    left=json.loads(matched.readline())
                    if left.get('step')!=step or any(left.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched joint source sampling differs')
                opt.zero_grad(set_to_none=True);total,metrics=_source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device);ui=np.empty(0,np.int64);pcd=dict(U_global=0.,U_local=0.)
                if target:
                    ui=rng.choice(U_COUNT,U_BATCH,replace=False);indices=body.base._tensor(ui,device,torch.long)
                    with torch.no_grad():weak=model(target_data['views'][0])
                    strong=[model(target_data['views'][j,indices]) for j in (1,2)];r=body.core.consistency(weak,*strong,prototypes,indices,neighbors);total=total+.1*r['loss']
                    pcd=dict(U_global=float(r['global'].detach().cpu()),U_local=float(r['local'].detach().cpu()))
                if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite native CE/source/PCD objective')
                total.backward();g=_gradients(model,prototypes) if step<=2 else None;params=[p for group in opt.param_groups for p in group['params']];norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite ordinary joint gradient norm')
                opt.step()
                if target:
                    with torch.no_grad():prototypes.copy_(F.normalize(prototypes,dim=1))
                for k in ('L','OE'):usage[k].update(map(int,sample[k]))
                usage['U'].update(map(int,ui));row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),loss_total=float(total.detach().cpu()),**metrics,**pcd,
                    source_anchor_weight=10.,U_PCD_weight=.1 if target else 0.,gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True,
                    student_signal_rows=256+(U_COUNT+2*U_BATCH)*int(target),native_identity_L_rows=192)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('JOINT_IDENTITY_PROGRESS',arm,step,STEPS,row['loss_total'],row['L_CE'],flush=True)
            if old_source.readline() or (matched is not None and matched.readline()):raise ValueError('Source traces contain extra updates')
        return trace,{k:len(v) for k,v in usage.items()}
    finally:
        if matched is not None:matched.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable joint identity training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device);snapshots=[(m,_snapshot(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def check():
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for model,before in snapshots for k,v in before.items()):raise ValueError('Original F0/r03/G changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source/U')
    pm,_,_,_=_reference(c);control,control_steps=fresh._source_control(c['reference_config'],pm);parent_identity=pm['model_identity']
    if b['model_identity']!=parent_identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original signal/F0/G ancestry differs')
    student=_student(b);initial=_snapshot(student);gradient=_gradient_smoke(student,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));roles={}
    try:
        banks,m,roles=body._source_cache(fresh._flat(c['reference_config']),b,out,check)
        if m!=control['source_feature_manifest']:raise ValueError('Original source replay handle differs')
        teacher=banks['OE']['teacher_u'].copy();rows={};U=None;U_manifest=None
        for arm in ARMS:
            target=arm=='TARGET_JOINT';prototypes=None
            if target:
                U,U_manifest=fresh._target_cache(c['reference_config'],b,out,device,check,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()));student=_student(b)
                centers,initialization=body.core.initialize_prototypes(body.base._tensor(U['parent_z'],device));prototypes=torch.nn.Parameter(centers)
            if any(not torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in initial.items()):raise ValueError('Both joint students must initialize independently from r03 and original F0 W')
            folder=out/arm;folder.mkdir();buffers={n:v.detach().clone() for n,v in student.named_buffers()};before={n:p.detach().cpu().clone() for n,p in student.named_parameters()}
            opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad]+([prototypes] if target else []),lr=c['lr'],weight_decay=.0001,eps=1e-8)
            trace,unique=_loop(student,banks,c,arm,device,opt,folder,control_steps,U,prototypes)
            if any(not torch.equal(v,dict(student.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Joint student buffers changed')
            changed={n:not torch.equal(v,dict(student.named_parameters())[n].detach().cpu()) for n,v in before.items()}
            if any(v for n,v in changed.items() if n.startswith(model_api.SINC_PREFIX)):raise ValueError('Frozen joint Sinc changed')
            if (not changed[model_api.HEAD_WEIGHT] or not any(v for n,v in changed.items() if n.startswith('signal.') and n!=model_api.HEAD_WEIGHT and not n.startswith(model_api.SINC_PREFIX))
                    or not any(v for n,v in changed.items() if n.startswith('projection.'))):raise ValueError('Body/projection/native head must actually update')
            if not np.array_equal(teacher,banks['OE']['teacher_u']):raise ValueError('Original OE topology teacher changed')
            check();identity=_identity(c,arm,parent_identity,m,U_manifest);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite joint network required')
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent_identity,parent_F0_identity=pm['parent_F0_identity'],source_feature_manifest=m),folder/'final.pt')
            if target:
                body._npz(folder/'prototype_state.npz',dict(prototypes=np.asarray(prototypes.detach().cpu().tolist(),np.float32)));_write(folder/'prototype_initialization.json',initialization)
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=identity,updates=STEPS,identity_head_updates=STEPS,
                source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,source_feature_manifest=m,
                initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,
                source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,signal_forward_rows=(256+(U_COUNT+2*U_BATCH)*int(target))*STEPS,first_step=trace[0],last_step=trace[-1],unique_training_rows=unique,
                optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=STEPS),identity_head_replaced=False,original_F0_frozen=True,prototype_deployed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('JOINT_IDENTITY_ARM_COMPLETE',arm,flush=True);del opt,student
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={arm:row['identity'] for arm,row in rows.items()},
            steps_per_arm=STEPS,total_optimizer_updates=2*STEPS,parent_signal_identity=parent_identity,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_feature_manifest=m,
            source_control_identity=control['identity'],source_control_frozen_before_training=True,source_sampling_matched=True,SOURCE_completed_before_U_open=True,source_only_SOURCE=True,source_only=False,target_U_reads=U_COUNT,U_manifest=U_manifest,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,all_student_heads_changed=True,original_OE_teacher_unchanged=True,
            real_parent_signal_forward_rows=sum(COUNTS.values())+U_COUNT,real_student_signal_forward_rows=sum(row['signal_forward_rows'] for row in rows.values()),G_forward_rows=U_COUNT,real_F_forward_rows=0,
            synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_signal_rows=gradient['synthetic_student_signal_rows'],
            identity_head_replaced=False,identity_head_jointly_trained=True,original_F0_frozen=True,source_density_fit_updates=0,threshold_fit_updates=0,prototype_deployed=False,per_record_U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():body.base.close_role(value)


def _initial_model(c,device):
    p=c['reference_config']['reference_config']['reference_config']['train_config'];parent,parent_identity=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    native=body.base.native.make_trainable_model(body.base._native_config(p['native_config']),device).eval().requires_grad_(False)
    if native._phase3_parent_identity!=parent.original_parent_identity or native._phase3_source_handle!=parent.source_data_handle:raise ValueError('Original F0 head and r03 must share exact source parent/roles')
    return model_api.JointIdentity(parent,native.id_backbone.encoder),parent_identity


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only SOURCE_JOINT or TARGET_JOINT may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');row=_read(root/arm/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Both joint identity arms must complete before load') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),steps_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_only_SOURCE=True,
        source_sampling_matched=True,SOURCE_completed_before_U_open=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,
        all_student_projections_changed=True,all_student_heads_changed=True,identity_head_replaced=False,original_F0_frozen=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows',{}).get(arm)!=row or done.get('identities',{}).get(arm)!=row.get('identity'):raise ValueError('Complete matched joint source/target identities required')
    target=arm=='TARGET_JOINT';required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,updates=STEPS,identity_head_updates=STEPS,source_only=not target,
        target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=done['U_manifest'] if target else None,initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,
        student_projection_changed=True,student_head_changed=True,model_buffers_unchanged=True,identity_head_replaced=False,original_F0_frozen=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(row.get(k)!=v for k,v in required.items()):raise ValueError('Joint source/U row boundary or freeze state differs')
    pm,_,_,_=_reference(c);model,parent_identity=_initial_model(c,device);identity=_identity(c,arm,parent_identity,done['source_feature_manifest'],done['U_manifest']);value=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if (parent_identity!=pm['model_identity'] or row['identity']!=identity or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity
            or value.get('parent_signal_identity')!=parent_identity or value.get('parent_F0_identity')!=pm['parent_F0_identity'] or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict source r03/original F0 head ancestry required')
    network=value.get('network')
    if not isinstance(network,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Full finite joint signal/projection/default head state required')
    original=model.state_dict();fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(model_api.SINC_PREFIX)}
    if any(k not in network or not torch.equal(original[k].detach().cpu(),network[k]) for k in fixed):raise ValueError('Frozen Sinc/persistent buffers cannot be replaced')
    try:model.load_state_dict(network,strict=True)
    except RuntimeError as exc:raise ValueError('Strict complete joint network including native default W required') from exc
    model.eval().requires_grad_(False);model._joint_identity=identity
    return model,identity


def main():
    print(json.dumps(dict(event='JOINT_IDENTITY_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
