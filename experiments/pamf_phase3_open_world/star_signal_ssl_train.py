"""Matched source replay and target-instance learning of independent P1 copies."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import star_signal_open_train as base
from . import star_signal_open_model as signalcore
from . import star_mixture_uda_run as mix
from . import star_representation_loss as views

best=mix.best
RUN_ID='20261031-phase3-signal-ssl-s2026092701-r01'
SCHEMA='pamf_phase3_signal_ssl_train_v1'
STATUS='SIGNAL_SSL_TRAINING_COMPLETE'
ARMS=('SOURCE','TARGET_SSL')
STEPS=444
U_COUNT=384
U_BATCH=64
U_SEED=2026103101
AUG_SEED=2026103102
COUNTS=dict(L=6300,OE=6134)
U_CAPSULE_ID=mix.U_CAPSULE_ID
U_SPLIT_ID=mix.U_SPLIT_ID
TRAIN_FIELDS={'run_id','code_commit','source_train_config','reference_prediction_root','U_public_root','U_capsule_id','U_split_id','output_root','steps','lr'}
FIELDS=TRAIN_FIELDS
ZEROS=('native_updates','identity_head_updates','validation_reads','novel_validation_reads','target_Q_reads','original_U_s_reads','private_truth_reads','U_truth_reads')
_read,_write,_snapshot=base._read,base._write,base._snapshot


def _core():
    from . import star_signal_ssl_core
    return star_signal_ssl_core


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=TRAIN_FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or not isinstance(c.get('source_train_config'),dict) or type(c.get('steps')) is not int or c['steps']!=STEPS
            or type(c.get('lr')) is not float or c['lr']!=.0001
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in TRAIN_FIELDS-{'source_train_config','steps','lr'})):
        raise ValueError('Exact matched source/target two-view training configuration required; no truth/Q/K')
    p=base.validate_config(c['source_train_config'])
    if (p['run_id'],p['code_commit'],p['arm'])!=(best.SIGNAL_RUN,best.SIGNAL_CODE,'E_SIGNAL'):raise ValueError('Only completed original P1 signal r03 may initialize both arms')
    u=Path(c['U_public_root'])
    if u.name!='public' or u.parent.name!='discovery' or c['U_capsule_id']!=U_CAPSULE_ID or c['U_split_id']!=U_SPLIT_ID:
        raise ValueError('Only fixed star-G10 discovery U allowed; clean and future Q are test only')
    out=Path(c['output_root']).resolve()
    for path in [c['reference_prediction_root'],c['U_public_root'],p['output_root'],p['IQ_root'],p['feature_root'],p['control_root'],p['native_config']['source_output'],p['native_config']['source_cache']]:
        path=Path(path).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):raise ValueError('New immutable training output cannot overlap source/public/model inputs')
    return c


def _reference(c):
    pm,state,tau,cal=mix._reference(c['reference_prediction_root'])
    if pm['config']['train_config']!=c['source_train_config']:raise ValueError('Best reference and both students must share the exact original r03 configuration')
    return pm,state,tau,cal


def _backend(c,device):
    reference=_reference(c);backend=best._backend(reference[0]['config'],device)
    return dict(backend,source_reference=reference)


def _student(parent):
    model=deepcopy(parent).requires_grad_(True).eval()
    if signalcore.parameter_report(model)!=dict(signal_feature_parameters=232033,projection_parameters=102688,signal_parameters=334721,combined_active_parameters=567714):
        raise ValueError('Complete live headless signal and metric projection required')
    return model


def _scalar(value):return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _instance(model,raw,generator,device):
    # CPU RNG and list bridge keep the source sampler and CUDA RNG independent.
    x=base._tensor(raw,'cpu');v0=views.phase_gain_view(x,generator);v1=views.phase_gain_view(x,generator)
    result=_core().instance_contrast(model(v0.to(device)),model(v1.to(device)))
    return result,dict(U_SSL=_scalar(result['loss']),U_positive_cosine=_scalar(result['positive_cosine']),
        U_negative_count=int(_scalar(result['negative_count'])),U_anchors=int(_scalar(result['anchors'])))


def _gradient_smoke(model,device):
    before=_snapshot(model);source=base._gradient_smoke(model,device)
    generator=torch.Generator(device='cpu').manual_seed(AUG_SEED)
    raw=np.random.default_rng(U_SEED).normal(size=(4,2,256)).astype(np.float32)
    try:
        model.zero_grad(set_to_none=True);result,metrics=_instance(model,raw,generator,device);result['loss'].backward();gradients=base._gradient_norms(model)
        if any(not np.isfinite(v) for v in gradients.values()) or any(gradients[k]<=0 for k in ('sinc','middle','last')):
            raise ValueError('Target instance objective must reach live Sinc/body/projection')
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step gradient smoke changed parameters/buffers')
        return dict(status='PASS',source=source,U_gradient_norms=gradients,U_metrics=metrics,synthetic_source_student_rows=8,synthetic_U_student_rows=8,
            optimizer_updates=0,parameters_buffers_unchanged=True,synthetic_IQ='P1-shaped synthetic only; no real source/target')
    finally:model.zero_grad(set_to_none=True)


def _public(c):
    from .star_relative_target import _open
    return _open(dict(U_public_root=c['U_public_root'],U_capsule_id=c['U_capsule_id'],U_split_id=c['U_split_id'],U_expected_count=U_COUNT),'U')


def _manifest(m):
    if (not isinstance(m,dict) or m.get('count')!=U_COUNT or m.get('capsule_id')!=U_CAPSULE_ID or m.get('split_id')!=U_SPLIT_ID
            or m.get('registered_classes')!=base.native.CLASSES):raise ValueError('Fixed star-G10 U public identity required')
    return m


def _restore_U(c,b,device,check):
    iq=None
    try:
        manifest,ids,iq=_public(c);_manifest(manifest)
        if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT:raise ValueError('Complete unique unlabelled U IDs required')
        rows=[]
        for start in range(0,U_COUNT,128):
            values=np.asarray(iq[start:start+128])
            if values.dtype!=np.float32 or values.shape!=(len(values),2,256) or not np.isfinite(values).all():raise ValueError('Finite star received full256 IQ required')
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                x=base._tensor(values,device);restored=b['dura'][1](x)[0].detach().clone()
            if restored.dtype!=torch.float32 or restored.shape!=x.shape or not bool(torch.isfinite(restored).all()):raise ValueError('Same fixed DURA must restore every U record once')
            rows.append(np.asarray(restored.cpu().tolist(),np.float32));check();print('SIGNAL_SSL_U_RESTORE_PROGRESS',start+len(values),U_COUNT,flush=True)
        return manifest,ids.copy(),np.concatenate(rows)
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()


def _identity(c,arm,parent,U_manifest):
    target=arm=='TARGET_SSL'
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,
        open_signal_updates=STEPS,projection_updates=STEPS,native_updates=0,original_parent_updates=11100,parent_signal_updates=2220,
        signal_feature_cumulative_updates=parent['signal_feature_cumulative_updates']+STEPS,projection_cumulative_updates=parent['projection_updates']+STEPS,
        signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,
        parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_contract=c['source_train_config']['native_config']['source_contract'],
        source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,
        input_path='cached_P1_IQ_only',identity_head_replaced=False,original_F0_frozen=True,BN_Dropout_eval=True)


def _loop(model,roles,teachers,c,arm,device,opt,out,U=None):
    p=c['source_train_config'];sampler=base.Sampler(teachers['L'],teachers['OE'],p['L_seed'],p['OE_seed'])
    target=arm=='TARGET_SSL';rng=np.random.default_rng(U_SEED) if target else None
    aug=torch.Generator(device='cpu').manual_seed(AUG_SEED) if target else None
    control=(Path(c['output_root'])/'SOURCE/steps.jsonl').open(encoding='utf-8') if target else None
    trace=[];usage=dict(L=set(),OE=set(),U=set())
    try:
        with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if control is not None:
                    line=control.readline()
                    if not line:raise ValueError('Matched SOURCE trace is incomplete')
                    row=json.loads(line)
                    if row.get('step')!=step or any(row.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Source replay indices differ between matched arms')
                opt.zero_grad(set_to_none=True)
                value,metrics=base._objective(model,roles['L'],roles['OE'],teachers,sample,device,step<=2)
                ui=np.empty(0,np.int64);u_metrics=dict(U_SSL=0.,U_positive_cosine=None,U_negative_count=0,U_anchors=0)
                if target:
                    ui=rng.choice(U_COUNT,U_BATCH,replace=False);instance,u_metrics=_instance(model,U[ui],aug,device);value=value+instance['loss']
                if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite source/target signal objective')
                value.backward();gradients=base._gradient_norms(model) if step<=2 else None
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite live signal gradients')
                norm=float(signalcore.clip(opt,model))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite physical gradient norm')
                opt.step()
                for key in ('L','OE'):usage[key].update(map(int,sample[key]))
                usage['U'].update(map(int,ui))
                row=dict(step=step,arm=arm,loss_total=float(value.detach().cpu()),**metrics,**u_metrics,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),
                    source_indices_equal=True,U_SSL_weight=1. if target else 0.,source_weight=1.,gradient_norms=gradients,gradient_norm_before_physical_clip=norm,
                    sinc_lr=float(opt.param_groups[1]['lr']),sinc_optimizer_step=opt._sinc_unit_steps,physical_clip_calls=opt._sinc_physical_clip_calls)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('SIGNAL_SSL_TRAIN_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if control is not None and control.readline():raise ValueError('Matched SOURCE trace contains extra updates')
        return trace,{key:len(value) for key,value in usage.items()}
    finally:
        if control is not None:control.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable matched signal training output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device);pm,_,_,_=b['source_reference'];parent=pm['model_identity']
    if b['model_identity']!=parent or b['reader_identity']!=pm['parent_F0_identity']:raise ValueError('Actual original F0 and frozen r03 checkpoint differ from source reference')
    actual=b['dura_identity'];expected=pm['dura_identity']
    if any(actual.get(k)!=v or expected.get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()) or any(actual.get(k)!=expected[k] for k in ('checkpoint','arm','training_commit','reuse','graph','native','precision','TF32') if k in expected):raise ValueError('Original DURA v1 path required')
    snapshots=[(m,best._snapshot_model(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def check():
        for model,before in snapshots:best._check_model(model,before)
    cp_smoke=best._smoke(b,device);check()
    if cp_smoke.get('status')!='PASS' or cp_smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source/U values')
    student=_student(b['model']);initial=_snapshot(b['model']);gradient=_gradient_smoke(student,device);check()
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=cp_smoke,gradient=gradient))
    m,_,_=base._metadata(c['source_train_config']);roles={};rows={};U_manifest=None;U=None
    try:
        roles,teachers=base._roles(c['source_train_config'],m);teacher_before=teachers['OE']['u'].copy()
        for arm in ARMS:
            if arm=='TARGET_SSL':
                U_manifest,U_ids,U=_restore_U(c,b,device,check)
                with (out/'U_restored.npz').open('xb') as stream:np.savez(stream,ids=U_ids,restored=U)
                _write(out/'U_manifest.json',U_manifest);student=_student(b['model'])
            if any(not torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in initial.items()):raise ValueError('Both students must initialize independently from the same r03 state')
            folder=out/arm;folder.mkdir();buffers={k:v.detach().clone() for k,v in student.named_buffers()};opt=signalcore.optimizer(student,c['lr'])
            trace,unique=_loop(student,roles,teachers,c,arm,device,opt,folder,U)
            if any(not torch.equal(value,dict(student.named_buffers())[key]) for key,value in buffers.items()):raise ValueError('Frozen BN/Sinc buffers changed during signal learning')
            if not np.array_equal(teacher_before,teachers['OE']['u']):raise ValueError('Original source OE teacher changed')
            check();identity=_identity(c,arm,parent,U_manifest);student.eval().requires_grad_(False)
            network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite learned signal checkpoint required')
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent,source_feature_manifest=m,
                source_only=arm=='SOURCE',model_buffers_unchanged=True),folder/'final.pt')
            target=arm=='TARGET_SSL';zero={k:0 for k in ZEROS}
            row=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,identity=identity,updates=STEPS,
                parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_feature_manifest=m,
                source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,
                source_training_counts=dict(COUNTS),source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,U_instance_exposures=U_BATCH*STEPS if target else 0,
                real_student_signal_forward_rows=(256+2*U_BATCH*int(target))*STEPS,unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],
                initial_state_equal_to_parent=True,model_buffers_unchanged=True,original_models_unchanged=True,original_OE_teacher_unchanged=True,
                optimizer=dict(name='AdamW',mode='CENTER_UNITS',lr=.0001,weight_decay=.0001,eps=1e-8,sinc_multiplier=1000.,sinc_eps=1e-11,
                    physical_clip=5.,physical_clip_calls=opt._sinc_physical_clip_calls,sinc_steps=opt._sinc_unit_steps),
                U_seed=U_SEED if target else None,augmentation_seed=AUG_SEED if target else None,U_views='two independent phase_gain draws' if target else None,
                final_phase3_complete=False,**zero)
            _write(folder/'complete.json',row);rows[arm]=row;print('SIGNAL_SSL_ARM_COMPLETE',arm,flush=True);del opt,student
        check()
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={arm:row['identity'] for arm,row in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_feature_manifest=m,
            source_only=False,target_U_reads=U_COUNT,U_manifest=U_manifest,SOURCE_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,
            real_student_signal_forward_rows=sum(row['real_student_signal_forward_rows'] for row in rows.values()),real_F_forward_rows=0,real_parent_signal_forward_rows=0,G_forward_rows=U_COUNT,
            synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_gradient_student_rows=16,
            source_density_fit_updates=0,threshold_fit_updates=0,per_record_U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        _write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():base.close_role(value)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only SOURCE or TARGET_SSL final frozen checkpoints allowed')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');row=_read(root/arm/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Both arms must be complete before any frozen model loads') from exc
    zero={k:0 for k in ZEROS}
    if (any(done.get(k)!=v for k,v in dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
            SOURCE_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,final_phase3_complete=False,**zero).items())
            or any(row.get(k)!=v for k,v in dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,updates=STEPS,
                source_only=arm=='SOURCE',target_U_used=arm=='TARGET_SSL',target_U_reads=U_COUNT if arm=='TARGET_SSL' else 0,
                model_buffers_unchanged=True,original_models_unchanged=True,initial_state_equal_to_parent=True,final_phase3_complete=False,**zero).items())
            or done.get('rows',{}).get(arm)!=row):raise ValueError('Complete matched source/target training identity required')
    _manifest(done.get('U_manifest'));pm,_,_,_=_reference(c);p=c['source_train_config'];model,parent=base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    if parent!=pm['model_identity'] or done.get('parent_signal_identity')!=parent:raise ValueError('Exact original r03 parent must initialize the loader')
    expected=_identity(c,arm,parent,done['U_manifest']);m,_,_=base._metadata(p)
    payload=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if (row.get('identity')!=expected or done.get('identities',{}).get(arm)!=expected or done.get('source_feature_manifest')!=m or row.get('source_feature_manifest')!=m
            or not isinstance(payload,dict) or payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('identity')!=expected
            or payload.get('parent_signal_identity')!=parent or payload.get('source_feature_manifest')!=m or payload.get('source_only')!=(arm=='SOURCE') or payload.get('model_buffers_unchanged') is not True):
        raise ValueError('Learned checkpoint source ancestry/target-U role differs')
    state=payload.get('network');original=model.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(original) or any(not torch.is_tensor(v) or v.shape!=original[k].shape or v.dtype!=original[k].dtype or not bool(torch.isfinite(v).all()) for k,v in state.items())):
        raise ValueError('Strict complete finite signal/projection checkpoint required')
    model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False),expected


def main(argv=None):
    print(json.dumps(dict(event='SIGNAL_SSL_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);train(_read(a.config),a.device)


if __name__=='__main__':main()
