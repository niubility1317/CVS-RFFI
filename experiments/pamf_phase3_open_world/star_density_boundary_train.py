"""Source-only r03 replay with point anchoring and fixed-density BCE."""
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
from . import star_signal_open_eval as signal_eval
from . import star_local_cov_run as reference

parent=reference.parent
RUN_ID='20261031-phase3-density-boundary-s2026092701-r01'
SCHEMA='pamf_phase3_density_boundary_train_v1'
STATUS='DENSITY_BOUNDARY_TRAINING_COMPLETE'
ARMS=('REPLAY','BOUNDARY')
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
TRAIN_FIELDS=FIELDS
ZEROS=('native_updates','identity_head_updates','validation_reads','novel_validation_reads','target_Q_reads','target_U_reads','original_U_s_reads','private_truth_reads','U_truth_reads')
_read,_write,_snapshot,_same=base._read,base._write,base._snapshot,reference._same


def _core():
    from . import star_density_boundary_core
    return star_density_boundary_core


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):raise ValueError('Exact source-only fixed-density boundary training configuration required; no target/truth/K')
    p=parent.validate_config(c['reference_config'])
    if p['code_commit']!=reference.PARENT_CODE:raise ValueError('Only the selected completed class-density source state may be inherited')
    source=base.validate_config(p['train_config'])
    if (source['run_id'],source['code_commit'],source['arm'])!=(parent.SIGNAL_RUN,parent.SIGNAL_CODE,'E_SIGNAL'):raise ValueError('Original r03 source checkpoint required')
    out=Path(c['output_root']).resolve()
    for path in [p['output_root'],p['reference_prediction_root'],source['output_root'],source['IQ_root'],source['feature_root'],source['control_root'],source['native_config']['source_output'],source['native_config']['source_cache']]:
        path=Path(path).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):raise ValueError('New immutable training output cannot overlap source/model inputs')
    return c


def _reference(c):
    """Training needs only frozen source ancestry and density, never V/tau."""
    p=c['reference_config'];root=Path(p['output_root']);m=_read(root/'complete.json');fit=_read(root/'models/complete.json')
    expected=dict(schema=parent.SCHEMA,status=parent.STATUS,run_id=parent.RUN_ID,code_commit=reference.PARENT_CODE,method_names=list(parent.METHODS),primary_method=parent.PRIMARY_METHOD,
        source_only=True,private_truth_reads=0,target_U_reads=0,formal_target_accessed=False,neural_updates=0,density_fit_updates=0,
        model_parameters_buffers_unchanged=True,source_states_unchanged=True)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('config')!=p:raise ValueError('Completed selected source-only class-density identity required')
    if any(fit.get(k)!=v for k,v in dict(schema=parent.SCHEMA,status=parent.MODELS_STATUS,run_id=parent.RUN_ID,code_commit=reference.PARENT_CODE,
            models_frozen=True,models_refitted=False,source_V_fit=False,novel_fit=False,target_fit=False).items()):raise ValueError('Fixed original source-L/OE density required without validation fitting')
    identity=m['model_identity'];sm=m['reference_prediction_identity'].get('signal_reference_identity',{})
    original=signal_eval._identity(identity,dict(code_commit=parent.SIGNAL_CODE,train_config=p['train_config']))
    if (sm.get('run_id')!=parent.SIGNAL_RUN or sm.get('code_commit')!=parent.SIGNAL_CODE or sm.get('config',{}).get('train_config')!=p['train_config']
            or sm.get('model_identity')!=identity or original!=m['parent_F0_identity'] or identity['dura_identity']!=m['dura_identity']):raise ValueError('Exact original F0/DURA/r03 source ancestry required')
    with np.load(root/'models/density.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
    parent.core.density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Only the original D160 ninety-mode source density allowed')
    for value in state.values():value.setflags(write=False)
    return m,state


def _backend(c,device):
    source=_reference(c);b=parent._backend(c['reference_config'],device)
    return dict(b,source_reference=source)


def _student(model):
    student=deepcopy(model).requires_grad_(True).eval()
    if signalcore.parameter_report(student)!=dict(signal_feature_parameters=232033,projection_parameters=102688,signal_parameters=334721,combined_active_parameters=567714):raise ValueError('Full differentiable r03 signal and projection required')
    return student


def _objective(model,L,OE,teachers,sample,device,boundary,arm):
    ix,oi=sample['L'],sample['OE'];lm,om=L['metadata'],OE['metadata']
    z=model(base._tensor(L['restored'][ix],device));oz=model(base._tensor(OE['restored'][oi],device))
    metadata=lambda m,k,rows:base._tensor(m[k][rows],device,torch.long)
    lrx,lday=metadata(lm,'rx_i',ix),metadata(lm,'day_i',ix);orx,oday=metadata(om,'rx_i',oi),metadata(om,'day_i',oi)
    known=base.loss.crossset_metric(z,oz,metadata(lm,'y',ix),lrx,lday,orx,oday)
    topology=base.loss.oe_structure(oz,base._tensor(teachers['OE']['u'][oi],device).detach(),orx,oday)
    oldL=base._tensor(teachers['L']['teacher_z'][ix],device).detach();oldOE=base._tensor(teachers['OE']['teacher_z'][oi],device).detach()
    anchor_L=.5*(z-oldL).square().sum(1).mean();anchor_OE=.5*(oz-oldOE).square().sum(1).mean();anchor=.5*(anchor_L+anchor_OE)
    bce=boundary.loss(z,oz,base._tensor(teachers['L']['closed'][ix],device,torch.long),base._tensor(teachers['OE']['closed'][oi],device,torch.long))
    source=known['loss']+topology['loss'];weight=1. if arm=='BOUNDARY' else 0.;total=source+anchor+weight*bce['loss']
    metrics=dict(source_loss=float(source.detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),OE_KL=float(topology['loss'].detach().cpu()),
        point_anchor=float(anchor.detach().cpu()),anchor_L=float(anchor_L.detach().cpu()),anchor_OE=float(anchor_OE.detach().cpu()),point_anchor_weight=1.,
        boundary_loss=float(bce['loss'].detach().cpu()),boundary_weight=weight,**{k:float(bce[k].detach().cpu()) for k in ('L_BCE','OE_BCE','L_mean_ratio','OE_mean_ratio')})
    return total,metrics,z,oz


def _gradient_smoke(model,boundary,device):
    before=_snapshot(model);source=base._gradient_smoke(model,device);rng=np.random.default_rng(2026103017)
    try:
        model.zero_grad(set_to_none=True)
        z=model(base._tensor(rng.normal(size=(4,2,256)).astype(np.float32),device));oz=model(base._tensor(rng.normal(size=(4,2,256)).astype(np.float32),device))
        # Fixed synthetic routes exercise the density graph, not TX supervision.
        routes=torch.arange(4,dtype=torch.long,device=device);loss=boundary.loss(z,oz,routes,routes.flip(0));loss['loss'].backward();gradients=base._gradient_norms(model)
        if any(not np.isfinite(v) for v in gradients.values()) or any(gradients[k]<=0 for k in ('sinc','middle','last')):raise ValueError('Fixed density BCE must reach the full live signal encoder and MLP')
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step density-gradient probe changed parameters/buffers')
        return dict(status='PASS',source=source,BCE_gradient_norms=gradients,BCE_loss=float(loss['loss'].detach().cpu()),synthetic_source_student_rows=8,synthetic_BCE_student_rows=8,
            synthetic_routes=True,optimizer_updates=0,parameters_buffers_unchanged=True,real_source_reads=0)
    finally:model.zero_grad(set_to_none=True)


def _cache_reference(b,roles,teachers,device,check):
    result={};arrays={}
    for role in ('L','OE'):
        a=roles[role];teacher=teachers[role];z=signal_eval._project(b['model'],a['restored'],device);closed=[]
        for start in range(0,len(a['ids']),128):
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                logits=b['reader'].classify_features(base._tensor(teacher['u'][start:start+128],device))
            if logits.dtype!=torch.float32 or logits.shape!=(min(128,len(a['ids'])-start),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Frozen F0 head-only predictions from aligned original u160 required')
            closed.extend(logits.argmax(1).cpu().tolist())
        closed=np.asarray(closed,np.int64);z.setflags(write=False);closed.setflags(write=False)
        result[role]=dict(teacher,teacher_z=z,closed=closed)
        arrays.update({role+'_ids':a['ids'].copy(),role+'_z':z,role+'_closed':closed});check()
        print('DENSITY_BOUNDARY_REFERENCE_PROGRESS',role,len(z),flush=True)
    return result,arrays


def _identity(c,arm,source):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,open_signal_updates=STEPS,projection_updates=STEPS,
        native_updates=0,original_parent_updates=11100,parent_signal_updates=2220,signal_feature_cumulative_updates=source['signal_feature_cumulative_updates']+STEPS,
        projection_cumulative_updates=source['projection_updates']+STEPS,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,original_F0_parameters=232993,
        parent_signal_identity=source,parent_F0_identity=source['parent_F0_identity'],dura_identity=source['dura_identity'],source_contract=c['reference_config']['train_config']['native_config']['source_contract'],
        source_only=True,target_U_used=False,target_U_reads=0,input_path='cached_P1_IQ_only',identity_head_replaced=False,original_F0_frozen=True,
        fixed_density=True,point_anchor_weight=1.,boundary_weight=1. if arm=='BOUNDARY' else 0.,BN_Dropout_eval=True)


def _loop(model,roles,teachers,c,arm,device,opt,out,boundary):
    p=c['reference_config']['train_config'];sampler=base.Sampler(teachers['L'],teachers['OE'],p['L_seed'],p['OE_seed']);trace=[];usage=dict(L=set(),OE=set())
    control=(Path(c['output_root'])/'REPLAY/steps.jsonl').open(encoding='utf-8') if arm=='BOUNDARY' else None
    try:
        with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if control is not None:
                    line=control.readline()
                    if not line:raise ValueError('Matched REPLAY trace incomplete')
                    row=json.loads(line)
                    if row.get('step')!=step or any(row.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched source replay indices differ')
                opt.zero_grad(set_to_none=True);value,metrics,z,oz=_objective(model,roles['L'],roles['OE'],teachers,sample,device,boundary,arm)
                if not bool(torch.isfinite(value)):raise FloatingPointError('Nonfinite anchored boundary objective')
                value.backward();gradients=base._gradient_norms(model) if step<=2 else None
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite live source signal gradients')
                norm=float(signalcore.clip(opt,model))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite physical gradient norm')
                opt.step()
                for key in usage:usage[key].update(map(int,sample[key]))
                row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),source_indices_equal=True,loss_total=float(value.detach().cpu()),**metrics,
                    gradient_norms=gradients,gradient_norm_before_physical_clip=norm,sinc_lr=float(opt.param_groups[1]['lr']),sinc_optimizer_step=opt._sinc_unit_steps,physical_clip_calls=opt._sinc_physical_clip_calls)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('DENSITY_BOUNDARY_TRAIN_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if control is not None and control.readline():raise ValueError('Matched REPLAY trace has extra updates')
        return trace,{key:len(value) for key,value in usage.items()}
    finally:
        if control is not None:control.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable density-boundary training output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device);pm,state=b['source_reference'];source=pm['model_identity']
    if b['model_identity']!=source or b['reader_identity']!=source['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual original F0/r03/DURA must match the completed source model')
    snapshots=[(model,reference._snapshot_model(model)) for model in (b['reader'],b['model'],b['dura'][0])];state_before=deepcopy(state)
    boundary=_core().FrozenDensity(state,device);density_before=_snapshot(boundary)
    if any(p.requires_grad for p in boundary.parameters()):raise ValueError('GMM must have no trainable parameters')
    def check():
        for model,before in snapshots:reference._check_model(model,before)
        if not _same(state,state_before) or any(not torch.equal(v,boundary.state_dict()[k].detach().cpu()) for k,v in density_before.items()):raise ValueError('Fixed source GMM state or buffers changed')
    smoke=parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source values')
    student=_student(b['model']);initial=_snapshot(b['model']);gradient=_gradient_smoke(student,boundary,device);check()
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));reference._npz(out/'density.npz',state)
    p=c['reference_config']['train_config'];m,_,_=base._metadata(p);roles={};rows={}
    try:
        roles,old_teachers=base._roles(p,m);teachers,arrays=_cache_reference(b,roles,old_teachers,device,check);reference._npz(out/'reference_signal.npz',arrays)
        teachers_before={role:{key:teachers[role][key].copy() for key in ('u','teacher_z','closed')} for role in ('L','OE')}
        def teacher_check():
            if any(not np.array_equal(teachers[role][key],value) for role,record in teachers_before.items() for key,value in record.items()):raise ValueError('Frozen source u/point teacher/predicted route changed')
        for arm in ARMS:
            if arm=='BOUNDARY':student=_student(b['model'])
            if any(not torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in initial.items()):raise ValueError('Both arms must independently initialize from the same r03 state')
            folder=out/arm;folder.mkdir();buffers={k:v.detach().clone() for k,v in student.named_buffers()};opt=signalcore.optimizer(student,c['lr'])
            trace,unique=_loop(student,roles,teachers,c,arm,device,opt,folder,boundary)
            if any(not torch.equal(v,dict(student.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Frozen student BN/Sinc buffers changed')
            check();teacher_check();identity=_identity(c,arm,source);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite learned signal state required')
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=source,source_feature_manifest=m,
                source_only=True,model_buffers_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True),folder/'final.pt')
            row=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,identity=identity,updates=STEPS,parent_signal_identity=source,
                parent_F0_identity=source['parent_F0_identity'],dura_identity=source['dura_identity'],source_feature_manifest=m,source_only=True,target_U_used=False,
                source_training_counts=dict(COUNTS),source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,real_student_signal_forward_rows=256*STEPS,unique_training_rows=unique,
                first_step=trace[0],last_step=trace[-1],initial_state_equal_to_parent=True,model_buffers_unchanged=True,original_models_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True,
                optimizer=dict(name='AdamW',mode='CENTER_UNITS',lr=.0001,weight_decay=.0001,eps=1e-8,sinc_multiplier=1000.,sinc_eps=1e-11,physical_clip=5.,
                    physical_clip_calls=opt._sinc_physical_clip_calls,sinc_steps=opt._sinc_unit_steps),final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('DENSITY_BOUNDARY_ARM_COMPLETE',arm,flush=True);del opt,student
        check();teacher_check();count=sum(COUNTS.values())
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={arm:row['identity'] for arm,row in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,parent_signal_identity=source,parent_F0_identity=source['parent_F0_identity'],dura_identity=source['dura_identity'],source_feature_manifest=m,
            reference_prediction_identity=pm,source_only=True,target_U_used=False,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True,
            real_student_signal_forward_rows=512*STEPS,real_parent_signal_forward_rows=count,F_head_only_rows=count,real_F_forward_rows=0,G_forward_rows=0,
            synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_gradient_student_rows=16,
            source_density_fit_updates=0,threshold_fit_updates=0,per_record_teacher_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        _write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():base.close_role(value)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only REPLAY or BOUNDARY fixed final checkpoints allowed')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');row=_read(root/arm/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Both source arms must complete before frozen loading') from exc
    zero={k:0 for k in ZEROS}
    flags=dict(source_only=True,target_U_used=False,original_models_unchanged=True,fixed_density_state_unchanged=True,point_teacher_unchanged=True,final_phase3_complete=False)
    if (any(done.get(k)!=v for k,v in dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
                source_sampling_matched=True,all_student_buffers_unchanged=True,**flags,**zero).items())
            or any(row.get(k)!=v for k,v in dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,updates=STEPS,model_buffers_unchanged=True,initial_state_equal_to_parent=True,**flags,**zero).items())
            or done.get('rows',{}).get(arm)!=row):raise ValueError('Complete matched source-only training and frozen teacher/model state required')
    pm,state=_reference(c);source=pm['model_identity']
    with np.load(root/'density.npz',allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
    if not _same(saved,state) or done.get('reference_prediction_identity')!=pm:raise ValueError('Original source density and model identity must remain exact')
    p=c['reference_config']['train_config'];model,parent_identity=base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    if parent_identity!=source or done.get('parent_signal_identity')!=source:raise ValueError('Original r03 checkpoint ancestry differs')
    expected=_identity(c,arm,source);m,_,_=base._metadata(p);payload=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if (row.get('identity')!=expected or done.get('identities',{}).get(arm)!=expected or row.get('source_feature_manifest')!=m or done.get('source_feature_manifest')!=m
            or not isinstance(payload,dict) or payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('identity')!=expected
            or payload.get('parent_signal_identity')!=source or payload.get('source_feature_manifest')!=m or any(payload.get(k) is not True for k in ('source_only','model_buffers_unchanged','fixed_density_state_unchanged','point_teacher_unchanged'))):
        raise ValueError('Learned source checkpoint provenance differs')
    network=payload.get('network');expected_state=model.state_dict()
    if (not isinstance(network,dict) or set(network)!=set(expected_state) or any(not torch.is_tensor(v) or v.dtype!=expected_state[k].dtype or v.shape!=expected_state[k].shape or not bool(torch.isfinite(v).all()) for k,v in network.items())):
        raise ValueError('Strict complete finite signal/projection checkpoint required')
    model.load_state_dict(network,strict=True);return model.eval().requires_grad_(False),expected


def main(argv=None):
    print(json.dumps(dict(event='DENSITY_BOUNDARY_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);train(_read(a.config),a.device)


if __name__=='__main__':main()
