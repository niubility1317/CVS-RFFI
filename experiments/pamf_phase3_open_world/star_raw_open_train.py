"""P0-only source signal branch with a separately frozen original-P0 teacher."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from . import runner,source_open_embedding_train as native
from . import star_signal_open_train as previous
from . import star_signal_open_model as core
from . import star_representation_cache as IQ

RUN_ID='20261031-phase3-star-raw-open-s2026092701-r01'
ARMS=('E_RAW',)
SCHEMA='pamf_phase3_star_raw_open_train_v1'
STATUS='STAR_RAW_OPEN_SOURCE_FROZEN'
CONTROL_RUN='20261030-phase3-star-macro12-risk-s2026092701-r01'
CONTROL_CODE='26fbecff1df7e5476bcc5f459d8367c6fa49fd36'
SIGNAL_RUN='20261030-phase3-star-signal-open-s2026092701-r03'
SIGNAL_CODE='dc01a402582f93e452dbaa3ef59a9fbb44151818'
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
ROLE_COUNTS=dict(L=6300,OE=6134,V=27000,novel=3150)
FIELDS={'run_id','code_commit','arm','native_config','IQ_root','control_root','control_train_config','output_root','steps','lr','L_seed','OE_seed'}
TRAIN_FIELDS=FIELDS
ZEROS=('native_updates','original_native_updates','identity_head_updates','G_forward_rows','target_reads','validation_reads','novel_validation_reads',
    'original_U_s_reads','private_truth_reads','OE_truth_reads','restored_cache_values_read','P1_teacher_values_read','clean_values_read')
_read,_write=runner._read,runner._write
_tensor,_snapshot,_gradient_norms=previous._tensor,previous._snapshot,previous._gradient_norms
_checkpoint_smoke=previous._checkpoint_smoke
Sampler=previous.Sampler
close_role=IQ.close_role


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm')!='E_RAW'
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None):raise ValueError('Exact P0-only source configuration required; no truth/U/Q/K')
    n=c['native_config']
    if (not isinstance(n,dict) or set(n)!=runner.COMMON|{'source_cache'} or n.get('run_id')!=RUN_ID or n.get('code_commit')!=c['code_commit']
            or n.get('row_id')!='SOURCE_E_RAW-s2026092701' or n.get('output_root')!=c['output_root']):raise ValueError('Matching original-F0 SOURCE_E_RAW constructor required')
    native.validate_config(previous._native_config(n));p=previous.validate_config(c['control_train_config'])
    if p['run_id']!=SIGNAL_RUN or p['code_commit']!=SIGNAL_CODE or p['arm']!='E_SIGNAL' or c['IQ_root']!=p['IQ_root']:
        raise ValueError('Fixed original r03 P1 control and same physical P0/P1 cache required')
    if any(n.get(k)!=p['native_config'].get(k) for k in ('source_output','source_cache','source_contract','base_release','frozen_sources','model_seed','registered_classes')):
        raise ValueError('Fresh source initialization must have the same original F0 ancestry')
    for k,v in dict(steps=STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018).items():
        if type(c[k]) is not type(v) or c[k]!=v:raise ValueError('Fixed matched steps, optimizer and independent source streams required')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('IQ_root','control_root','output_root')):raise ValueError('Nonempty source roots required')
    previous.parent._separate(c['output_root'],[c['IQ_root'],c['control_root'],p['output_root'],p['feature_root'],n['source_output'],n['source_cache']])
    return c


def _metadata(c):
    m=_read(Path(c['control_root'])/'complete.json');p=c['control_train_config'];done=_read(Path(p['output_root'])/'complete.json')
    expected=dict(schema='pamf_phase3_star_macro12_prediction_v1',status='STAR_MACRO12_PREDICTIONS_COMPLETE',run_id=CONTROL_RUN,code_commit=CONTROL_CODE,
        source_only=True,private_truth_reads=0,target_reads=0,models_frozen_before_V=True,all_thresholds_frozen_before_novel=True,
        controls_reused_exactly=True,model_parameters_buffers_unchanged=True)
    if (any(m.get(k)!=v for k,v in expected.items()) or m.get('config',{}).get('train_config')!=p
            or any(done.get(k)!=v for k,v in dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=SIGNAL_RUN,code_commit=SIGNAL_CODE,
                arm='E_SIGNAL',config=p,updates=2220,open_signal_updates=2220,native_updates=0,source_only=True,model_buffers_unchanged=True,
                target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,private_truth_reads=0).items())
            or m.get('parent_model_identity')!=done.get('identity')):raise ValueError('Exact completed macro12 control and r03 training metadata required')
    identity=m['parent_model_identity']
    if (identity.get('parent_F0_identity')!=m.get('parent_F0_identity') or identity.get('dura_identity')!=m.get('dura_identity')
            or identity.get('input_path')!='cached_P1_IQ_only' or identity.get('open_signal_updates')!=2220):raise ValueError('Frozen P1 control source identity differs')
    for role in COUNTS:
        manifest=m.get('source_IQ_manifests',{}).get(role,{})
        if (manifest.get('role')!=role or manifest.get('count')!=COUNTS[role] or manifest.get('parent_F0_identity')!=m['parent_F0_identity']
                or manifest.get('dura_identity')!=m['dura_identity']):raise ValueError('Original P0/P1 physical role ancestry required')
    return m,Path(p['output_root'])/'steps.jsonl'


def load_received(role_directory,role):
    """Map P0 only. Never open restored, old P1 teacher or baseline NPY values."""
    if role not in ROLE_COUNTS:raise ValueError('Only the original four source roles allowed')
    root=Path(role_directory);m=_read(root/'manifest.json');n=ROLE_COUNTS[role]
    required=dict(schema=IQ.ROLE_SCHEMA,status='STAR_REPRESENTATION_ROLE_COMPLETE',run_id=IQ.RUN_ID,role=role,count=n,iq_shape=[n,2,256],
        precision='FP32',paths=['P0','P1'],received_input_path='P0',restored_input_path='P1',source_only=True,all_new_IQ_satellite=True,
        metadata_fields=sorted(IQ.METADATA_FIELDS[role]))
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Existing VALIDATED_ONCE source P0 role handle required')
    ids=np.load(root/'ids.npy',allow_pickle=False)
    with np.load(root/'metadata.npz',allow_pickle=False) as z:metadata={k:z[k].copy() for k in z.files}
    IQ._metadata(role,ids,metadata)
    received=np.load(root/'received.npy',mmap_mode='r',allow_pickle=False)
    if not isinstance(received,np.memmap) or received.dtype!=np.float32 or received.shape!=(n,2,256):
        if isinstance(received,np.memmap):received._mmap.close()
        raise ValueError('Original FP32 P0 IQ shape required')
    return dict(manifest=m,ids=ids,metadata=metadata,received=received)


def load_role(root,role):
    if role not in COUNTS:raise ValueError('Training may only open L/OE P0 values')
    return load_received(Path(root)/role,role)


def _roles(c,m):
    roles={}
    try:
        for role in ('L','OE'):
            a=load_role(c['IQ_root'],role);roles[role]=a;previous.parent._metadata(a['ids'],a['metadata'],role)
            # Only IDs/public metadata from the already-derived parent cache;
            # frozen P1 u/c feature values are never indexed or used as teacher.
            with np.load(Path(c['control_train_config']['feature_root'])/(role+'.npz'),allow_pickle=False) as z:
                same=np.array_equal(a['ids'],z['ids']) and all(k in z.files and np.array_equal(v,z[k]) for k,v in a['metadata'].items())
            if a['manifest']!=m['source_IQ_manifests'][role] or not same:raise ValueError('Exact source physical ID order/metadata required for matched r03 indices')
        if set(roles['L']['ids'].tolist())&set(roles['OE']['ids'].tolist()):raise ValueError('Source L and OE physical records must be disjoint')
        return roles
    except BaseException:
        for a in roles.values():close_role(a)
        raise


def _construct(c,device,*,smoke=False):
    original=native.make_trainable_model(previous._native_config(c['native_config']),device).eval()
    check=_checkpoint_smoke(original,device) if smoke else None
    # Copy the live differentiable source BEFORE freezing its teacher instance.
    student=core.SignalOpen(core.HeadlessSignal(original.id_backbone.encoder)).to(device)
    student.original_parent_identity=original._phase3_parent_identity;student.source_data_handle=original._phase3_source_handle;student.checkpoint_smoke=check
    original.eval().requires_grad_(False)
    if core.parameter_report(student)!=dict(signal_feature_parameters=232033,projection_parameters=102688,signal_parameters=334721,combined_active_parameters=567714):
        raise ValueError('Complete original signal and zero-last projection required')
    return student,original


def _teacher_provenance(c,m):
    return dict(schema='pamf_phase3_star_raw_open_teacher_v1',run_id=RUN_ID,code_commit=c['code_commit'],role='OE',count=COUNTS['OE'],input_path='P0',
        model_identity=m['parent_F0_identity'],source_manifest=m['source_IQ_manifests']['OE'],teacher_parameters_frozen=True,teacher_state_unchanged=True,
        new_teacher_updates=0,teacher_parameters=232993,original_parent_updates=11100,unit_policy='raw fused FP64 canonical unit to FP32',P1_teacher_reused=False)


def _teacher_u(teacher,OE,device):
    if any(p.requires_grad for p in teacher.parameters()) or any(m.training for m in teacher.modules()):raise ValueError('Independent original F0 teacher must be frozen eval')
    before=_snapshot(teacher);parts=[]
    for start in range(0,len(OE['ids']),128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            raw=teacher.id_backbone.encoder.features(_tensor(OE['received'][start:start+128],device));u=core.unit(raw)
        parts.append(np.asarray(u.detach().cpu().tolist(),np.float32))
    if any(not torch.equal(v,teacher.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('P0 teacher parameters/buffers changed')
    result=np.concatenate(parts)
    if result.shape!=(COUNTS['OE'],160) or not np.isfinite(result).all():raise ValueError('Finite fixed original-P0 OE teacher required')
    result.setflags(write=False);return result


def _objective(model,L,OE,teacher_u,sample,device,probe=False):
    ix,oi=sample['L'],sample['OE'];lm,om=L['metadata'],OE['metadata']
    z=model(_tensor(L['received'][ix],device));oz=model(_tensor(OE['received'][oi],device))
    get=lambda meta,k,index:_tensor(meta[k][index],device,torch.long)
    lr,ld=get(lm,'rx_i',ix),get(lm,'day_i',ix);orx,od=get(om,'rx_i',oi),get(om,'day_i',oi)
    known=previous.loss.crossset_metric(z,oz,get(lm,'y',ix),lr,ld,orx,od)
    other=previous.loss.oe_structure(oz,_tensor(teacher_u[oi],device).detach(),orx,od);total=known['loss']+other['loss'];gradients=None
    if probe:gradients={k:dict(norm=float(g.detach().norm().cpu()),minimum_row_norm=float(g.detach().norm(dim=1).min().cpu()))
        for k,g in zip(('L','OE'),torch.autograd.grad(known['loss'],(z,oz),retain_graph=True))}
    return total,dict(loss=float(total.detach().cpu()),L_metric=float(known['loss'].detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),
        OE_KL=float(other['loss'].detach().cpu()),L_weight=1.,OE_weight=1.,L_anchors=len(ix),OE_candidates=len(oi),OE_anchors=len(oi),
        candidate_set_priors=[.5,.5],crossset_embedding_gradients=gradients,OE_teacher_fixed=True,OE_OE_hard_negatives=False,input_path='P0',teacher_input_path='P0')


def _gradient_smoke(model,device):
    before=_snapshot(model);rng=np.random.default_rng(2026103017);rx=np.array([1,3,1,3],np.int64);day=np.array([1,2,1,2],np.int64)
    def role():return dict(received=rng.normal(size=(4,2,256)).astype(np.float32),metadata=dict(rx_i=rx,day_i=day))
    L,OE=role(),role();L['metadata']['y']=np.array([0,0,1,1],np.int64)
    try:
        model.zero_grad(set_to_none=True);value,metrics=_objective(model,L,OE,rng.normal(size=(4,160)).astype(np.float32),dict(L=np.arange(4),OE=np.arange(4)),device,True)
        value.backward();gradients=_gradient_norms(model)
        if any(not np.isfinite(v) for v in gradients.values()) or any(gradients[k]<=0 for k in ('sinc','middle','last')):raise ValueError('Actual P0 signal gradient smoke failed')
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step smoke changed source signal state')
        return dict(status='PASS',synthetic_L_signal_rows=4,synthetic_OE_signal_rows=4,optimizer_updates=0,parameters_buffers_unchanged=True,
            gradient_norms=gradients,metrics=metrics,synthetic_IQ='P0-shaped synthetic; no G/real IQ')
    finally:model.zero_grad(set_to_none=True)


def _identity(c,m,model):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='E_RAW',checkpoint=str(Path(c['output_root'])/'final.pt'),updates=STEPS,
        open_signal_updates=STEPS,projection_updates=STEPS,native_updates=0,original_native_updates=0,original_parent_updates=11100,
        signal_feature_cumulative_updates=11100+STEPS,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
        original_F0_parameters=232993,combined_active_parameters=567714,initialization_seed=2026092701,parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],
        original_parent_identity=model.original_parent_identity,identity_head_replaced=False,original_F0_frozen=True,input_path='cached_P0_IQ_only',
        teacher_input_path='P0',OE_embedding_training=True,OE_topology_regularization=True,crossset_separation=True)


def _loop(model,roles,teacher,c,device,opt,control,on_step):
    sampler=Sampler(roles['L'],roles['OE'],c['L_seed'],c['OE_seed']);usage=dict(L=set(),OE=set());trace=[]
    with Path(control).open(encoding='utf-8') as saved:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);line=saved.readline()
            if not line:raise ValueError('Matched r03 source trace incomplete')
            old=json.loads(line)
            if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in usage):raise ValueError('Fixed r03 source indices differ')
            opt.zero_grad(set_to_none=True);value,metrics=_objective(model,roles['L'],roles['OE'],teacher,sample,device,step<=2);value.backward()
            gradients=_gradient_norms(model) if step<=2 else None
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite P0 signal gradients')
            norm=float(core.clip(opt,model))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite physical global gradient norm')
            opt.step()
            for k in usage:usage[k].update(map(int,sample[k]))
            row=dict(step=step,open_signal_updates=step,projection_updates=step,native_updates=0,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),
                control_indices_equal=True,gradient_norms=gradients,gradient_norm_before_physical_clip=norm,sinc_lr=float(opt.param_groups[1]['lr']),
                sinc_optimizer_step=opt._sinc_unit_steps,physical_clip_calls=opt._sinc_physical_clip_calls,**metrics)
            trace.append(row);on_step(row)
            if step==1 or step%100==0 or step==STEPS:print('STAR_RAW_OPEN_TRAIN_PROGRESS',step,STEPS,metrics['loss'],flush=True)
        if saved.readline():raise ValueError('Matched r03 source trace contains extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable P0 source training output required')
    m,control=_metadata(c);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,original=_construct(c,device,smoke=True);probe=_gradient_smoke(model,device)
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=model.checkpoint_smoke,gradient=probe));roles={}
    try:
        roles=_roles(c,m);teacher=_teacher_u(original,roles['OE'],device);provenance=_teacher_provenance(c,m)
        with (out/'teacher_P0.npz').open('xb') as f:np.savez(f,ids=roles['OE']['ids'],u=teacher,**roles['OE']['metadata'])
        _write(out/'teacher_manifest.json',provenance);del original
        buffers={k:v.detach().clone() for k,v in model.named_buffers()};opt=core.optimizer(model,c['lr'])
        with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
            def record(row):log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            trace,unique=_loop(model,roles,teacher,c,device,opt,control,record)
        if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Signal buffers changed')
        identity=_identity(c,m,model);zero={k:0 for k in ZEROS}
        torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=model.state_dict(),control_identity=m,teacher_provenance=provenance,
            original_parent_identity=model.original_parent_identity,source_data_handle=model.source_data_handle,source_only=True,**zero),out/'final.pt')
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm='E_RAW',config=c,identity=identity,updates=STEPS,open_signal_updates=STEPS,
            projection_updates=STEPS,control_identity=m,source_manifests={r:a['manifest'] for r,a in roles.items()},parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],
            teacher_provenance=provenance,teacher_state_unchanged=True,model_buffers_unchanged=True,source_only=True,identity_head_replaced=False,signal_parameters=334721,
            L_training_rows_loaded=COUNTS['L'],OE_training_rows_loaded=COUNTS['OE'],L_sampled_exposures=192*STEPS,OE_sampled_exposures=64*STEPS,unique_training_rows=unique,
            real_F_forward_rows=COUNTS['OE'],synthetic_F_forward_rows=2,real_signal_forward_rows=256*STEPS,synthetic_signal_forward_rows=8,
            optimizer=dict(name='AdamW',mode='CENTER_UNITS',lr=.0001,weight_decay=.0001,eps=1e-8,sinc_multiplier=1000.,sinc_eps=1e-11,physical_clip=5.,
                physical_clip_calls=opt._sinc_physical_clip_calls,sinc_steps=opt._sinc_unit_steps),first_step=trace[0],second_step=trace[1],final_step=trace[-1],
            final_phase3_complete=False,**zero)
        _write(out/'training_summary.json',done);_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for a in roles.values():close_role(a)


def load_trained_model(c,checkpoint,device='cuda:0'):
    validate_config(c)
    if Path(checkpoint).resolve()!=Path(c['output_root'],'final.pt').resolve():raise ValueError('Only fixed final P0 signal checkpoint allowed')
    m,_=_metadata(c);done=_read(Path(c['output_root'])/'complete.json');provenance=_teacher_provenance(c,m)
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm='E_RAW',config=c,updates=STEPS,open_signal_updates=STEPS,
        control_identity=m,teacher_provenance=provenance,teacher_state_unchanged=True,model_buffers_unchanged=True,source_only=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete source-only P0 signal checkpoint metadata required')
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=done.get('identity') or value.get('control_identity')!=m
            or value.get('teacher_provenance')!=provenance or value.get('source_only') is not True or any(value.get(k)!=0 for k in ZEROS)):
        raise ValueError('Original-P0 teacher and source-only checkpoint provenance differs')
    model,original=_construct(c,device);del original
    if (value.get('identity')!=_identity(c,m,model) or value.get('original_parent_identity')!=model.original_parent_identity
            or value.get('source_data_handle')!=model.source_data_handle):raise ValueError('Original source F0 ancestor differs')
    state=value.get('network',{});expected=model.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(expected) or any(not torch.is_tensor(v) or v.dtype!=expected[k].dtype
            or v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items())):raise ValueError('Strict complete finite P0 signal state required')
    model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False),value['identity']


def load_trained(c,device='cuda:0'):return load_trained_model(c,Path(c['output_root'])/'final.pt',device)


def main(argv=None):
    print(json.dumps(dict(event='STAR_RAW_OPEN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv);train(_read(a.config),a.device)


if __name__=='__main__':main()
