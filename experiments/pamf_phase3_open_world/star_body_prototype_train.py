"""Matched live-P1 body/projection learning with frozen Sinc and buffers."""
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
from torch.nn import functional as F
from . import star_class_density_run as parent
from . import star_signal_open_train as base
from . import star_signal_open_model as signalcore
from . import star_signal_ssl_train as ssl
from . import star_representation_loss as views
from . import star_target_prototype_core as core
from .star_local_cov_run import _reference

RUN_ID='20261031-phase3-body-prototype-s2026092701-r01'
SCHEMA='pamf_phase3_body_prototype_train_v1'
STATUS='BODY_PROTOTYPE_TRAINING_COMPLETE'
ROW_STATUS='BODY_PROTOTYPE_STUDENT_FROZEN'
ARMS=('SOURCE','TARGET_PCD')
STEPS=444
COUNTS=dict(L=6300,OE=6134)
U_COUNT=384
U_BATCH=64
U_SEED=2026103112
AUG_SEED=2026103114
PARENT_CODE='4af0668331682636ee212077b38cd912857a1b31'
U_CAPSULE_ID,U_SPLIT_ID=ssl.U_CAPSULE_ID,ssl.U_SPLIT_ID
FIELDS={'run_id','code_commit','reference_config','U_public_root','U_capsule_id','U_split_id','output_root','steps','lr'}
TRAIN_FIELDS=FIELDS
ZEROS=('native_updates','identity_head_updates','Sinc_optimizer_steps','physical_clip_calls','target_Q_reads','validation_reads','novel_validation_reads','original_U_s_reads','private_truth_reads','U_truth_reads')
_read,_write,_snapshot=base._read,base._write,base._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in FIELDS-{'reference_config','steps','lr'})):
        raise ValueError('Exact matched live-IQ body/prototype configuration required; no truth/Q/K')
    p=parent.validate_config(c['reference_config']);u=Path(c['U_public_root'])
    if p['code_commit']!=PARENT_CODE or u.name!='public' or u.parent.name!='discovery' or c['U_capsule_id']!=U_CAPSULE_ID or c['U_split_id']!=U_SPLIT_ID:
        raise ValueError('Only original source-only r03/class-density and fixed star-G10 discovery U may be used')
    out=Path(c['output_root']).resolve();t=p['train_config']
    for path in (c['U_public_root'],p['output_root'],p['reference_prediction_root'],p['dura_checkpoint'],p['dura_release'],t['output_root'],t['IQ_root'],t['feature_root'],t['control_root']):
        path=Path(path).resolve()
        if out.is_relative_to(path) or path.is_relative_to(out):raise ValueError('Immutable output cannot overlap source/model/public input')
    return c


def _backend(c,device):return parent._backend(c['reference_config'],device)


def _student(model):
    child=deepcopy(model).eval().requires_grad_(True)
    frozen=[]
    for name,p in child.named_parameters():
        if name.startswith(signalcore.SINC_PREFIX):p.requires_grad_(False);frozen.append((name,p))
    expected={signalcore.SINC_PREFIX+suffix for suffix in ('low_coordinate','band_coordinate')}
    if (set(n for n,p in frozen)!=expected or any(p.shape!=(24,1) for n,p in frozen)
            or signalcore.parameter_report(child)['signal_parameters']!=334721
            or sum(p.numel() for p in child.parameters() if p.requires_grad)!=334673):
        raise ValueError('Only two 24x1 Sinc leaves freeze; body231985 plus projection102688 must train')
    return child


def _npz(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez(stream,**data)


def _parent_z(model,values,device):
    blocks=[]
    for start in range(0,len(values),128):
        with torch.no_grad():r=model(base._tensor(values[start:start+128],device))
        blocks.append(np.asarray(r.detach().cpu().tolist(),np.float32))
    result=np.concatenate(blocks)
    if result.shape!=(len(values),160) or not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result.astype(np.float64),axis=1),1.,rtol=0,atol=2e-6):
        raise ValueError('Finite unit FP32 parent z160 anchor required')
    return result


def _source_cache(c,b,out,check):
    p=c['reference_config']['train_config'];m,_,_=base._metadata(p);roles={};banks={}
    try:
        roles,teachers=base._roles(p,m)
        for role in ('L','OE'):
            a=roles[role];z=_parent_z(b['model'],a['restored'],next(b['model'].parameters()).device);check()
            banks[role]=dict(iq=a['restored'],parent_z=z,ids=a['ids'].copy(),metadata={k:v.copy() for k,v in a['metadata'].items()})
            if role=='OE':banks[role]['teacher_u']=teachers['OE']['u'].copy()
            _npz(out/'source_features'/(role+'.npz'),dict(ids=banks[role]['ids'],parent_z=z,**banks[role]['metadata'],**({'teacher_u':banks[role]['teacher_u']} if role=='OE' else {})))
        return banks,m,roles
    except BaseException:
        for value in roles.values():base.close_role(value)
        raise


def _restore_U(c,b,device,check):return ssl._restore_U(c,b,device,check)


def _target_cache(c,b,out,device,check,source_ids):
    manifest,ids,restored=_restore_U(c,b,device,check)
    if source_ids&set(ids.tolist()):raise ValueError('Target U must be physically disjoint from source roles')
    generator=torch.Generator(device='cpu').manual_seed(AUG_SEED);raw=base._tensor(restored,'cpu')
    arrays=[restored]+[np.asarray(views.phase_gain_view(raw,generator).tolist(),np.float32) for _ in range(2)]
    weak=_parent_z(b['model'],restored,device);check()
    cache=dict(ids=ids,views=np.stack(arrays),parent_z=weak,neighbors=np.asarray(core.nearest_neighbors(base._tensor(weak,'cpu'),ids).tolist(),np.int64))
    _npz(out/'U_features.npz',cache);_write(out/'U_manifest.json',manifest)
    return cache,manifest


def _runtime(bank,device):
    return {k:base._tensor(bank[k],device) for k in ('iq','parent_z','teacher_u','views') if k in bank}|{
        'metadata':{k:base._tensor(v,device,torch.long) for k,v in bank.get('metadata',{}).items()}}


def _source_objective(model,L,OE,ix,oi,device):
    i=base._tensor(ix,device,torch.long);o=base._tensor(oi,device,torch.long);lm,om=L['metadata'],OE['metadata']
    z=model(L['iq'][i]);oz=model(OE['iq'][o])
    known=base.loss.crossset_metric(z,oz,lm['y'][i],lm['rx_i'][i],lm['day_i'][i],om['rx_i'][o],om['day_i'][o])
    other=base.loss.oe_structure(oz,OE['teacher_u'][o].detach(),om['rx_i'][o],om['day_i'][o])
    anchor=.5*(z-L['parent_z'][i].detach()).square().sum(1).mean()+.5*(oz-OE['parent_z'][o].detach()).square().sum(1).mean()
    total=known['loss']+other['loss']+10*anchor
    return total,dict(source_loss=float(total.detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),OE_KL=float(other['loss'].detach().cpu()),source_anchor=float(anchor.detach().cpu()))


def _gradients(model,prototypes=None):
    def norm(parameters):return sum(float(p.grad.detach().double().square().sum().cpu()) for p in parameters if p.grad is not None)**.5
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(signalcore.SINC_PREFIX)):raise ValueError('Frozen Sinc received a gradient')
    result=base._gradient_norms(model)
    return dict(result,body=norm(p for n,p in model.named_parameters() if n.startswith('signal.') and not n.startswith(signalcore.SINC_PREFIX)),
        prototype=0. if prototypes is None or prototypes.grad is None else float(prototypes.grad.detach().norm().cpu()))


def _gradient_smoke(model,device):
    before=_snapshot(model)
    try:
        with torch.random.fork_rng(devices=base.native._cuda_devices(model)):
            torch.manual_seed(2026103115);iq=torch.randn(32,2,256,device=device)
            with torch.no_grad():weak=model(iq)
            centers,_=core.initialize_prototypes(weak);prototypes=torch.nn.Parameter(centers);neighbors=core.nearest_neighbors(weak)
            generator=torch.Generator(device='cpu').manual_seed(AUG_SEED);cpu_iq=iq[:8].detach().cpu()
            strong=[views.phase_gain_view(cpu_iq,generator).to(device) for _ in range(2)]
            first,second=(model(x) for x in strong);source_probe=(first*torch.linspace(.1,1.,160,device=device)).sum(1).mean()
            r=core.consistency(weak,first,second,prototypes,torch.arange(8,device=device),neighbors)
            (source_probe+.1*r['loss']).backward();g=_gradients(model,prototypes)
            if any(not np.isfinite(x) for x in g.values()) or any(g[k]<=0 for k in ('body','middle','hidden','last','prototype')):raise ValueError('Synthetic live body/projection/prototype gradient paths must be live')
            if any(not torch.equal(value,model.state_dict()[key].detach().cpu()) for key,value in before.items()):raise ValueError('No-step live-IQ smoke changed parameters/buffers')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,Sinc_gradients_none=True,parameters_buffers_unchanged=True,synthetic_student_signal_rows=48,
                objective='P1-shaped synthetic live signal linear probe plus prototype consistency; no source/U reads')
    finally:model.zero_grad(set_to_none=True)


def _identity(c,arm,parent_identity,m,U_manifest):
    target=arm=='TARGET_PCD'
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,
        projection_updates=STEPS,new_projection_updates=STEPS,projection_cumulative_updates=parent_identity['projection_updates']+STEPS,new_body_updates=STEPS,open_signal_updates=STEPS,signal_encoder_updates=STEPS,
        native_updates=0,identity_head_updates=0,Sinc_optimizer_steps=0,original_parent_updates=11100,parent_signal_updates=2220,
        signal_feature_cumulative_updates=parent_identity['signal_feature_cumulative_updates']+STEPS,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
        trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,auxiliary_prototype_parameters=2560 if target else 0,prototype_deployed=False,encoder_frozen=False,Sinc_frozen=True,
        parent_signal_identity=parent_identity,parent_F0_identity=parent_identity['parent_F0_identity'],dura_identity=parent_identity['dura_identity'],source_feature_manifest=m,
        source_handles=m['source_manifests'],source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,
        input_path='cached_P1_IQ_only',identity_head_replaced=False,original_F0_frozen=True,BN_Dropout_eval=True)


def _loop(model,banks,c,arm,device,opt,out,U=None,prototypes=None):
    p=c['reference_config']['train_config'];sampler=base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed']);data={k:_runtime(v,device) for k,v in banks.items()}
    target=arm=='TARGET_PCD';target_data=_runtime(U,device) if target else None;neighbors=base._tensor(U['neighbors'],device,torch.long) if target else None
    rng=np.random.default_rng(U_SEED) if target else None;control=(Path(c['output_root'])/'SOURCE/steps.jsonl').open(encoding='utf-8') if target else None
    usage={k:set() for k in ('L','OE','U')};trace=[]
    try:
        with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if target:
                    old=json.loads(control.readline())
                    if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched source sampling streams differ')
                opt.zero_grad(set_to_none=True);total,metrics=_source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device)
                ui=np.empty(0,np.int64);pcd=dict(U_global=0.,U_local=0.,prototype_mass_min=None,prototype_mass_max=None)
                if target:
                    ui=rng.choice(U_COUNT,U_BATCH,replace=False);indices=base._tensor(ui,device,torch.long)
                    with torch.no_grad():weak=model(target_data['views'][0])
                    strong=[model(target_data['views'][j,indices]) for j in (1,2)]
                    r=core.consistency(weak,*strong,prototypes,indices,neighbors);total=total+.1*r['loss']
                    pcd=dict(U_global=float(r['global'].detach().cpu()),U_local=float(r['local'].detach().cpu()),prototype_mass_min=float(r['prototype_mass_min'].cpu()),prototype_mass_max=float(r['prototype_mass_max'].cpu()))
                if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source/prototype objective')
                total.backward();gradients=_gradients(model,prototypes) if step<=2 else None
                params=[p for group in opt.param_groups for p in group['params']];norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite ordinary body/projection gradient norm')
                opt.step()
                if target:
                    with torch.no_grad():prototypes.copy_(F.normalize(prototypes,dim=1))
                for k in ('L','OE'):usage[k].update(map(int,sample[k]))
                usage['U'].update(map(int,ui));row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),
                    loss_total=float(total.detach().cpu()),**metrics,**pcd,source_anchor_weight=10.,U_PCD_weight=.1 if target else 0.,gradient_norms=gradients,
                    gradient_norm_before_clip=norm,ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('BODY_PROTOTYPE_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if control is not None and control.readline():raise ValueError('SOURCE sampling trace contains extra updates')
        return trace,{k:len(v) for k,v in usage.items()}
    finally:
        if control is not None:control.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable body/prototype training output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device)
    snapshots=[(model,_snapshot(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def check():
        for model,before in snapshots:
            if any(not torch.equal(value,model.state_dict()[key].detach().cpu()) for key,value in before.items()):raise ValueError('Original F0/r03/DURA state changed')
    smoke=parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint smoke must precede source/U values')
    pm,_,_,_=_reference(c);parent_identity=pm['model_identity']
    if b['model_identity']!=parent_identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original source F0/r03/DURA identity differs')
    student=_student(b['model']);initial=_snapshot(b['model']);gradient=_gradient_smoke(student,device);check()
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient))
    roles={}
    try:
        banks,m,roles=_source_cache(c,b,out,check);teacher_before=banks['OE']['teacher_u'].copy();rows={};U=None;U_manifest=None
        for arm in ARMS:
            prototypes=None
            if arm=='TARGET_PCD':
                U,U_manifest=_target_cache(c,b,out,device,check,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()));student=_student(b['model'])
                centers,initialization=core.initialize_prototypes(base._tensor(U['parent_z'],device));prototypes=torch.nn.Parameter(centers)
            if any(not torch.equal(value,student.state_dict()[key].detach().cpu()) for key,value in initial.items()):raise ValueError('Both students must independently initialize from original r03')
            folder=out/arm;folder.mkdir();buffers={k:v.detach().clone() for k,v in student.named_buffers()};before={k:v.detach().cpu().clone() for k,v in student.named_parameters()}
            params=[p for p in student.parameters() if p.requires_grad]+([prototypes] if prototypes is not None else []);opt=torch.optim.AdamW(params,lr=c['lr'],weight_decay=.0001,eps=1e-8)
            trace,unique=_loop(student,banks,c,arm,device,opt,folder,U,prototypes)
            if any(not torch.equal(value,dict(student.named_buffers())[key]) for key,value in buffers.items()):raise ValueError('Student persistent buffers changed')
            changed={k:not torch.equal(value,dict(student.named_parameters())[k].detach().cpu()) for k,value in before.items()}
            if any(v for k,v in changed.items() if k.startswith(signalcore.SINC_PREFIX)):raise ValueError('Frozen Sinc changed')
            if not any(v for k,v in changed.items() if k.startswith('signal.') and not k.startswith(signalcore.SINC_PREFIX)) or not any(v for k,v in changed.items() if k.startswith('projection.')):raise ValueError('Body and projection must actually update')
            if not np.array_equal(teacher_before,banks['OE']['teacher_u']):raise ValueError('Original OE topology teacher changed')
            check();identity=_identity(c,arm,parent_identity,m,U_manifest);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
            if any(not bool(torch.isfinite(value).all()) for value in network.values()):raise ValueError('Finite full SignalOpen checkpoint required')
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent_identity,source_feature_manifest=m),folder/'final.pt')
            if prototypes is not None:
                _npz(folder/'prototype_state.npz',dict(prototypes=np.asarray(prototypes.detach().cpu().tolist(),np.float32)));_write(folder/'prototype_initialization.json',initialization)
            target=arm=='TARGET_PCD';row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,identity=identity,updates=STEPS,
                source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,source_feature_manifest=m,
                initial_state_equal_to_parent=True,student_encoder_unchanged=False,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,
                model_buffers_unchanged=True,original_models_unchanged=True,source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,
                signal_forward_rows=(256+(U_COUNT+2*U_BATCH)*int(target))*STEPS,unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],
                optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=STEPS),prototype_deployed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('BODY_PROTOTYPE_ARM_COMPLETE',arm,flush=True);del opt,student
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={arm:row['identity'] for arm,row in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,parent_signal_identity=parent_identity,parent_F0_identity=parent_identity['parent_F0_identity'],dura_identity=parent_identity['dura_identity'],
            source_feature_manifest=m,source_only=False,target_U_reads=U_COUNT,U_manifest=U_manifest,SOURCE_completed_before_U_open=True,source_sampling_matched=True,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,original_OE_teacher_unchanged=True,
            real_parent_signal_forward_rows=sum(COUNTS.values())+U_COUNT,real_student_signal_forward_rows=sum(row['signal_forward_rows'] for row in rows.values()),real_F_forward_rows=0,G_forward_rows=U_COUNT,
            synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_signal_rows=gradient['synthetic_student_signal_rows'],
            source_density_fit_updates=0,threshold_fit_updates=0,per_record_U_bank_deployed=False,prototype_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():base.close_role(value)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only SOURCE or TARGET_PCD frozen checkpoints may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');row=_read(root/arm/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Both body/prototype arms must be complete before loading') from exc
    zero={k:0 for k in ZEROS};expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        SOURCE_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_bodies_changed=True,all_student_projections_changed=True,final_phase3_complete=False,**zero)
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows',{}).get(arm)!=row or done.get('identities',{}).get(arm)!=row.get('identity'):raise ValueError('Complete matched body/source-target training identity required')
    if any(row.get(k)!=v for k,v in dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,config=c,updates=STEPS,source_only=arm=='SOURCE',
            target_U_used=arm=='TARGET_PCD',target_U_reads=U_COUNT if arm=='TARGET_PCD' else 0,U_manifest=done['U_manifest'] if arm=='TARGET_PCD' else None,initial_state_equal_to_parent=True,
            student_encoder_unchanged=False,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,final_phase3_complete=False,**zero).items()):raise ValueError('Frozen row identity or source/U boundary differs')
    pm,_,_,_=_reference(c);p=c['reference_config']['train_config'];model,parent_identity=base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    identity=_identity(c,arm,parent_identity,done['source_feature_manifest'],done['U_manifest']);value=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if (parent_identity!=pm['model_identity'] or row['identity']!=identity or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity
            or value.get('parent_signal_identity')!=parent_identity or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict source r03 ancestry and body-prototype identity required')
    network=value.get('network')
    if not isinstance(network,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite SignalOpen state required')
    original=model.state_dict()
    # Runtime/nonpersistent caches are reconstructed, never checkpoint entries.
    fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(signalcore.SINC_PREFIX)}
    if any(k not in network or not torch.equal(original[k].detach().cpu(),network[k]) for k in fixed):raise ValueError('Frozen source Sinc/buffers cannot be replaced')
    try:model.load_state_dict(network,strict=True)
    except RuntimeError as exc:raise ValueError('Strict full SignalOpen body/projection checkpoint required') from exc
    model.eval().requires_grad_(False);model._body_prototype_identity=identity
    return model,identity


def main():
    print(json.dumps(dict(event='BODY_PROTOTYPE_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
