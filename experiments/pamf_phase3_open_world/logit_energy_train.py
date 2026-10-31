"""Independent complete-F0 source increments for uniform versus paired energy."""
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
from . import logit_energy_core as core
from . import native_identity_train as identity
from . import star_representation_cache as cache

RUN_ID='20261031-phase3-logit-energy-s2026092701-r01'
SCHEMA='pamf_phase3_logit_energy_train_v1'
STATUS='LOGIT_ENERGY_TRAIN_COMPLETE'
ROW_STATUS='LOGIT_ENERGY_STUDENT_FROZEN'
ARMS=core.ARMS
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='f4b5a0b80cafcf237c3ae9dac8ecd5b24c91bff7'
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
ZEROS=('target_U_reads','U_ID_metadata_reads','target_Q_reads','validation_reads','novel_validation_reads','private_truth_reads','U_truth_reads',
       'OE_truth_reads','original_U_s_reads','source_Us_reads','Sinc_optimizer_steps','physical_clip_calls','source_density_fit_updates','threshold_fit_updates')
_read,_write,_snapshot=identity._read,identity._write,identity._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact source logit-energy config required; no U/Q/truth/K or budget overrides')
    r=identity.reg.fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Original completed fresh f4b5 source ancestry required')
    source=identity._source_config(c);out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],source['output_root'],source['IQ_root'],source['native_config']['source_output']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable source-energy output overlaps an input')
    return c


def _reference(c):return identity._reference(c)
def _backend(c,device):return identity._backend(c,device)
def _load_PCD(c,device):return identity._load_PCD(c,device)


def _initial_model(c,pm,device):
    wrapper=identity._initial_model(c,pm,device)
    return core.NativeEnergy(wrapper.id_backbone.encoder).to(device),deepcopy(wrapper._phase3_parent_identity)


def _roles(c,pm):
    root=Path(identity._source_config(c)['IQ_root']);roles={}
    try:
        for role in ('L','OE'):
            a=cache.load_role(root/role,role);roles[role]=a;n=COUNTS[role]
            if (a['manifest']['count']!=n or a['manifest']['parent_F0_identity']!=pm['parent_F0_identity'] or a['manifest']['dura_identity']!=pm['dura_identity']
                    or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n
                    or a['restored'].shape!=(n,2,256) or a['restored'].dtype!=np.float32 or not np.isfinite(a['restored']).all()):
                raise ValueError('Original source P1 role/opaque IDs/F0/G identity required')
            if role=='L' and (a['teacher_logits'].shape!=(n,6) or a['teacher_logits'].dtype!=np.float32 or not np.isfinite(a['teacher_logits']).all()):raise ValueError('Existing same-P1 original-F0 cached L teacher required')
            core._metadata(a['metadata'],role=='L')
        if set(roles['L']['ids'].tolist())&set(roles['OE']['ids'].tolist()):raise ValueError('Known L and source exposure roles must be disjoint')
        return roles
    except BaseException:
        for value in roles.values():cache.close_role(value)
        raise


def _gradients(model):
    b=model.reader.core.id_backbone
    norm=lambda ps:sum(float(p.grad.detach().double().square().sum().cpu()) for p in ps if p.grad is not None)**.5
    shallow=b.time_fuse if hasattr(b,'time_fuse') else b.t1
    result=dict(head=norm([b.cls_head.weight]),shallow=norm(shallow.parameters()),middle=norm(b.t2.parameters()),Sinc=norm(b.sinc.parameters()),
        body=norm(p for n,p in model.named_parameters() if n!=core.HEAD_WEIGHT and not n.startswith(core.SINC_PREFIX)))
    if any(p.grad is not None for p in b.sinc.parameters()) or any(not np.isfinite(v) for v in result.values()) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite native gradient or frozen Sinc gradient')
    return result


def _smoke(model,b,pcd,device):
    from .star_open_feature_export import _synthetic
    state=_snapshot(model);buffers={n:v.detach().clone() for n,v in model.named_buffers()};trials={}
    try:
        with torch.random.fork_rng(devices=identity.native._cuda_devices(model)):
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                restored=b['dura'][1](_synthetic(device))[0].detach().clone()
                original=b['reader'].classify_features(b['reader'].features(restored));r03=b['model'](restored);final=pcd(restored)
                raw,initial=model(restored)
            if (original.shape!=(2,6) or raw.shape!=(2,160) or r03.shape!=(2,160) or final.shape!=(2,160)
                    or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (restored,original,r03,final,raw))):raise ValueError('Actual original F0/r03/PCD/G synthetic outputs differ')
            if not torch.allclose(initial,original,rtol=identity.LOGIT_RTOL,atol=identity.LOGIT_ATOL):raise ValueError('Initial full native logits differ from original F0')
            for arm in ARMS:
                model.zero_grad(set_to_none=True);_,logits=model(torch.cat((restored,restored)))
                terms=core.compute(logits[:2],torch.tensor([0,1],device=device),original.detach(),torch.tensor([0,1],device=device),logits[2:],torch.tensor([0,1],device=device),arm)
                terms['total'].backward();g=_gradients(model)
                if any(g[k]<=0 for k in ('head','body','shallow','middle')):raise ValueError('Actual native default head and shallow/middle/body gradients must be finite and nonzero')
                identity._unchanged(model,state,'No-step native energy smoke')
                if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('No-step native energy smoke changed buffers')
                trials[arm]=dict(status='PASS',gradient_norms=g,loss=float(terms['total'].detach().cpu()))
            return dict(status='PASS',actual_F0_r03_DURA_PCD_checkpoints=True,actual_original_F0_equal=True,
                initial_logit_max_abs_difference=float((initial-original).abs().max().cpu()),logit_rtol=identity.LOGIT_RTOL,logit_atol=identity.LOGIT_ATOL,
                trials=trials,Sinc_gradients_none=True,parameters_buffers_unchanged=True,optimizer_updates=0,real_data_reads=0,
                synthetic_F_forward_rows=2,synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_forward_rows=10)
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,v in model.named_buffers():v.copy_(buffers[n])


def _runtime(a,device):
    return dict(iq=identity._tensor(a['restored'],device),metadata={k:identity._tensor(v,device,torch.long) for k,v in a['metadata'].items()},
        **({'teacher_logits':identity._tensor(a['teacher_logits'],device)} if 'teacher_logits' in a else {}))


def _loop(c,arm,model,roles,device,folder,check):
    sampler=core.SourceSampler(roles['L']['metadata'],roles['OE']['metadata']);data={k:_runtime(v,device) for k,v in roles.items()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    matched=(Path(c['output_root'])/'rows/UNIFORM/steps.jsonl').open(encoding='utf-8') if arm=='RXPAIR' else None
    trace=[];usage={k:set() for k in ('L','OE')}
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if matched is not None:
                    line=matched.readline()
                    if not line:raise ValueError('Matched UNIFORM trajectory is incomplete')
                    row=json.loads(line)
                    if row['step']!=step or any(row[k+'_indices']!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched source L/OE index streams differ')
                li=identity._tensor(sample['L'],device,torch.long);oi=identity._tensor(sample['OE'],device,torch.long);L,OE=data['L'],data['OE']
                opt.zero_grad(set_to_none=True);_,logits=model(torch.cat((L['iq'][li],OE['iq'][oi])))
                terms=core.compute(logits[:192],L['metadata']['y'][li],L['teacher_logits'][li],L['metadata']['domain'][li],logits[192:],OE['metadata']['domain'][oi],arm)
                if not bool(torch.isfinite(terms['total'])):raise FloatingPointError('Nonfinite complete native energy objective')
                terms['total'].backward();gradients=_gradients(model);unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite native clip norm')
                opt.step()
                for key in usage:usage[key].update(map(int,sample[key]))
                row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),source_indices_equal=True,
                    **{k:float(v.detach().cpu()) for k,v in terms.items() if torch.is_tensor(v)},pair_count=terms['pair_count'],
                    CE_weight=1.,KD_weight=1.,KD_temperature=2.,OE_weight=1.,energy_temperature=1.,gradient_norms=gradients,
                    gradient_norm_before_clip=norm,ordinary_clip_calls=step,unused_gradient_parameters=unused,
                    optimizer_updates=step,identity_head_updates=step,Sinc_optimizer_steps=0,physical_clip_calls=0,
                    L_count=192,OE_count=64,student_forward_rows=256,cumulative_student_forward_rows=256*step,
                    source_L_exposures=192*step,source_OE_exposures=64*step,target_U_reads=0,target_Q_reads=0)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:
                    check();print('LOGIT_ENERGY_PROGRESS',arm,step,STEPS,row['total'],row['OE'],row['s_L'],row['s_OE'],flush=True)
        if matched is not None and matched.readline():raise ValueError('Matched UNIFORM trajectory contains extra updates')
    finally:
        if matched is not None:matched.close()
    return trace,{k:len(v) for k,v in usage.items()}


def _identity(c,arm,pm,original_parent,handles):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/'rows'/arm/'final.pt'),updates=STEPS,
        reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,
        identity_head_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,initial_native_logits_equal_original_F0=True,
        original_parent_updates=11100,cumulative_updates=11100+STEPS,original_parent_identity=original_parent,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_handles=handles,
        source_only=True,target_U_used=False,target_U_reads=0,target_Q_reads=0,input_path='cached_P1_IQ_only',BN_Dropout_eval=True,
        initialization='independent_complete_original_F0_E50',deployment_state='complete_reader_only',unused_wrapper_parameters_not_deployed=667729,target_REG=False)


def _changed(model,initial):
    state=model.state_dict();parameters=dict(model.named_parameters());fixed=(set(dict(model.named_buffers()))&set(state))|{n for n in parameters if n.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[n],state[n].detach().cpu()) for n in fixed):raise ValueError('Frozen Sinc/persistent buffers differ from original F0')
    head=not torch.equal(initial[core.HEAD_WEIGHT],state[core.HEAD_WEIGHT].detach().cpu())
    body=any(not torch.equal(initial[n],p.detach().cpu()) for n,p in parameters.items() if n!=core.HEAD_WEIGHT and not n.startswith(core.SINC_PREFIX))
    if not head or not body:raise ValueError('Original default head and native body must both actually update')


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable native energy training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,_,_,_=_reference(c);b=_backend(c,device);pcd,pcd_identity=_load_PCD(c,device)
    if b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity'] or b['model_identity']!=pm['model_identities']['E0'] or pcd_identity!=pm['model_identities']['TARGET_PCD']:raise ValueError('Original F0/pure-r03/PCD/G ancestry differs')
    originals=(b['reader'],b['model'],pcd,b['dura'][0])
    if any(p.requires_grad or p.grad is not None for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):raise ValueError('All original models must remain frozen eval')
    snapshots=[(m,_snapshot(m)) for m in originals]
    def check():
        for model,before in snapshots:
            identity._unchanged(model,before,'Original F0/r03/PCD/G')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Original frozen model received gradients')
    model,original_parent=_initial_model(c,pm,device);initial=_snapshot(model);original=_snapshot(b['reader'])
    if set(original)!={n.removeprefix('reader.') for n in initial} or any(not torch.equal(v,initial['reader.'+n]) for n,v in original.items()):raise ValueError('Complete original F0 body and default head initialization required')
    smoke=_smoke(model,b,pcd,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_original_F0_equal') is not True:raise ValueError('Actual native energy smoke must pass before source values')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);roles={};rows={}
    try:
        roles=_roles(c,pm);handles={k:v['manifest'] for k,v in roles.items()}
        for arm in ARMS:
            if arm=='RXPAIR':
                model,parent=_initial_model(c,pm,device)
                if parent!=original_parent:raise ValueError('Both students need the same original F0 parent')
            identity._unchanged(model,initial,'Independent complete original-F0 initial state')
            buffers={n:v.detach().clone() for n,v in model.named_buffers()};folder=out/'rows'/arm;folder.mkdir(parents=True)
            trace,unique=_loop(c,arm,model,roles,device,folder,check);check();_changed(model,initial)
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Native student buffers changed')
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite complete native reader checkpoint required')
            i=_identity(c,arm,pm,original_parent,handles);model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,original_parent_identity=original_parent,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_handles=handles),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,updates=STEPS,identity_head_updates=STEPS,
                source_only=True,target_U_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,student_head_changed=True,
                model_buffers_unchanged=True,original_models_unchanged=True,source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,student_forward_rows=256*STEPS,
                unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('LOGIT_ENERGY_ARM_COMPLETE',arm,flush=True);del model
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={k:v['identity'] for k,v in rows.items()},
            steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_sampling_matched=True,source_only=True,new_target_UDA_updates=0,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,all_student_heads_changed=True,all_student_bodies_changed=True,
            all_students_initialized_from_original_F0=True,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],original_parent_identity=original_parent,source_handles=handles,
            reader_parameters=232993,trainable_parameters_per_student=232945,default_head_parameters=960,unused_wrapper_parameters_not_deployed=667729,
            source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],cached_L_teacher_logits_reads=COUNTS['L'],real_student_forward_rows=512*STEPS,
            real_parent_signal_forward_rows=0,real_F_forward_rows=0,real_F_teacher_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            L_sampling_seed=core.L_SEED,OE_sampling_seed=core.OE_SEED,optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=2*STEPS),
            **{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'},
            source_bank_deployed=False,source_teacher_deployed=False,formal_target_accessed=False,target_REG=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():cache.close_role(role)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only the fixed two source-energy students may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={name:_read(root/'rows'/name/'complete.json') for name in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both complete source-energy rows required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_sampling_matched=True,source_only=True,new_target_UDA_updates=0,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        all_student_heads_changed=True,all_student_bodies_changed=True,all_students_initialized_from_original_F0=True,real_student_forward_rows=512*STEPS,
        real_parent_signal_forward_rows=0,real_F_forward_rows=0,G_forward_rows=0,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows:raise ValueError('Complete matched pure-source energy training required')
    pm,_,_,_=_reference(c);model,parent=_initial_model(c,pm,device)
    if done.get('original_parent_identity')!=parent or done.get('parent_F0_identity')!=pm['parent_F0_identity'] or done.get('dura_identity')!=pm['dura_identity']:raise ValueError('Original F0/G source ancestry differs')
    handles=done['source_handles']
    if set(handles)!={'L','OE'} or any(v.get('count')!=COUNTS[k] or v.get('parent_F0_identity')!=pm['parent_F0_identity'] or v.get('dura_identity')!=pm['dura_identity'] for k,v in handles.items()):raise ValueError('Only original source L/OE handles may train the reader')
    for name,row in rows.items():
        i=_identity(c,name,pm,parent,handles)
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=name,identity=i,updates=STEPS,identity_head_updates=STEPS,
            source_only=True,target_U_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,student_head_changed=True,
            model_buffers_unchanged=True,original_models_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or done['identities'].get(name)!=i:raise ValueError('Strict source-only native row/default-head identity required')
    payload=torch.load(root/'rows'/arm/'final.pt',map_location='cpu',weights_only=False);i=rows[arm]['identity']
    if (payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('identity')!=i or payload.get('original_parent_identity')!=parent
            or payload.get('parent_F0_identity')!=pm['parent_F0_identity'] or payload.get('dura_identity')!=pm['dura_identity'] or payload.get('source_handles')!=handles):raise ValueError('Strict original full-F0 checkpoint ancestry required')
    network=payload.get('network');initial=_snapshot(model)
    if not isinstance(network,dict) or set(network)!=set(initial) or any(not torch.is_tensor(v) or v.dtype!=initial[k].dtype or v.shape!=initial[k].shape or not bool(torch.isfinite(v).all()) for k,v in network.items()):raise ValueError('Full finite original reader/default-head state required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{n for n,p in model.named_parameters() if n.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[n],network[n]) for n in fixed):raise ValueError('Frozen Sinc and persistent buffers cannot change')
    model.load_state_dict(network,strict=True);_changed(model,initial);model.eval().requires_grad_(False);model._logit_energy_identity=i
    return model,i


def main():
    print(json.dumps(dict(event='LOGIT_ENERGY_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
