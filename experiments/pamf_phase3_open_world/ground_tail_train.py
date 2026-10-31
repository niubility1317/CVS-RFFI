"""Matched domain-mean versus domain-tail native source energy training."""
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

from . import ground_open_train as old
from . import ground_open_run as previous
from . import ground_tail_core as core

RUN_ID='20261101-phase3-ground-tail-s2026092701-r01'
SCHEMA='pamf_phase3_ground_tail_train_v1'
STATUS='GROUND_TAIL_TRAIN_COMPLETE'
ROW_STATUS='GROUND_TAIL_STUDENT_FROZEN'
ARMS=('DMEAN','DTAIL')
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='d7e0f875397b2f9801df4605ba0051d221aa4e2a'
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
native=old.core
ZEROS=('target_U_reads','U_ID_metadata_reads','target_Q_reads','validation_reads','novel_validation_reads',
       'CAL_reads','source_Q_reads','source_Us_reads','source_U_s_reads','source_U_GT_reads','Us_ID_metadata_reads',
       'private_truth_reads','U_truth_reads','OE_truth_reads','Sinc_optimizer_steps','physical_clip_calls',
       'source_density_fit_updates','threshold_fit_updates','new_data_G_forward_rows')
_read,_write,_snapshot=old._read,old._write,old._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS
            or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field fixed-budget ground-tail config required; no Us/Q/truth inputs')
    r=previous.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:
        raise ValueError('Only completed ground r03/D7 and its unchanged data ancestor may be used')
    out=Path(c['output_root']).resolve();t=r['train_config']
    roots=[r['output_root'],r['data_root'],t['output_root'],t['reference_config']['output_root']]
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in roots):
        raise ValueError('New immutable training output cannot overlap old ground/data/model outputs')
    return c


def _reference(c):
    r=c['reference_config'];t=r['train_config'];root=Path(r['output_root'])
    prediction=_read(root/'complete.json')
    expected=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=r['run_id'],code_commit=REFERENCE_CODE,config=r,
        all_12_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0,final_phase3_complete=False)
    if any(prediction.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed fixed old ground prediction metadata required; no old query arrays are read')
    trained=_read(Path(t['output_root'])/'complete.json')
    expected_train=dict(schema=old.SCHEMA,status=old.STATUS,run_id=old.RUN_ID,code_commit=REFERENCE_CODE,config=t,
        steps_per_arm=old.STEPS,updates_per_arm=old.STEPS,total_optimizer_updates=2*old.STEPS,
        source_L_OE_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_buffers_unchanged=True,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_U_GT_reads=0,final_phase3_complete=False)
    if any(trained.get(k)!=v for k,v in expected_train.items()):
        raise ValueError('Original D7 ground training/data ancestry required, without loading its students')
    pm,_,_,_=old._reference(t)
    ground,handles=old._ground_metadata(t,pm,with_us=False)
    identity=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=r['run_id'],
                  code_commit=REFERENCE_CODE,output_root=r['output_root'])
    return pm,ground,handles,identity


def _backend(c,device):return old._backend(c['reference_config']['train_config'],device)


def _initial_model(c,pm,device):return old._initial_model(c['reference_config']['train_config'],pm,device)


def _load_PCD(c,device):return old._load_PCD(c['reference_config']['train_config'],device)


def _smoke(model,b,pcd,device):
    """Reuse the base twelve rows, then check each new loss on four rows."""
    captured={}
    def capture(module,args,output):
        if not captured:
            captured.update(iq=args[0].detach().clone(),teacher=output[1].detach().clone())
    handle=model.register_forward_hook(capture)
    try:base=old._smoke(model,b,pcd,device)
    finally:handle.remove()
    if base.get('status')!='PASS' or base.get('actual_original_F0_equal') is not True or captured.get('iq') is None:
        raise ValueError('Actual original F0/G/native initialization smoke must pass')
    initial=_snapshot(model);buffers={n:v.detach().clone() for n,v in model.named_buffers()};trials={}
    try:
        x=captured['iq'];teacher=captured['teacher'];domains=torch.tensor([0,1],device=device,dtype=torch.long)
        if len(x)!=2:raise ValueError('Base no-query smoke must expose exactly two synthetic restored rows')
        for arm in ARMS:
            model.zero_grad(set_to_none=True);_,logits=model(torch.cat((x,x)))
            terms=core.compute(logits[:2],domains,teacher,domains,logits[2:],domains,arm)
            terms['total'].backward();g=old._gradients(model)
            if any(g[k]<=0 for k in ('head','body','shallow','middle')):
                raise ValueError('New mean/tail loss must reach the actual default head and shallow/middle/body')
            old.identity._unchanged(model,initial,'No-step new domain-tail smoke')
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):
                raise ValueError('No-step domain-tail smoke changed frozen buffers')
            trials[arm]=dict(status='PASS',loss=float(terms['total'].detach().cpu()),gradient_norms=g,
                domain_count=terms['domain_count'],pair_count=terms['pair_count'],selected_pair_count=terms['selected_pair_count'])
        return dict(base,new_loss_trials=trials,new_loss_gradients_verified=True,synthetic_student_forward_rows=base['synthetic_student_forward_rows']+8)
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,v in model.named_buffers():v.copy_(buffers[n])


def _loop(c,arm,model,roles,device,folder,check):
    lm=old._metadata(roles['L']['metadata'],roles['L']['ids'],True)
    om=old._metadata(roles['OE']['metadata'],roles['OE']['ids'],False)
    sampler=native.SourceSampler(lm,om);values={k:old._runtime(v,device) for k,v in roles.items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    matched=(Path(c['output_root'])/'rows/DMEAN/steps.jsonl').open(encoding='utf-8') if arm=='DTAIL' else None
    trace=[];usage={k:set() for k in ('L','OE')}
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as stream:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if matched is not None:
                    line=matched.readline();other=json.loads(line) if line else {}
                    if other.get('step')!=step or any(other.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):
                        raise ValueError('Domain mean/tail source samples must match at every step')
                li=old.identity._tensor(sample['L'],device,torch.long);oi=old.identity._tensor(sample['OE'],device,torch.long)
                L,OE=values['L'],values['OE'];optimizer.zero_grad(set_to_none=True)
                _,logits=model(torch.cat((L['iq'][li],OE['iq'][oi])))
                terms=core.compute(logits[:192],L['metadata']['y'][li],L['teacher_logits'][li],L['metadata']['domain'][li],
                                   logits[192:],OE['metadata']['domain'][oi],arm)
                if terms['domain_count']!=15:raise ValueError('All fifteen public source RX/day domains must contribute every step')
                if not bool(torch.isfinite(terms['total'])):raise FloatingPointError('Nonfinite native mean/tail objective')
                terms['total'].backward();g=old._gradients(model)
                unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite native gradient norm')
                optimizer.step()
                for k in usage:usage[k].update(map(int,sample[k]))
                scalars={k:float(v.detach().cpu()) if torch.is_tensor(v) else v for k,v in terms.items()}
                row=dict(scalars,step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),source_indices_equal=True,
                    CE_weight=1.,KD_weight=1.,KD_temperature=2.,OE_weight=1.,energy_temperature=1.,PL_weight=0.,PL=0.,
                    gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,
                    unused_gradient_parameters=unused,optimizer_updates=step,identity_head_updates=step,Sinc_optimizer_steps=0,
                    physical_clip_calls=0,L_count=192,OE_count=64,student_forward_rows=256,cumulative_student_forward_rows=256*step,
                    source_L_exposures=192*step,source_OE_exposures=64*step,source_Us_reads=0,target_U_reads=0,CAL_reads=0,source_Q_reads=0)
                trace.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if step==1 or step%100==0 or step==STEPS:
                    check();print('GROUND_TAIL_PROGRESS',arm,step,STEPS,row['total'],row['OE'],row['selected_pair_count'],row['pair_count'],flush=True)
        if matched is not None and matched.readline():raise ValueError('Mean-control trace contains extra updates')
    finally:
        if matched is not None:matched.close()
    return trace,{k:len(v) for k,v in usage.items()}


def _identity(c,arm,pm,parent,ground,handles,reference):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/'rows'/arm/'final.pt'),updates=STEPS,
        reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,
        identity_head_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,projection_updates=0,
        initial_native_logits_equal_original_F0=True,original_parent_updates=11100,cumulative_updates=11100+STEPS,
        original_parent_identity=parent,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        source_ground_reference_identity=reference,ground_data_identity=ground,source_handles=handles,same_input_teacher=True,
        source_only=True,ground_clean_training=True,source_U_s_used=False,source_Us_reads=0,source_U_s_reads=0,source_U_s_exposures=0,
        target_U_used=False,target_U_reads=0,target_Q_reads=0,source_U_GT_reads=0,input_path='G(clean)',BN_Dropout_eval=True,
        initialization='independent_complete_original_F0_E50',deployment_state='complete_reader_only',unused_wrapper_parameters_not_deployed=667729,
        OE_loss_reduction='equal_public_RX_day_domains',tail_fraction=1. if arm=='DMEAN' else .25,target_REG=False)


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('Ground-tail outputs are immutable; preserve existing run')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,ground,handles,reference=_reference(c);b=_backend(c,device);pcd,pcd_identity=_load_PCD(c,device)
    if (b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']
            or b['model_identity']!=pm['model_identities']['E0'] or pcd_identity!=pm['model_identities']['TARGET_PCD']):
        raise ValueError('Original pure-source F0/r03 and fixed PCD/G ancestors differ')
    originals=(b['reader'],b['model'],pcd,b['dura'][0]);snapshots=[(model,_snapshot(model)) for model in originals]
    if any(p.requires_grad or p.grad is not None for model in originals for p in model.parameters()) or any(m.training for model in originals for m in model.modules()):
        raise ValueError('Every original model must remain frozen eval with no gradients')
    def check():
        for model,before in snapshots:
            old.identity._unchanged(model,before,'Original frozen ground-tail ancestors')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('An original frozen model received gradients')
    model,parent=_initial_model(c,pm,device);initial=_snapshot(model);original=_snapshot(b['reader'])
    if set(original)!={n.removeprefix('reader.') for n in initial} or any(not torch.equal(v,initial['reader.'+n]) for n,v in original.items()):
        raise ValueError('Each student must initialize the complete original F0 reader and head')
    smoke=_smoke(model,b,pcd,device);check()
    if smoke.get('status')!='PASS' or smoke.get('new_loss_gradients_verified') is not True:
        raise ValueError('Actual new-loss checkpoint smoke must pass before opening source IQ')
    print('GROUND_TAIL_CHECKPOINT_SMOKE_PASS',flush=True)
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    roles={};rows={}
    try:
        for role in ('L','OE'):roles[role]=old._role(c['reference_config']['train_config'],pm,role,handles[role])
        if set(roles['L']['ids'].tolist())&set(roles['OE']['ids'].tolist()):raise ValueError('L/OE physical records overlap')
        for arm in ARMS:
            if arm!='DMEAN':
                model,other=_initial_model(c,pm,device)
                if other!=parent:raise ValueError('Matched students need the same original F0 parent')
            old.identity._unchanged(model,initial,'Independent original F0 initialization')
            buffers={n:v.detach().clone() for n,v in model.named_buffers()};folder=out/'rows'/arm;folder.mkdir(parents=True)
            trace,unique=_loop(c,arm,model,roles,device,folder,check);check();old._changed(model,initial)
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Native student buffers changed')
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite complete native checkpoint required')
            i=_identity(c,arm,pm,parent,ground,deepcopy(handles),reference);model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,original_parent_identity=parent,
                source_ground_reference_identity=reference,ground_data_identity=ground,source_handles=deepcopy(handles)),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,
                updates=STEPS,identity_head_updates=STEPS,source_only=True,ground_clean_training=True,same_input_teacher=True,
                source_U_s_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,
                student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,
                source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,student_forward_rows=256*STEPS,
                unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('GROUND_TAIL_ARM_COMPLETE',arm,flush=True);del model
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={k:v['identity'] for k,v in rows.items()},
            steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_L_OE_sampling_matched=True,source_sampling_matched=True,
            teacher_source_sampling_matched=True,same_input_teacher=True,source_only=True,ground_clean_training=True,new_target_UDA_updates=0,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,all_student_heads_changed=True,
            all_student_bodies_changed=True,all_students_initialized_from_original_F0=True,parent_F0_identity=pm['parent_F0_identity'],
            dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],original_parent_identity=parent,
            source_ground_reference_identity=reference,ground_data_identity=ground,source_handles=handles,
            reader_parameters=232993,trainable_parameters_per_student=232945,default_head_parameters=960,unused_wrapper_parameters_not_deployed=667729,
            source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],cached_L_teacher_logits_reads=COUNTS['L'],
            real_student_forward_rows=512*STEPS,real_student_forward_rows_by_arm={arm:256*STEPS for arm in ARMS},
            real_parent_signal_forward_rows=0,real_F_forward_rows=0,real_F_teacher_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            L_sampling_seed=native.L_SEED,OE_sampling_seed=native.OE_SEED,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=2*STEPS),
            **{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'},
            source_bank_deployed=False,source_teacher_deployed=False,formal_target_accessed=False,target_REG=False,
            full_goal_status='ACTIVE',final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():old._data().close(role)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only the two complete domain mean/tail native students may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/'rows'/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both completed ground-tail students required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),
        steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_L_OE_sampling_matched=True,source_sampling_matched=True,
        source_only=True,ground_clean_training=True,same_input_teacher=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_buffers_unchanged=True,all_student_heads_changed=True,all_student_bodies_changed=True,
        real_student_forward_rows=512*STEPS,new_target_UDA_updates=0,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows:raise ValueError('Complete source-only matched ground-tail run required')
    pm,ground,handles,reference=_reference(c);model,parent=_initial_model(c,pm,device)
    if any(done.get(k)!=v for k,v in dict(original_parent_identity=parent,ground_data_identity=ground,
                                         source_ground_reference_identity=reference,source_handles=handles).items()):
        raise ValueError('New student versus old unchanged ground-data ancestry differs')
    for name,row in rows.items():
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=name,
            updates=STEPS,identity_head_updates=STEPS,source_only=True,ground_clean_training=True,same_input_teacher=True,
            source_U_s_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,
            student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or row.get('identity')!=done.get('identities',{}).get(name):
            raise ValueError('Complete frozen native ground-tail row required')
    i=_identity(c,arm,pm,parent,ground,handles,reference)
    if rows[arm]['identity']!=i:raise ValueError('Own native row identity differs')
    payload=torch.load(root/'rows'/arm/'final.pt',map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in dict(schema=SCHEMA,config=c,identity=i,original_parent_identity=parent,
        source_ground_reference_identity=reference,ground_data_identity=ground,source_handles=handles).items()):
        raise ValueError('Strict new native checkpoint and unchanged D7 ground ancestry required')
    network=payload.get('network');initial=_snapshot(model)
    if (not isinstance(network,dict) or set(network)!=set(initial)
            or any(not torch.is_tensor(v) or v.dtype!=initial[k].dtype or v.shape!=initial[k].shape or not bool(torch.isfinite(v).all()) for k,v in network.items())):
        raise ValueError('Full finite native reader/default head state required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{n for n in initial if n.startswith(native.SINC_PREFIX)}
    if any(not torch.equal(initial[n],network[n]) for n in fixed):raise ValueError('Frozen Sinc/persistent buffers changed')
    model.load_state_dict(network,strict=True);old._changed(model,initial);model.eval().requires_grad_(False)
    return model,i


def main():
    print(json.dumps(dict(event='GROUND_TAIL_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
