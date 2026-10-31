"""Two complete-original-F0 source students matched to the fixed CPAIR trace."""
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

from . import ground_contrast_core as core
from . import ground_centered_train as current
from . import ground_centered_run as previous

RUN_ID='20261101-phase3-ground-contrast-s2026092701-r01'
SCHEMA='pamf_phase3_ground_contrast_train_v1'
STATUS='GROUND_CONTRAST_TRAIN_COMPLETE'
ROW_STATUS='GROUND_CONTRAST_STUDENT_FROZEN'
ARMS=core.ARMS
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='1ae125733a0f25eb9c887351eb63f11f730d8cea'
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
old=current.old
native=current.native
ZEROS=current.ZEROS
_read,_write,_snapshot=old._read,old._write,old._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS
            or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field fixed-budget contrast source configuration required; no Us/U/Q/truth inputs')
    r=previous.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Only the fixed completed centered1ae control is allowed')
    out=Path(c['output_root']).resolve();tail=r['reference_config'];ground=tail['reference_config']
    roots=[r['output_root'],r['train_config']['output_root'],tail['output_root'],tail['train_config']['output_root'],
        ground['output_root'],ground['data_root'],ground['train_config']['output_root']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):
        raise ValueError('New contrast training output cannot overlap unchanged source/control outputs')
    return c


def _indices_path(c):return Path(c['reference_config']['train_config']['output_root'])/'rows/CPAIR/steps.jsonl'


def _reference(c):
    r=c['reference_config'];t=r['train_config'];marker=_read(Path(r['output_root'])/'complete.json')
    expected=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=r['run_id'],code_commit=REFERENCE_CODE,config=r,
        all_16_points_complete=True,all_four_DURA_conditions_complete=True,all_conditions_through_DURA=True,
        source_controls_reused_exactly=True,all_thresholds_frozen_before_Q=True,private_truth_reads=0,target_U_reads=0,final_phase3_complete=False)
    if any(marker.get(k)!=v for k,v in expected.items()):raise ValueError('Complete unchanged centered control metadata required; no query arrays are opened')
    pm,ground,handles,tail_reference=current._reference(t)
    trained=_read(Path(t['output_root'])/'complete.json');row=_read(Path(t['output_root'])/'rows/CPAIR/complete.json')
    required=dict(schema=current.SCHEMA,status=current.STATUS,run_id=current.RUN_ID,code_commit=REFERENCE_CODE,config=t,arms=['CPAIR'],
        steps_per_arm=current.STEPS,updates_per_arm=current.STEPS,total_optimizer_updates=current.STEPS,
        real_student_forward_rows=256*current.STEPS,source_L_OE_sampling_matched=True,source_sampling_matched=True,same_input_teacher=True,
        original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        source_indices_match_DMEAN=True,source_Us_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,
        source_tail_reference_identity=tail_reference,source_ground_reference_identity=tail_reference['source_ground_reference_identity'],
        ground_data_identity=ground,source_handles=handles,final_phase3_complete=False)
    row_required=dict(schema=current.SCHEMA,status=current.ROW_STATUS,run_id=current.RUN_ID,code_commit=REFERENCE_CODE,
        config=t,arm='CPAIR',updates=current.STEPS,source_indices_match_DMEAN=True,final_phase3_complete=False)
    if (any(trained.get(k)!=v for k,v in required.items()) or any(row.get(k)!=v for k,v in row_required.items())
            or trained.get('rows')!={'CPAIR':row} or set(trained.get('identities',{}))!={'CPAIR'}
            or row.get('identity')!=trained['identities']['CPAIR']
            or marker.get('source_tail_reference_identity')!=tail_reference
            or marker.get('source_ground_reference_identity')!=tail_reference['source_ground_reference_identity']
            or not _indices_path(c).is_file()):raise ValueError('Completed CPAIR source training and original tail/D7 handles required')
    i=row['identity']
    for key,value in dict(run_id=current.RUN_ID,code_commit=REFERENCE_CODE,arm='CPAIR',updates=current.STEPS,
        reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,identity_head_jointly_trained=True,
        identity_head_replaced=False,initial_native_logits_equal_original_F0=True,source_only=True,source_Us_reads=0,target_U_reads=0,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        source_tail_reference_identity=tail_reference,source_ground_reference_identity=tail_reference['source_ground_reference_identity'],
        ground_data_identity=ground,source_handles=handles).items():
        if i.get(key)!=value:raise ValueError('Pure-source CPAIR ancestor identity differs: '+key)
    reference=dict(schema=previous.SCHEMA,status=previous.STATUS,run_id=r['run_id'],code_commit=REFERENCE_CODE,
        output_root=r['output_root'],source_tail_reference_identity=tail_reference,
        source_ground_reference_identity=tail_reference['source_ground_reference_identity'])
    return pm,ground,handles,reference


def _backend(c,device):return current._backend(c['reference_config']['train_config'],device)


def _initial_model(c,pm,device):return current._initial_model(c['reference_config']['train_config'],pm,device)


def _ground_train_config(c):return current._ground_train_config(c['reference_config']['train_config'])


def _smoke(models,b,device):
    """Actual G/F0 synthetic2, initialization2 and both contrast losses6 each."""
    snapshots={arm:_snapshot(m) for arm,m in models.items()}
    buffers={arm:{n:v.detach().clone() for n,v in m.named_buffers()} for arm,m in models.items()}
    try:
        first=models[ARMS[0]]
        with torch.random.fork_rng(devices=old.identity.native._cuda_devices(first)):
            t=torch.arange(256,device=device,dtype=torch.float32)
            x=torch.stack([torch.stack((torch.cos(w*t),torch.sin(w*t))) for w in (.07,.11)])
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                restored=b['dura'][1](x)[0].detach().clone()
                original=b['reader'].classify_features(b['reader'].features(restored));_,initial=first(restored)
            if (restored.dtype!=torch.float32 or restored.shape!=(2,2,256) or original.dtype!=torch.float32 or original.shape!=(2,6)
                    or not bool(torch.isfinite(restored).all()) or not bool(torch.isfinite(original).all())
                    or not torch.allclose(initial,original,rtol=old.identity.LOGIT_RTOL,atol=old.identity.LOGIT_ATOL)):
                raise ValueError('Actual fixed G/original F0 and independent complete-native initialization smoke failed')
            y=torch.tensor([0,1,0,1],device=device,dtype=torch.long)
            domains=torch.tensor([0,0,4,4],device=device,dtype=torch.long);outside_domains=torch.tensor([0,4],device=device,dtype=torch.long)
            teacher=original.repeat(2,1).detach();inputs=torch.cat((restored.repeat(2,1,1),torch.flip(restored,[-1])))
            diagnostics={}
            for arm,model in models.items():
                model.zero_grad(set_to_none=True);raw,logits=model(inputs)
                terms=core.compute(raw[:4],logits[:4],y,teacher,domains,raw[4:],logits[4:],outside_domains,arm)
                if not torch.allclose(logits[:2].detach(),original,rtol=old.identity.LOGIT_RTOL,atol=old.identity.LOGIT_ATOL):
                    raise ValueError('Both contrast students must independently preserve original initial logits')
                terms['total'].backward();g=old._gradients(model)
                if any(g[k]<=0 for k in ('head','body','shallow','middle')):raise ValueError('Contrast smoke must reach default head and shallow/middle/body')
                old.identity._unchanged(model,snapshots[arm],'No-step contrast smoke')
                if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers[arm].items()):raise ValueError('Contrast smoke changed frozen buffers')
                diagnostics[arm]=dict(gradient_norms=g,CE=float(terms['CE'].detach().cpu()),KD=float(terms['KD'].detach().cpu()),
                    OE=float(terms['OE'].detach().cpu()),contrast=float(terms['contrast'].detach().cpu()),
                    known_only_contrast=float(terms['known_only_contrast'].detach().cpu()),with_OE_contrast=float(terms['with_OE_contrast'].detach().cpu()),
                    positive_pair_count=terms['positive_pair_count'],selected_OE_negative_pair_count=terms['selected_OE_negative_pair_count'])
            return dict(status='PASS',actual_original_F0_DURA=True,actual_original_F0_equal=True,new_loss_gradients_verified=True,
                initial_logit_max_abs_difference=float((initial-original).abs().max().cpu()),arms=diagnostics,
                synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_forward_rows=14,
                synthetic_r03_forward_rows=0,synthetic_PCD_forward_rows=0,optimizer_updates=0,real_data_reads=0,
                Sinc_gradients_none=True,parameters_buffers_unchanged=True)
    finally:
        for arm,model in models.items():
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n,v in model.named_buffers():v.copy_(buffers[arm][n])


def _loop(c,arm,model,roles,device,folder,check):
    lm=old._metadata(roles['L']['metadata'],roles['L']['ids'],True);om=old._metadata(roles['OE']['metadata'],roles['OE']['ids'],False)
    sampler=native.SourceSampler(lm,om);values={k:old._runtime(v,device) for k,v in roles.items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    trace=[];usage={k:set() for k in ('L','OE')}
    with _indices_path(c).open(encoding='utf-8') as matched,(folder/'steps.jsonl').open('x',encoding='utf-8') as stream:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);line=matched.readline();reference=json.loads(line) if line else {}
            if reference.get('step')!=step or any(reference.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):
                raise ValueError('Source L/OE indices differ from the fixed CPAIR step')
            li=old.identity._tensor(sample['L'],device,torch.long);oi=old.identity._tensor(sample['OE'],device,torch.long)
            L,OE=values['L'],values['OE'];optimizer.zero_grad(set_to_none=True)
            raw,logits=model(torch.cat((L['iq'][li],OE['iq'][oi])))
            terms=core.compute(raw[:192],logits[:192],L['metadata']['y'][li],L['teacher_logits'][li],L['metadata']['domain'][li],
                raw[192:],logits[192:],OE['metadata']['domain'][oi],arm)
            if terms['domain_count']!=15:raise ValueError('Every step must retain all fifteen public source domains')
            if not bool(torch.isfinite(terms['total'])):raise FloatingPointError('Nonfinite contrast native objective')
            terms['total'].backward();g=old._gradients(model);unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite contrast native gradient norm')
            optimizer.step()
            for k in usage:usage[k].update(map(int,sample[k]))
            scalars={k:float(v.detach().cpu()) if torch.is_tensor(v) else v for k,v in terms.items()}
            row=dict(scalars,step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),
                source_indices_equal=True,source_indices_match_CPAIR=True,reference_arm='CPAIR',reference_step=step,
                CE_weight=1.,KD_weight=1.,KD_temperature=2.,OE_weight=1.,energy_temperature=1.,PL_weight=0.,PL=0.,
                gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,physical_clip_calls=0,
                unused_gradient_parameters=unused,optimizer_updates=step,identity_head_updates=step,Sinc_optimizer_steps=0,
                L_count=192,OE_count=64,student_forward_rows=256,cumulative_student_forward_rows=256*step,
                source_L_exposures=192*step,source_OE_exposures=64*step,source_Us_reads=0,target_U_reads=0,CAL_reads=0,source_Q_reads=0)
            trace.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if step==1 or step%100==0 or step==STEPS:
                check();print('GROUND_CONTRAST_PROGRESS',arm,step,STEPS,row['total'],row['contrast'],row['with_OE_minus_known'],flush=True)
        if matched.readline():raise ValueError('The fixed CPAIR reference contains extra steps')
    return trace,{k:len(v) for k,v in usage.items()}


def _identity(c,arm,pm,parent,ground,handles,reference):
    if arm not in ARMS:raise ValueError('Only KCON/OECON may be loaded')
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/'rows'/arm/'final.pt'),updates=STEPS,
        reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,
        identity_head_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,projection_updates=0,
        initial_native_logits_equal_original_F0=True,original_parent_updates=11100,cumulative_updates=11100+STEPS,
        original_parent_identity=parent,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        source_centered_reference_identity=reference,source_tail_reference_identity=reference['source_tail_reference_identity'],
        source_ground_reference_identity=reference['source_ground_reference_identity'],ground_data_identity=ground,source_handles=handles,
        same_input_teacher=True,source_indices_match_CPAIR=True,matched_source_arm='CPAIR',source_indices_reference=str(_indices_path(c)),
        source_only=True,ground_clean_training=True,original_F0_source_Us_ancestor=True,source_U_s_used=False,
        source_Us_reads=0,source_U_s_reads=0,source_U_s_exposures=0,target_U_used=False,target_U_reads=0,target_Q_reads=0,source_U_GT_reads=0,
        input_path='G(clean)',BN_Dropout_eval=True,initialization='independent_complete_original_F0_E50',
        deployment_state='complete_reader_only',unused_wrapper_parameters_not_deployed=667729,
        OE_loss_reduction='equal_public_RX_day_domains',evidence='logsumexp_minus_mean_minus_log6',
        contrast_weight=core.CONTRAST_WEIGHT,contrast_temperature=core.TEMPERATURE,OE_contrast_negatives=(arm=='OECON'),target_REG=False)


def _ancestry(parent,ground,handles,reference):
    return dict(original_parent_identity=parent,source_centered_reference_identity=reference,
        source_tail_reference_identity=reference['source_tail_reference_identity'],source_ground_reference_identity=reference['source_ground_reference_identity'],
        ground_data_identity=ground,source_handles=handles)


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable contrast source output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,ground,handles,reference=_reference(c);b=_backend(c,device)
    if b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity'] or b['model_identity']!=pm['model_identities']['E0']:
        raise ValueError('Original frozen source F0/r03/G ancestry differs')
    originals=(b['reader'],b['model'],b['dura'][0]);snapshots=[(m,_snapshot(m)) for m in originals]
    if any(p.requires_grad or p.grad is not None for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):
        raise ValueError('All original models must remain frozen eval without gradients')
    def check():
        for model,before in snapshots:
            old.identity._unchanged(model,before,'Frozen contrast source ancestors')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('An original frozen model received gradients')
    models={};initials={};parents={};original=_snapshot(b['reader'])
    for arm in ARMS:
        model,parent=_initial_model(c,pm,device);initial=_snapshot(model)
        if set(original)!={n.removeprefix('reader.') for n in initial} or any(not torch.equal(v,initial['reader.'+n]) for n,v in original.items()):
            raise ValueError('Each contrast student must copy complete original F0 including its default head')
        models[arm]=model;initials[arm]=initial;parents[arm]=parent
    if parents[ARMS[0]]!=parents[ARMS[1]]:raise ValueError('Matched contrast students require the same original parent')
    smoke=_smoke(models,b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('new_loss_gradients_verified') is not True:raise ValueError('Actual contrast checkpoint smoke must pass before source IQ')
    print('GROUND_CONTRAST_CHECKPOINT_SMOKE_PASS',flush=True)
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);roles={};rows={};identities={};traces={}
    try:
        for role in ('L','OE'):roles[role]=old._role(_ground_train_config(c),pm,role,handles[role])
        if set(roles['L']['ids'].tolist())&set(roles['OE']['ids'].tolist()):raise ValueError('Original source L/OE records overlap')
        for arm,model in models.items():
            buffers={n:v.detach().clone() for n,v in model.named_buffers()};folder=out/'rows'/arm;folder.mkdir(parents=True)
            trace,unique=_loop(c,arm,model,roles,device,folder,check);check();old._changed(model,initials[arm]);traces[arm]=trace
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Contrast student buffers changed')
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite complete native checkpoint required')
            i=_identity(c,arm,pm,parents[arm],ground,deepcopy(handles),reference);model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,**_ancestry(parents[arm],ground,deepcopy(handles),reference)),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,
                updates=STEPS,identity_head_updates=STEPS,source_only=True,ground_clean_training=True,same_input_teacher=True,
                source_U_s_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,
                student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,source_indices_match_CPAIR=True,
                source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,student_forward_rows=256*STEPS,
                unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;identities[arm]=i
        if any(a[k]!=b[k] for a,b in zip(traces[ARMS[0]],traces[ARMS[1]]) for k in ('L_indices','OE_indices')):
            raise ValueError('KCON/OECON source sampling must match step by step')
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities=identities,
            steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_L_OE_sampling_matched=True,source_sampling_matched=True,
            source_indices_match_CPAIR=True,source_indices_reference=str(_indices_path(c)),matched_source_arm='CPAIR',same_input_teacher=True,
            source_only=True,ground_clean_training=True,new_target_UDA_updates=0,original_models_unchanged=True,
            all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,all_student_heads_changed=True,all_student_bodies_changed=True,
            all_students_initialized_from_original_F0=True,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
            **_ancestry(parents[ARMS[0]],ground,handles,reference),original_F0_source_Us_ancestor=True,
            reader_parameters=232993,trainable_parameters_per_student=232945,default_head_parameters=960,unused_wrapper_parameters_not_deployed=667729,
            source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],cached_L_teacher_logits_reads=COUNTS['L'],
            real_student_forward_rows=512*STEPS,real_student_forward_rows_by_arm={arm:256*STEPS for arm in ARMS},
            real_parent_signal_forward_rows=0,real_F_forward_rows=0,real_F_teacher_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            L_sampling_seed=native.L_SEED,OE_sampling_seed=native.OE_SEED,
            contrast_weight=core.CONTRAST_WEIGHT,contrast_temperature=core.TEMPERATURE,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=2*STEPS),
            **{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'},
            source_bank_deployed=False,source_teacher_deployed=False,formal_target_accessed=False,target_REG=False,
            full_goal_status='ACTIVE',final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():old._data().close(role)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only complete KCON/OECON native students may load')
    root=Path(c['output_root'])
    try:
        done=_read(root/'complete.json');rows={a:_read(root/'rows'/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both completed contrast native students required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),
        steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_L_OE_sampling_matched=True,
        source_sampling_matched=True,source_indices_match_CPAIR=True,matched_source_arm='CPAIR',same_input_teacher=True,
        source_only=True,ground_clean_training=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_buffers_unchanged=True,all_student_heads_changed=True,all_student_bodies_changed=True,
        real_student_forward_rows=512*STEPS,new_target_UDA_updates=0,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows or set(done.get('identities',{}))!=set(ARMS):
        raise ValueError('Complete source-only contrast run required')
    for a,row in rows.items():
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=a,
            updates=STEPS,identity_head_updates=STEPS,source_only=True,ground_clean_training=True,same_input_teacher=True,
            source_U_s_used=False,initial_state_equal_to_original_F0=True,student_Sinc_unchanged=True,student_body_changed=True,
            student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,source_indices_match_CPAIR=True,
            final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or row.get('identity')!=done['identities'][a]:raise ValueError('Both complete frozen contrast rows required')
    pm,ground,handles,reference=_reference(c);model,parent=_initial_model(c,pm,device)
    i=_identity(c,arm,pm,parent,ground,handles,reference);ancestry=_ancestry(parent,ground,handles,reference)
    if any(done.get(k)!=v for k,v in ancestry.items()) or rows[arm]['identity']!=i:raise ValueError('Own contrast student and unchanged source/control ancestry differ')
    payload=torch.load(root/'rows'/arm/'final.pt',map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in dict(schema=SCHEMA,config=c,identity=i,**ancestry).items()):raise ValueError('Strict complete contrast checkpoint ancestry required')
    network=payload.get('network');initial=_snapshot(model)
    if (not isinstance(network,dict) or set(network)!=set(initial)
            or any(not torch.is_tensor(v) or v.dtype!=initial[k].dtype or v.shape!=initial[k].shape or not bool(torch.isfinite(v).all()) for k,v in network.items())):
        raise ValueError('Full finite native reader and unchanged default-head structure required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{n for n in initial if n.startswith(native.SINC_PREFIX)}
    if any(not torch.equal(initial[n],network[n]) for n in fixed):raise ValueError('Frozen native Sinc/persistent buffers differ')
    model.load_state_dict(network,strict=True);old._changed(model,initial);model.eval().requires_grad_(False)
    return model,i


def main():
    print(json.dumps(dict(event='GROUND_CONTRAST_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    train(_read(a.config),a.device)


if __name__=='__main__':main()
