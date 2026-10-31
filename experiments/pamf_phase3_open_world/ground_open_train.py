"""Source ground-clean native training: matched supervised and ordinary PL."""
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
from . import logit_energy_train as base

core,identity=base.core,base.identity
RUN_ID='20261101-phase3-ground-open-s2026092701-r03'
SCHEMA='pamf_phase3_ground_open_train_v1'
STATUS='GROUND_OPEN_TRAIN_COMPLETE'
ROW_STATUS='GROUND_OPEN_STUDENT_FROZEN'
ARMS=('SUP','SSL')
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
US_COUNT=56700
US_SEED=2026110102
REFERENCE_CODE=base.REFERENCE_CODE
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','data_root','output_root','steps','lr'}
ZEROS=('target_U_reads','U_ID_metadata_reads','target_Q_reads','validation_reads','novel_validation_reads','CAL_reads','source_Q_reads',
       'private_truth_reads','U_truth_reads','OE_truth_reads','source_U_GT_reads','Sinc_optimizer_steps','physical_clip_calls','source_density_fit_updates','threshold_fit_updates')
_read,_write,_snapshot=base._read,base._write,base._snapshot
_reference,_backend,_load_PCD=base._reference,base._backend,base._load_PCD
_initial_model,_gradients,_changed=base._initial_model,base._gradients,base._changed


def _data():
    from . import ground_open_data
    return ground_open_data


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('data_root','output_root'))):
        raise ValueError('Exact seven-field ground source config required; no CAL/Q/truth/U or budget options')
    r=identity.reg.fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Original completed fresh f4b5 ancestry required')
    source=identity._source_config(c);out=Path(c['output_root']).resolve();data=Path(c['data_root']).resolve()
    roots=[data,r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],source['output_root'],source['IQ_root'],source['native_config']['source_output']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable ground training output overlaps an input')
    return c


def _ground_metadata(c,pm,with_us=False):
    root=Path(c['data_root']);m=_read(root/'complete.json');module=_data()
    expected=dict(schema=module.SCHEMA,status='GROUND_OPEN_DATA_COMPLETE',run_id=RUN_ID,code_commit=c['code_commit'],source_only=True,
        phase3_data_status='VALIDATED_ONCE',capsule_id=RUN_ID+'-source-clean-v1',split_id='sourceV-half-y-rx-day-s2026110101',
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'])
    if any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Current source ground-clean G(clean) data and original F0/DURA ancestry required')
    source=m.get('source_config',{});native=identity._source_config(c)['native_config'];reader=source.get('reader_config',{})
    if (source.get('source_contract')!=native['source_contract'] or any(reader.get(k)!=native[k] for k in ('source_output','base_release','model_seed'))):
        raise ValueError('Ground data source constructor/contract differs from the original F0')
    handles={}
    for role in (('L','OE','Us') if with_us else ('L','OE')):
        value=_read(root/'roles'/role/'manifest.json');n=US_COUNT if role=='Us' else COUNTS[role]
        required=dict(schema=module.SCHEMA,status='GROUND_OPEN_ROLE_COMPLETE',run_id=RUN_ID,code_commit=c['code_commit'],role=role,count=n,iq_shape=[n,2,256],precision='FP32',
            input_path='G(clean)',source_only=True,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],capsule_id=m['capsule_id'],split_id=m['split_id'])
        if any(value.get(k)!=v for k,v in required.items()):raise ValueError('Exact source G(clean) role metadata required: '+role)
        handles[role]=value
    keys=('schema','status','run_id','code_commit','capsule_id','split_id','parent_F0_identity','dura_identity','loaded_dura_identity')
    return {k:m[k] for k in keys},handles


def _metadata(meta,ids,known):
    if not isinstance(meta,dict) or set(meta)!={'ids','rx_i','day_i','domain'}|({'y'} if known else set()) or not np.array_equal(meta['ids'],ids):
        raise ValueError('Only aligned opaque IDs, public source RX/day/domain and L labels are permitted')
    return core._metadata({k:v for k,v in meta.items() if k!='ids'},known)


def _role(c,pm,role,handle):
    if role not in ('L','OE','Us'):raise ValueError('Training may load only L/OE and SSL Us')
    a=_data().load_role(c['data_root'],role);n=US_COUNT if role=='Us' else COUNTS[role]
    try:
        fields={'manifest','ids','metadata','iq'}|({'teacher_logits'} if role=='L' else set())
        if (set(a)!=fields or a['manifest']!=handle or a['ids'].shape!=(n,) or a['ids'].dtype.kind!='U' or len(set(a['ids'].tolist()))!=n or np.any(np.char.strip(a['ids'])=='')
                or a['iq'].shape!=(n,2,256) or a['iq'].dtype!=np.float32 or not np.isfinite(a['iq']).all()):raise ValueError('Exact finite source G(clean) records required')
        _metadata(a['metadata'],a['ids'],role=='L')
        if role=='L' and (a['teacher_logits'].shape!=(n,6) or a['teacher_logits'].dtype!=np.float32 or not np.isfinite(a['teacher_logits']).all()):raise ValueError('Original F0 same-G(clean) L teacher cache required')
        return a
    except BaseException:
        _data().close(a);raise


class UsSampler:
    def __init__(self,metadata):
        if not isinstance(metadata,dict) or 'ids' not in metadata:raise ValueError('Public Us metadata required')
        a=_metadata(metadata,metadata['ids'],False);self.domains={d:np.flatnonzero(a['domain']==d) for d in range(15)}
        self.rng=np.random.default_rng(US_SEED)
    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive Us optimization step required')
        blocks=[self.rng.choice(self.domains[d],8,replace=True) for d in range(15)]
        blocks.append(np.asarray([self.rng.choice(self.domains[int(d)]) for d in self.rng.integers(15,size=8)],np.int64))
        return np.concatenate(blocks).astype(np.int64)


def source_loss(logits,y,teacher,domains,OE,OE_domains):
    return core.compute(logits,y,teacher,domains,OE,OE_domains,'RXPAIR')


def pseudo_loss(logits):
    core._logits(logits);confidence,pseudo=F.softmax(logits.detach(),dim=1).max(1);mask=confidence>=.95
    loss=(F.cross_entropy(logits,pseudo,reduction='none')*mask).sum()/len(logits)
    accepted=int(mask.sum().cpu());hist=torch.bincount(pseudo[mask],minlength=6).cpu().tolist()
    return loss,dict(count=len(logits),accepted=accepted,acceptance_rate=accepted/len(logits),pseudo_histogram=hist,
        confidence_mean=float(confidence.mean().cpu()),confidence_min=float(confidence.min().cpu()),confidence_max=float(confidence.max().cpu()),threshold=.95)


def _smoke(model,b,pcd,device):
    from .star_open_feature_export import _synthetic
    before=_snapshot(model);buffers={n:v.detach().clone() for n,v in model.named_buffers()};trials={}
    try:
        with torch.random.fork_rng(devices=identity.native._cuda_devices(model)):
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                restored=b['dura'][1](_synthetic(device))[0].detach().clone();original=b['reader'].classify_features(b['reader'].features(restored))
                r03=b['model'](restored);final=pcd(restored);_,initial=model(restored)
            if (original.shape!=(2,6) or r03.shape!=(2,160) or final.shape!=(2,160) or any(v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in (restored,original,r03,final))
                    or not torch.allclose(initial,original,rtol=identity.LOGIT_RTOL,atol=identity.LOGIT_ATOL)):
                raise ValueError('Actual original F0/r03/PCD/G and independent native initialization smoke required')
            for arm in ARMS:
                model.zero_grad(set_to_none=True);_,logits=model(torch.cat([restored]*(3 if arm=='SSL' else 2)))
                domains=torch.tensor([0,1],device=device);terms=source_loss(logits[:2],domains,original.detach(),domains,logits[2:4],domains)
                pl,info=pseudo_loss(logits[4:]) if arm=='SSL' else (logits.new_zeros(()),dict(accepted=0))
                loss=terms['total']+pl;loss.backward();g=_gradients(model)
                if any(g[k]<=0 for k in ('head','body','shallow','middle')):raise ValueError('Actual head/shallow/middle/body smoke gradients must be finite nonzero')
                identity._unchanged(model,before,'No-step ground native smoke')
                if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('No-step ground smoke changed buffers')
                trials[arm]=dict(status='PASS',loss=float(loss.detach().cpu()),gradient_norms=g,PL_accepted=info['accepted'])
            return dict(status='PASS',actual_F0_r03_DURA_PCD_checkpoints=True,actual_original_F0_equal=True,initial_logit_max_abs_difference=float((initial-original).abs().max().cpu()),
                trials=trials,parameters_buffers_unchanged=True,Sinc_gradients_none=True,optimizer_updates=0,real_data_reads=0,
                synthetic_F_forward_rows=2,synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_forward_rows=12)
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,v in model.named_buffers():v.copy_(buffers[n])


def _runtime(a,device):
    meta=_metadata(a['metadata'],a['ids'],'teacher_logits' in a)
    return dict(iq=identity._tensor(a['iq'],device),metadata={k:identity._tensor(v,device,torch.long) for k,v in meta.items()},
        **({'teacher_logits':identity._tensor(a['teacher_logits'],device)} if 'teacher_logits' in a else {}))


def _loop(c,arm,model,roles,device,folder,check):
    lm=_metadata(roles['L']['metadata'],roles['L']['ids'],True);om=_metadata(roles['OE']['metadata'],roles['OE']['ids'],False)
    sampler=core.SourceSampler(lm,om);us=UsSampler(roles['Us']['metadata']) if arm=='SSL' else None
    values={k:_runtime(roles[k],device) for k in ('L','OE')};opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    matched=(Path(c['output_root'])/'rows/SUP/steps.jsonl').open(encoding='utf-8') if arm=='SSL' else None
    usage={k:set() for k in ('L','OE','Us')};trace=[]
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as stream:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step);ui=us.sample(step) if us is not None else np.asarray([],np.int64)
                if matched is not None:
                    line=matched.readline();previous=json.loads(line) if line else {}
                    if previous.get('step')!=step or any(previous.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched SUP L/OE sampling differs')
                li=identity._tensor(sample['L'],device,torch.long);oi=identity._tensor(sample['OE'],device,torch.long);L,OE=values['L'],values['OE']
                inputs=[L['iq'][li],OE['iq'][oi]]
                if us is not None:inputs.append(identity._tensor(roles['Us']['iq'][ui],device))
                opt.zero_grad(set_to_none=True);_,logits=model(torch.cat(inputs));terms=source_loss(logits[:192],L['metadata']['y'][li],L['teacher_logits'][li],L['metadata']['domain'][li],logits[192:256],OE['metadata']['domain'][oi])
                pl,info=pseudo_loss(logits[256:]) if us is not None else (logits.new_zeros(()),dict(count=0,accepted=0,acceptance_rate=None,pseudo_histogram=[0]*6,confidence_mean=None,confidence_min=None,confidence_max=None,threshold=.95))
                loss=terms['total']+pl
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite ground CE/KD/RXPAIR/PL objective')
                loss.backward();g=_gradients(model);unused=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite native gradient norm')
                opt.step()
                for k in ('L','OE'):usage[k].update(map(int,sample[k]))
                usage['Us'].update(map(int,ui));n=384 if us is not None else 256
                row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),Us_indices=ui.tolist(),source_indices_equal=True,
                    total=float(loss.detach().cpu()),source_loss=float(terms['total'].detach().cpu()),PL=float(pl.detach().cpu()),
                    **{k:float(v.detach().cpu()) for k,v in terms.items() if torch.is_tensor(v) and k!='total'},pair_count=terms['pair_count'],
                    CE_weight=1.,KD_weight=1.,KD_temperature=2.,OE_weight=1.,PL_weight=1.,energy_temperature=1.,
                    PL_count=info['count'],PL_accepted=info['accepted'],PL_acceptance_rate=info['acceptance_rate'],PL_threshold=.95,PL_denominator=info['count'],
                    pseudo_histogram=info['pseudo_histogram'],confidence_mean=info['confidence_mean'],confidence_min=info['confidence_min'],confidence_max=info['confidence_max'],pseudo_label_source='current_student_detached',
                    gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,unused_gradient_parameters=unused,optimizer_updates=step,identity_head_updates=step,Sinc_optimizer_steps=0,physical_clip_calls=0,
                    L_count=192,OE_count=64,Us_count=len(ui),student_forward_rows=n,cumulative_student_forward_rows=n*step,source_L_exposures=192*step,source_OE_exposures=64*step,source_U_s_exposures=len(ui)*step,target_U_reads=0,CAL_reads=0,source_Q_reads=0)
                trace.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if step==1 or step%100==0 or step==STEPS:check();print('GROUND_OPEN_PROGRESS',arm,step,STEPS,row['total'],row['PL'],row['PL_accepted'],flush=True)
        if matched is not None and matched.readline():raise ValueError('SUP trace contains extra updates')
    finally:
        if matched is not None:matched.close()
    return trace,{k:len(v) for k,v in usage.items()}


def _identity(c,arm,pm,parent,data_identity,handles):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/'rows'/arm/'final.pt'),updates=STEPS,
        reader_parameters=232993,trainable_parameters=232945,default_head_parameters=960,frozen_Sinc_parameters=48,Sinc_frozen=True,identity_head_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,
        projection_updates=0,initial_native_logits_equal_original_F0=True,original_parent_updates=11100,cumulative_updates=11100+STEPS,original_parent_identity=parent,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],ground_data_identity=data_identity,source_handles=handles,
        source_only=True,ground_clean_training=True,source_U_s_used=arm=='SSL',source_U_s_reads=US_COUNT if arm=='SSL' else 0,source_U_s_exposures=128*STEPS if arm=='SSL' else 0,
        target_U_used=False,target_U_reads=0,target_Q_reads=0,source_U_GT_reads=0,input_path='G(clean)',BN_Dropout_eval=True,initialization='independent_complete_original_F0_E50',deployment_state='complete_reader_only',unused_wrapper_parameters_not_deployed=667729,target_REG=False)


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable ground-open training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,_,_,_=_reference(c);ground,handles=_ground_metadata(c,pm);b=_backend(c,device);pcd,pcd_identity=_load_PCD(c,device)
    if b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity'] or b['model_identity']!=pm['model_identities']['E0'] or pcd_identity!=pm['model_identities']['TARGET_PCD']:raise ValueError('Original frozen F0/r03/PCD/G ancestry differs')
    originals=(b['reader'],b['model'],pcd,b['dura'][0]);snapshots=[(m,_snapshot(m)) for m in originals]
    if any(p.requires_grad or p.grad is not None for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):raise ValueError('Original models must remain frozen eval')
    def check():
        for model,before in snapshots:
            identity._unchanged(model,before,'Original F0/r03/PCD/G')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen original model received gradients')
    model,parent=_initial_model(c,pm,device);initial=_snapshot(model);original=_snapshot(b['reader'])
    if set(original)!={n.removeprefix('reader.') for n in initial} or any(not torch.equal(v,initial['reader.'+n]) for n,v in original.items()):raise ValueError('Complete original F0 native initialization required')
    smoke=_smoke(model,b,pcd,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_original_F0_equal') is not True:raise ValueError('Actual checkpoint no-query smoke must pass before source IQ')
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke);roles={};rows={}
    try:
        for k in ('L','OE'):roles[k]=_role(c,pm,k,handles[k])
        if set(roles['L']['ids'].tolist())&set(roles['OE']['ids'].tolist()):raise ValueError('L/OE records overlap')
        for arm in ARMS:
            if arm=='SSL':
                ground,ssl_handles=_ground_metadata(c,pm,with_us=True);handles=ssl_handles;roles['Us']=_role(c,pm,'Us',handles['Us'])
                if set(roles['Us']['ids'].tolist())&(set(roles['L']['ids'].tolist())|set(roles['OE']['ids'].tolist())):raise ValueError('Us overlaps labeled or exposure roles')
                model,other_parent=_initial_model(c,pm,device)
                if other_parent!=parent:raise ValueError('Both ground students need the same original parent')
            identity._unchanged(model,initial,'Independent original-F0 initialization');buffers={n:v.detach().clone() for n,v in model.named_buffers()}
            folder=out/'rows'/arm;folder.mkdir(parents=True);trace,unique=_loop(c,arm,model,roles,device,folder,check);check();_changed(model,initial)
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Frozen ground student buffers changed')
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite native checkpoint required')
            i=_identity(c,arm,pm,parent,ground,deepcopy(handles));model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,original_parent_identity=parent,ground_data_identity=ground,source_handles=deepcopy(handles)),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,updates=STEPS,identity_head_updates=STEPS,
                source_only=True,ground_clean_training=True,source_U_s_used=arm=='SSL',source_U_s_reads=US_COUNT if arm=='SSL' else 0,initial_state_equal_to_original_F0=True,
                student_Sinc_unchanged=True,student_body_changed=True,student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,
                source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,source_U_s_exposures=(128*STEPS if arm=='SSL' else 0),student_forward_rows=(384 if arm=='SSL' else 256)*STEPS,
                unique_training_rows=unique,PL_accepted_exposures=sum(t['PL_accepted'] for t in trace),PL_effective_steps=sum(t['PL_accepted']>0 for t in trace),
                accepted_pseudo_histogram=np.asarray([t['pseudo_histogram'] for t in trace],np.int64).sum(0).tolist(),first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('GROUND_OPEN_ARM_COMPLETE',arm,flush=True);del model
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={k:v['identity'] for k,v in rows.items()},
            steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_L_OE_sampling_matched=True,source_sampling_matched=True,source_only=True,ground_clean_training=True,new_target_UDA_updates=0,
            original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,all_student_heads_changed=True,all_student_bodies_changed=True,
            all_students_initialized_from_original_F0=True,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],original_parent_identity=parent,ground_data_identity=ground,
            reader_parameters=232993,trainable_parameters_per_student=232945,default_head_parameters=960,unused_wrapper_parameters_not_deployed=667729,
            source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],source_U_s_reads=US_COUNT,source_U_s_exposures=128*STEPS,cached_L_teacher_logits_reads=COUNTS['L'],
            real_student_forward_rows=640*STEPS,real_student_forward_rows_by_arm=dict(SUP=256*STEPS,SSL=384*STEPS),real_parent_signal_forward_rows=0,real_F_forward_rows=0,real_F_teacher_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            L_sampling_seed=core.L_SEED,OE_sampling_seed=core.OE_SEED,Us_sampling_seed=US_SEED,optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=2*STEPS),
            **{k:v for k,v in smoke.items() if k.startswith('synthetic_') or k=='G_synthetic_forward_rows'},source_bank_deployed=False,source_teacher_deployed=False,formal_target_accessed=False,target_REG=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():_data().close(role)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only SUP/SSL native ground students may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/'rows'/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both completed ground students required') from exc
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),steps_per_arm=STEPS,updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_L_OE_sampling_matched=True,source_only=True,ground_clean_training=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        all_student_heads_changed=True,all_student_bodies_changed=True,real_student_forward_rows=640*STEPS,new_target_UDA_updates=0,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in required.items()) or done.get('rows')!=rows:raise ValueError('Complete source-only matched ground training required')
    pm,_,_,_=_reference(c);ground,handles=_ground_metadata(c,pm,with_us=arm=='SSL');model,parent=_initial_model(c,pm,device)
    if done.get('original_parent_identity')!=parent or done.get('ground_data_identity')!=ground:raise ValueError('Original F0/G(clean) training ancestry differs')
    for name,row in rows.items():
        expected=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=name,updates=STEPS,identity_head_updates=STEPS,
            source_only=True,ground_clean_training=True,source_U_s_used=name=='SSL',source_U_s_reads=US_COUNT if name=='SSL' else 0,initial_state_equal_to_original_F0=True,
            student_Sinc_unchanged=True,student_body_changed=True,student_head_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in expected.items()) or row.get('identity')!=done.get('identities',{}).get(name):raise ValueError('Complete own native ground row required')
    i=_identity(c,arm,pm,parent,ground,handles)
    if rows[arm]['identity']!=i:raise ValueError('Own complete ground-clean identity differs')
    payload=torch.load(root/'rows'/arm/'final.pt',map_location='cpu',weights_only=False)
    if any(payload.get(k)!=v for k,v in dict(schema=SCHEMA,config=c,identity=i,original_parent_identity=parent,ground_data_identity=ground,source_handles=handles).items()):raise ValueError('Strict ground native checkpoint lineage required')
    network=payload.get('network');initial=_snapshot(model)
    if not isinstance(network,dict) or set(network)!=set(initial) or any(not torch.is_tensor(v) or v.dtype!=initial[k].dtype or v.shape!=initial[k].shape or not bool(torch.isfinite(v).all()) for k,v in network.items()):raise ValueError('Full finite native reader/head checkpoint required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{n for n in initial if n.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[n],network[n]) for n in fixed):raise ValueError('Frozen Sinc/persistent buffers differ')
    model.load_state_dict(network,strict=True);_changed(model,initial);model.eval().requires_grad_(False);return model,i


def main():
    print(json.dumps(dict(event='GROUND_OPEN_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
