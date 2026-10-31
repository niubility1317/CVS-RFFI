"""Matched source replay with frozen-F0 pseudo labels on the original source Us."""
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

RUN_ID='20261031-phase3-source-us-replay-s2026092701-r01'
SCHEMA='pamf_phase3_source_us_replay_train_v1'
STATUS='SOURCE_US_REPLAY_TRAINING_COMPLETE'
ARM='SOURCE_US'
STEPS=444
COUNTS=dict(L=6300,OE=6134)
US_COUNT=56700
US_BATCH=128
US_SEED=2026103150
AUG_SEED=2026103151
US_CAPSULE_ID='pamf-source-Us392005-G10-MIX01-20261031-r01'
US_SPLIT_ID='392005-Us-source5rx-3day-eq1'
US_SCHEMA='pamf_phase3_source_us_cache_role_v1'
US_STATUS='SOURCE_US_G10_ROLE_COMPLETE'
REFERENCE_CODE='4273e88392abe0717c1bd94fca1370d19c67cbb3'
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','Us_public_root','Us_capsule_id','Us_split_id','Us_expected_count','output_root','steps','lr'}
ZEROS=('native_updates','identity_head_updates','Sinc_optimizer_steps','physical_clip_calls','target_U_reads','target_Q_reads','validation_reads','novel_validation_reads','private_truth_reads','Us_ground_truth_reads','U_truth_reads')
_read,_write,_snapshot=body._read,body._write,body._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or type(c.get('Us_expected_count')) is not int or c['Us_expected_count']!=US_COUNT
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('Us_public_root','Us_capsule_id','Us_split_id','output_root'))):
        raise ValueError('Exact source-Us replay configuration required; no truth, target U/Q or raw locator')
    p=reference.validate_config(c['reference_config']);us=Path(c['Us_public_root']);t=p['reference_config']['train_config']
    if (p['code_commit']!=REFERENCE_CODE or us.name!='U_s' or c['Us_capsule_id']!=US_CAPSULE_ID or c['Us_split_id']!=US_SPLIT_ID
            or t['L_seed']!=2026103017 or t['OE_seed']!=2026103018):raise ValueError('Original SOURCE444 and the original split392005 source-Us handle required')
    out=Path(c['output_root']).resolve()
    roots=[c['Us_public_root'],p['output_root'],p['train_config']['output_root'],t['output_root'],t['IQ_root'],t['feature_root'],t['control_root']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable replay output overlaps an input')
    return c


def _source(c):return dict(reference_config=c['reference_config']['reference_config'])
def _reference(c):return body._reference(_source(c))
def _backend(c,device):return body._backend(_source(c),device)


def _source_control(c,pm):
    t=c['reference_config']['train_config'];root=Path(t['output_root']);row=_read(root/'SOURCE/complete.json')
    expected=dict(schema=body.SCHEMA,status=body.ROW_STATUS,run_id=body.RUN_ID,code_commit=reference.TRAIN_CODE,arm='SOURCE',config=t,updates=STEPS,
        source_only=True,target_U_used=False,target_U_reads=0,U_manifest=None,initial_state_equal_to_parent=True,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,original_models_unchanged=True)
    i=row.get('identity',{})
    if (any(row.get(k)!=v for k,v in expected.items()) or any(i.get(k)!=v for k,v in dict(run_id=body.RUN_ID,code_commit=reference.TRAIN_CODE,arm='SOURCE',updates=STEPS,source_only=True,target_U_used=False,target_U_reads=0).items())
            or i.get('parent_signal_identity')!=pm['model_identity'] or i.get('parent_F0_identity')!=pm['parent_F0_identity'] or i.get('dura_identity')!=pm['dura_identity']):
        raise ValueError('Frozen source-only a336 SOURCE444 ancestry required')
    return row,root/'SOURCE/steps.jsonl'


def _public(c):
    from .source_us_cache import load_public
    return load_public(c['Us_public_root'],c['Us_capsule_id'],c['Us_split_id'],c['Us_expected_count'])


def _close_public(value):
    if value is not None:
        for name in ('received','restored'):
            a=value.get(name)
            if hasattr(a,'_mmap'):a._mmap.close()


def _check_Us(c,b,value,source_ids):
    if not isinstance(value,dict) or set(value)!={'manifest','ids','received','restored','metadata'}:raise ValueError('Exact public source Us bundle required')
    m=value['manifest'];ids=value['ids'];meta=value['metadata'];x=value['restored']
    expected=dict(schema=US_SCHEMA,status=US_STATUS,run_id=RUN_ID,role='U_s',count=US_COUNT,data_status='VALIDATED_ONCE',capsule_id=c['Us_capsule_id'],split_id=c['Us_split_id'],
        input_path='cached_P1_only',received_input_path='P0',restored_input_path='P1',source_only=True,parent_F0_identity=b['reader_identity'],dura_identity=b['dura_identity'])
    if (any(m.get(k)!=v for k,v in expected.items()) or ids.shape!=(US_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=US_COUNT
            or np.any(np.char.strip(ids)=='') or source_ids&set(ids.tolist()) or x.shape!=(US_COUNT,2,256) or x.dtype!=np.float32
            or set(meta)!={'rx_i','day_i','domain'} or any(a.shape!=(US_COUNT,) or a.dtype!=np.int64 for a in meta.values())
            or not np.isin(meta['rx_i'],[1,3,4,6,8]).all() or not np.isin(meta['day_i'],[1,2,3]).all()
            or not np.array_equal(meta['domain'],np.searchsorted([1,3,4,6,8],meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Source Us must expose only matching frozen P1, opaque IDs and RX/day/domain; no y/TX')


class PermutationStream:
    def __init__(self,count,seed):
        if type(count) is not int or count<1:raise ValueError('Positive source Us count required')
        self.count=count;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(count);self.position=0

    def sample(self,count):
        if type(count) is not int or count<1:raise ValueError('Positive batch size required')
        parts=[];left=count
        while left:
            if self.position==self.count:self.order=self.rng.permutation(self.count);self.position=0
            n=min(left,self.count-self.position);parts.append(self.order[self.position:self.position+n]);self.position+=n;left-=n
        return np.concatenate(parts).astype(np.int64,copy=False)


def training_prototypes(z,y):
    if (z.ndim!=2 or z.shape[1]!=160 or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()) or y.shape!=(len(z),) or y.dtype!=torch.long
            or z.device!=y.device or set(y.detach().cpu().tolist())!=set(range(6))):raise ValueError('Legal source L unit160 and six labels required')
    means=torch.stack([z[y==c].detach().double().mean(0) for c in range(6)])
    if bool(means.norm(dim=1).eq(0).any()):raise ValueError('Nonzero fixed source L class means required')
    return F.normalize(means,dim=1).float().detach()


def pseudo_loss(z,pseudo,confidence,prototypes):
    n=len(z)
    if (z.ndim!=2 or z.shape!=(n,160) or not n or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all())
            or pseudo.shape!=(n,) or pseudo.dtype!=torch.long or bool(((pseudo<0)|(pseudo>=6)).any())
            or confidence.shape!=(n,) or not confidence.is_floating_point() or not bool(torch.isfinite(confidence).all()) or bool(((confidence<0)|(confidence>1)).any())
            or prototypes.shape!=(6,160) or prototypes.dtype!=torch.float32 or not bool(torch.isfinite(prototypes).all())
            or any(a.device!=z.device for a in (pseudo,confidence,prototypes))):raise ValueError('Finite student unit160, frozen six prototypes and teacher confidence/prediction required')
    mask=confidence.detach()>=.95;logits=z@prototypes.detach().T/.1
    value=(F.cross_entropy(logits,pseudo.detach(),reduction='none')*mask.to(z.dtype)).mean()
    return dict(loss=value,accepted=int(mask.sum()),count=n)


def _teacher(reader,values,device,out):
    blocks=[]
    for start in range(0,len(values),128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):logits=reader(body.base._tensor(values[start:start+128],device))
        if logits.shape!=(min(128,len(values)-start),6) or logits.dtype!=torch.float32 or not bool(torch.isfinite(logits).all()):raise ValueError('Original F0 six logits on the same source Us P1 required')
        blocks.append(np.asarray(logits.detach().cpu().tolist(),np.float32))
        if start==0 or (start+128)%4096==0 or start+128>=len(values):print('SOURCE_US_TEACHER_PROGRESS',min(start+128,len(values)),len(values),flush=True)
    logits=np.concatenate(blocks);score=logits.astype(np.float64);score-=score.max(1,keepdims=True);prob=np.exp(score);prob/=prob.sum(1,keepdims=True)
    return dict(logits=logits,prediction=logits.argmax(1).astype(np.int64),confidence=prob.max(1))


def _gradient_smoke(model,device):
    before=_snapshot(model)
    try:
        with torch.random.fork_rng(devices=body.base.native._cuda_devices(model)):
            torch.manual_seed(2026103152);x=torch.randn(8,2,256,device=device);z=model(x);prototypes=F.normalize(torch.randn(6,160,device=device),dim=1)
            value=pseudo_loss(z,torch.arange(8,device=device)%6,torch.ones(8,device=device),prototypes)['loss'];value.backward();g=body._gradients(model)
            if any(not np.isfinite(v) for v in g.values()) or any(g[k]<=0 for k in ('body','hidden','last')):raise ValueError('Synthetic PL must connect body and projection gradients')
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step source-Us smoke changed model state')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,parameters_buffers_unchanged=True,Sinc_gradients_none=True,synthetic_student_signal_rows=8)
    finally:model.zero_grad(set_to_none=True)


def _identity(c,parent,m,manifest,control):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=ARM,checkpoint=str(Path(c['output_root'])/'final.pt'),updates=STEPS,projection_updates=STEPS,new_projection_updates=STEPS,
        new_body_updates=STEPS,open_signal_updates=STEPS,signal_encoder_updates=STEPS,projection_cumulative_updates=parent['projection_updates']+STEPS,signal_feature_cumulative_updates=parent['signal_feature_cumulative_updates']+STEPS,
        signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,trainable_parameters=334673,body_trainable_parameters=231985,trainable_projection_parameters=102688,
        encoder_frozen=False,Sinc_frozen=True,Sinc_optimizer_steps=0,identity_head_updates=0,identity_head_replaced=False,original_F0_frozen=True,original_parent_updates=11100,parent_signal_updates=2220,
        parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_feature_manifest=m,source_handles=m['source_manifests'],
        source_control_identity=control['identity'],source_only=True,target_U_used=False,target_U_reads=0,source_Us_used=True,source_Us_count=US_COUNT,source_Us_sampled_exposures=US_BATCH*STEPS,
        Us_manifest=manifest,Us_ground_truth_reads=0,input_path='cached_P1_IQ_only',BN_Dropout_eval=True,auxiliary_prototype_parameters=0,prototype_deployed=False)


def _loop(model,banks,c,device,opt,out,Us,teacher,prototypes,control_steps):
    p=c['reference_config']['reference_config']['train_config'];sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed'])
    data={k:body._runtime(v,device) for k,v in banks.items()};stream=PermutationStream(US_COUNT,US_SEED);aug=torch.Generator(device='cpu').manual_seed(AUG_SEED)
    pseudo=body.base._tensor(teacher['prediction'],device,torch.long);confidence=body.base._tensor(teacher['confidence'],device,torch.float64);trace=[];usage={k:set() for k in ('L','OE','Us')}
    with control_steps.open(encoding='utf-8') as control,(out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);line=control.readline()
            if not line:raise ValueError('Matched SOURCE sampling trace is incomplete')
            old=json.loads(line)
            if old.get('step')!=step or any(old.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Matched SOURCE sampling differs')
            opt.zero_grad(set_to_none=True);source,metrics=body._source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device)
            ui=stream.sample(US_BATCH);indices=body.base._tensor(ui,device,torch.long);strong=body.views.phase_gain_view(body.base._tensor(Us['restored'][ui],'cpu'),aug).to(device)
            r=pseudo_loss(model(strong),pseudo[indices],confidence[indices],prototypes);total=source+r['loss']
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source Us replay objective')
            total.backward();g=body._gradients(model) if step<=2 else None;params=[p for group in opt.param_groups for p in group['params']]
            norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite body/projection gradient norm')
            opt.step()
            for k in ('L','OE'):usage[k].update(map(int,sample[k]))
            usage['Us'].update(map(int,ui));accepted=teacher['confidence'][ui]>=.95
            row=dict(step=step,arm=ARM,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),Us_indices=ui.tolist(),source_indices_equal=True,
                loss_total=float(total.detach().cpu()),**metrics,Us_PL_loss=float(r['loss'].detach().cpu()),Us_PL_weight=1.,Us_count=US_BATCH,Us_accepted=r['accepted'],Us_acceptance=r['accepted']/US_BATCH,
                Us_pseudo_histogram=np.bincount(teacher['prediction'][ui],minlength=6).tolist(),Us_accepted_histogram=np.bincount(teacher['prediction'][ui][accepted],minlength=6).tolist(),
                source_anchor_weight=10.,gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0)
            trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%100==0 or step==STEPS:print('SOURCE_US_REPLAY_PROGRESS',step,STEPS,row['loss_total'],r['accepted'],US_BATCH,flush=True)
        if control.readline():raise ValueError('SOURCE sampling trace contains extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source Us training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;b=_backend(c,device);snapshots=[(model,_snapshot(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def check():
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for model,before in snapshots for k,v in before.items()):raise ValueError('Original F0/r03/DURA changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual original three-checkpoint smoke must precede source/Us')
    pm,_,_,_=_reference(c);control,control_steps=_source_control(c,pm);parent=pm['model_identity']
    if b['model_identity']!=parent or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original source F0/r03/G identity differs')
    student=body._student(b['model']);gradient=_gradient_smoke(student,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));roles={};Us=None
    try:
        banks,m,roles=body._source_cache(_source(c),b,out,check)
        if m!=control['source_feature_manifest']:raise ValueError('Matched SOURCE role handles differ')
        prototypes=training_prototypes(body.base._tensor(banks['L']['parent_z'],device),body.base._tensor(banks['L']['metadata']['y'],device,torch.long))
        body._npz(out/'training_prototypes.npz',dict(prototypes=np.asarray(prototypes.detach().cpu().tolist(),np.float32)));Us=_public(c)
        _check_Us(c,b,Us,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()));teacher=_teacher(b['reader'],Us['restored'],device,out);check()
        body._npz(out/'Us_teacher.npz',dict(ids=Us['ids'],**teacher));before={n:p.detach().cpu().clone() for n,p in student.named_parameters()};buffers={n:p.detach().clone() for n,p in student.named_buffers()}
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8);trace,unique=_loop(student,banks,c,device,opt,out,Us,teacher,prototypes,control_steps)
        if any(not torch.equal(v,dict(student.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Student buffers changed')
        changed={n:not torch.equal(v,dict(student.named_parameters())[n].detach().cpu()) for n,v in before.items()}
        if any(v for n,v in changed.items() if n.startswith(body.signalcore.SINC_PREFIX)):raise ValueError('Frozen Sinc changed')
        if not any(v for n,v in changed.items() if n.startswith('signal.') and not n.startswith(body.signalcore.SINC_PREFIX)) or not any(v for n,v in changed.items() if n.startswith('projection.')):raise ValueError('Body and projection must actually update')
        if US_BATCH*STEPS>=US_COUNT and unique['Us']!=US_COUNT:raise ValueError('Source Us permutation failed full physical coverage')
        check();identity=_identity(c,parent,m,Us['manifest'],control);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite full SignalOpen checkpoint required')
        torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent,source_feature_manifest=m,Us_manifest=Us['manifest']),out/'final.pt')
        accepted=teacher['confidence']>=.95
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,identity=identity,updates=STEPS,total_optimizer_updates=STEPS,
            parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_feature_manifest=m,Us_manifest=Us['manifest'],
            source_control_identity=control['identity'],source_control_training_run_id=body.RUN_ID,source_control_training_code_commit=reference.TRAIN_CODE,source_control_source_only=True,source_control_frozen_before_training=True,source_sampling_matched=True,
            source_only=True,target_U_used=False,original_models_unchanged=True,all_student_Sinc_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
            student_body_changed=True,student_projection_changed=True,initial_state_equal_to_parent=True,source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,
            source_Us_count=US_COUNT,source_Us_sampled_exposures=US_BATCH*STEPS,source_Us_unique_records=unique['Us'],source_Us_teacher_accepted=int(accepted.sum()),source_Us_teacher_acceptance=float(accepted.mean()),
            source_Us_teacher_pseudo_histogram=np.bincount(teacher['prediction'],minlength=6).tolist(),source_Us_teacher_accepted_histogram=np.bincount(teacher['prediction'][accepted],minlength=6).tolist(),
            source_Us_accepted_exposures=sum(r['Us_accepted'] for r in trace),real_parent_signal_forward_rows=sum(COUNTS.values()),real_student_signal_forward_rows=(256+US_BATCH)*STEPS,
            real_F_forward_rows=US_COUNT,G_forward_rows=0,synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_signal_rows=gradient['synthetic_student_signal_rows'],
            first_step=trace[0],last_step=trace[-1],unique_training_rows=unique,source_density_fit_updates=0,threshold_fit_updates=0,prototype_deployed=False,per_record_Us_bank_deployed=False,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=STEPS),PL_confidence=.95,PL_temperature=.1,PL_weight=1.,Us_sampling_seed=US_SEED,Us_view_seed=AUG_SEED,
            original_F0_has_clean_Us_ancestor=True,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        _close_public(Us)
        for value in roles.values():body.base.close_role(value)


def load_trained_model(c,device='cuda:0'):
    validate_config(c);root=Path(c['output_root'])
    try:done=_read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Complete source Us training required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,updates=STEPS,total_optimizer_updates=STEPS,source_sampling_matched=True,
        source_control_source_only=True,source_control_frozen_before_training=True,source_only=True,target_U_used=False,original_models_unchanged=True,
        all_student_Sinc_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,student_body_changed=True,student_projection_changed=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen source-only matched Us completion required')
    pm,_,_,_=_reference(c);control,_=_source_control(c,pm);p=c['reference_config']['reference_config']['train_config'];model,parent=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    identity=_identity(c,parent,done['source_feature_manifest'],done['Us_manifest'],control);payload=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if (parent!=pm['model_identity'] or done['identity']!=identity or done['source_control_identity']!=control['identity'] or payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('identity')!=identity
            or payload.get('parent_signal_identity')!=parent or payload.get('source_feature_manifest')!=done['source_feature_manifest'] or payload.get('Us_manifest')!=done['Us_manifest']):raise ValueError('Strict original r03/source SOURCE/Us ancestry required')
    network=payload.get('network')
    if not isinstance(network,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite SignalOpen state required')
    original=model.state_dict();fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(body.signalcore.SINC_PREFIX)}
    if any(k not in network or not torch.equal(original[k].detach().cpu(),network[k]) for k in fixed):raise ValueError('Frozen Sinc/persistent buffers cannot change')
    try:model.load_state_dict(network,strict=True)
    except RuntimeError as exc:raise ValueError('Strict full source body/projection state required') from exc
    model.eval().requires_grad_(False);model._source_us_identity=identity;return model,identity


def main():
    print(json.dumps(dict(event='SOURCE_US_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
