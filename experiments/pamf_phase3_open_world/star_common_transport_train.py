"""One matched CCD student from r03, keeping original identifier/G frozen."""
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
from . import star_common_transport_core as core

RUN_ID='20261031-phase3-common-transport-s2026092701-r01'
SCHEMA='pamf_phase3_common_transport_train_v1'
STATUS='COMMON_TRANSPORT_TRAINING_COMPLETE'
ARM='TARGET_CCD'
ARMS=(ARM,)
STEPS=444
WARMUP_STEPS=100
U_COUNT=384
U_BATCH=64
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE='4273e88392abe0717c1bd94fca1370d19c67cbb3'
FIELDS=TRAIN_FIELDS=set(body.FIELDS)
ZEROS=body.ZEROS
_read,_write,_snapshot=body._read,body._write,body._snapshot


def _flat(c):return dict(c,run_id=body.RUN_ID,reference_config=c['reference_config']['reference_config'])


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001):
        raise ValueError('Exact nine-field common-transport configuration required; no truth/Q/K')
    r=reference.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed body r02 controls required')
    body.validate_config(_flat(c));out=Path(c['output_root']).resolve()
    for value in (r['output_root'],r['train_config']['output_root']):
        old=Path(value).resolve()
        if out.is_relative_to(old) or old.is_relative_to(out):raise ValueError('New immutable CCD output cannot overlap the fixed controls')
    if any(c[k]!=r['train_config'][k] for k in ('U_public_root','U_capsule_id','U_split_id')):raise ValueError('Matched star-G10 U handle required')
    return c


def _reference(c):return body._reference(_flat(c))


def _controls(c,pm):
    """Only frozen control headers and trace paths; never old source/Q values."""
    r=c['reference_config'];tc=r['train_config'];p=_read(Path(r['output_root'])/'complete.json');t=_read(Path(tc['output_root'])/'complete.json')
    expected=dict(schema=reference.SCHEMA,status=reference.STATUS,run_id=reference.RUN_ID,code_commit=REFERENCE_CODE,config=r,
        reference_prediction_identity=pm,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],training_run_id=body.RUN_ID,
        training_code_commit=reference.TRAIN_CODE,training_reused=True,this_run_optimizer_updates=0,private_truth_reads=0,source_states_unchanged=True,
        frozen_parameters_buffers_unchanged=True,all_three_DURA_conditions_complete=True)
    required=dict(schema=body.SCHEMA,status=body.STATUS,run_id=body.RUN_ID,code_commit=reference.TRAIN_CODE,config=tc,arms=['SOURCE','TARGET_PCD'],
        updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,source_sampling_matched=True,SOURCE_completed_before_U_open=True,
        original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True)
    if any(p.get(k)!=v for k,v in expected.items()) or any(t.get(k)!=v for k,v in required.items()):raise ValueError('Completed matched body r01/r02 metadata ancestry required')
    if p.get('model_identities',{}).get('E0')!=pm['model_identity']:raise ValueError('Control r03 model identity differs')
    for arm in ('SOURCE','TARGET_PCD'):
        row=_read(Path(tc['output_root'])/arm/'complete.json')
        if (row!=t.get('rows',{}).get(arm) or row.get('identity')!=t.get('identities',{}).get(arm)
                or row.get('identity')!=p.get('model_identities',{}).get(arm) or row.get('updates')!=STEPS):raise ValueError('Frozen control row/model identity differs')
    return dict(prediction=p,training=t,source_trace=Path(tc['output_root'])/'SOURCE/steps.jsonl',target_trace=Path(tc['output_root'])/'TARGET_PCD/steps.jsonl')


def _centers(bank,device):
    z=bank['parent_z'].astype(np.float64);y=bank['metadata']['y'];means=[]
    for label in range(6):
        if not np.any(y==label):raise ValueError('All six registered source classes required for fixed centers')
        mean=z[y==label].mean(0);norm=float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm==0:raise ValueError('Nonzero source class mean required')
        means.append(mean/norm)
    return body.base._tensor(np.asarray(means,np.float32),device).detach()


def _ccd_gradient_smoke(model,device):
    before=_snapshot(model)
    try:
        with torch.random.fork_rng(devices=body.base.native._cuda_devices(model)):
            torch.manual_seed(2026103121);iq=torch.randn(16,2,256,device=device);centers=F.normalize(torch.randn(6,160,device=device),dim=1)
            first,second=model(iq[:8]),model(iq[8:]);assignment=dict(labels=torch.arange(8,device=device)%6,mask=torch.ones(8,dtype=torch.bool,device=device))
            value=core.classification_loss(first,second,centers,torch.arange(8,device=device),assignment)['loss'];value.backward();g=body._gradients(model)
            if any(not np.isfinite(v) for v in g.values()) or any(g[k]<=0 for k in ('body','middle','hidden','last')):raise ValueError('CCD-only live body/projection gradient smoke failed')
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step CCD smoke changed model state')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,Sinc_gradients_none=True,parameters_buffers_unchanged=True,synthetic_student_signal_rows=16,
                assignment='fixed synthetic six-slot supervision solely for gradient probe; no target/source records')
    finally:model.zero_grad(set_to_none=True)


def _identity(c,parent_identity,m,U_manifest):
    value=body._identity(_flat(c),'TARGET_PCD',parent_identity,m,U_manifest)
    value.update(run_id=RUN_ID,arm=ARM,checkpoint=str(Path(c['output_root'])/'final.pt'),updates=STEPS,projection_updates=STEPS,new_projection_updates=STEPS,
        new_body_updates=STEPS,open_signal_updates=STEPS,signal_encoder_updates=STEPS,projection_cumulative_updates=parent_identity['projection_updates']+STEPS,
        signal_feature_cumulative_updates=parent_identity['signal_feature_cumulative_updates']+STEPS,target_U_reads=U_COUNT,CCD_weight=.1,CCD_start_step=WARMUP_STEPS+1,
        CCD_centers_training_only=True,CCD_beta_deployed=False,control_training_run_id=body.RUN_ID,control_training_code_commit=reference.TRAIN_CODE)
    return value


def _loop(model,banks,c,device,opt,out,U,prototypes,centers,controls):
    p=c['reference_config']['reference_config']['train_config'];sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed'])
    data={k:body._runtime(v,device) for k,v in banks.items()};target=body._runtime(U,device);neighbors=body.base._tensor(U['neighbors'],device,torch.long)
    rng=np.random.default_rng(body.U_SEED);filling=torch.Generator(device='cpu').manual_seed(core.FILL_SEED);beta=torch.full((6,),1/6,dtype=torch.float64,device=device)
    usage={k:set() for k in ('L','OE','U')};trace=[]
    with controls['source_trace'].open(encoding='utf-8') as source,controls['target_trace'].open(encoding='utf-8') as control,(out/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);ui=rng.choice(U_COUNT,U_BATCH,replace=False);indices=body.base._tensor(ui,device,torch.long)
            old_s=json.loads(source.readline());old_t=json.loads(control.readline())
            if (old_s.get('step')!=step or old_t.get('step')!=step or any(old_s.get(k+'_indices')!=sample[k].tolist() or old_t.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE'))
                    or old_t.get('U_indices')!=ui.tolist()):raise ValueError('Matched source/U sampling differs from frozen SOURCE/PCD controls')
            opt.zero_grad(set_to_none=True);total,metrics=body._source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device)
            with torch.no_grad():weak=model(target['views'][0])
            strong=[model(target['views'][j,indices]) for j in (1,2)]
            pcd=body.core.consistency(weak,*strong,prototypes,indices,neighbors);total=total+.1*pcd['loss']
            active=step>WARMUP_STEPS;ccd=(strong[0].sum()+strong[1].sum())*0;selected=0;diag=None
            if active:
                assigned=core.assign(weak,centers,beta,filling);beta=assigned['beta'];diag=assigned['diagnostics']
                result=core.classification_loss(*strong,centers,indices,assigned);ccd=result['loss'];selected=result['selected_batch_count'];total=total+.1*ccd
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source/PCD/CCD objective')
            total.backward();gradients=body._gradients(model,prototypes) if step<=2 or step==WARMUP_STEPS+1 else None
            params=[p for group in opt.param_groups for p in group['params']];norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite ordinary body/projection gradient norm')
            opt.step()
            with torch.no_grad():prototypes.copy_(F.normalize(prototypes,dim=1))
            for k in ('L','OE'):usage[k].update(map(int,sample[k]))
            usage['U'].update(map(int,ui));row=dict(step=step,arm=ARM,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),loss_total=float(total.detach().cpu()),**metrics,
                U_global=float(pcd['global'].detach().cpu()),U_local=float(pcd['local'].detach().cpu()),source_anchor_weight=10.,U_PCD_weight=.1,CCD_weight=.1,
                CCD_active=active,CCD_loss=float(ccd.detach().cpu()),CCD_selected_batch_count=selected,CCD_selected_U_count=0 if diag is None else diag['selected_count'],
                CCD_histogram=[0]*6 if diag is None else diag['selected_histogram'],CCD_beta=beta.cpu().tolist(),CCD_transport=diag,gradient_norms=gradients,
                gradient_norm_before_clip=norm,ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True,U_indices_equal=True)
            trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%100==0 or step==STEPS or step==WARMUP_STEPS+1:print('COMMON_TRANSPORT_PROGRESS',step,STEPS,row['loss_total'],row['CCD_selected_U_count'],flush=True)
        if source.readline() or control.readline():raise ValueError('Frozen control sampling traces contain extra updates')
    return trace,{k:len(v) for k,v in usage.items()},beta


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable common-transport training output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;flat=_flat(c);b=body._backend(flat,device)
    snapshots=[(m,_snapshot(m)) for m in (b['reader'],b['model'],b['dura'][0])]
    def check():
        for model,before in snapshots:
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('Original F0/r03/DURA changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual three-checkpoint no-query smoke must precede source/U')
    pm,_,_,_=_reference(c);controls=_controls(c,pm);parent_identity=pm['model_identity']
    if b['model_identity']!=parent_identity or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original source model/identifier/G identity differs')
    student=body._student(b['model']);gradient=body._gradient_smoke(student,device);ccd_probe=_ccd_gradient_smoke(student,device);check()
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient,CCD_only_gradient=ccd_probe));roles={}
    try:
        banks,m,roles=body._source_cache(flat,b,out,check);teacher=banks['OE']['teacher_u'].copy();centers=_centers(banks['L'],device)
        U,U_manifest=body._target_cache(flat,b,out,device,check,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()))
        if U_manifest!=controls['training']['U_manifest']:raise ValueError('Matched U observation/physical handle differs from frozen control')
        prototypes,initialization=body.core.initialize_prototypes(body.base._tensor(U['parent_z'],device));prototypes=torch.nn.Parameter(prototypes)
        initial={n:p.detach().cpu().clone() for n,p in student.named_parameters()};buffers={n:p.detach().clone() for n,p in student.named_buffers()}
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad]+[prototypes],lr=c['lr'],weight_decay=.0001,eps=1e-8)
        trace,unique,beta=_loop(student,banks,c,device,opt,out,U,prototypes,centers,controls)
        if any(not torch.equal(v,dict(student.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Student buffers changed')
        changed={n:not torch.equal(v,dict(student.named_parameters())[n].detach().cpu()) for n,v in initial.items()}
        if any(v for n,v in changed.items() if n.startswith(body.signalcore.SINC_PREFIX)):raise ValueError('Frozen Sinc changed')
        if not any(v for n,v in changed.items() if n.startswith('signal.') and not n.startswith(body.signalcore.SINC_PREFIX)) or not any(v for n,v in changed.items() if n.startswith('projection.')):raise ValueError('Student body/projection must actually update')
        if not np.array_equal(teacher,banks['OE']['teacher_u']):raise ValueError('Original OE teacher changed')
        check();identity=_identity(c,parent_identity,m,U_manifest);student.eval().requires_grad_(False);network={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite complete SignalOpen checkpoint required')
        torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=network,parent_signal_identity=parent_identity,source_feature_manifest=m),out/'final.pt')
        body._npz(out/'training_only.npz',dict(source_centers=np.asarray(centers.cpu().tolist(),np.float32),beta=np.asarray(beta.cpu().tolist(),np.float64),prototypes=np.asarray(prototypes.detach().cpu().tolist(),np.float32)))
        _write(out/'prototype_initialization.json',initialization)
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,identity=identity,updates=STEPS,total_optimizer_updates=STEPS,
            parent_signal_identity=parent_identity,parent_F0_identity=parent_identity['parent_F0_identity'],dura_identity=parent_identity['dura_identity'],source_feature_manifest=m,
            parent_old_control_prediction=controls['prediction'],control_training_run_id=body.RUN_ID,control_training_code_commit=reference.TRAIN_CODE,
            source_only=False,target_U_used=True,target_U_reads=U_COUNT,U_manifest=U_manifest,controls_frozen_before_training=True,source_sampling_matched=True,U_sampling_matched=True,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,original_OE_teacher_unchanged=True,
            initial_state_equal_to_parent=True,student_encoder_unchanged=False,student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,model_buffers_unchanged=True,
            source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,real_parent_signal_forward_rows=sum(COUNTS.values())+U_COUNT,
            real_student_signal_forward_rows=(256+U_COUNT+2*U_BATCH)*STEPS,real_F_forward_rows=0,G_forward_rows=U_COUNT,
            synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,synthetic_student_signal_rows=gradient['synthetic_student_signal_rows']+ccd_probe['synthetic_student_signal_rows'],
            CCD_only_gradient_smoke=ccd_probe,CCD_first_active_step=WARMUP_STEPS+1,CCD_weight=.1,CCD_centers_training_only=True,CCD_beta_deployed=False,
            unique_training_rows=unique,first_step=trace[0],last_step=trace[-1],optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=STEPS),
            source_density_fit_updates=0,threshold_fit_updates=0,prototype_deployed=False,per_record_U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():body.base.close_role(value)


def load_trained_model(c,device='cuda:0'):
    validate_config(c);root=Path(c['output_root'])
    try:done=_read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Complete common-transport training required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=ARM,updates=STEPS,total_optimizer_updates=STEPS,
        source_sampling_matched=True,U_sampling_matched=True,original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_bodies_changed=True,all_student_projections_changed=True,source_only=False,target_U_used=True,target_U_reads=U_COUNT,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen matched common-transport completion/roles required')
    pm,_,_,_=_reference(c);controls=_controls(c,pm);p=c['reference_config']['reference_config']['train_config'];model,parent_identity=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    identity=_identity(c,parent_identity,done['source_feature_manifest'],done['U_manifest']);value=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if (parent_identity!=pm['model_identity'] or done['identity']!=identity or done.get('parent_old_control_prediction')!=controls['prediction']
            or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity or value.get('parent_signal_identity')!=parent_identity
            or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict source r03 and frozen control ancestry required')
    network=value.get('network')
    if not isinstance(network,dict) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite SignalOpen state required')
    original=model.state_dict();fixed=(set(dict(model.named_buffers()))&set(original))|{n for n,p in model.named_parameters() if n.startswith(body.signalcore.SINC_PREFIX)}
    if any(k not in network or not torch.equal(original[k].detach().cpu(),network[k]) for k in fixed):raise ValueError('Frozen source Sinc/persistent buffers cannot be replaced')
    try:model.load_state_dict(network,strict=True)
    except RuntimeError as exc:raise ValueError('Strict complete SignalOpen body/projection checkpoint required') from exc
    model.eval().requires_grad_(False);model._common_transport_identity=identity
    return model,identity


def main():
    print(json.dumps(dict(event='COMMON_TRANSPORT_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
