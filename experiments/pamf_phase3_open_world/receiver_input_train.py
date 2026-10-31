"""Source-first bounded receiver input learning through frozen F0 and PCD."""
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
from . import receiver_input_core as core
from . import pcd_cluster_registry_run as reg
from . import fresh_pcd_train as training
from . import star_representation_cache as cache

RUN_ID='20261031-phase3-receiver-input-s2026092701-r01'
SCHEMA='pamf_phase3_receiver_input_train_v1'
STATUS='RECEIVER_INPUT_TRAINING_COMPLETE'
ROW_STATUS='RECEIVER_INPUT_ADAPTER_FROZEN'
ARMS=('SOURCE_INPUT','TARGET_INPUT')
STEPS=200
L_COUNT=6300
L_BATCH=128
U_COUNT=576
U_BATCH=64
L_SEED=2026103160
U_SEED=2026103161
REFERENCE_CODE=reg.REFERENCE_CODE
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','reference_prediction_root','output_root','steps','lr'}
ZEROS=('NN_optimizer_updates','target_Q_reads','validation_reads','novel_validation_reads','private_truth_reads','U_truth_reads','source_Us_reads','OE_reads')
_read,_write=reg._read,reg._write


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.01
            or any(not isinstance(c.get(k),str) or not c[k].strip() for k in ('reference_prediction_root','output_root'))):raise ValueError('Exact receiver input configuration required; no private/source-validation/query inputs')
    t=reg.fresh._train().validate_config(c['reference_config'])
    if t['code_commit']!=REFERENCE_CODE:raise ValueError('Original fresh f4b5 final PCD training required')
    root=Path(c['reference_prediction_root']);out=Path(c['output_root']).resolve()
    if root.resolve()==Path(t['output_root']).resolve():raise ValueError('Prediction reference must be separate from the trained checkpoint root')
    roots=[root,t['output_root'],t['U_public_root'],t['reference_config']['reference_config']['train_config']['IQ_root']]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable adapter output overlaps a model/public/reference input')
    return c


def _reference(c):
    header=_read(Path(c['reference_prediction_root'])/'complete.json');r=header['config']
    if r.get('train_config')!=c['reference_config'] or r.get('code_commit')!=REFERENCE_CODE or Path(r['output_root']).resolve()!=Path(c['reference_prediction_root']).resolve():
        raise ValueError('Actual completed fresh prediction and exact training config must match')
    pm,state,tau,cal=reg._reference(dict(reference_config=r))
    return r,pm,state,tau,cal


def _models(c,r,device):
    b=reg._backend(dict(reference_config=r),device);pcd,identity=reg._load_model(dict(reference_config=r),device)
    return b,pcd,identity


def _L(c,pm):
    t=c['reference_config']['reference_config']['reference_config']['train_config'];a=cache.load_role(Path(t['IQ_root'])/'L','L');m=a['manifest']
    if m['count']!=L_COUNT or m['parent_F0_identity']!=pm['parent_F0_identity'] or m['dura_identity']!=pm['dura_identity'] or 'teacher_logits' not in a:
        cache.close_role(a);raise ValueError('Original source L P1/y/cached F0 teacher with matching ancestry required')
    return a


def _public_U(c):return training._public(c['reference_config'])


def _U(c,Lids):
    t=c['reference_config'];root=Path(t['output_root']);done=_read(root/'complete.json');manifest=done.get('U_manifest',{});public=None
    try:
        declared,ids,public=_public_U(c)
        expected=dict(count=U_COUNT,capsule_id=t['U_capsule_id'],split_id=t['U_split_id'])
        if (done.get('status')!=training.STATUS or done.get('source_only') is not False or done.get('target_U_reads')!=U_COUNT or done.get('original_models_unchanged') is not True
                or any(manifest.get(k)!=v or declared.get(k)!=v for k,v in expected.items())):raise ValueError('Original frozen U capsule/split/count ancestry differs')
        with np.load(root/'U_features.npz',allow_pickle=False) as z:
            if set(z.files)!={'ids','views','parent_z','neighbors'} or not np.array_equal(z['ids'],ids):raise ValueError('Original public and cached U IDs/order must match')
            views=z['views']
            if views.shape!=(3,U_COUNT,2,256) or views.dtype!=np.float32:raise ValueError('Original fixed U P1 view cache required')
            weak=views[0].copy()
        if ids.shape!=(U_COUNT,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=U_COUNT or set(ids.tolist())&set(Lids.tolist()) or not np.isfinite(weak).all():raise ValueError('Finite disjoint opaque U P1 records required')
        return dict(ids=ids.copy(),restored=weak,manifest=manifest)
    finally:
        if hasattr(public,'_mmap'):public._mmap.close()


def _tensor(a,device,dtype=torch.float32):return torch.tensor(np.asarray(a).tolist(),device=device,dtype=dtype)
def _logits(reader,x):return reader.classify_features(reader.features(x))


def fixed_mask(logits,margin):
    logits=np.asarray(logits);margin=np.asarray(margin);n=len(logits)
    if logits.shape!=(n,6) or logits.dtype!=np.float32 or not np.isfinite(logits).all() or margin.shape!=(n,) or margin.dtype.kind!='f' or not np.isfinite(margin).all():raise ValueError('Finite original F0 logits and PCD predicted margins required')
    v=logits.astype(np.float64);v-=v.max(1,keepdims=True);p=np.exp(v);p/=p.sum(1,keepdims=True);confidence=p.max(1);closed=logits.argmax(1).astype(np.int64)
    return (confidence>=.95)&(margin>=0),closed,confidence


def _teacher(b,pcd,U,state,tau,device):
    values=[]
    for start in range(0,U_COUNT,128):
        with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
            x=_tensor(U['restored'][start:start+128],device);logits=_logits(b['reader'],x);z=pcd(x)
        values.append(dict(default_logits=np.asarray(logits.cpu().tolist(),np.float32),features=np.asarray(z.cpu().tolist(),np.float32)))
    a={k:np.concatenate([v[k] for v in values]) for k in values[0]};closed=a['default_logits'].argmax(1).astype(np.int64);a['margin']=reg.fresh._raw(state,a['features'],closed)[:,1]-tau[1,closed]
    a['mask'],a['pseudo'],a['confidence']=fixed_mask(a['default_logits'],a['margin']);a['ids']=U['ids'];return a


def _gradient_smoke(b,pcd,models,device):
    devices=[] if torch.device(device).type!='cuda' else [torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2026103162);adapter=core.BoundedIQ().to(device);x=torch.randn(8,2,256,device=device);adjusted=adapter(x)
        logits=_logits(b['reader'],adjusted);z=pcd(adjusted);probe=(z*torch.linspace(.1,1.,160,device=device)).sum(1).mean();(F.cross_entropy(logits,torch.arange(8,device=device)%6)+probe).backward()
        norm=float(adapter.theta.grad.norm()) if adapter.theta.grad is not None else 0.
        if not np.isfinite(norm) or norm<=0 or any(p.grad is not None for m in models for p in m.parameters()) or bool(adapter.theta.detach().ne(0).any()):raise ValueError('Synthetic input gradients must reach theta only without an update')
        return dict(status='PASS',theta_gradient_norm=norm,NN_gradients_none=True,optimizer_updates=0,synthetic_adapter_rows=8,synthetic_F_forward_rows=8,synthetic_PCD_forward_rows=8)


def _coefficients(adapter):return {k:np.asarray(v.detach().cpu().tolist(),np.float32).tolist() for k,v in adapter.components().items()}


def _coefficient_reports_equal(reported,computed):
    """Compare derived FP32 metadata within two local CPU/CUDA ULPs."""
    if not isinstance(reported,dict) or not isinstance(computed,dict) or set(reported)!=set(computed) or set(reported)!={'fir','gamma'}:return False
    for key,shape in (('fir',(7,2)),('gamma',(2,))):
        try:a,b=np.asarray(reported[key]),np.asarray(computed[key])
        except (TypeError,ValueError):return False
        if a.shape!=shape or b.shape!=shape or a.dtype.kind!='f' or b.dtype.kind!='f' or not np.isfinite(a).all() or not np.isfinite(b).all():return False
        with np.errstate(over='ignore',invalid='ignore'):
            a,b=a.astype(np.float32),b.astype(np.float32)
            spacing=np.maximum(np.abs(np.spacing(a)).astype(np.float64),np.abs(np.spacing(b)).astype(np.float64))
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(spacing).all():return False
        if not np.all(np.abs(a.astype(np.float64)-b.astype(np.float64))<=2*spacing):return False
    return True


def _identity(c,arm,pm):
    target=arm=='TARGET_INPUT'
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'adapter.pt'),updates=STEPS,adapter_parameters=16,
        parent_F0_identity=pm['parent_F0_identity'],parent_PCD_identity=pm['model_identities']['TARGET_PCD'],dura_identity=pm['dura_identity'],
        adapter_source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,system_PCD_has_target_U_ancestor=True,NN_optimizer_updates=0,input_path='DURA_P1_then_bounded_input')


def _loop(c,arm,adapter,L,U,teacher,b,pcd,device,folder,check):
    target=arm=='TARGET_INPUT';rng=np.random.default_rng(L_SEED);ur=np.random.default_rng(U_SEED) if target else None;source=None
    if target:source=(Path(c['output_root'])/'SOURCE_INPUT/steps.jsonl').open(encoding='utf-8')
    opt=torch.optim.Adam(adapter.parameters(),lr=.01,weight_decay=0.);trace=[];usage={k:set() for k in ('L','U')}
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                ix=rng.choice(L_COUNT,L_BATCH,replace=False)
                if target:
                    line=source.readline()
                    if not line or json.loads(line).get('L_indices')!=ix.tolist():raise ValueError('Matched source L sampling differs')
                opt.zero_grad(set_to_none=True);x=adapter(_tensor(L['restored'][ix],device));s=core.source_loss(_logits(b['reader'],x),_tensor(L['metadata']['y'][ix],device,torch.long),_tensor(L['teacher_logits'][ix],device),adapter)
                total=s['total'];ui=np.empty(0,np.int64);pl=anchor=0.;accepted=0;hist=[0]*6
                if target:
                    ui=ur.choice(U_COUNT,U_BATCH,replace=False);ux=adapter(_tensor(U['restored'][ui],device));r=core.target_loss(_logits(b['reader'],ux),_tensor(teacher['pseudo'][ui],device,torch.long),_tensor(teacher['mask'][ui],device,torch.bool),pcd(ux),_tensor(teacher['features'][ui],device))
                    total=total+r['total'];pl=float(r['PL'].detach().cpu());anchor=float(r['z_anchor'].detach().cpu());accepted=int(teacher['mask'][ui].sum());hist=np.bincount(teacher['pseudo'][ui][teacher['mask'][ui]],minlength=6).tolist()
                if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite bounded input loss')
                total.backward();norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite input gradient')
                if any(p.grad is not None for model in (b['reader'],b['model'],b['dura'][0],pcd) for p in model.parameters()):raise ValueError('Frozen neural network received a parameter gradient')
                opt.step();usage['L'].update(map(int,ix));usage['U'].update(map(int,ui));check()
                row=dict(step=step,arm=arm,L_indices=ix.tolist(),U_indices=ui.tolist(),source_indices_equal=True,loss_total=float(total.detach().cpu()),CE=float(s['CE'].detach().cpu()),KD=float(s['KD'].detach().cpu()),prior=float(s['prior'].detach().cpu()),
                    PL=pl,z_anchor=anchor,U_accepted=accepted,U_count=len(ui),U_accepted_histogram=hist,gradient_norm_before_clip=norm,gradient_clip_calls=step,NN_gradients_none=True,
                    theta_norm=float(adapter.theta.detach().norm().cpu()),theta_gradient=None if step!=1 else adapter.theta.grad.detach().cpu().tolist(),coefficients=_coefficients(adapter))
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%50==0 or step==STEPS:print('RECEIVER_INPUT_PROGRESS',arm,step,STEPS,row['loss_total'],accepted,flush=True)
        if source is not None and source.readline():raise ValueError('Source sampling trace has extra updates')
        return trace,{k:len(v) for k,v in usage.items()}
    finally:
        if source is not None:source.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable receiver-input training root required')
    r,pm,state,tau,_=_reference(c);b,pcd,identity=_models(c,r,device)
    if identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Actual original F0/finalPCD/DURA identities differ')
    models=(b['reader'],b['model'],b['dura'][0],pcd)
    if any(p.requires_grad or p.dtype!=torch.float32 or p.grad is not None for m in models for p in m.parameters()) or any(v.training for m in models for v in m.modules()):raise ValueError('Original neural networks must be FP32 frozen eval with no parameter gradients')
    snapshots=[(m,reg.fresh.best._snapshot_model(m)) for m in models];source_state=deepcopy(state);source_tau=tau.copy()
    def check():
        for model,before in snapshots:reg.fresh.best._check_model(model,before)
        if not reg._same(state,source_state) or not np.array_equal(tau,source_tau):raise ValueError('Original density or thresholds changed')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;smoke=reg._smoke(b,pcd,device);check();gradient=_gradient_smoke(b,pcd,models,device);check()
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));L=None;U=teacher=None;rows={}
    try:
        L=_L(c,pm)
        for arm in ARMS:
            if arm=='TARGET_INPUT':
                if not (out/'SOURCE_INPUT/complete.json').is_file():raise ValueError('Source adapter must finish before target U opens')
                U=_U(c,L['ids']);teacher=_teacher(b,pcd,U,state,tau,device);check();reg.fresh.best._save(out/'U_teacher.npz',teacher)
            adapter=core.BoundedIQ().to(device);folder=out/arm;folder.mkdir();trace,unique=_loop(c,arm,adapter,L,U,teacher,b,pcd,device,folder,check)
            if not bool(adapter.theta.detach().ne(0).any()):raise ValueError('Input adapter did not change from identity')
            adapter.eval().requires_grad_(False);i=_identity(c,arm,pm);theta=adapter.theta.detach().cpu().tolist()
            torch.save(dict(schema=SCHEMA,config=c,arm=arm,identity=i,state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}),folder/'adapter.pt')
            target=arm=='TARGET_INPUT';row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,updates=STEPS,theta=theta,coefficients=_coefficients(adapter),
                initial_theta_zero=True,adapter_changed=True,adapter_source_only=not target,target_U_reads=U_COUNT if target else 0,source_L_exposures=L_BATCH*STEPS,target_U_exposures=U_BATCH*STEPS if target else 0,
                unique_records=unique,U_accepted_exposures=sum(v['U_accepted'] for v in trace),original_models_unchanged=True,NN_optimizer_updates=0,first_step=trace[0],last_step=trace[-1],final_phase3_complete=False)
            check();_write(folder/'complete.json',row);rows[arm]=row;print(ROW_STATUS,arm,flush=True)
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={a:rows[a]['identity'] for a in ARMS},updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
            source_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_adapters_changed=True,source_L_reads=L_COUNT,cached_L_teacher_logits_reads=L_COUNT,target_U_reads=U_COUNT,
            source_L_manifest=L['manifest'],U_manifest=U['manifest'],parent_F0_identity=pm['parent_F0_identity'],parent_PCD_identity=identity,dura_identity=pm['dura_identity'],reference_prediction_identity=pm,
            fixed_teacher_mask=True,U_teacher_accepted=int(teacher['mask'].sum()),U_teacher_acceptance=float(teacher['mask'].mean()),U_teacher_prediction_histogram=np.bincount(teacher['pseudo'],minlength=6).tolist(),
            real_F_forward_rows=(2*L_BATCH+U_BATCH)*STEPS+U_COUNT,real_PCD_forward_rows=U_BATCH*STEPS+U_COUNT,real_r03_forward_rows=0,G_forward_rows=0,
            synthetic_F_forward_rows=10,synthetic_PCD_forward_rows=10,synthetic_r03_forward_rows=2,G_synthetic_forward_rows=2,synthetic_adapter_rows=8,
            optimizer=dict(name='Adam',lr=.01,weight_decay=0.,norm_clip=5.),L_sampling_seed=L_SEED,U_sampling_seed=U_SEED,adapter_parameters_per_arm=16,
            NN_parameters_unchanged=True,NN_buffers_unchanged=True,NN_gradients_none=True,source_input_theta_source_only=True,complete_system_PCD_has_target_U_ancestor=True,
            formal_target_accessed=False,target_REG=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        if L is not None:cache.close_role(L)


def load_adapters(c,device='cuda:0'):
    validate_config(c);root=Path(c['output_root'])
    try:done=_read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Both completed receiver-input arms required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_completed_before_U_open=True,source_sampling_matched=True,original_models_unchanged=True,all_adapters_changed=True,source_L_reads=L_COUNT,target_U_reads=U_COUNT,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Complete source-first matched input training required')
    _,pm,_,_,_=_reference(c);adapters={}
    if done.get('parent_PCD_identity')!=pm['model_identities']['TARGET_PCD'] or done.get('parent_F0_identity')!=pm['parent_F0_identity'] or done.get('dura_identity')!=pm['dura_identity']:raise ValueError('Frozen parent model lineage differs')
    for arm in ARMS:
        row=_read(root/arm/'complete.json');identity=_identity(c,arm,pm);payload=torch.load(root/arm/'adapter.pt',map_location='cpu',weights_only=False)
        if (done.get('rows',{}).get(arm)!=row or done.get('identities',{}).get(arm)!=identity or row.get('status')!=ROW_STATUS or row.get('updates')!=STEPS or row.get('identity')!=identity
                or payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('arm')!=arm or payload.get('identity')!=identity):raise ValueError('Strict receiver input arm/config/parent identity required')
        state=payload.get('state');theta=state.get('theta') if isinstance(state,dict) else None
        if not isinstance(state,dict) or set(state)!={'theta'} or not torch.is_tensor(theta) or theta.dtype!=torch.float32 or theta.shape!=(16,) or not bool(torch.isfinite(theta).all()) or not bool(theta.ne(0).any()):raise ValueError('Only finite learned FP32 theta[16] allowed')
        model=core.BoundedIQ();model.load_state_dict(state,strict=True)
        if row.get('theta')!=theta.tolist() or not _coefficient_reports_equal(row.get('coefficients'),_coefficients(model)):raise ValueError('Stored adapter parameters and coefficient report differ')
        adapters[arm]=model.to(device).eval().requires_grad_(False)
    return adapters,done


def main():
    print(json.dumps(dict(event='RECEIVER_INPUT_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
