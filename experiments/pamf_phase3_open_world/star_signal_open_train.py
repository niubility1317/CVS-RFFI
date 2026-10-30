"""Source-only independent signal encoder; original F0 identity stays frozen."""
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
from . import star_crossset_metric_train as parent
from . import star_crossset_metric_model as loss
from . import star_signal_open_model as core
from . import star_consensus_run as base
from . import star_representation_cache as IQ

RUN_ID='20261030-phase3-star-signal-open-s2026092701-r03'
ARMS=('E_SIGNAL',)
SCHEMA='pamf_phase3_star_signal_open_train_v1'
STATUS='STAR_SIGNAL_OPEN_SOURCE_FROZEN'
CONTROL_CODE='fb016284e6c1f4715da4c690e2333efce89103b5'
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
FIELDS={'run_id','code_commit','arm','native_config','IQ_root','feature_root','control_root','output_root','steps','lr','L_seed','OE_seed'}
TRAIN_FIELDS=FIELDS
ZEROS=('native_updates','original_native_updates','identity_head_updates','G_forward_rows','target_reads','validation_reads',
    'novel_validation_reads','original_U_s_reads','private_truth_reads','OE_truth_reads','received_cache_values_read')
_read,_write=runner._read,runner._write
Sampler=parent.Sampler


def _native_config(n):return dict(n,arm='supervised_only',steps=2220,lr=.0001,L_seed=2026092701,OE_seed=2026102907)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None):
        raise ValueError('Exact independent source signal configuration required; no truth/U/Q/K')
    n=c['native_config']
    if (not isinstance(n,dict) or set(n)!=runner.COMMON|{'source_cache'} or n.get('run_id')!=RUN_ID
            or n.get('code_commit')!=c['code_commit'] or n.get('row_id')!='E_SIGNAL-s2026092701' or n.get('output_root')!=c['output_root']):
        raise ValueError('Matching original F0 constructor and E_SIGNAL row required')
    native.validate_config(_native_config(n))
    for k,v in dict(steps=STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018).items():
        if type(c[k]) is not type(v) or c[k]!=v:raise ValueError('Fixed matched 2220 updates, LR and source streams required')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('IQ_root','feature_root','control_root','output_root')):
        raise ValueError('Nonempty immutable output and legal source roots required')
    parent._separate(c['output_root'],[c[k] for k in ('IQ_root','feature_root','control_root')]+[n['source_output'],n['source_cache']])
    return c


def _metadata(c):
    """Only completed source/control metadata; never V/novel prediction values."""
    m=parent.validate_cache_manifest(_read(Path(c['feature_root'])/'complete.json'))
    d=_read(Path(c['control_root'])/'complete.json')
    expected=dict(schema='pamf_phase3_star_crossset_metric_prediction_v1',status='STAR_CROSSSET_METRIC_PREDICTIONS_COMPLETE',
        run_id=parent.RUN_ID,code_commit=CONTROL_CODE,arm='S_SEP_TOPO',source_only=True,target_reads=0,private_truth_reads=0,
        source_states_unchanged=True,model_parameters_buffers_unchanged=True)
    if (m['code_commit']!=CONTROL_CODE or any(d.get(k)!=v for k,v in expected.items())
            or d.get('feature_manifest')!=m or d.get('source_feature_manifest')!=m['source_feature_manifest']):
        raise ValueError('Fixed completed S_SEP_TOPO control and original-F0 teacher features required')
    n=c['native_config'];p=m['parent_F0_identity'];source=m['config']['source_config']
    for k in ('source_output','source_contract','base_release','model_seed'):
        if p.get(k)!=n[k] or source.get(k)!=n[k]:raise ValueError('Original F0 constructor/feature cache source identity differs')
    t=d.get('config',{}).get('train_config',{})
    if any(t.get(k)!=v for k,v in dict(steps=2220,L_seed=2026103017,OE_seed=2026103018).items()) or not t.get('output_root'):
        raise ValueError('Saved matched control sampling trace required')
    return m,d,Path(t['output_root'])/'steps.jsonl'


def _export(model):
    # Preserve the actual, strictly loaded differentiable SequenceReference.
    return core.HeadlessSignal(model.id_backbone.encoder)


def _checkpoint_smoke(model,device):
    before=_snapshot(model);model.eval()
    result=base._smoke(dict(reader=model.id_backbone.encoder),device)
    if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('Actual F0 smoke changed source state')
    if result.get('status')!='PASS' or result.get('actual_F0_checkpoint') is not True or result.get('G_forward_rows')!=0:
        raise ValueError('Actual original F0 synthetic G10 no-query smoke required')
    return result


def _model(c,device,*,smoke=False):
    original=native.make_trainable_model(_native_config(c['native_config']),device).eval()
    check=_checkpoint_smoke(original,device) if smoke else None
    model=core.SignalOpen(_export(original)).to(device)
    model.original_parent_identity=original._phase3_parent_identity;model.source_data_handle=original._phase3_source_handle
    model.checkpoint_smoke=check
    if core.parameter_report(model)!=dict(signal_feature_parameters=232033,projection_parameters=102688,signal_parameters=334721,combined_active_parameters=567714):
        raise ValueError('Actual complete headless232033 plus projection102688 required')
    return model


def load_role(root,role):
    if role not in COUNTS:raise ValueError('Training may only open L/OE P1 role handles')
    return IQ.load_role(Path(root)/role,role)


close_role=IQ.close_role


def _roles(c,m):
    a={};teachers={}
    try:
        for role in ('L','OE'):
            teachers[role]=parent.load_role(c['feature_root'],role);a[role]=load_role(c['IQ_root'],role)
            t=teachers[role];b=a[role];parent._metadata(b['ids'],b['metadata'],role)
            if (t['manifest']!=m or b['manifest']!=m['source_manifests'][role] or not np.array_equal(b['ids'],t['ids'])
                    or any(not np.array_equal(v,t['metadata'][k]) for k,v in b['metadata'].items())
                    or b['restored'].shape!=(COUNTS[role],2,256) or b['restored'].dtype!=np.float32):
                raise ValueError('Same P1 physical ID order, role metadata and original teacher ancestry required')
        if set(a['L']['ids'].tolist())&set(a['OE']['ids'].tolist()):raise ValueError('Source L and OE physical roles must be disjoint')
        return a,teachers
    except BaseException:
        for value in a.values():close_role(value)
        raise


def _tensor(x,device,dtype=torch.float32):return torch.tensor(np.asarray(x).tolist(),device=device,dtype=dtype)


def _objective(model,L,OE,teacher,sample,device,probe=False):
    ix,oi=sample['L'],sample['OE'];lm,om=L['metadata'],OE['metadata']
    z=model(_tensor(L['restored'][ix],device));oz=model(_tensor(OE['restored'][oi],device))
    metadata=lambda m,k,indices:_tensor(m[k][indices],device,torch.long)
    lrx,lday=metadata(lm,'rx_i',ix),metadata(lm,'day_i',ix);orx,oday=metadata(om,'rx_i',oi),metadata(om,'day_i',oi)
    known=loss.crossset_metric(z,oz,metadata(lm,'y',ix),lrx,lday,orx,oday)
    fixed=_tensor(teacher['OE']['u'][oi],device).detach();other=loss.oe_structure(oz,fixed,orx,oday)
    total=known['loss']+other['loss'];gradients=None
    if probe:
        gradients={k:dict(norm=float(g.detach().norm().cpu()),minimum_row_norm=float(g.detach().norm(dim=1).min().cpu()))
            for k,g in zip(('L','OE'),torch.autograd.grad(known['loss'],(z,oz),retain_graph=True))}
    return total,dict(loss=float(total.detach().cpu()),L_metric=float(known['loss'].detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),
        OE_KL=float(other['loss'].detach().cpu()),L_weight=1.,OE_weight=1.,L_anchors=len(ix),OE_candidates=len(oi),OE_anchors=len(oi),
        candidate_set_priors=[.5,.5],crossset_embedding_gradients=gradients,OE_teacher_fixed=True,OE_OE_hard_negatives=False)


def _gradient_norms(model):
    r=model.signal.encoder;h=r.core.id_backbone.cls_head
    modules=dict(sinc=r.core.id_backbone.sinc,shallow=r.core.id_backbone.t1 if hasattr(r.core.id_backbone,'t1') else r.core.id_backbone.t2,
        middle=r.core.id_backbone.t2,time_frequency=h.base_norm,PA=h.pa_norm,reference=r.response_projection,signature=r.sequence_pool,
        hidden=model.projection.network[0],last=model.projection.network[-1])
    return {k:sum(float(p.grad.detach().double().square().sum().cpu()) for p in module.parameters() if p.grad is not None)**.5 for k,module in modules.items()}


def _gradient_smoke(model,device):
    before=_snapshot(model);rng=np.random.default_rng(2026103017);rx=np.array([1,3,1,3],np.int64);day=np.array([1,2,1,2],np.int64)
    def role():return dict(restored=rng.normal(size=(4,2,256)).astype(np.float32),metadata=dict(rx_i=rx,day_i=day))
    L,OE=role(),role();L['metadata']['y']=np.array([0,0,1,1],np.int64)
    teacher=dict(OE=dict(u=rng.normal(size=(4,160)).astype(np.float32)))
    try:
        model.zero_grad(set_to_none=True);value,metrics=_objective(model,L,OE,teacher,dict(L=np.arange(4),OE=np.arange(4)),device,True);value.backward()
        gradients=_gradient_norms(model)
        if any(not np.isfinite(v) for v in gradients.values()) or any(gradients[k]<=0 for k in ('sinc','middle','last')):
            raise ValueError('Live Sinc/body/projection signal gradient smoke failed')
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step signal smoke changed parameters/buffers')
        return dict(status='PASS',synthetic_L_signal_rows=4,synthetic_OE_signal_rows=4,optimizer_updates=0,parameters_buffers_unchanged=True,
            gradient_norms=gradients,metrics=metrics,synthetic_IQ='P1-shaped synthetic; no G/real IQ')
    finally:model.zero_grad(set_to_none=True)


def _identity(c,m,model):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm='E_SIGNAL',checkpoint=str(Path(c['output_root'])/'final.pt'),updates=STEPS,
        open_signal_updates=STEPS,projection_updates=STEPS,native_updates=0,original_native_updates=0,original_parent_updates=11100,
        signal_feature_cumulative_updates=11100+STEPS,signal_parameters=334721,signal_feature_parameters=232033,metric_parameters=102688,
        original_F0_parameters=232993,combined_active_parameters=567714,initialization_seed=2026092701,
        parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],original_parent_identity=model.original_parent_identity,
        identity_head_replaced=False,original_F0_frozen=True,input_path='cached_P1_IQ_only',OE_embedding_training=True,
        OE_topology_regularization=True,crossset_separation=True)


def _loop(model,roles,teachers,c,device,opt,control_steps,on_step):
    sampler=Sampler(teachers['L'],teachers['OE'],c['L_seed'],c['OE_seed']);trace=[];usage=dict(L=set(),OE=set())
    with Path(control_steps).open(encoding='utf-8') as control:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);line=control.readline()
            if not line:raise ValueError('Control sampling trace is incomplete')
            expected=json.loads(line)
            if expected.get('step')!=step or any(expected.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):
                raise ValueError('Matched control indices differ from fixed source streams')
            opt.zero_grad(set_to_none=True);value,metrics=_objective(model,roles['L'],roles['OE'],teachers,sample,device,step<=2);value.backward()
            gradients=_gradient_norms(model) if step<=2 else None
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite signal gradients')
            norm=float(core.clip(opt,model))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite physical global gradient norm')
            opt.step()
            for k in usage:usage[k].update(map(int,sample[k]))
            row=dict(step=step,open_signal_updates=step,projection_updates=step,native_updates=0,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),
                control_indices_equal=True,gradient_norms=gradients,gradient_norm_before_physical_clip=norm,
                sinc_lr=float(opt.param_groups[1]['lr']),sinc_optimizer_step=opt._sinc_unit_steps,physical_clip_calls=opt._sinc_physical_clip_calls,**metrics)
            trace.append(row);on_step(row)
            if step==1 or step%100==0 or step==STEPS:print('STAR_SIGNAL_OPEN_TRAIN_PROGRESS',step,STEPS,metrics['loss'],flush=True)
        if control.readline():raise ValueError('Control sampling trace has extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable signal training output required')
    m,reference,control=_metadata(c);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(c,device,smoke=True);gradient_smoke=_gradient_smoke(model,device)
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=model.checkpoint_smoke,gradient=gradient_smoke))
    roles={}
    try:
        roles,teachers=_roles(c,m);opt=core.optimizer(model,c['lr']);buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
            def record(row):log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            trace,unique=_loop(model,roles,teachers,c,device,opt,control,record)
        if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Signal BN/Sinc buffers changed')
        identity=_identity(c,m,model);zero={k:0 for k in ZEROS}
        torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=model.state_dict(),cache_manifest=m,original_parent_identity=model.original_parent_identity,
            source_data_handle=model.source_data_handle,source_only=True,**zero),out/'final.pt')
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm='E_SIGNAL',config=c,identity=identity,updates=STEPS,
            open_signal_updates=STEPS,projection_updates=STEPS,cache_manifest=m,parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],
            control_prediction_identity=reference,source_only=True,model_buffers_unchanged=True,identity_head_replaced=False,signal_parameters=334721,
            L_training_rows_loaded=COUNTS['L'],OE_training_rows_loaded=COUNTS['OE'],L_sampled_exposures=192*STEPS,OE_sampled_exposures=64*STEPS,
            unique_training_rows=unique,real_signal_forward_rows=256*STEPS,synthetic_signal_forward_rows=8,synthetic_F_forward_rows=2,real_F_forward_rows=0,
            optimizer=dict(name='AdamW',mode='CENTER_UNITS',lr=.0001,weight_decay=.0001,eps=1e-8,sinc_multiplier=1000.,sinc_eps=1e-11,
                physical_clip=5.,physical_clip_calls=opt._sinc_physical_clip_calls,sinc_steps=opt._sinc_unit_steps),
            first_step=trace[0],second_step=trace[1],final_step=trace[-1],statistical_background_uses_OE=True,
            final_phase3_complete=False,**zero)
        _write(out/'training_summary.json',done);_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():close_role(role)


def load_trained_model(c,checkpoint,device='cuda:0'):
    validate_config(c)
    if Path(checkpoint).resolve()!=Path(c['output_root'],'final.pt').resolve():raise ValueError('Only the fixed final signal checkpoint may load')
    m,_,_=_metadata(c);done=_read(Path(c['output_root'])/'complete.json')
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm='E_SIGNAL',config=c,updates=STEPS,open_signal_updates=STEPS,
        cache_manifest=m,source_only=True,model_buffers_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete source-only signal checkpoint metadata required')
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=done.get('identity')
            or value.get('cache_manifest')!=m or value.get('source_only') is not True or any(value.get(k)!=0 for k in ZEROS)):
        raise ValueError('Signal checkpoint source provenance differs')
    model=_model(c,device)
    if (value.get('identity')!=_identity(c,m,model) or value.get('original_parent_identity')!=model.original_parent_identity
            or value.get('source_data_handle')!=model.source_data_handle):raise ValueError('Actual original source ancestor differs')
    state=value.get('network',{});expected=model.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(expected)
            or any(not torch.is_tensor(v) or v.dtype!=expected[k].dtype or v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items())):
        raise ValueError('Strict complete finite headless signal/projection state required')
    model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False),value['identity']


def load_trained(c,device='cuda:0'):return load_trained_model(c,Path(c['output_root'])/'final.pt',device)
def _snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main(argv=None):
    print(json.dumps(dict(event='STAR_SIGNAL_OPEN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    train(_read(a.config),a.device)


if __name__=='__main__':main()
