"""Matched source-P1/source-pair/target-PCD learning on fixed same-record IQ."""
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
from . import momentum_prototype_train as ancestry
from .paired_signal_model import PairedSignal,parameter_report

body=ancestry.body
fresh=ancestry.fresh
reference=ancestry.reference
RUN_ID='20261031-phase3-paired-signal-s2026092701-r01'
SCHEMA='pamf_phase3_paired_signal_train_v1'
STATUS='PAIRED_SIGNAL_TRAIN_COMPLETE'
ROW_STATUS='PAIRED_SIGNAL_STUDENT_FROZEN'
ARMS=('SOURCE_P1','SOURCE_PAIR','TARGET_PAIR')
STEPS=444
U_COUNT=576
U_BATCH=64
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE=ancestry.REFERENCE_CODE
FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
TRAIN_FIELDS=FIELDS
ZEROS=body.ZEROS
_read,_write,_snapshot=body._read,body._write,body._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
        or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None or type(c.get('steps')) is not int or c['steps']!=STEPS
        or type(c.get('lr')) is not float or c['lr']!=.0001 or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact independent six-field paired training required; no truth/K/query input')
    r=reference.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Exact completed fresh-world reference required')
    t=r['train_config'];p=fresh._flat(t)['reference_config'];source=p['train_config'];out=Path(c['output_root']).resolve()
    roots=[r['output_root'],t['output_root'],t['U_public_root'],r['source_control_config']['output_root'],p['output_root'],p['reference_prediction_root'],p['dura_checkpoint'],p['dura_release'],source['output_root'],source['IQ_root'],source['feature_root'],source['control_root']]+[v['public_root'] for v in r['Q_conditions'].values()]
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable pair output overlaps frozen inputs')
    return c


def _load_U(c,manifest,source_ids):
    original=ancestry._load_U(c,manifest,source_ids);t=c['reference_config']['train_config'];iq=None
    try:
        public,ids,iq=fresh._public(t)
        if public!=manifest or not np.array_equal(ids,original['ids']):raise ValueError('Same canonical public P0/cache P1 ID order required')
        p0=np.asarray(iq[:],np.float32).copy()
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    with np.load(Path(t['output_root'])/'U_features.npz',allow_pickle=False) as z:neighbors=z['neighbors'].copy()
    if (neighbors.shape!=(U_COUNT,) or neighbors.dtype!=np.int64 or np.any((neighbors<0)|(neighbors>=U_COUNT))
        or np.any(neighbors==np.arange(U_COUNT))):raise ValueError('Original fixed nonself parent-z nearest indices required')
    if p0.shape!=(U_COUNT,2,256) or not np.isfinite(p0).all():raise ValueError('Finite original same-record received IQ required')
    generators=[torch.Generator(device='cpu').manual_seed(body.AUG_SEED) for _ in range(2)]
    received=body.base._tensor(p0,'cpu');restored=body.base._tensor(original['views'][0],'cpu');views=[p0]
    for j in (1,2):
        raw=body.views.phase_gain_view(received,generators[0]);replayed=body.views.phase_gain_view(restored,generators[1])
        if not np.array_equal(np.asarray(replayed.tolist(),np.float32),original['views'][j]):raise ValueError('P0/P1 phase-gain synchronization differs from original fixed P1 cache')
        views.append(np.asarray(raw.tolist(),np.float32))
    return dict(ids=original['ids'],P0_views=np.stack(views),P1_views=original['views'],parent_z=original['parent_z'],neighbors=neighbors)


def _runtime(bank,device,paired):
    result=body._runtime(bank,device)
    result['received']=body.base._tensor(bank['received'] if paired else bank['iq'],device)
    return result


def _forward(model,p0,p1,norms):
    value=model.features(p0,p1)
    if norms is not None:norms.append(value['delta'].detach().norm(dim=1).max())
    return value['z']


def _source_objective(model,L,OE,ix,oi,device,norms=None):
    i=body.base._tensor(ix,device,torch.long);o=body.base._tensor(oi,device,torch.long);lm,om=L['metadata'],OE['metadata']
    z=_forward(model,L['received'][i],L['iq'][i],norms);oz=_forward(model,OE['received'][o],OE['iq'][o],norms)
    known=body.base.loss.crossset_metric(z,oz,lm['y'][i],lm['rx_i'][i],lm['day_i'][i],om['rx_i'][o],om['day_i'][o])
    other=body.base.loss.oe_structure(oz,OE['teacher_u'][o].detach(),om['rx_i'][o],om['day_i'][o])
    anchor=.5*(z-L['parent_z'][i].detach()).square().sum(1).mean()+.5*(oz-OE['parent_z'][o].detach()).square().sum(1).mean()
    total=known['loss']+other['loss']+10*anchor
    return total,dict(source_loss=float(total.detach().cpu()),crossset_metric=float(known['loss'].detach().cpu()),OE_KL=float(other['loss'].detach().cpu()),source_anchor=float(anchor.detach().cpu()))


def _gradients(model,prototypes=None):
    result=body._gradients(model.base,prototypes)
    result['adapter']=sum(float(p.grad.detach().double().square().sum().cpu()) for p in model.adapter.parameters() if p.grad is not None)**.5
    return result


def _gradient_smoke(model,device):
    before=_snapshot(model)
    try:
        with torch.random.fork_rng(devices=body.base.native._cuda_devices(model)):
            torch.manual_seed(2026103115);p0=torch.randn(32,2,256,device=device);p1=.97*p0
            with torch.no_grad():weak=model(p0,p1)
            centers,_=body.core.initialize_prototypes(weak);head=torch.nn.Parameter(centers);neighbors=body.core.nearest_neighbors(weak)
            generators=[torch.Generator(device='cpu').manual_seed(body.AUG_SEED) for _ in range(2)]
            first=[body.views.phase_gain_view(p0[:8].detach().cpu(),generators[0]).to(device) for _ in range(2)]
            second=[body.views.phase_gain_view(p1[:8].detach().cpu(),generators[1]).to(device) for _ in range(2)]
            strong=[model(first[j],second[j]) for j in range(2)]
            pcd=body.core.consistency(weak,*strong,head,torch.arange(8,device=device),neighbors)
            probe=(strong[0]*torch.linspace(.1,1.,160,device=device)).sum(1).mean()
            (probe+.1*pcd['loss']).backward();g=_gradients(model,head)
            if any(not np.isfinite(v) for v in g.values()) or any(g[k]<=0 for k in ('body','middle','hidden','last','adapter','prototype')):
                raise ValueError('Synthetic same-record source-probe/PCD base and adapter gradients must be live')
            if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()):raise ValueError('No-step Pair smoke changed parameters/buffers')
            return dict(status='PASS',gradient_norms=g,optimizer_updates=0,Sinc_gradients_none=True,parameters_buffers_unchanged=True,
                synthetic_student_composite_rows=48,synthetic_student_base_rows=96,objective='same-record synthetic linear source probe plus original global/local PCD; no real source/U')
    finally:model.zero_grad(set_to_none=True)


def _identity(c,arm,parent,m,U_manifest,control):
    target=arm=='TARGET_PAIR';flat=fresh._flat(c['reference_config']['train_config'])
    value=body._identity(flat,'TARGET_PCD' if target else 'SOURCE',parent,m,U_manifest)
    value.update(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),
        signal_parameters=386081,base_signal_parameters=334721,pair_adapter_parameters=51360,trainable_parameters=386033,
        paired_inputs=arm!='SOURCE_P1',source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,
        U_manifest=U_manifest if target else None,updates=STEPS,identity_head_updates=0,source_control_identity=control['identity'],source_control_source_only=True,
        temporary_head_parameters=2560 if target else 0,prototype_deployed=False,per_record_U_bank_deployed=False,
        initialization='independent_pure_source_r03_with_zero_pair_adapter',initial_P1_equivalence='mathematical_not_bitwise_FP32')
    return value


def _loop(model,banks,c,arm,device,opt,folder,control_path,U=None,prototypes=None):
    p=fresh._flat(c['reference_config']['train_config'])['reference_config']['train_config'];sampler=body.base.Sampler(banks['L'],banks['OE'],p['L_seed'],p['OE_seed'])
    data={k:_runtime(v,device,model.paired) for k,v in banks.items()};target=arm=='TARGET_PAIR'
    if target:
        uv={k:body.base._tensor(U[k],device) for k in ('P0_views','P1_views')};neighbors=body.base._tensor(U['neighbors'],device,torch.long);rng=np.random.default_rng(body.U_SEED)
    trace=[];usage={k:set() for k in ('L','OE','U')}
    with control_path.open(encoding='utf-8') as old,(folder/'steps.jsonl').open('x',encoding='utf-8') as log:
        for step in range(1,STEPS+1):
            sample=sampler.sample(step);source=json.loads(old.readline())
            if source.get('step')!=step or any(source.get(k+'_indices')!=sample[k].tolist() for k in ('L','OE')):raise ValueError('Source sampling differs from frozen SOURCE444')
            opt.zero_grad(set_to_none=True);norms=[];total,metrics=_source_objective(model,data['L'],data['OE'],sample['L'],sample['OE'],device,norms)
            ui=np.empty(0,np.int64);terms=dict(U_global=0.,U_local=0.)
            if target:
                ui=rng.choice(U_COUNT,U_BATCH,replace=False);i=body.base._tensor(ui,device,torch.long)
                with torch.no_grad():weak=_forward(model,uv['P0_views'][0],uv['P1_views'][0],norms)
                strong=[_forward(model,uv['P0_views'][j,i],uv['P1_views'][j,i],norms) for j in (1,2)]
                loss=body.core.consistency(weak,*strong,prototypes,i,neighbors);total=total+.1*loss['loss']
                terms=dict(U_global=float(loss['global'].detach().cpu()),U_local=float(loss['local'].detach().cpu()))
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite matched Pair source/PCD objective')
            total.backward();g=_gradients(model,prototypes);norm=float(torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],5.))
            if not np.isfinite(norm) or any(not np.isfinite(v) for v in g.values()):raise FloatingPointError('Nonfinite Pair gradient')
            opt.step()
            if target:
                with torch.no_grad():prototypes.copy_(F.normalize(prototypes,dim=1))
            for k in ('L','OE'):usage[k].update(map(int,sample[k]))
            usage['U'].update(map(int,ui));composite=(256+(U_COUNT+2*U_BATCH)*int(target))*step
            row=dict(step=step,arm=arm,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),U_indices=ui.tolist(),loss_total=float(total.detach().cpu()),**metrics,**terms,
                source_anchor_weight=10.,U_PCD_weight=.1 if target else 0.,gradient_norms=g,gradient_norm_before_clip=norm,
                delta_norm_max=float(torch.stack(norms).max().cpu()),source_L_exposures=192*step,source_OE_exposures=64*step,
                U_weak_exposures=U_COUNT*step if target else 0,U_strong_exposures=2*U_BATCH*step if target else 0,
                composite_forward_rows=composite,base_signal_forward_rows=2*composite,student_optimizer_updates=step,
                ordinary_clip_calls=step,Sinc_optimizer_steps=0,physical_clip_calls=0,source_indices_equal=True)
            trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            if step==1 or step%100==0 or step==STEPS:print('PAIRED_SIGNAL_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if old.readline():raise ValueError('SOURCE control trace contains extra updates')
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable Pair training output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;t=c['reference_config']['train_config'];flat=fresh._flat(t)
    b=body._backend(flat,device);snapshots=[(model,_snapshot(model)) for model in (b['reader'],b['model'],b['dura'][0])]
    def check():
        if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for model,state in snapshots for k,v in state.items()):raise ValueError('Original F0/r03/DURA changed')
    smoke=body.parent._smoke(b,device);check()
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_signal_DURA_checkpoints') is not True:raise ValueError('Actual no-query checkpoint smoke must precede source/U')
    pm,_,_,_=fresh._reference(t);control,control_steps=fresh._source_control(t,pm);_,U_manifest=ancestry._fresh_metadata(c,pm,control);parent=pm['model_identity']
    if b['model_identity']!=parent or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Pure source r03/F0/G ancestry required')
    probe=PairedSignal(body._student(b['model']),True);gradient=_gradient_smoke(probe,device);del probe;check()
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=smoke,gradient=gradient));roles={}
    try:
        banks,m,roles=body._source_cache(flat,b,out,check)
        if m!=control['source_feature_manifest']:raise ValueError('Source handles differ from original SOURCE444')
        for role,bank in banks.items():
            if not np.array_equal(bank['ids'],roles[role]['ids']) or roles[role]['received'].shape!=bank['iq'].shape or roles[role]['received'].dtype!=np.float32:
                raise ValueError('Same-ID cached source received/restored pairs required')
            bank['received']=roles[role]['received']
        initial=_snapshot(b['model']);rows={};source_trajectory=None;U=None;oe_teacher=banks['OE']['teacher_u'].copy()
        for arm in ARMS:
            target=arm=='TARGET_PAIR';head=None
            if target:
                if any(not (out/a/'complete.json').is_file() for a in ARMS[:2]):raise ValueError('Both new source students must freeze before U')
                U=_load_U(c,U_manifest,set(banks['L']['ids'].tolist())|set(banks['OE']['ids'].tolist()))
                _write(out/'U_manifest.json',U_manifest);_write(out/'U_cache_reference.json',dict(path=str(Path(t['output_root'])/'U_features.npz'),
                    cached_P1_reused=True,new_P0_rows=U_COUNT,new_DURA_rows=0,new_physical_observations=0,paired_augmentations_synchronized=True))
                centers,initialization=body.core.initialize_prototypes(body.base._tensor(U['parent_z'],device));head=torch.nn.Parameter(centers);_write(out/'prototype_initialization.json',initialization)
            model=PairedSignal(body._student(b['model']),arm!='SOURCE_P1')
            if any(not torch.equal(v,model.base.state_dict()[k].detach().cpu()) for k,v in initial.items()):raise ValueError('All three bases must independently initialize from original pure r03')
            if parameter_report(model)!=dict(signal_parameters=386081,base_signal_parameters=334721,pair_adapter_parameters=51360,trainable_parameters=386033):raise ValueError('Actual full paired model parameter contract differs')
            folder=out/arm;folder.mkdir();before=_snapshot(model);buffers={k:v.detach().clone() for k,v in model.named_buffers()}
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]+([head] if target else []),lr=c['lr'],weight_decay=.0001,eps=1e-8)
            trace,unique=_loop(model,banks,c,arm,device,opt,folder,control_steps,U,head)
            trajectory=[(x['L_indices'],x['OE_indices']) for x in trace]
            if source_trajectory is not None and trajectory!=source_trajectory:raise ValueError('Three matched source streams differ')
            source_trajectory=trajectory
            if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items()):raise ValueError('Pair buffers changed')
            state=_snapshot(model);changed={k:not torch.equal(v,state[k]) for k,v in before.items()}
            if any(v for k,v in changed.items() if k.startswith('base.'+body.signalcore.SINC_PREFIX)):raise ValueError('Frozen base Sinc changed')
            if (not any(v for k,v in changed.items() if k.startswith('base.signal.') and not k.startswith('base.'+body.signalcore.SINC_PREFIX))
                or not any(v for k,v in changed.items() if k.startswith('base.projection.')) or not any(v for k,v in changed.items() if k.startswith('adapter.'))):raise ValueError('Base body/projection and pair adapter must actually update')
            if any(not bool(torch.isfinite(v).all()) for v in state.values()) or not np.array_equal(oe_teacher,banks['OE']['teacher_u']):raise ValueError('Finite Pair state and frozen OE teacher required')
            check();identity=_identity(c,arm,parent,m,U_manifest,control)
            torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=state,parent_signal_identity=parent,source_feature_manifest=m),folder/'final.pt')
            if target:body._npz(folder/'prototype_state.npz',dict(prototypes=np.asarray(head.detach().cpu().tolist(),np.float32)))
            composite=(256+(U_COUNT+2*U_BATCH)*int(target))*STEPS
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=identity,updates=STEPS,
                source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,U_manifest=U_manifest if target else None,
                source_feature_manifest=m,initial_base_state_equal_to_parent=True,adapter_initially_zero=True,student_Sinc_unchanged=True,student_body_changed=True,
                student_projection_changed=True,adapter_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,composite_forward_rows=composite,
                base_signal_forward_rows=2*composite,source_L_exposures=192*STEPS,source_OE_exposures=64*STEPS,unique_training_rows=unique,
                first_step=trace[0],last_step=trace[-1],prototype_deployed=False,per_record_U_bank_deployed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('PAIRED_SIGNAL_ARM_COMPLETE',arm,flush=True);del model,opt,head
        composite=sum(row['composite_forward_rows'] for row in rows.values())
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={a:v['identity'] for a,v in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=3*STEPS,new_target_UDA_updates=STEPS,parent_signal_identity=parent,parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],
            source_control_identity=control['identity'],source_pair_identity=rows['SOURCE_PAIR']['identity'],source_control_source_only=True,source_control_frozen_before_training=True,
            SOURCE_completed_before_U_open=True,source_sampling_matched=True,source_feature_manifest=m,U_manifest=U_manifest,target_U_reads=U_COUNT,source_only=False,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,all_adapters_changed=True,
            target_U_P0_reads=U_COUNT,U_P1_cache_reused=True,paired_augmentations_synchronized=True,source_P0_cache_rows=sum(COUNTS.values()),
            real_parent_signal_forward_rows=sum(COUNTS.values()),real_student_composite_forward_rows=composite,real_student_base_signal_forward_rows=2*composite,
            real_student_signal_forward_rows=2*composite,G_forward_rows=0,real_F_forward_rows=0,synthetic_F_forward_rows=2,synthetic_parent_signal_forward_rows=2,G_synthetic_forward_rows=2,
            synthetic_student_composite_rows=gradient['synthetic_student_composite_rows'],synthetic_student_base_rows=gradient['synthetic_student_base_rows'],
            prototype_deployed=False,per_record_U_bank_deployed=False,formal_target_accessed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for value in roles.values():body.base.close_role(value)


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only the three completed Pair arms may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('All three Pair students must complete before prediction') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=3*STEPS,new_target_UDA_updates=STEPS,
        source_control_source_only=True,source_control_frozen_before_training=True,SOURCE_completed_before_U_open=True,source_sampling_matched=True,source_only=False,target_U_reads=U_COUNT,
        original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_projections_changed=True,all_adapters_changed=True,
        final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows:raise ValueError('Complete matched pure-source/target Pair metadata required')
    t=c['reference_config']['train_config'];pm,_,_,_=fresh._reference(t);control,_=fresh._source_control(t,pm);_,manifest=ancestry._fresh_metadata(c,pm,control)
    p=fresh._flat(t)['reference_config']['train_config'];base,parent=body.base.load_trained_model(p,Path(p['output_root'])/'final.pt',device)
    if parent!=pm['model_identity'] or done.get('parent_signal_identity')!=parent or done.get('source_control_identity')!=control['identity'] or done.get('U_manifest')!=manifest:raise ValueError('Strict pure r03/SOURCE444/paired-U ancestry required')
    for a,row in rows.items():
        identity=_identity(c,a,parent,done['source_feature_manifest'],manifest,control);target=a=='TARGET_PAIR'
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=a,identity=identity,updates=STEPS,
            source_only=not target,target_U_used=target,target_U_reads=U_COUNT if target else 0,initial_base_state_equal_to_parent=True,adapter_initially_zero=True,
            student_Sinc_unchanged=True,student_body_changed=True,student_projection_changed=True,adapter_changed=True,model_buffers_unchanged=True,original_models_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or done.get('identities',{}).get(a)!=identity:raise ValueError('Full per-arm Pair identity differs')
    identity=done['identities'][arm];model=PairedSignal(body._student(base),arm!='SOURCE_P1')
    value=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False)
    if value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=identity or value.get('parent_signal_identity')!=parent or value.get('source_feature_manifest')!=done['source_feature_manifest']:raise ValueError('Strict Pair checkpoint config/identity/source handles required')
    state=value.get('network');original=model.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(original) or any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in state.values())
        or any(state[k].shape!=original[k].shape or state[k].dtype!=original[k].dtype for k in original)):raise ValueError('Finite strict full Pair tensor shapes/dtypes required')
    fixed=(set(dict(model.named_buffers()))&set(original))|{k for k,p in model.named_parameters() if k.startswith('base.'+body.signalcore.SINC_PREFIX)}
    if any(not torch.equal(original[k].detach().cpu(),state[k]) for k in fixed):raise ValueError('Frozen Pair Sinc/persistent buffers changed')
    if not any(not torch.equal(original[k].detach().cpu(),state[k]) for k in state if k.startswith('adapter.')):raise ValueError('Actual trained pair adapter required')
    try:model.load_state_dict(state,strict=True)
    except RuntimeError as exc:raise ValueError('Strict full shared-base Pair network required') from exc
    if parameter_report(model)['signal_parameters']!=386081:raise ValueError('Full 386081-parameter Pair model required')
    model.eval().requires_grad_(False);model._paired_signal_identity=identity;return model,identity


def main():
    print(json.dumps(dict(event='PAIRED_SIGNAL_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0');a=parser.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
