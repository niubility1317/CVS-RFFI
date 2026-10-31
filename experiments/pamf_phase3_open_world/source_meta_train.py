"""Matched native identity/open ERM and exact second-order source-RX meta DG."""
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
from . import source_meta_core as core
from . import source_style_train as style

identity,body=style.identity,style.body
RUN_ID='20261031-phase3-source-meta-s2026092701-r01'
SCHEMA='pamf_phase3_source_meta_train_v1'
STATUS='SOURCE_META_TRAIN_COMPLETE'
ROW_STATUS='SOURCE_META_STUDENT_FROZEN'
ARMS=('ERM_NATIVE','META_NATIVE')
STEPS=444
COUNTS=dict(L=6300,OE=6134)
META_ALPHA,META_BETA=.001,.5
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
ZEROS=('target_U_reads','target_Q_reads','validation_reads','novel_validation_reads','private_truth_reads','U_truth_reads',
       'original_U_s_reads','source_Us_reads','Sinc_optimizer_steps','physical_clip_calls','source_density_fit_updates','threshold_fit_updates')
_read,_write,_snapshot=style._read,style._write,style._snapshot
_reference,_backend,_load_PCD=style._reference,style._backend,style._load_PCD
_pure_source,_model,_source_cache=style._pure_source,style._model,style._source_cache
_runtime,_gradients,_changed=style._runtime,style._gradients,style._changed


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field source-meta config required; no target/truth/Us/K inputs')
    r=identity.reg.fresh.validate_config(c['reference_config'])
    if r['code_commit']!=identity.REFERENCE_CODE:raise ValueError('Original completed fresh-world reference metadata required')
    source=identity._source_config(c);out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],source['output_root'],source['IQ_root'],source['native_config']['source_output']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):raise ValueError('Immutable source-meta output overlaps an input')
    return c


def _smoke(model,b,device):
    """Real native checkpoint Hessian path, synthetic only, without an update."""
    from .star_open_feature_export import _synthetic
    state=_snapshot(model);buffers={n:v.detach().clone() for n,v in model.named_buffers()};trials={}
    try:
        with torch.random.fork_rng(devices=identity.native._cuda_devices(model)):
            with torch.no_grad():
                x=b['dura'][1](_synthetic(device))[0].detach().clone()
                original=b['reader'].classify_features(b['reader'].features(x));_,initial=model(x)
            if not torch.allclose(initial,original,rtol=identity.LOGIT_RTOL,atol=identity.LOGIT_ATOL):raise ValueError('Initial complete native logits differ from original F0')
            y=torch.tensor([0,1],device=device);direction=torch.linspace(.1,1.,160,device=device)
            def probe(call,flip=False):
                z,logits=call(x.flip(0) if flip else x)
                return F.cross_entropy(logits,y)+(z*direction).sum(1).mean()
            for arm in ARMS:
                model.zero_grad(set_to_none=True)
                result=core.bilevel_loss(model,lambda call:probe(call),lambda call:probe(call,True),alpha=META_ALPHA if arm=='META_NATIVE' else 0.,beta=META_BETA)
                result['total'].backward();g=_gradients(model)
                if any(g[k]<=0 for k in ('head','projection','body','shallow','middle')):raise ValueError('Live finite native head/body/projection/shallow/middle second-order gradients required')
                identity._unchanged(model,state,'No-step virtual native smoke')
                if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Virtual native smoke changed a buffer')
                trials[arm]=dict(status='PASS',gradient_norms=g,**result['diagnostics'])
            return dict(status='PASS',actual_original_F0_equal=True,initial_logit_max_abs_difference=float((initial-original).abs().max().cpu()),
                full_second_order=True,virtual_updates_committed=False,parameters_buffers_unchanged=True,Sinc_gradients_none=True,optimizer_updates=0,real_data_reads=0,
                trials=trials,synthetic_student_forward_rows=10,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
                initial_open_equivalence_to_r03='not_claimed; body initializes from original F0')
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,v in model.named_buffers():v.copy_(buffers[n])


def _objective(call,L,OE,li,oi,*,L_rows=None,OE_rows=None):
    lm,om=L['metadata'],OE['metadata'];n=len(li)
    z,logits=call(torch.cat((L['iq'][li],OE['iq'][oi])));lz,oz=z[:n],z[n:]
    a=torch.arange(n,device=z.device) if L_rows is None else L_rows
    o=torch.arange(len(oi),device=z.device) if OE_rows is None else OE_rows
    values=core.anchor_losses(lz,oz,lm['y'][li],lm['rx_i'][li],lm['day_i'][li],om['rx_i'][oi],om['day_i'][oi],OE['teacher_u'][oi],L_rows=L_rows,OE_rows=OE_rows)
    anchor=.5*(lz[a]-L['parent_z'][li[a]].detach()).square().sum(1).mean()+.5*(oz[o]-OE['parent_z'][oi[o]].detach()).square().sum(1).mean()
    supervised=identity.source_loss(logits[:n][a],lm['y'][li[a]],L['teacher_logits'][li[a]])
    total=values['crossset_metric']+values['OE_KL']+10*anchor+supervised['loss']
    return total,dict(crossset_metric=values['crossset_metric'],OE_KL=values['OE_KL'],source_anchor=anchor,CE=supervised['CE'],KD=supervised['KD'])


def _split_check(banks,sample):
    held=sample['held_rx']
    for role in ('L','OE'):
        rx=banks[role]['metadata']['rx_i'];left=rx[sample['train_'+role]];right=rx[sample['held_'+role]]
        if np.any(left==held) or np.any(right!=held) or set(left.tolist())!=set(core.RECEIVERS)-{held}:raise ValueError('Inner source RX and held RX must be disjoint')


def _loop(c,arm,model,banks,device,folder,check):
    sampler=core.ReceiverSampler(banks['L'],banks['OE']);data={k:_runtime(v,device) for k,v in banks.items()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    alpha=META_ALPHA if arm=='META_NATIVE' else 0.;trace=[];usage={k:set() for k in ('train_L','train_OE','held_L','held_OE')}
    matched=(Path(c['output_root'])/'ERM_NATIVE/steps.jsonl').open(encoding='utf-8') if arm=='META_NATIVE' else None
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step);_split_check(banks,sample)
                if matched is not None:
                    line=matched.readline()
                    if not line:raise ValueError('Matched ERM trajectory is incomplete')
                    old=json.loads(line)
                    if old['step']!=step or old['held_rx']!=sample['held_rx'] or any(old[k+'_indices']!=sample[k].tolist() for k in usage):raise ValueError('Matched source/held index trajectories differ')
                ix={k:identity._tensor(sample[k],device,torch.long) for k in usage}
                all_L=torch.cat((ix['train_L'],ix['held_L']));all_OE=torch.cat((ix['train_OE'],ix['held_OE']))
                held_L=torch.arange(len(ix['train_L']),len(all_L),device=device);held_OE=torch.arange(len(ix['train_OE']),len(all_OE),device=device)
                train_metrics={};held_metrics={}
                def train_loss(call):
                    value,metrics=_objective(call,data['L'],data['OE'],ix['train_L'],ix['train_OE']);train_metrics.update(metrics);return value
                def held_loss(call):
                    value,metrics=_objective(call,data['L'],data['OE'],all_L,all_OE,L_rows=held_L,OE_rows=held_OE);held_metrics.update(metrics);return value
                opt.zero_grad(set_to_none=True);result=core.bilevel_loss(model,train_loss,held_loss,alpha=alpha,beta=META_BETA)
                result['total'].backward();gradients=_gradients(model)
                unused_outer=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite native outer gradient norm')
                opt.step();check()
                for k in usage:usage[k].update(map(int,sample[k]))
                row=dict(step=step,arm=arm,held_rx=sample['held_rx'],receiver_splits_disjoint=True,source_indices_equal=True,
                    **{k+'_indices':sample[k].tolist() for k in usage},loss_total=float(result['total'].detach().cpu()),train_loss=float(result['train'].detach().cpu()),held_loss=float(result['held'].detach().cpu()),
                    train_components={k:float(v.detach().cpu()) for k,v in train_metrics.items()},held_components={k:float(v.detach().cpu()) for k,v in held_metrics.items()},
                    **result['diagnostics'],gradient_norms=gradients,unused_outer_parameters=unused_outer,gradient_norm_before_clip=norm,ordinary_clip_calls=step,
                    optimizer_updates=step,identity_head_updates=step,projection_updates=step,Sinc_optimizer_steps=0,physical_clip_calls=0,
                    base_forward_rows=192,virtual_forward_rows=288,student_forward_rows=480,cumulative_student_forward_rows=480*step,
                    inner_L_count=128,inner_OE_count=64,held_L_anchor_count=64,held_OE_anchor_count=32,
                    virtual_L_candidate_count=192,virtual_OE_candidate_count=96,student_candidates_detached=False,source_anchor_weight=10.,CE_weight=1.,KD_weight=1.,KD_temperature=2.)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('SOURCE_META_PROGRESS',arm,step,STEPS,row['loss_total'],sample['held_rx'],flush=True)
        if matched is not None and matched.readline():raise ValueError('Matched ERM trajectory contains extra updates')
    finally:
        if matched is not None:matched.close()
    return trace,{k:len(v) for k,v in usage.items()}


def _identity(c,arm,pm,parent,manifest):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,
        signal_parameters=335681,reader_parameters=232993,projection_parameters=102688,trainable_parameters=335633,
        Sinc_frozen=True,frozen_Sinc_parameters=48,unused_wrapper_parameters_not_deployed=667729,
        identity_head_updates=STEPS,projection_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,
        parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=parent,dura_identity=pm['dura_identity'],source_feature_manifest=manifest,
        source_only=True,target_U_used=False,target_U_reads=0,target_Q_reads=0,initialization='original_F0_reader_plus_pure_source_r03_projection',
        initial_native_logits_equal_original_F0=True,initial_open_equivalence_to_r03='not_claimed; body initializes from original F0',
        meta_enabled=arm=='META_NATIVE',meta_alpha=META_ALPHA if arm=='META_NATIVE' else 0.,meta_beta=META_BETA,second_order=True,
        full_second_order=True,second_order_contribution_active=arm=='META_NATIVE',virtual_updates_committed=False,
        deployment_state='reader_and_projection_only',target_REG=False)


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-meta output required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,state,tau,_=_reference(c);b=_backend(c,device);pcd,pcd_identity=_load_PCD(c,device)
    _pure_source(b['model'],b['model_identity'],pm)
    if pcd_identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Original frozen F0/r03/PCD/G ancestry differs')
    originals=(b['reader'],b['model'],b['dura'][0],pcd)
    if any(p.requires_grad or p.grad is not None for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):raise ValueError('Original models must remain frozen eval')
    snapshots=[(m,_snapshot(m)) for m in originals];fixed_state=deepcopy(state);fixed_tau=tau.copy()
    def check():
        for model,before in snapshots:
            identity._unchanged(model,before,'Original F0/r03/PCD/G')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen original model received gradients')
        if not identity.reg._same(state,fixed_state) or not np.array_equal(tau,fixed_tau):raise ValueError('Original source statistics/thresholds changed')
    checkpoint=identity.reg._smoke(b,pcd,device);check()
    if checkpoint.get('status')!='PASS' or checkpoint.get('actual_F0_r03_DURA_PCD_checkpoints') is not True:raise ValueError('Actual original no-query smoke required before source values')
    model=_model(c,pm,b['model'],device);initial=_snapshot(model);original=_snapshot(b['reader'])
    if set(original)!={n.removeprefix('reader.') for n in initial if n.startswith('reader.')} or any(not torch.equal(v,initial['reader.'+n]) for n,v in original.items()):raise ValueError('Strict complete original F0 reader initialization required')
    smoke=_smoke(model,b,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=checkpoint,native_meta=smoke))
    roles={};rows={}
    try:
        banks,manifest,roles=_source_cache(c,b,out,check);banks['L']['teacher_logits']=np.asarray(roles['L']['teacher_logits'],np.float32)
        for arm in ARMS:
            if arm=='META_NATIVE':model=_model(c,pm,b['model'],device)
            identity._unchanged(model,initial,'Independent original F0/source-projection initial state')
            buffers={n:v.detach().clone() for n,v in model.named_buffers()};folder=out/arm;folder.mkdir()
            trace,unique=_loop(c,arm,model,banks,device,folder,check);_changed(model,initial);check()
            if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Student buffers changed during source meta training')
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Complete finite native/open state required')
            i=_identity(c,arm,pm,b['model_identity'],manifest);model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=b['model_identity'],source_feature_manifest=manifest),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,updates=STEPS,
                identity_head_updates=STEPS,projection_updates=STEPS,source_only=True,target_U_used=False,initial_state_equal_to_parents=True,
                student_Sinc_unchanged=True,model_buffers_unchanged=True,student_body_changed=True,student_head_changed=True,student_projection_changed=True,original_models_unchanged=True,
                full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,unique_training_rows=unique,
                source_L_sampled_exposures=192*STEPS,source_OE_sampled_exposures=96*STEPS,student_forward_rows=480*STEPS,base_forward_rows=192*STEPS,virtual_forward_rows=288*STEPS,
                first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('SOURCE_META_ARM_COMPLETE',arm,flush=True);del model
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={a:r['identity'] for a,r in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,new_target_UDA_updates=0,source_only=True,same_sampling=True,source_sampling_matched=True,
            parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=b['model_identity'],dura_identity=pm['dura_identity'],source_feature_manifest=manifest,
            baseline_PCD_identity=pcd_identity,baseline_PCD_has_target_U_ancestor=True,all_students_initialized_from_original_F0=True,original_models_unchanged=True,
            all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_heads_changed=True,all_student_projections_changed=True,wrapper_outside_reader_unchanged=True,
            full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,meta_alpha=META_ALPHA,meta_beta=META_BETA,
            source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],cached_L_teacher_logits_reads=COUNTS['L'],real_parent_signal_forward_rows=sum(COUNTS.values()),
            real_student_forward_rows=960*STEPS,real_student_base_forward_rows=384*STEPS,real_student_virtual_forward_rows=576*STEPS,
            real_F_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            synthetic_F_forward_rows=checkpoint['synthetic_F_forward_rows']+smoke['synthetic_F_forward_rows'],synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,
            G_synthetic_forward_rows=checkpoint['G_synthetic_forward_rows']+smoke['G_synthetic_forward_rows'],synthetic_student_forward_rows=smoke['synthetic_student_forward_rows'],
            signal_parameters=335681,trainable_parameters_per_student=335633,wrapper_parameters_not_deployed=667729,source_sampling_seeds=core.SEEDS,
            optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,eps=1e-8,ordinary_clip=5.,ordinary_clip_calls=2*STEPS),
            source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():body.base.close_role(role)


def _initial_for_load(c,pm,device):
    source=identity._source_config(c);parent,parent_identity=body.base.load_trained_model(source,Path(source['output_root'])/'final.pt',device)
    _pure_source(parent,parent_identity,pm);return _model(c,pm,parent,device),parent_identity


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only the two source-native meta students may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both complete source-only students required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_only=True,same_sampling=True,source_sampling_matched=True,all_students_initialized_from_original_F0=True,original_models_unchanged=True,
        all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_heads_changed=True,all_student_projections_changed=True,wrapper_outside_reader_unchanged=True,
        full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,meta_alpha=META_ALPHA,meta_beta=META_BETA,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows:raise ValueError('Complete matched source meta training required')
    pm,_,_,_=_reference(c);model,parent=_initial_for_load(c,pm,device)
    if done.get('parent_signal_identity')!=parent or done.get('parent_F0_identity')!=pm['parent_F0_identity'] or done.get('dura_identity')!=pm['dura_identity']:raise ValueError('Original pure-source model ancestry differs')
    for name,row in rows.items():
        i=_identity(c,name,pm,parent,done['source_feature_manifest'])
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=name,identity=i,updates=STEPS,
            source_only=True,target_U_used=False,initial_state_equal_to_parents=True,student_Sinc_unchanged=True,model_buffers_unchanged=True,student_body_changed=True,student_head_changed=True,student_projection_changed=True,
            original_models_unchanged=True,full_second_order=True,all_receiver_splits_disjoint=True,virtual_updates_committed=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or done['identities'].get(name)!=i:raise ValueError('Strict source-native row and shared head/body/projection identity required')
    payload=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False);i=rows[arm]['identity']
    if (payload.get('schema')!=SCHEMA or payload.get('config')!=c or payload.get('identity')!=i or payload.get('parent_F0_identity')!=pm['parent_F0_identity']
            or payload.get('parent_signal_identity')!=parent or payload.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict original F0/pure-r03 source lineage required')
    network=payload.get('network');initial=_snapshot(model)
    if not isinstance(network,dict) or set(network)!=set(initial) or any(not torch.is_tensor(v) or v.shape!=initial[k].shape or v.dtype!=initial[k].dtype or not bool(torch.isfinite(v).all()) for k,v in network.items()):raise ValueError('Full finite native/projection state with exact shapes/dtypes required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{n for n,p in model.named_parameters() if n.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[n],network[n]) for n in fixed):raise ValueError('Frozen Sinc and persistent buffers must equal original parents')
    model.load_state_dict(network,strict=True);_changed(model,initial);model.eval().requires_grad_(False);model._source_meta_identity=i
    return model,i


def main():
    print(json.dumps(dict(event='SOURCE_META_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(_read(a.config),a.device)


if __name__=='__main__':main()
