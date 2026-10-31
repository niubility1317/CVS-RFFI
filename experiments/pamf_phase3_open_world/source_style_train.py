"""Two matched source-only native identity/open students, with source MixStyle."""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import sys
import numpy as np
import torch
from torch.nn import functional as F
from . import source_style_core as core
from . import native_identity_train as identity
from . import fresh_pcd_train as fresh

body=fresh.body
RUN_ID='20261031-phase3-source-style-s2026092701-r01'
SCHEMA='pamf_phase3_source_style_train_v1'
STATUS='SOURCE_STYLE_TRAIN_COMPLETE'
ROW_STATUS='SOURCE_STYLE_STUDENT_FROZEN'
ARMS=('SOURCE_NATIVE','STYLE_NATIVE')
STEPS=444
COUNTS=dict(L=6300,OE=6134)
REFERENCE_CODE=identity.REFERENCE_CODE
FIELDS=TRAIN_FIELDS={'run_id','code_commit','reference_config','output_root','steps','lr'}
ZEROS=('target_U_reads','target_Q_reads','validation_reads','novel_validation_reads','private_truth_reads','U_truth_reads',
       'original_U_s_reads','source_Us_reads','Sinc_optimizer_steps','physical_clip_calls','source_density_fit_updates','threshold_fit_updates')
_read,_write,_snapshot=identity._read,identity._write,identity._snapshot


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('steps')) is not int or c['steps']!=STEPS or type(c.get('lr')) is not float or c['lr']!=.0001
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact six-field source-style config required; no target/truth/K inputs')
    r=identity.reg.fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Original completed fresh-world metadata and source handles required')
    source=identity._source_config(c);out=Path(c['output_root']).resolve()
    roots=[r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],source['output_root'],source['IQ_root'],source['native_config']['source_output']]
    roots.extend(v['public_root'] for v in r['Q_conditions'].values())
    if any(out.is_relative_to(Path(v).resolve()) or Path(v).resolve().is_relative_to(out) for v in roots):
        raise ValueError('New immutable source-style output overlaps an input')
    return c


def _reference(c):return identity._reference(c)
def _backend(c,device):return identity._backend(c,device)
def _load_PCD(c,device):return identity._load_PCD(c,device)


def _pure_source(model,parent_identity,pm):
    # The historical r03 identity has no source_only member. Its existing
    # strict loader checks source_only in complete.json and the checkpoint.
    if (parent_identity!=pm['model_identities']['E0'] or parent_identity.get('run_id')!=body.base.RUN_ID
            or parent_identity.get('arm')!='E_SIGNAL' or parent_identity.get('signal_parameters')!=334721
            or ('source_only' in parent_identity and parent_identity['source_only'] is not True)
            or sum(p.numel() for p in model.projection.parameters())!=102688):
        raise ValueError('Only the complete pure-source r03 projection may initialize a student')


def _model(c,pm,parent,device):
    wrapper=identity._initial_model(c,pm,device)
    return core.NativeOpen(wrapper.id_backbone.encoder,parent.projection).to(device)


def _source_cache(c,b,out,check):
    return body._source_cache(fresh._flat(c['reference_config']['train_config']),b,out,check)


def _gradients(model):
    r=model.reader;b=r.core.id_backbone
    norm=lambda ps:sum(float(p.grad.detach().double().square().sum().cpu()) for p in ps if p.grad is not None)**.5
    g=dict(head=norm([b.cls_head.weight]),projection=norm(model.projection.parameters()),
           projection_hidden=norm(model.projection.network[0].parameters()),projection_last=norm(model.projection.network[-1].parameters()),
           shallow=norm(b.time_down.parameters()),middle=norm(b.t2.parameters()),Sinc=norm(b.sinc.parameters()),
           body=norm(p for n,p in r.named_parameters() if n!=identity.READER_HEAD and not n.startswith(identity.READER_SINC)))
    if any(p.grad is not None for p in b.sinc.parameters()) or any(not np.isfinite(v) for v in g.values()) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite source-style gradients or frozen Sinc received a gradient')
    return g


def _smoke(model,b,device):
    from .star_open_feature_export import _synthetic
    before=_snapshot(model);counts={path:len(model.reader.get_submodule(path)._forward_hooks) for path in core.SITES}
    try:
        with torch.random.fork_rng(devices=identity.native._cuda_devices(model)):
            with torch.no_grad():
                x=b['dura'][1](_synthetic(device))[0].detach().clone()
                original=b['reader'].classify_features(b['reader'].features(x));_,initial=model(x)
            if not torch.allclose(initial,original,rtol=identity.LOGIT_RTOL,atol=identity.LOGIT_ATOL):
                raise ValueError('Initial actual native logits differ from original F0')
            # Two synthetic paired-L rows plus two synthetic OE passthrough
            # rows make the actual-checkpoint OE assertion non-vacuous.
            sx=torch.cat((x,x),dim=0)
            with core.style_context(model.reader,core.style_plan(1,2),np.random.default_rng(core.STYLE_SEED),force=True) as sites:
                z,logits=model(sx)
                loss=F.cross_entropy(logits[:2],torch.tensor([0,0],device=device))+(z*torch.linspace(.1,1.,160,device=device)).sum(1).mean()
            loss.backward();g=_gradients(model)
            if any(g[k]<=0 for k in ('head','projection','body','shallow','middle')):raise ValueError('Native head/body/projection and style content gradients must be live')
            if any(len(model.reader.get_submodule(path)._forward_hooks)!=count for path,count in counts.items()):raise ValueError('Source style hooks leaked after context exit')
            with torch.no_grad():_,after=model(x)
            if not torch.equal(initial,after):raise ValueError('Exiting style context must restore initial same-input logits exactly')
            identity._unchanged(model,before,'No-step native/open/style smoke')
            return dict(status='PASS',actual_original_F0_equal=True,initial_logit_max_abs_difference=float((initial-original).abs().max().cpu()),
                logit_rtol=identity.LOGIT_RTOL,logit_atol=identity.LOGIT_ATOL,gradient_norms=g,style_sites=sites,
                hooks_removed=True,unhooked_logits_restored_exactly=True,parameters_buffers_unchanged=True,Sinc_gradients_none=True,
                optimizer_updates=0,real_data_reads=0,synthetic_student_forward_rows=8,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2,
                initial_open_equivalence_to_r03='not_claimed; body initializes from original F0')
    finally:model.zero_grad(set_to_none=True)


def _runtime(bank,device):
    return dict(iq=identity._tensor(bank['iq'],device),parent_z=identity._tensor(bank['parent_z'],device),
                metadata={k:identity._tensor(v,device,torch.long) for k,v in bank['metadata'].items()},
                **({'teacher_u':identity._tensor(bank['teacher_u'],device)} if 'teacher_u' in bank else {}),
                **({'teacher_logits':identity._tensor(bank['teacher_logits'],device)} if 'teacher_logits' in bank else {}))


def _objective(model,L,OE,sample,style_rng,styled):
    li=identity._tensor(np.concatenate((sample['A'],sample['B'])),L['iq'].device,torch.long)
    oi=identity._tensor(sample['OE'],L['iq'].device,torch.long)
    x=torch.cat((L['iq'][li],OE['iq'][oi]));context=core.style_context(model.reader,core.style_plan(),style_rng) if styled else nullcontext({})
    with context as sites:z,logits=model(x)
    lz,oz=z[:256],z[256:];lm,om=L['metadata'],OE['metadata']
    metric=body.base.loss.crossset_metric(lz,oz,lm['y'][li],lm['rx_i'][li],lm['day_i'][li],om['rx_i'][oi],om['day_i'][oi])
    other=body.base.loss.oe_structure(oz,OE['teacher_u'][oi].detach(),om['rx_i'][oi],om['day_i'][oi])
    anchor=.5*(lz-L['parent_z'][li].detach()).square().sum(1).mean()+.5*(oz-OE['parent_z'][oi].detach()).square().sum(1).mean()
    supervised=identity.source_loss(logits[:256],lm['y'][li],L['teacher_logits'][li])
    total=metric['loss']+other['loss']+10*anchor+supervised['loss']
    return total,dict(crossset_metric=float(metric['loss'].detach().cpu()),OE_KL=float(other['loss'].detach().cpu()),
        source_anchor=float(anchor.detach().cpu()),CE=float(supervised['CE'].detach().cpu()),KD=float(supervised['KD'].detach().cpu())),sites


def _loop(c,arm,model,banks,device,folder,check):
    sampler=core.SourceSampler(banks['L'],banks['OE']);data={k:_runtime(v,device) for k,v in banks.items()}
    styled=arm=='STYLE_NATIVE';rng=np.random.default_rng(core.STYLE_SEED)
    matched=(Path(c['output_root'])/'SOURCE_NATIVE/steps.jsonl').open(encoding='utf-8') if styled else None
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=.0001,eps=1e-8)
    trace=[];usage={k:set() for k in ('A','B','OE')}
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,STEPS+1):
                sample=sampler.sample(step)
                if matched is not None:
                    left=json.loads(matched.readline())
                    if left['step']!=step or any(left[k+'_indices']!=sample[k].tolist() for k in sample):raise ValueError('Matched complete source A/B/OE trajectory differs')
                meta=banks['L']['metadata'];a,partner=sample['A'],sample['B']
                if not np.array_equal(meta['y'][a],meta['y'][partner]) or np.any(meta['rx_i'][a]==meta['rx_i'][partner]):raise ValueError('Every source pair must share TX and differ RX')
                opt.zero_grad(set_to_none=True);total,metrics,sites=_objective(model,data['L'],data['OE'],sample,rng,styled)
                if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source-style joint objective')
                total.backward();g=_gradients(model);norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.))
                if not np.isfinite(norm):raise FloatingPointError('Nonfinite ordinary source-style gradient norm')
                opt.step();check()
                for key in usage:usage[key].update(map(int,sample[key]))
                row=dict(step=step,arm=arm,**{k+'_indices':v.tolist() for k,v in sample.items()},loss_total=float(total.detach().cpu()),**metrics,
                    source_anchor_weight=10.,CE_weight=1.,KD_weight=1.,KD_temperature=2.,style_sites=sites,source_pairs_same_TX_different_RX=True,
                    source_indices_equal=True,gradient_norms=g,gradient_norm_before_clip=norm,ordinary_clip_calls=step,
                    student_optimizer_updates=step,identity_head_updates=step,projection_updates=step,Sinc_optimizer_steps=0,physical_clip_calls=0,
                    student_forward_rows=320,cumulative_student_forward_rows=320*step,source_L_exposures=256*step,source_OE_exposures=64*step,
                    style_hooks_removed=True,source_only=True,target_U_reads=0,target_Q_reads=0)
                trace.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                if step==1 or step%100==0 or step==STEPS:print('SOURCE_STYLE_PROGRESS',arm,step,STEPS,row['loss_total'],flush=True)
        if matched is not None and matched.readline():raise ValueError('Source control has extra updates')
        return trace,{k:len(v) for k,v in usage.items()}
    finally:
        if matched is not None:matched.close()


def _identity(c,arm,pm,parent,manifest):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=arm,checkpoint=str(Path(c['output_root'])/arm/'final.pt'),updates=STEPS,
        signal_parameters=335681,reader_parameters=232993,projection_parameters=102688,trainable_parameters=335633,
        Sinc_frozen=True,frozen_Sinc_parameters=48,unused_wrapper_parameters_not_deployed=667729,
        identity_head_updates=STEPS,projection_updates=STEPS,identity_head_jointly_trained=True,identity_head_replaced=False,
        parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=parent,dura_identity=pm['dura_identity'],source_feature_manifest=manifest,
        source_only=True,target_U_used=False,target_U_reads=0,target_Q_reads=0,initialization='original_F0_reader_plus_pure_source_r03_projection',
        initial_native_logits_equal_original_F0=True,initial_open_equivalence_to_r03='not_claimed; body initializes from original F0',
        style_enabled=arm=='STYLE_NATIVE',source_pair_recipe='same_TX_different_RX',deployment_state='reader_and_projection_only',target_REG=False)


def _changed(model,initial):
    state=model.state_dict();params=dict(model.named_parameters());fixed=(set(dict(model.named_buffers()))&set(state))|{n for n in params if n.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[n],state[n].detach().cpu()) for n in fixed):raise ValueError('Source student Sinc/persistent buffers changed')
    conditions=[not torch.equal(initial[core.HEAD_WEIGHT],state[core.HEAD_WEIGHT].detach().cpu()),
        any(not torch.equal(initial[n],p.detach().cpu()) for n,p in params.items() if n.startswith('reader.') and n!=core.HEAD_WEIGHT and not n.startswith(core.SINC_PREFIX)),
        any(not torch.equal(initial[n],p.detach().cpu()) for n,p in params.items() if n.startswith('projection.'))]
    if not all(conditions):raise ValueError('Native body, original head and source projection must all actually update')


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source-style training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pm,state,tau,_=_reference(c);b=_backend(c,device);pcd,pcd_identity=_load_PCD(c,device)
    _pure_source(b['model'],b['model_identity'],pm)
    if pcd_identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:
        raise ValueError('Original frozen F0/r03/PCD/G ancestry differs')
    originals=(b['reader'],b['model'],b['dura'][0],pcd)
    if any(p.requires_grad or p.grad is not None for m in originals for p in m.parameters()) or any(v.training for m in originals for v in m.modules()):raise ValueError('Original models must remain frozen eval')
    snapshots=[(m,_snapshot(m)) for m in originals];fixed_state=deepcopy(state);fixed_tau=tau.copy()
    def check():
        for model,before in snapshots:
            identity._unchanged(model,before,'Original F0/r03/PCD/G')
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen original model received gradients')
        if not identity.reg._same(state,fixed_state) or not np.array_equal(tau,fixed_tau):raise ValueError('Original source readout state changed')
    checkpoint=identity.reg._smoke(b,pcd,device);check()
    if checkpoint.get('status')!='PASS' or checkpoint.get('actual_F0_r03_DURA_PCD_checkpoints') is not True:raise ValueError('Actual no-query checkpoint smoke must pass before source values')
    model=_model(c,pm,b['model'],device);initial=_snapshot(model)
    original=_snapshot(b['reader'])
    if any(not torch.equal(v,initial['reader.'+k]) for k,v in original.items()) or set(original)!={k.removeprefix('reader.') for k in initial if k.startswith('reader.')}:
        raise ValueError('Strict complete original F0 reader initialization required')
    smoke=_smoke(model,b,device);check();out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',dict(checkpoint=checkpoint,native_open_style=smoke))
    roles={};rows={}
    try:
        banks,manifest,roles=_source_cache(c,b,out,check)
        banks['L']['teacher_logits']=roles['L']['teacher_logits']
        for arm in ARMS:
            if arm=='STYLE_NATIVE':model=_model(c,pm,b['model'],device)
            identity._unchanged(model,initial,'Independent original-F0/source-projection initialization')
            folder=out/arm;folder.mkdir();trace,unique=_loop(c,arm,model,banks,device,folder,check);_changed(model,initial);check()
            network=_snapshot(model)
            if any(not bool(torch.isfinite(v).all()) for v in network.values()):raise ValueError('Finite complete native/open state required')
            i=_identity(c,arm,pm,b['model_identity'],manifest);model.eval().requires_grad_(False)
            torch.save(dict(schema=SCHEMA,config=c,identity=i,network=network,parent_F0_identity=pm['parent_F0_identity'],
                parent_signal_identity=b['model_identity'],source_feature_manifest=manifest),folder/'final.pt')
            row=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=arm,identity=i,updates=STEPS,
                identity_head_updates=STEPS,projection_updates=STEPS,source_only=True,target_U_used=False,initial_state_equal_to_parents=True,
                student_Sinc_unchanged=True,model_buffers_unchanged=True,student_body_changed=True,student_head_changed=True,student_projection_changed=True,
                original_models_unchanged=True,style_hooks_removed=True,source_pairs_same_TX_different_RX=True,unique_training_rows=unique,
                source_L_exposures=256*STEPS,source_OE_exposures=64*STEPS,student_forward_rows=320*STEPS,
                first_step=trace[0],last_step=trace[-1],final_phase3_complete=False,**{k:0 for k in ZEROS})
            _write(folder/'complete.json',row);rows[arm]=row;print('SOURCE_STYLE_ARM_COMPLETE',arm,flush=True);del model
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),rows=rows,identities={a:r['identity'] for a,r in rows.items()},
            updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,new_target_UDA_updates=0,source_only=True,same_sampling=True,source_sampling_matched=True,
            parent_F0_identity=pm['parent_F0_identity'],parent_signal_identity=b['model_identity'],dura_identity=pm['dura_identity'],source_feature_manifest=manifest,
            baseline_PCD_identity=pcd_identity,baseline_PCD_has_target_U_ancestor=True,all_students_initialized_from_original_F0=True,
            original_models_unchanged=True,all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,
            all_student_bodies_changed=True,all_student_heads_changed=True,all_student_projections_changed=True,wrapper_outside_reader_unchanged=True,
            all_style_hooks_removed=True,all_source_pairs_same_TX_different_RX=True,source_L_reads=COUNTS['L'],source_OE_reads=COUNTS['OE'],
            cached_L_teacher_logits_reads=COUNTS['L'],real_parent_signal_forward_rows=sum(COUNTS.values()),real_student_forward_rows=640*STEPS,
            real_F_forward_rows=0,real_PCD_forward_rows=0,G_forward_rows=0,
            synthetic_F_forward_rows=checkpoint['synthetic_F_forward_rows']+smoke['synthetic_F_forward_rows'],synthetic_r03_forward_rows=2,synthetic_PCD_forward_rows=2,
            G_synthetic_forward_rows=checkpoint['G_synthetic_forward_rows']+smoke['G_synthetic_forward_rows'],synthetic_student_forward_rows=smoke['synthetic_student_forward_rows'],
            signal_parameters=335681,trainable_parameters_per_student=335633,wrapper_parameters_not_deployed=667729,
            L_sampling_seed=core.L_SEED,partner_sampling_seed=core.PAIR_SEED,OE_sampling_seed=core.OE_SEED,style_sampling_seed=core.STYLE_SEED,
            source_bank_deployed=False,U_bank_deployed=False,formal_target_accessed=False,target_REG=False,final_phase3_complete=False,**{k:0 for k in ZEROS})
        check();_write(out/'complete.json',done);print(STATUS,flush=True);return done
    finally:
        for role in roles.values():body.base.close_role(role)


def _initial_for_load(c,pm,device):
    source=identity._source_config(c);parent,parent_identity=body.base.load_trained_model(source,Path(source['output_root'])/'final.pt',device)
    _pure_source(parent,parent_identity,pm)
    return _model(c,pm,parent,device),parent_identity


def load_trained_model(c,arm,device='cuda:0'):
    validate_config(c)
    if arm not in ARMS:raise ValueError('Only two source-style students may load')
    root=Path(c['output_root'])
    try:done=_read(root/'complete.json');rows={a:_read(root/a/'complete.json') for a in ARMS}
    except (OSError,ValueError) as exc:raise ValueError('Both complete source-only students required') from exc
    expected=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arms=list(ARMS),updates_per_arm=STEPS,total_optimizer_updates=2*STEPS,
        source_only=True,same_sampling=True,source_sampling_matched=True,all_students_initialized_from_original_F0=True,original_models_unchanged=True,
        all_student_buffers_unchanged=True,all_student_Sinc_unchanged=True,all_student_bodies_changed=True,all_student_heads_changed=True,all_student_projections_changed=True,
        all_style_hooks_removed=True,wrapper_outside_reader_unchanged=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if any(done.get(k)!=v for k,v in expected.items()) or done.get('rows')!=rows:raise ValueError('Complete matched source-only training required')
    pm,_,_,_=_reference(c);model,parent=_initial_for_load(c,pm,device)
    if done.get('parent_signal_identity')!=parent or done.get('parent_F0_identity')!=pm['parent_F0_identity'] or done.get('dura_identity')!=pm['dura_identity']:
        raise ValueError('Pure-source model ancestry differs')
    for name,row in rows.items():
        i=_identity(c,name,pm,parent,done['source_feature_manifest'])
        required=dict(schema=SCHEMA,status=ROW_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,arm=name,identity=i,updates=STEPS,
            source_only=True,target_U_used=False,initial_state_equal_to_parents=True,student_Sinc_unchanged=True,model_buffers_unchanged=True,
            student_body_changed=True,student_head_changed=True,student_projection_changed=True,style_hooks_removed=True,original_models_unchanged=True,
            final_phase3_complete=False,**{k:0 for k in ZEROS})
        if any(row.get(k)!=v for k,v in required.items()) or done['identities'].get(name)!=i:raise ValueError('Strict source-only row and shared-body/head/projection identity required')
    value=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=False);i=rows[arm]['identity']
    if (value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=i or value.get('parent_F0_identity')!=pm['parent_F0_identity']
            or value.get('parent_signal_identity')!=parent or value.get('source_feature_manifest')!=done['source_feature_manifest']):raise ValueError('Strict original F0/pure-r03 source lineage required')
    saved=value.get('network');initial=_snapshot(model)
    if (not isinstance(saved,dict) or set(saved)!=set(initial) or any(not torch.is_tensor(v) or v.shape!=initial[k].shape or v.dtype!=initial[k].dtype or not bool(torch.isfinite(v).all()) for k,v in saved.items())):
        raise ValueError('Full finite reader/projection state with exact shapes and dtypes required')
    fixed=(set(dict(model.named_buffers()))&set(initial))|{k for k,p in model.named_parameters() if k.startswith(core.SINC_PREFIX)}
    if any(not torch.equal(initial[k],saved[k]) for k in fixed):raise ValueError('Frozen Sinc and persistent buffers must equal original parents')
    model.load_state_dict(saved,strict=True);_changed(model,initial);model.eval().requires_grad_(False);model._source_style_identity=i
    return model,i


def main():
    print(json.dumps(dict(event='SOURCE_STYLE_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();train(_read(args.config),args.device)


if __name__=='__main__':main()
