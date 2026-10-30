"""Matched native PAMF increments for fixed versus episodic member learning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import re
import time
import numpy as np
import torch
from . import runner
from . import source_open_embedding_train as native
from . import star_representation_train as previous
from . import star_joint_ova_train as joint
from . import star_open_world_model as ow
from . import star_open_world_episode as episode
from . import star_open_world_loss as losses

RUN_ID='20261030-phase3-star-open-episodic-s2026092701-r02'
ARMS=('S0','S1','M')
SCHEMA='pamf_phase3_star_open_world_train_v1'
STATUS='STAR_OPEN_WORLD_SOURCE_FROZEN'
STEPS=2220
MODEL_PARAMETERS=284834
COUNTS=dict(L=6300,OE=6134)
FIELDS={'run_id','code_commit','arm','native_config','L_root','OE_root','output_root','steps','lr','sampling_seed','slot_seed'}
NATIVE_FIELDS=runner.COMMON|{'source_cache'}
load_L,load_OE,close_role=previous.load_L,previous.load_OE,previous.close_role
_tensor,_optimizer,_clip,_write=previous._tensor,previous._optimizer,previous._clip,runner._write


def _native_config(c):
    # Legacy constructor constants are not this experiment's actual samplers.
    return dict(c['native_config'],arm='supervised_only',steps=c['steps'],lr=c['lr'],L_seed=2026092701,OE_seed=2026102907)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or not isinstance(c.get('native_config'),dict) or set(c['native_config'])!=NATIVE_FIELDS):
        raise ValueError('Only the fixed three-arm native PAMF-OW schema is allowed')
    fixed=dict(steps=STEPS,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)
    if any(type(c[k]) is not type(v) or c[k]!=v for k,v in fixed.items()):
        raise ValueError('Fixed 2220 updates, LR and source episode sampling seeds required')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('L_root','OE_root','output_root')):
        raise ValueError('Separate source roles and immutable output root required')
    n=c['native_config']
    if (n.get('run_id')!=RUN_ID or n.get('row_id')!=c['arm']+'-s2026092701'
            or n.get('code_commit')!=c['code_commit'] or n.get('output_root')!=c['output_root']):
        raise ValueError('Native run/arm/code/output identity differs')
    roots=(c['L_root'],c['OE_root'],n['source_output'],n['source_cache'])
    out=Path(c['output_root']).resolve()
    if (Path(c['L_root']).resolve()==Path(c['OE_root']).resolve()
            or any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in roots)):
        raise ValueError('Output must be separate from original input roles/model')
    native.validate_config(_native_config(c))
    return c


def _cached_as(raw,metadata,epoch,step):
    # These slots replay the same fixed satellite P1; no second channel draw.
    return raw


def _model(c,device):
    model=native.make_trainable_model(_native_config(c),device)
    ow.install(model,seed=2026092701)
    model._cd_practical=_cached_as
    p=ow.parameter_report(model)
    if (p['native_identity_parameters']!=232993 or p['open_world_parameters']!=51841
            or p['active_parameters']!=MODEL_PARAMETERS or p['training_container_parameters']!=952563):
        raise ValueError('Measured native/open/deployment parameter counts differ')
    return model


def _train_mode(model):
    # The native L hook needs the container's train flag. Per-record encoder
    # evaluation removes dropout and running-stat updates without cutting grad.
    model.train()
    model.id_backbone.encoder.eval()
    for module in model.modules():
        if isinstance(module,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.modules.dropout._DropoutNd)):
            module.eval()


def _identity(c,model,L=None):
    p=ow.parameter_report(model)
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],checkpoint=str(Path(c['output_root'])/'final.pt'),
        updates=STEPS,cumulative_updates=11100+STEPS,original_parent_identity=model._phase3_parent_identity,
        identifier_parameters=p['active_parameters'],open_world_parameters=p['open_world_parameters'],
        training_container_parameters=p['training_container_parameters'],identity_head_replaced=False,input_path='cached_P1_only',
        parent_F0_identity=None if L is None else L['manifest']['parent_F0_identity'])


def _smoke(model,device):
    # Existing builder generates synthetic G10 P0; no source/query IQ is opened.
    result=previous._smoke(model,device)
    from .star_channel import observe
    mixture=runner._read(Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json')
    t=np.arange(256);rows=[]
    for i,w in enumerate((.07,.11)):
        x=np.stack((np.cos(w*t),np.sin(w*t)))
        y,_=observe(x,physical_id='star-open-world-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
            episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)
        rows.append(y)
    state={k:v.detach().clone() for k,v in model.state_dict().items()}
    caches=[(module,key,value) for module in model.modules() for key,value in vars(module).items()
            if 'cache' in key and key not in ('_buffers','_parameters','_modules')]
    try:
        model.eval();model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out=ow.forward(model,_tensor(np.asarray(rows,np.float32),device))
            base=model.id_backbone.encoder.classify_features(out['baseline_fused160'])
            if not torch.equal(base,out['logits']):raise ValueError('Zero-gain initialization changed original default-head logits')
            logits=ow.relation(model,out['open64'],out['open64'].flip(0))
            torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.eye(2,device=device)).backward()
        named=dict(model.named_parameters())
        prefixes=(native.SINC_PREFIX,'open_world.projector.','open_world.relation_scorer.')
        norms={p:sum(float(v.grad.detach().square().sum()) for n,v in named.items() if n.startswith(p) and v.grad is not None)**.5 for p in prefixes}
        if any(not np.isfinite(v) or v<=0 for v in norms.values()):raise ValueError('Real native/open/relation gradients required')
        if any(not torch.equal(state[k],v) for k,v in model.state_dict().items()):raise ValueError('No-step smoke changed model state')
        return dict(result,status='PASS',open_world_gradient_norms=norms,parameter_report=ow.parameter_report(model),
            synthetic_open_forward_rows=2,zero_gain_original_logits_exact=True)
    finally:
        model.zero_grad(set_to_none=True);ow.clear(model)
        for module,key,value in caches:setattr(module,key,value)


def _loop(model,L,OE,c,device,optimizer,on_step=None):
    meta=L['metadata'];sampler=episode.EpisodeSampler(meta['y'],meta['rx_i'],meta['day_i'],L['ids'],OE['metadata']['domain'],seed=c['sampling_seed'],slot_seed=c['slot_seed'])
    augmentor,no_pa,augstate=native._augmentor(model._phase3_parent_args)
    named=list(model.named_parameters());params=[v for _,v in named];trace=[]
    torch.manual_seed(c['sampling_seed']);np.random.seed(c['sampling_seed']);random.seed(c['sampling_seed'])
    if torch.device(device).type=='cuda':torch.cuda.manual_seed(c['sampling_seed'])
    for step in range(1,c['steps']+1):
        started=time.perf_counter();sample=sampler.sample(step)
        ix=sample['native_L'];si=sample['support'].reshape(-1);qi=sample['query'].reshape(-1);oi=sample['OE']
        x=_tensor(L['restored'][ix],device);y=_tensor(meta['y'][ix],device,torch.long);domains=_tensor(meta['domain'][ix],device,torch.long)
        lm={k:_tensor(v[ix],device,torch.long) for k,v in meta.items() if k!='y'}
        optimizer.zero_grad(set_to_none=True);_train_mode(model)
        full_l,nm=native.supervised_loss(model,x,y,domains,lm,augmentor,no_pa,step)
        if (int(nm['T1_L_packets'])!=768 or float(nm['CRIC_supcon_weight'])!=.1 or int(nm['T1_PL_seen'])!=0
                or int(nm['T1_sat_seen'])!=0 or float(nm['T1_As_KL_weight'])!=0):
            raise ValueError('Actual full L scaffold or U-off/no-new-channel execution differs')
        model.eval()
        raw=ow.forward(model,x);kd=losses.kd_loss(raw['logits'],_tensor(L['teacher_logits'][ix],device),temperature=2.)
        anchor=full_l+kd
        mx=np.concatenate((L['restored'][si],L['restored'][qi],OE['restored'][oi]))
        out=ow.forward(model,_tensor(mx,device));z=out['open64']
        zs=z[:96].reshape(6,16,64);zq=z[96:]
        if c['arm']=='S0':zs=zs.detach();zq=zq.detach()
        p,labels=losses.prototypes(zs,sample['slot_order'],sample['held_class'],c['arm'])
        query_y=np.concatenate((meta['y'][qi],np.full(64,-1,np.int64)))
        member=losses.member_loss(ow.relation(model,zq,p),_tensor(query_y,device,torch.long),labels)
        weighted=.1*member['member_loss']
        grads,diag=losses.protected_gradients(anchor,weighted,named,auxiliary_prefix='open_world.relation_scorer.')
        for (_,param),grad in zip(named,grads):param.grad=grad
        if any(v.grad is not None and not bool(torch.isfinite(v.grad).all()) for v in params):raise FloatingPointError('Nonfinite final gradient')
        physical=_clip(optimizer,params)
        if not np.isfinite(float(physical)):raise FloatingPointError('Nonfinite physical gradient norm')
        optimizer.step();ow.clear(model)
        row=dict(step=step,cumulative_updates=11100+step,arm=c['arm'],lr=float(optimizer.param_groups[0]['lr']),
            anchor_loss=float(anchor.detach()),full_L_loss=float(full_l.detach()),KD=float(kd.detach()),member_loss=float(member['member_loss'].detach()),
            pair_bce=float(member['pair_bce'].detach()),set_bce=float(member['set_bce'].detach()),member_weight=.1,member_counts=member['counts'],
            held_class=int(sample['held_class']),held_rx=int(sample['held_rx']),held_day=int(sample['held_day']),slot_order=sample['slot_order'].tolist(),
            source_L_native_samples=128,L_support_samples=96,L_query_samples=96,OE_samples=64,
            native_L_packet_views=768,KD_packet_views=128,episode_packet_views=256,total_F_packet_views=1152,
            As_cached_P1_replay_samples=64,As_new_channel_generated_samples=0,physical_grad_norm=float(physical),gradient=diag,
            seconds=time.perf_counter()-started,native_metrics={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in nm.items()})
        trace.append(row)
        if on_step:on_step(row)
        del anchor,full_l,kd,weighted,member,raw,out,z,zs,zq,p,grads
    return trace,augstate


def train(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('A new immutable PAMF-OW source root required')
    started=time.perf_counter();model=_model(c,device);smoke=_smoke(model,device)
    if smoke['status']!='PASS':raise ValueError('First actual-checkpoint no-query smoke must pass')
    output.mkdir(parents=True,exist_ok=False);_write(output/'resolved_config.json',c);_write(output/'smoke.json',smoke);print('STAR_OPEN_WORLD_GRADIENT_SMOKE_PASS',flush=True)
    L=OE=None
    try:
        L=load_L(c['L_root']);previous._role(L,'L',c);OE=load_OE(c['OE_root']);previous._role(OE,'OE',c);joint._same_roles(L,OE)
        before={n:v.detach().cpu().clone() for n,v in model.named_parameters()};buffers={n:v.detach().cpu().clone() for n,v in model.named_buffers()}
        optimizer=_optimizer(model,list(model.parameters()),lr=c['lr'],weight_decay=float(model._phase3_parent_args['weight_decay']))
        def log(row):
            with (output/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if row['step']==1 or row['step']%25==0 or row['step']==STEPS:print('STAR_OPEN_WORLD_STEP',json.dumps({k:row[k] for k in ('step','arm','anchor_loss','member_loss','seconds')}),flush=True)
        trace,augstate=_loop(model,L,OE,c,device,optimizer,log)
        if any(not torch.equal(buffers[n],v.detach().cpu()) for n,v in model.named_buffers()):raise ValueError('Frozen running buffers changed')
        changed=[n for n,v in model.named_parameters() if not torch.equal(before[n],v.detach().cpu())]
        if (native.HEAD_NAME not in changed or not any(n.startswith(native.SINC_PREFIX) for n in changed)
                or not any(n.startswith('open_world.relation_scorer.') for n in changed)):
            raise ValueError('Expected actual Sinc/default-head/member updates absent')
        identity=_identity(c,model,L)
        payload=dict(schema=SCHEMA,identity=identity,config=c,model={n:v.detach().cpu() for n,v in model.state_dict().items()},
            source_data_handle=model._phase3_source_handle,cache_handles=dict(L=L['manifest'],OE=OE['manifest']),
            updates=STEPS,cumulative_updates=13320,optimizer_reset=True,target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0)
        torch.save(payload,output/'final.pt')
        done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,identity=identity,
            parameter_report=ow.parameter_report(model),source_only=True,cache_handles=payload['cache_handles'],parent_F0_identity=L['manifest']['parent_F0_identity'],
            dura_identity=L['manifest']['dura_identity'],updates=STEPS,cumulative_updates=13320,optimizer_reset=True,
            target_reads=0,validation_reads=0,novel_validation_reads=0,original_U_s_reads=0,OE_truth_reads=0,received_cache_values_read=0,legacy_clean_IQ_reads=0,
            G_updates=0,G_forward_rows=0,encoder_updates=STEPS,identity_head_updates=STEPS,model_buffers_unchanged=True,identity_head_replaced=False,
            input_path='cached_P1_only',source_L_count=6300,source_OE_count=6134,L_native_sampled_exposures=128*STEPS,
            L_support_exposures=96*STEPS,L_query_exposures=96*STEPS,OE_exposures=64*STEPS,F_forward_rows=1152*STEPS,
            As_cached_P1_replay_exposures=64*STEPS,As_new_channel_generated_exposures=0,ordinary_augmentation_state=augstate,
            changed_parameters=changed,final_step=trace[-1],seconds=time.perf_counter()-started,final_phase3_complete=False)
        _write(output/'training_summary.json',done);_write(output/'complete.json',done);print(STATUS,c['arm'],flush=True);return done
    finally:
        ow.clear(model)
        if OE is not None:close_role(OE)
        if L is not None:close_role(L)


def load_trained_model(c,checkpoint,device='cuda:0'):
    validate_config(c)
    if Path(checkpoint).resolve()!=Path(c['output_root'],'final.pt').resolve():raise ValueError('Only this row frozen final checkpoint allowed')
    model=_model(c,device);value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ident=value.get('identity',{});expected=_identity(c,model)
    if (value.get('schema')!=SCHEMA or value.get('config')!=c or any(ident.get(k)!=v for k,v in expected.items() if k!='parent_F0_identity')
            or value.get('source_data_handle')!=model._phase3_source_handle or value.get('updates')!=STEPS
            or value.get('cumulative_updates')!=13320 or value.get('optimizer_reset') is not True
            or any(value.get(k)!=0 for k in ('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','OE_truth_reads'))
            or ident.get('parent_F0_identity')!=value.get('cache_handles',{}).get('L',{}).get('parent_F0_identity')):
        raise ValueError('Frozen native PAMF-OW source-only identity/contract differs')
    model.load_state_dict(value['model'],strict=True)
    return model.eval().requires_grad_(False),ident


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train(runner._read(a.config),a.device)


if __name__=='__main__':main()
