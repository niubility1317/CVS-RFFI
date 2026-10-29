"""Source-only fixed-step P1 representation training from original F0 E50.

All model forwards use eval mode with gradients enabled. No native training
hook, ordinary/As augmentation, clean IQ loader, U, V, novel or target is used.
The raw fused160 is additive branch/reference/signature fusion; its global
normalization occurs only inside the unchanged cosine identity classifier.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch

from . import runner
from . import source_open_embedding_train as native
from . import star_representation_loss as losses

RUN_ID='20261029-phase3-star-representation-s2026092701-r01'
SCHEMA='pamf_phase3_star_representation_train_v1'
STATUS='STAR_REPRESENTATION_SOURCE_FROZEN'
ARMS=('anchor','oe','oe_vic')
STEPS=2220
BATCH=128
COUNTS=dict(L=6300,OE=6134)
NATIVE_FIELDS=runner.COMMON|{'source_cache'}
FIELDS={'run_id','code_commit','arm','native_config','L_root','OE_root','output_root','steps','lr','L_seed','OE_seed','view_seed'}
_write=runner._write


def _native_config(config):
    # Constructor only: supervised mapping disables historical U/EMA machinery.
    return dict(config['native_config'],arm='supervised_only',steps=config['steps'],lr=config['lr'],
        L_seed=config['L_seed'],OE_seed=config['OE_seed'])


def validate_config(config):
    if (not isinstance(config,dict) or set(config)!=FIELDS or config.get('run_id')!=RUN_ID or config.get('arm') not in ARMS
            or not isinstance(config.get('code_commit'),str) or re.fullmatch(r'[0-9a-f]{40}',config['code_commit']) is None
            or not isinstance(config.get('native_config'),dict) or set(config['native_config'])!=NATIVE_FIELDS):
        raise ValueError('Only the fixed source representation input schema is accepted')
    fixed=dict(steps=STEPS,lr=.0001,L_seed=2026092701,OE_seed=2026102907,view_seed=2026102908)
    if any(type(config[k]) is not type(v) or config[k]!=v for k,v in fixed.items()):
        raise ValueError('Fixed 2220 updates, constant LR and independent sampling/view seeds required')
    if any(not isinstance(config[k],str) or not config[k].strip() for k in ('L_root','output_root')):
        raise ValueError('Visible L cache and a new output root are required')
    if ((config['arm']=='anchor' and config['OE_root'] is not None)
            or (config['arm']!='anchor' and (not isinstance(config['OE_root'],str) or not config['OE_root'].strip()))):
        raise ValueError('Anchor cannot receive an OE path; OE arms require only the source exposure cache')
    n=config['native_config']
    if (n.get('run_id')!=RUN_ID or n.get('code_commit')!=config['code_commit'] or n.get('output_root')!=config['output_root']
            or n.get('row_id')!=config['arm']+'-s2026092701'):
        raise ValueError('The constructor must belong to this independent run/arm/output')
    roots=[config['L_root'],n['source_output'],n['source_cache']]+([config['OE_root']] if config['OE_root'] else [])
    if any(Path(config['output_root']).resolve().is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError('Training output cannot replace an input or original source root')
    native.validate_config(_native_config(config))
    return config


def load_L(root):
    from .star_representation_cache import load_role
    return load_role(root,'L')


def load_OE(root):
    from .star_representation_cache import load_role
    return load_role(root,'OE')


def close_role(bundle):
    from .star_representation_cache import close_role as close
    close(bundle)


def _role(bundle,role,config):
    fields={'manifest','ids','received','restored','metadata'}|({'teacher_logits'} if role=='L' else set())
    if not isinstance(bundle,dict) or set(bundle)!=fields:
        raise ValueError('Only this source cache role and permitted metadata may be loaded')
    n=COUNTS[role];ids=bundle['ids'];x=bundle['restored'];meta=bundle['metadata'];manifest=bundle['manifest']
    if (not isinstance(manifest,dict) or manifest.get('role')!=role or manifest.get('count')!=n
            or ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or x.shape!=(n,2,256) or x.dtype!=np.float32 or not isinstance(meta,dict)
            or set(meta)!=({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            or any(value.shape!=(n,) or value.dtype!=np.int64 for value in meta.values())
            or not np.isin(meta['rx_i'],native.RX).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(native.RX,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Aligned restored P1 source cache count/IDs/RX/day layout required')
    parent=manifest.get('parent_F0_identity');dura=manifest.get('dura_identity')
    if (not isinstance(parent,dict) or parent.get('model')!='original_F0_E50' or parent.get('model_seed')!=2026092701
            or parent.get('epoch')!=50 or parent.get('updates')!=11100 or parent.get('parameters')!=232993
            or parent.get('source_output')!=config['native_config']['source_output']
            or manifest.get('restored_input_path')!='P1' or not isinstance(dura,dict)
            or dura.get('deployment_id')!='dura_v1' or dura.get('parameters')!=266951
            or dura.get('epoch')!=20 or dura.get('updates')!=4940):
        raise ValueError('Source teacher cache must belong to the original same-input F0')
    if role=='L':
        teacher=bundle['teacher_logits']
        if (set(meta['y'].tolist())!=set(range(6)) or teacher.shape!=(n,6) or teacher.dtype!=np.float32
                or manifest.get('teacher_input_path')!='P1'):
            raise ValueError('Visible original six labels and fixed full-six teacher logits required')
    return bundle


def _tensor(value,device='cpu',dtype=torch.float32):
    # NumPy2 / historical Torch2.1 compatible bridge.
    return torch.tensor(np.asarray(value).tolist(),dtype=dtype,device=device)


def _observe(raw,**kwargs):
    from .star_channel import observe
    return observe(raw,**kwargs)


def _smoke(model,device):
    # Actual F checkpoint, synthetic star received only. The cache builder's
    # separate first smoke covers the frozen G+F P0/P1 paths.
    path=Path(__file__).resolve().parents[2]/'configs/pamf_phase2_practical_mixture_20261028.json'
    numpy_rng,python_rng=np.random.get_state(),random.getstate()
    try:
        with torch.random.fork_rng(devices=native._cuda_devices(model)):
            mixture=runner._read(path);t=np.arange(256);rows=[]
            for i,w in enumerate((.07,.11)):
                raw=np.stack((np.cos(w*t),np.sin(w*t)))
                received,_=_observe(raw,physical_id='star-representation-synthetic-'+str(i),receiver='1-19',day='2021_03_08',
                    episode=dict(id='MIX01',scene_seed=2026103001,channel_seed=2026104001),mixture=mixture)
                rows.append(received)
            x=_tensor(np.stack(rows).astype(np.float32),device)
            result=native.gradient_smoke(model,x,torch.tensor([0,1],dtype=torch.long,device=device))
    finally:
        np.random.set_state(numpy_rng);random.setstate(python_rng)
    return dict(result,actual_F0_checkpoint=True,input_path='P0_synthetic_G10',G_forwards=0,
        real_cache_IQ_reads=0,training_input_path='cached_P1_only',cache_G_F_smoke_scope='performed by cache builder')


def forward(model,x):
    """Same differentiable forward: raw fused160 and complete default6 head."""
    reader=model.id_backbone.encoder
    if model.training or any(module.training for module in reader.modules()):
        raise ValueError('Representation training uses eval mode with gradient tracking')
    with torch.enable_grad(),torch.autocast(device_type=x.device.type,enabled=False):
        z=reader.features(x.float());logits=reader.classify_features(z)
    if (z.shape!=(len(x),160) or logits.shape!=(len(x),6) or z.dtype!=torch.float32 or logits.dtype!=torch.float32
            or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(logits).all())):
        raise ValueError('Finite raw fused160 and full default-six logits required')
    return z,logits


def _optimizer(model,params,**kwargs):
    from experiments.cric_dura_comparison.f_optimizer import build
    return build(model,params,'CENTER_UNITS',**kwargs)


def _clip(optimizer,params):
    from experiments.cric_dura_comparison.f_optimizer import clip_physical
    return clip_physical(optimizer,params,5.)


def _loop(model,labeled,oe,config,device,optimizer,clip,*,steps=None,on_step=None):
    steps=config['steps'] if steps is None else steps
    model.eval().requires_grad_(True);params=list(model.parameters());named=dict(model.named_parameters())
    sinc=[named[native.SINC_PREFIX+key] for key in ('low_coordinate','band_coordinate')];head=named[native.HEAD_NAME]
    L_rng=torch.Generator().manual_seed(config['L_seed'])
    L_view=torch.Generator().manual_seed(config['view_seed']);OE_view=torch.Generator().manual_seed(config['view_seed']+1)
    sampler=native.DomainSampler(oe['metadata']['domain'],config['OE_seed']) if oe is not None else None
    trace=[]
    for step in range(1,steps+1):
        started=time.perf_counter();ix=torch.randint(len(labeled['ids']),(BATCH,),generator=L_rng).tolist()
        x0=_tensor(labeled['restored'][ix]);x1=losses.phase_gain_view(x0,L_view)
        y=_tensor(labeled['metadata']['y'][ix],device,torch.long);teacher=_tensor(labeled['teacher_logits'][ix],device)
        before_sinc=[q.detach().clone() for q in sinc];before_head=head.detach().clone();optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _,l0=forward(model,x0.to(device));_,l1=forward(model,x1.to(device));known=losses.labeled(l0,l1,y,teacher)
            zero=known['total'].new_zeros(());oe_loss=zero;vic=dict(inv=zero,std=zero,cov=zero,total=zero)
            if oe is not None:
                oi=sampler.sample(BATCH);o0=_tensor(oe['restored'][oi]);o1=losses.phase_gain_view(o0,OE_view)
                z0,p0=forward(model,o0.to(device));z1,p1=forward(model,o1.to(device));oe_loss=losses.uniform_oe(p0,p1)
                if config['arm']=='oe_vic':vic=losses.vicreg(z0,z1)
            total=known['total']+.1*oe_loss+.1*vic['total']
            if not bool(torch.isfinite(total)):raise FloatingPointError('Nonfinite source representation loss')
            total.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
            raise FloatingPointError('Nonfinite source representation gradient')
        sn=sum(float(q.grad.norm()) for q in sinc if q.grad is not None);hn=float(head.grad.norm()) if head.grad is not None else 0.
        grad=clip(optimizer,params)
        if not bool(torch.isfinite(torch.as_tensor(grad))):raise FloatingPointError('Nonfinite physical global gradient')
        optimizer.step()
        row=dict(step=step,cumulative_updates=11100+step,lr=float(optimizer.param_groups[0]['lr']),
            loss=float(total.detach()),L_ce=float(known['ce'].detach()),L_KD=float(known['kd'].detach()),
            OE_uniform=float(oe_loss.detach()),vicreg_inv=float(vic['inv'].detach()),vicreg_std=float(vic['std'].detach()),
            vicreg_cov=float(vic['cov'].detach()),vicreg_total=float(vic['total'].detach()),
            sinc_coordinate_gradient_norm=sn,head_gradient_norm=hn,physical_global_gradient_norm=float(grad),
            sinc_change_hz_norm=sum(float((q.detach()-before).norm())*25000. for q,before in zip(sinc,before_sinc)),
            head_change_norm=float((head.detach()-before_head).norm()),L_sampled_records=BATCH,L_unique_in_batch=len(set(ix)),
            L_forward_samples=2*BATCH,OE_sampled_records=BATCH if oe is not None else 0,OE_forward_samples=2*BATCH if oe is not None else 0,
            vicreg_evaluations=int(config['arm']=='oe_vic'),F_forward_calls=2+2*int(oe is not None),seconds=time.perf_counter()-started)
        trace.append(row)
        if on_step is not None:on_step(row)
    return trace


def _identity(config):
    return dict(run_id=config['run_id'],arm=config['arm'],row_id=config['arm']+'-s2026092701',code_commit=config['code_commit'],
        native_config=config['native_config'],L_root=config['L_root'],OE_root=config['OE_root'],steps=config['steps'],lr=config['lr'],
        L_seed=config['L_seed'],OE_seed=config['OE_seed'],view_seed=config['view_seed'])


def run(config,device='cuda:0'):
    validate_config(config);output=Path(config['output_root'])
    if output.exists():raise FileExistsError('A new immutable representation run output is required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=native.make_trainable_model(_native_config(config),device).eval().requires_grad_(True)
    smoke=_smoke(model,device)
    if smoke.get('status')!='PASS':raise ValueError('Actual native checkpoint gradient smoke must pass')
    output.mkdir(parents=True,exist_ok=False);_write(output/'smoke.json',smoke)
    L=OE=None;started=time.perf_counter()
    try:
        L=load_L(config['L_root']);_role(L,'L',config)
        if config['arm']!='anchor':
            OE=load_OE(config['OE_root']);_role(OE,'OE',config)
            if set(L['ids'].tolist())&set(OE['ids'].tolist()):raise ValueError('Source L and OE physical IDs overlap')
            if (L['manifest']['parent_F0_identity']!=OE['manifest']['parent_F0_identity']
                    or L['manifest']['dura_identity']!=OE['manifest']['dura_identity']):
                raise ValueError('L and OE must use the same original F0 and frozen P1 restorer')
        buffers={k:v.detach().clone() for k,v in model.named_buffers()}
        params=list(model.parameters());optimizer=_optimizer(model,params,lr=config['lr'],
            weight_decay=float(model._phase3_parent_args['weight_decay']),betas=(.9,.999),eps=1e-8,foreach=False)
        if optimizer.state:raise ValueError('The source representation optimizer must cold-start')
        _write(output/'resolved_config.json',config)
        with (output/'steps.jsonl').open('x',encoding='utf-8') as stream:
            def progress(row):
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print('STAR_REPRESENTATION_PROGRESS',config['arm'],row['step'],config['steps'],'loss',row['loss'],flush=True)
            trace=_loop(model,L,OE,config,device,optimizer,_clip,on_step=progress)
        updates=config['steps']
        if (len(trace)!=updates or optimizer._sinc_unit_steps!=updates or optimizer._sinc_physical_clip_calls!=updates
                or sum(r['L_sampled_records'] for r in trace)!=BATCH*updates
                or sum(r['OE_sampled_records'] for r in trace)!=(0 if OE is None else BATCH*updates)):
            raise ValueError('Exactly the declared source representation updates/exposures must complete')
        if any(not torch.equal(value,dict(model.named_buffers())[key]) for key,value in buffers.items()):
            raise ValueError('Eval-mode representation training must not update persistent model buffers')
        model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
        handles=dict(L=L['manifest'],OE=None if OE is None else OE['manifest'])
        result=dict(status=STATUS,schema=SCHEMA,config=config,arm=config['arm'],run_id=RUN_ID,code_commit=config['code_commit'],
            identity=_identity(config),original_parent_identity=model._phase3_parent_identity,cache_handles=handles,
            updates=updates,parent_updates=11100,cumulative_updates=11100+updates,optimizer_reset=True,
            optimizer='AdamW_CENTER_UNITS',lr=config['lr'],lr_schedule='constant',physical_global_clip=5.,
            source_only=True,target_reads=0,U_reads=0,validation_reads=0,novel_validation_reads=0,OE_truth_reads=0,
            legacy_clean_IQ_reads=0,SourceDataset_calls=0,received_cache_values_read=0,
            model_classes=6,parameters=232993,default_head_joint=True,identity_head_replaced=False,
            input_path='cached_P1_only',model_mode='eval_with_grad',dropout_enabled=False,BN_updates=0,
            model_buffers_unchanged=True,precision='FP32',TF32=False,
            source_L_count=COUNTS['L'],source_OE_count=0 if OE is None else COUNTS['OE'],
            L_sampled_exposures=BATCH*updates,L_forward_samples=2*BATCH*updates,
            OE_sampled_exposures=0 if OE is None else BATCH*updates,OE_forward_samples=0 if OE is None else 2*BATCH*updates,
            vicreg_evaluations=updates if config['arm']=='oe_vic' else 0,
            CE_label_smoothing=0.,KD_weight=1.,KD_temperature=2.,OE_weight=0. if OE is None else .1,
            VICReg_weight=.1 if config['arm']=='oe_vic' else 0.,VICReg_coefficients=[25.,25.,1.],VICReg_epsilon=.0001,
            feature='raw_additive_fused160_before_head_normalization',projector=False,
            ordinary_augmentation=False,As_generated=0,CRIC_evaluations=0,PL_evaluations=0,EMA_updates=0,
            budget_scope='fixed iid updates; not ten natural traversals of source records',
            view_seeds=dict(L=config['view_seed'],OE=config['view_seed']+1),checkpoint='final.pt',seconds=time.perf_counter()-started,
            final_phase3_complete=False)
        payload=dict(schema=SCHEMA,config=config,identity=_identity(config),original_parent_identity=model._phase3_parent_identity,
            cache_handles=handles,updates=updates,cumulative_updates=11100+updates,optimizer_reset=True,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),training_summary=result)
        with (output/'final.pt').open('xb') as stream:torch.save(payload,stream)
        _write(output/'training.json',result);_write(output/'complete.json',result)
        print(STATUS,config['arm'],flush=True)
        return result
    finally:
        if OE is not None:close_role(OE)
        if L is not None:close_role(L)


def load_trained(config,checkpoint,device='cuda:0'):
    """Verify this new objective, rebuild original F0, strict-load own final state."""
    validate_config(config);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (payload.get('schema')!=SCHEMA or payload.get('config')!=config or payload.get('identity')!=_identity(config)
            or payload.get('updates')!=config['steps'] or payload.get('cumulative_updates')!=11100+config['steps']
            or payload.get('optimizer_reset') is not True):
        raise ValueError('Only this independently trained star representation checkpoint may load')
    summary=payload.get('training_summary',{})
    if (summary.get('status')!=STATUS or summary.get('input_path')!='cached_P1_only'
            or summary.get('source_only') is not True or summary.get('default_head_joint') is not True
            or summary.get('identity_head_replaced') is not False or summary.get('parameters')!=232993
            or any(summary.get(k)!=0 for k in ('target_reads','U_reads','validation_reads','novel_validation_reads','OE_truth_reads'))):
        raise ValueError('The complete source-only P1 training boundary must be retained')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=native.make_trainable_model(_native_config(config),device)
    if payload.get('original_parent_identity')!=model._phase3_parent_identity:
        raise ValueError('Original F0 parent identity differs')
    model.load_state_dict(payload['model'],strict=True);model.eval().requires_grad_(False)
    model._star_representation_identity=dict(run_id=RUN_ID,arm=config['arm'],row_id=config['arm']+'-s2026092701',
        code_commit=config['code_commit'],checkpoint=str(checkpoint),updates=config['steps'],cumulative_updates=11100+config['steps'],
        original_parent_identity=model._phase3_parent_identity)
    return model


def main(argv=None):
    print(json.dumps(dict(event='STAR_REPRESENTATION_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),
        argv=list(sys.argv) if argv is None else list(argv)),ensure_ascii=False),flush=True)
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    command=sub.add_parser('train');command.add_argument('--config',required=True);command.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv);done=run(runner._read(args.config),args.device)
    print(json.dumps(dict(status=done['status'],arm=done['arm'])),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
