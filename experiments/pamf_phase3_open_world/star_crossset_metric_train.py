"""Shared frozen-feature preparation and independent cross-set metric source training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import torch
from . import runner
from . import star_consensus_run as base
from . import star_protected_open_cache as source_cache
from . import star_density_core as geometry
from . import star_crossset_metric_model as core

RUN_ID='20261030-phase3-star-crossset-metric-s2026092701-r01'
ARMS=('S_SEP','S_SEP_TOPO')
SCHEMA='pamf_phase3_star_crossset_metric_train_v1'
STATUS='STAR_CROSSSET_METRIC_SOURCE_FROZEN'
CACHE_SCHEMA='pamf_phase3_star_crossset_metric_cache_v1'
CACHE_STATUS='STAR_CROSSSET_METRIC_FEATURES_COMPLETE'
SOURCE_CODE='7dd90671968ddf58aab5a5b302b4d56371c8b9c9'
STEPS=2220
COUNTS=dict(L=6300,OE=6134)
PREPARE_FIELDS={'run_id','code_commit','source_config','feature_root','output_root','batch_size'}
TRAIN_FIELDS={'run_id','code_commit','arm','cache_root','output_root','steps','lr','L_seed','OE_seed'}
FIELDS=TRAIN_FIELDS
ZEROS=('target_reads','validation_reads','novel_validation_reads','original_U_s_reads','private_truth_reads','OE_truth_reads','G_forward_rows','native_updates','identity_head_updates','IQ_reads')
_read,_write=runner._read,runner._write


def _config(c,fields):
    if not isinstance(c,dict) or set(c)!=fields or c.get('run_id')!=RUN_ID or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None:
        raise ValueError('Exact source-only cross-set metric configuration required; no truth/U/Q/K')
    for k in ('output_root',):
        if not isinstance(c[k],str) or not c[k].strip():raise ValueError('Nonempty immutable output path required')


def _separate(output,inputs):
    out=Path(output).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in inputs):
        raise ValueError('Immutable output must be separate from every source input')


def validate_prepare_config(c):
    _config(c,PREPARE_FIELDS);s=c.get('source_config')
    if not isinstance(s,dict) or set(s)!=runner.COMMON|{'source_cache'}:raise ValueError('Exact original F0 source constructor handle required')
    runner.validate_config('calibrate',dict(s,retention=.99))
    if (s['run_id']!=RUN_ID or s['code_commit']!=c['code_commit'] or s['row_id']!='star-crossset-metric-prepare-s2026092701'
            or s['output_root']!=c['output_root'] or s['model_seed']!=2026092701 or s['registered_classes']!=base.native.CLASSES
            or type(c['batch_size']) is not int or c['batch_size']!=128
            or not isinstance(c['feature_root'],str) or not c['feature_root'].strip()):
        raise ValueError('Fixed F0 original-six constructor and batch size 128 required')
    _separate(c['output_root'],(c['feature_root'],s['source_output'],s['source_cache']));return c


def validate_config(c):
    _config(c,TRAIN_FIELDS)
    if c['arm'] not in ARMS or not isinstance(c['cache_root'],str) or not c['cache_root'].strip():raise ValueError('Declared arm and prepared source feature cache required')
    for k,v in dict(steps=STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018).items():
        if type(c[k]) is not type(v) or c[k]!=v:raise ValueError('Fixed adapter steps, learning rate and independent source seeds required')
    _separate(c['output_root'],(c['cache_root'],));return c


def _backend(c,device):
    from .star_channel import _channel_api
    _channel_api();return base._backend(c,device)


_smoke=base._smoke
_old_features=source_cache.load_features


def _source_manifest(m,source):
    source_cache._manifest(m)
    p=m['parent_F0_identity']
    if m['code_commit']!=SOURCE_CODE or any(p.get(k)!=source[k] for k in ('source_output','source_contract','base_release','model_seed')):
        raise ValueError('Only actual original F0/code7dd protected source features are allowed')
    return m


def prepare(c,device='cuda:0'):
    validate_prepare_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable common feature cache required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b,_=_backend(c['source_config'],device);snapshots=base._snapshot(b);smoke=_smoke(b,device);base._check_models(snapshots)
    if smoke.get('status')!='PASS' or smoke.get('actual_F0_checkpoint') is not True or b.get('dura') is not None:
        raise ValueError('Actual original F0 no-query smoke must pass, without G')
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    parent=None;seen=set()
    for role in COUNTS:
        a=_old_features(c['feature_root'],role);m=_source_manifest(a['manifest'],c['source_config'])
        if m['parent_F0_identity']!=b['reader_identity'] or (parent is not None and m!=parent):raise ValueError('Actual F0 and both source feature roles must share exact ancestry')
        parent=m
        if len(a['ids'])!=COUNTS[role] or seen&set(a['ids'].tolist()):raise ValueError('Full source L/OE role counts and disjoint IDs required')
        seen.update(a['ids'].tolist())
        u=geometry.unit(a['raw_original_fused160']).astype(np.float32);contribution=geometry.contribution_unit(a['contributions'])
        with (out/(role+'.npz')).open('xb') as stream:np.savez(stream,ids=a['ids'],u=u,c=contribution,**a['metadata'])
        base._check_models(snapshots);print('STAR_CROSSSET_PREPARE_PROGRESS',role,len(u),flush=True)
    done=dict(schema=CACHE_SCHEMA,status=CACHE_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,counts=dict(COUNTS),
        parent_F0_identity=parent['parent_F0_identity'],dura_identity=parent['dura_identity'],source_feature_manifest=parent,
        source_manifests=parent['source_manifests'],source_only=True,input_path='cached_P1_features_only',model_parameters_buffers_unchanged=True,
        real_F_forward_rows=0,synthetic_F_forward_rows=2,optimizer_updates=0,neural_updates=0,identity_head_replaced=False,**{k:0 for k in ZEROS})
    _write(out/'complete.json',done);print(CACHE_STATUS,flush=True);return done


def validate_cache_manifest(m):
    expected=dict(schema=CACHE_SCHEMA,status=CACHE_STATUS,run_id=RUN_ID,counts=COUNTS,source_only=True,input_path='cached_P1_features_only',
        model_parameters_buffers_unchanged=True,real_F_forward_rows=0,synthetic_F_forward_rows=2,optimizer_updates=0,neural_updates=0,
        identity_head_replaced=False,**{k:0 for k in ZEROS})
    if not isinstance(m,dict) or any(m.get(k)!=v for k,v in expected.items()):raise ValueError('Complete immutable source feature-pair cache required')
    c=validate_prepare_config(m['config']);old=_source_manifest(m['source_feature_manifest'],c['source_config'])
    if (m['code_commit']!=c['code_commit'] or m['parent_F0_identity']!=old['parent_F0_identity'] or m['dura_identity']!=old['dura_identity']
            or m['source_manifests']!=old['source_manifests']):raise ValueError('Prepared feature pair source ancestry differs')
    return m


def _metadata(ids,meta,role):
    n=COUNTS[role];fields={'rx_i','day_i','domain'}|({'y'} if role=='L' else set())
    if (ids.shape!=(n,) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
            or set(meta)!=fields or any(x.shape!=(n,) or x.dtype!=np.int64 for x in meta.values())
            or not np.isin(meta['rx_i'],[1,3,4,6,8]).all() or not np.isin(meta['day_i'],[1,2,3]).all()
            or not np.array_equal(meta['domain'],np.searchsorted([1,3,4,6,8],meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Exact opaque IDs and role-public source metadata required; OE TX/y/K forbidden')
    if role=='L' and (not np.isin(meta['y'],range(6)).all() or n%90 or not np.all(np.bincount(meta['y']*15+meta['domain'],minlength=90)==n//90)):
        raise ValueError('Original balanced source-L six-class/domain grid required')


def load_role(root,role):
    if role not in COUNTS:raise ValueError('Only prepared source-L or OE feature pairs may be opened')
    root=Path(root);m=validate_cache_manifest(_read(root/'complete.json'));fields={'ids','u','c','rx_i','day_i','domain'}|({'y'} if role=='L' else set())
    with np.load(root/(role+'.npz'),allow_pickle=False) as saved:
        if set(saved.files)!=fields:raise ValueError('Exact prepared role arrays required')
        a={k:saved[k].copy() for k in saved.files}
    meta={k:a[k] for k in fields-{'ids','u','c'}};_metadata(a['ids'],meta,role)
    for k,d in (('u',160),('c',640)):
        if a[k].shape!=(COUNTS[role],d) or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all():raise ValueError('Finite FP32 u160/c640 pairs required')
        if not np.allclose(np.linalg.norm(a[k].astype(np.float64),axis=1),1.,atol=2e-6,rtol=0.):raise ValueError('Prepared features must be nonzero unit vectors')
    for value in a.values():value.setflags(write=False)
    return dict(manifest=m,ids=a['ids'],metadata=meta,u=a['u'],c=a['c'])


class Sampler:
    def __init__(self,L,OE=None,L_seed=2026103017,OE_seed=2026103018):
        if OE is None:raise ValueError('Both cross-set arms require the OE candidate pool')
        self.L=self._cells(L,True);self.OE=None;self.L_rng=np.random.default_rng(L_seed)
        if OE is not None:self.OE=self._cells(OE,False);self.OE_rng=np.random.default_rng(OE_seed)

    @staticmethod
    def _cells(a,labeled):
        role='L' if labeled else 'OE';_metadata(a['ids'],a['metadata'],role);m=a['metadata'];result={}
        for label in range(6) if labeled else (None,):
            for domain in range(15):
                mask=m['domain']==domain
                if labeled:mask=mask&(m['y']==label)
                rows=np.flatnonzero(mask)
                if len(rows)<(3 if labeled else 5):raise ValueError('Each source cell needs sufficient no-replacement samples')
                result[label,domain]=rows
        return result

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive step index required')
        t=step-1;known=[];outside=[]
        for label in range(6):
            extra=(2*t+label)%15
            for domain in range(15):known.extend(self.L_rng.choice(self.L[label,domain],2+int(domain in (extra,(extra+1)%15)),replace=False))
        if self.OE is not None:
            extra={(4*t+j)%15 for j in range(4)}
            for domain in range(15):outside.extend(self.OE_rng.choice(self.OE[None,domain],4+int(domain in extra),replace=False))
        return dict(L=np.asarray(known,np.int64),OE=np.asarray(outside,np.int64))


def _tensor(x,device,dtype=torch.float32):return torch.tensor(np.asarray(x).tolist(),device=device,dtype=dtype)


def _model(c,device):return core.MetricResidual().to(device=device,dtype=torch.float32)


def _objective(model,L,OE,sample,device,topology=False,track_crossset_grad=False):
    if OE is None:raise ValueError('Both arms require the same L and OE candidate sets')
    ix=sample['L'];m=L['metadata'];u=_tensor(L['u'][ix],device);z=model(u,_tensor(L['c'][ix],device))
    oi=sample['OE'];om=OE['metadata'];teacher=_tensor(OE['u'][oi],device);oz=model(teacher,_tensor(OE['c'][oi],device))
    loss=core.crossset_metric(z,oz,_tensor(m['y'][ix],device,torch.long),_tensor(m['rx_i'][ix],device,torch.long),
        _tensor(m['day_i'][ix],device,torch.long),_tensor(om['rx_i'][oi],device,torch.long),_tensor(om['day_i'][oi],device,torch.long))
    cross_gradients=None
    if track_crossset_grad:
        gradients=torch.autograd.grad(loss['loss'],(z,oz),retain_graph=True)
        cross_gradients={key:dict(norm=float(g.detach().norm().cpu()),minimum_row_norm=float(g.detach().norm(dim=1).min().cpu()))
            for key,g in zip(('L','OE'),gradients)}
    other=None
    if topology:
        other=core.oe_structure(oz,teacher,_tensor(om['rx_i'][oi],device,torch.long),_tensor(om['day_i'][oi],device,torch.long))
    drift_oe=float((1-(oz*torch.nn.functional.normalize(teacher,dim=1)).sum(1)).mean().detach().cpu())
    total=loss['loss']+(other['loss'] if other is not None else 0.)
    return total,dict(loss=float(total.detach().cpu()),L_metric=float(loss['loss'].detach().cpu()),crossset_metric=float(loss['loss'].detach().cpu()),
        L_metric_definition='L anchors vs L+OE; equal set/domain priors; L-only positives',OE_KL=0. if other is None else float(other['loss'].detach().cpu()),
        L_weight=1.,OE_weight=0. if other is None else 1.,L_anchors=len(ix),OE_candidates=len(sample['OE']),
        OE_anchors=len(sample['OE']) if topology else 0,candidate_set_priors=[.5,.5],crossset_embedding_gradients=cross_gradients,
        L_feature_drift=float((1-(z*torch.nn.functional.normalize(u,dim=1)).sum(1)).mean().detach().cpu()),OE_feature_drift=drift_oe)


def _gradient_norms(model):
    return {key:sum(float(p.grad.detach().double().square().sum().cpu()) for p in layer.parameters() if p.grad is not None)**.5
        for key,layer in (('hidden',model.network[0]),('last',model.network[-1]))}


def _gradient_smoke(model,arm,device):
    state={k:v.detach().clone() for k,v in model.state_dict().items()};generator=torch.Generator().manual_seed(2026103017)
    rx=np.asarray([1,3,1,3],np.int64);day=np.asarray([1,2,1,2],np.int64)
    def role(name,known):
        u=torch.nn.functional.normalize(torch.randn(4,160,generator=generator),dim=1)
        c=torch.nn.functional.normalize(torch.randn(4,640,generator=generator),dim=1)
        meta=dict(rx_i=rx,day_i=day)
        if known:meta['y']=np.asarray([0,0,1,1],np.int64)
        return dict(u=np.asarray(u.tolist(),np.float32),c=np.asarray(c.tolist(),np.float32),metadata=meta)
    try:
        L=role('L',True);OE=role('OE',False)
        sample=dict(L=np.arange(4),OE=np.arange(4));model.zero_grad(set_to_none=True)
        loss,metrics=_objective(model,L,OE,sample,device,topology=arm=='S_SEP_TOPO',track_crossset_grad=True);loss.backward();norms=_gradient_norms(model)
        if not np.isfinite(list(norms.values())).all() or norms['last']<=0:raise ValueError('Actual metric checkpoint gradient smoke failed')
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in state.items()):raise ValueError('No-step smoke changed network parameters')
        return dict(status='PASS',optimizer_updates=0,synthetic_L_feature_rows=4,synthetic_OE_feature_rows=4 if OE is not None else 0,
            first_gradient_norms=norms,parameters_unchanged=True,metrics=metrics,F_forward_rows=0,G_forward_rows=0)
    finally:model.zero_grad(set_to_none=True)


def _identity(c,manifest):
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],checkpoint=str(Path(c['output_root'])/'final.pt'),
        updates=STEPS,native_updates=0,original_parent_updates=11100,new_parameters=102688,original_F0_parameters=232993,
        combined_active_parameters=335681,initialization_seed=2026092701,parent_F0_identity=manifest['parent_F0_identity'],dura_identity=manifest['dura_identity'],
        identity_head_replaced=False,original_F0_frozen=True,input_path='cached_P1_features_only',OE_embedding_training=True,
        OE_topology_regularization=c['arm']=='S_SEP_TOPO',crossset_separation=True)


def _loop(model,L,OE,c,device,optimizer,on_step):
    sampler=Sampler(L,OE,c['L_seed'],c['OE_seed']);usage=dict(L=set(),OE=set());trace=[]
    for step in range(1,c['steps']+1):
        sample=sampler.sample(step);optimizer.zero_grad(set_to_none=True)
        total,metrics=_objective(model,L,OE,sample,device,topology=c['arm']=='S_SEP_TOPO',track_crossset_grad=step<=2);total.backward()
        gradients=_gradient_norms(model)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite open metric gradient')
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
        optimizer.step()
        for key in usage:usage[key].update(map(int,sample[key]))
        row=dict(step=step,open_network_updates=step,native_updates=0,L_indices=sample['L'].tolist(),OE_indices=sample['OE'].tolist(),
            gradient_norms=gradients,gradient_norm_before_clip=norm,lr=float(optimizer.param_groups[0]['lr']),**metrics)
        trace.append(row);on_step(row)
        if step==1 or step%100==0 or step==c['steps']:print('STAR_CROSSSET_METRIC_TRAIN_PROGRESS',c['arm'],step,c['steps'],metrics['loss'],flush=True)
    return trace,{k:len(v) for k,v in usage.items()}


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable open metric training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=_model(c,device);smoke=_gradient_smoke(model,c['arm'],device)
    if smoke.get('status')!='PASS':raise ValueError('No-query actual new-network gradient smoke must pass first')
    out.mkdir(parents=True,exist_ok=False);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    L=load_role(c['cache_root'],'L');m=L['manifest'];OE=load_role(c['cache_root'],'OE')
    if m['code_commit']!=c['code_commit'] or (OE is not None and (OE['manifest']!=m or set(L['ids'].tolist())&set(OE['ids'].tolist()))):
        raise ValueError('Same-source prepared cache, code and disjoint role IDs required')
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=.0001)
    with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
        def record(row):log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
        trace,unique=_loop(model,L,OE,c,device,optimizer,record)
    if len(trace)!=STEPS or trace[-1]['step']!=STEPS:raise ValueError('All fixed open metric updates must finish')
    identity=_identity(c,m);zeros={k:0 for k in ZEROS}
    torch.save(dict(schema=SCHEMA,config=c,identity=identity,network=model.state_dict(),cache_manifest=m,source_only=True,**zeros),out/'final.pt')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,identity=identity,updates=STEPS,
        source_only=True,cache_manifest=m,parent_F0_identity=m['parent_F0_identity'],dura_identity=m['dura_identity'],new_parameters=102688,
        real_F_forward_rows=0,synthetic_F_forward_rows=0,L_training_rows_loaded=len(L['ids']),OE_training_rows_loaded=0 if OE is None else len(OE['ids']),
        L_sampled_exposures=192*STEPS,OE_sampled_exposures=64*STEPS if OE is not None else 0,unique_training_rows=unique,
        optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,gradient_clip_L2=5.),first_step=trace[0],second_step=trace[1] if len(trace)>1 else None,
        final_step=trace[-1],statistical_background_uses_OE=True,embedding_training_only=True,final_phase3_complete=False,**zeros)
    _write(out/'training_summary.json',done);_write(out/'complete.json',done);print(STATUS,c['arm'],flush=True);return done


def load_trained_model(c,checkpoint,device='cuda:0'):
    validate_config(c)
    if Path(checkpoint).resolve()!=Path(c['output_root'],'final.pt').resolve():raise ValueError('Only the fixed final network checkpoint may load')
    done=_read(Path(c['output_root'])/'complete.json');m=validate_cache_manifest(_read(Path(c['cache_root'])/'complete.json'))
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,updates=STEPS,
        identity=_identity(c,m),cache_manifest=m,source_only=True,final_phase3_complete=False,**{k:0 for k in ZEROS})
    if m['code_commit']!=c['code_commit'] or any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete matching source-only metric network required')
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('config')!=c or value.get('identity')!=done['identity']
            or value.get('cache_manifest')!=m or value.get('source_only') is not True or any(value.get(k)!=0 for k in ZEROS)):
        raise ValueError('New-network checkpoint source provenance differs')
    model=_model(c,device);state=value.get('network',{});expected=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(expected) or any(not torch.is_tensor(v) or v.dtype!=torch.float32 or v.shape!=expected[k].shape or not bool(torch.isfinite(v).all()) for k,v in state.items()):
        raise ValueError('Strict complete finite new-network state required')
    model.load_state_dict(state,strict=True);return model.eval().requires_grad_(False),value['identity']


def load_trained(c,device='cuda:0'):return load_trained_model(c,Path(c['output_root'])/'final.pt',device)


def _snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main(argv=None):
    print(json.dumps(dict(event='STAR_CROSSSET_METRIC_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','train'));p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    {'prepare':prepare,'train':train}[a.mode](_read(a.config),a.device)


if __name__=='__main__':main()
