"""Matched native E51 increments with an additional clean or frozen-G view."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import development_transfer as frozen, source_open_embedding_train as native, runner
from .source_open_export import _tensor

RUN='20261029-phase3-source-practical-encoder-s2026092701-r01'
ARMS=('clean_increment','practical_increment')
VIEWS=('clean','practical_low_suburban','practical_mid_urban','practical_low_urban')
FIELDS={'run_id','code_commit','arm','source_row_root','risk_prediction_root','cache_root','output_root',
        'steps','lr','L_seed','OE_seed','scene_seed'}
SCHEMA='pamf_phase3_source_practical_encoder_v1'
STATUS='SOURCE_PRACTICAL_ENCODER_FROZEN'
PARENT_UPDATES=11322
STEPS=222


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS:raise ValueError('Only fixed source training inputs; no V/target/truth/counts')
    for k in FIELDS-{'steps','lr','L_seed','OE_seed','scene_seed'}:
        if not isinstance(c[k],str) or not c[k].strip():raise ValueError('Nonempty source identity/path '+k)
    if (c['run_id']!=RUN or c['arm'] not in ARMS or len(c['code_commit'])!=40
            or any(x not in '0123456789abcdef' for x in c['code_commit'])):raise ValueError('Fixed run/arms and actual commit required')
    for k,v in dict(steps=222,lr=.0001,L_seed=2026092701,OE_seed=2026102907,scene_seed=2026102932).items():
        if type(c[k]) is not type(v) or c[k]!=v:raise ValueError('Fixed budget/seed/optimizer: '+k)
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(c[k]).resolve()) for k in ('source_row_root','risk_prediction_root','cache_root')):
        raise ValueError('New training output cannot overwrite its source/cache input')
    return c


def initial_model(c,device='cuda:0'):
    """Validate E51, rebuild native E52 hooks, strict-load the same E51 weights."""
    source=frozen._source_bundle(c)
    parent=deepcopy(source['marker']['encoder_state_identity'])
    old=runner._read(source['marker']['config']['train_config'])
    verified=native.load_trained_model(old,parent['checkpoint'],'cpu')
    original=runner._read(Path(old['source_output'])/'complete.json')['config']
    nc=native._training_config(old,original)
    nc.update(run_id=RUN,row_id=c['arm']+'-s2026092701',output_root=c['output_root'],
        step_offset=PARENT_UPDATES,epoch_offset=51,segment_epochs=1)
    model=native._new_native(old,nc,device)
    model.load_state_dict(verified.state_dict(),strict=True)
    model._phase3_parent_args=deepcopy(verified._phase3_parent_args)
    model._phase3_source_handle=deepcopy(verified._phase3_source_handle)
    model._phase3_parent_identity=deepcopy(verified._phase3_parent_identity)
    model._phase3_E51_identity=parent
    model._phase3_native_config=nc
    del verified
    model.train().requires_grad_(True);native._assert_trainable(model)
    if parent.get('cumulative_updates')!=11322 or parent.get('arm')!='uniform_oe':raise ValueError('Verified uniform-OE E51 endpoint required')
    return model


def select_view(arrays,indices,scenes,arm):
    indices=np.asarray(indices);scenes=np.asarray(scenes)
    if (arm not in ARMS or indices.ndim!=1 or indices.dtype.kind not in 'iu' or not len(indices)
            or scenes.shape!=indices.shape or scenes.dtype.kind not in 'iu' or np.any(scenes<0) or np.any(scenes>2)):
        raise ValueError('Aligned class-independent fixed-view selection required')
    if np.any(indices<0) or np.any(indices>=len(arrays['clean'])):raise ValueError('Cache row index outside matching source role')
    if arm=='clean_increment':return np.ascontiguousarray(arrays['clean'][indices],dtype=np.float32)
    result=np.empty((len(indices),2,256),np.float32)
    for j,view in enumerate(VIEWS[1:]):
        mask=scenes==j
        if np.any(mask):result[mask]=arrays[view][indices[mask]]
    if not np.isfinite(result).all():raise ValueError('Nonfinite source practical cached view')
    return result


def _histogram(hist,key,scenes,arm):
    counts=hist.setdefault(key,dict.fromkeys(VIEWS,0))
    if arm=='clean_increment':counts['clean']+=len(scenes)
    else:
        for j,view in enumerate(VIEWS[1:]):counts[view]+=int(np.sum(scenes==j))


def labeled_loss(arrays,arm,generator,histogram):
    def combined(model,raw,y,domains,meta,augmentor,no_pa,step):
        loss,metrics=native.supervised_loss(model,raw,y,domains,meta,augmentor,no_pa,step)
        idx=np.asarray(meta['cache_index'].tolist(),np.int64)
        scene=np.asarray(torch.randint(3,(len(idx),),generator=generator).tolist(),np.int64)
        extra=_tensor(select_view(arrays,idx,scene,arm),raw.device)
        ce=F.cross_entropy(native._reader_logits(model,extra),y,label_smoothing=.01)
        _histogram(histogram,'additional_L',scene,arm)
        metrics=dict(metrics,extra_CE_loss=float(ce.detach()),extra_CE_weight=.68,extra_L_packet_views=len(idx))
        return loss+.68*ce,metrics
    return combined


class CachedL(torch.utils.data.Dataset):
    role='L_s'
    def __init__(self,arrays,metadata):
        self.x=_tensor(arrays['clean'],'cpu');self.metadata=metadata
    def __len__(self):return len(self.x)
    def __getitem__(self,index):
        m={k:torch.tensor(int(self.metadata[k][index]),dtype=torch.long) for k in ('rx_i','day_i','eq_i','sig_i','base_index')}
        m['tx_label_visible']=True
        m['cache_index']=torch.tensor(index,dtype=torch.long)
        return self.x[index],torch.tensor(int(self.metadata['y'][index])),torch.tensor(int(self.metadata['domain'][index])),m


def _close(arrays):
    for value in arrays.values():
        if isinstance(value,np.memmap):value._mmap.close()


def train(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable encoder training root required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=initial_model(c,device)
    # Smoke uses no real source/query and preserves the model/RNG/counters.
    t=torch.arange(256,dtype=torch.float32,device=device)
    synthetic=torch.stack((torch.stack((torch.cos(.11*t),torch.sin(.11*t))),torch.stack((torch.cos(.19*t),torch.sin(.19*t)))))
    smoke=native.gradient_smoke(model,synthetic,torch.tensor([0,1],dtype=torch.long,device=device))
    oe_smoke=native.gradient_smoke(model,synthetic)
    print('SOURCE_PRACTICAL_ENCODER_GRADIENT_SMOKE_PASS',c['arm'],json.dumps(dict(known=smoke,OE=oe_smoke)),flush=True)
    from .source_practical_encoder_cache import cache_marker,load_role
    cache=cache_marker(c['cache_root'])
    if (cache['encoder_state_identity']!=model._phase3_E51_identity
            or any(cache['config'][k]!=c[k] for k in ('run_id','code_commit','source_row_root','risk_prediction_root'))):
        raise ValueError('Training cache and verified E51/source identity differ')
    views=('clean',) if c['arm']=='clean_increment' else VIEWS
    arrays,meta=load_role(c['cache_root'],'source_L',views=views)
    oe_arrays,oe_meta=load_role(c['cache_root'],'source_OE',views=views)
    try:
        if set(meta['ids'].tolist())&set(oe_meta['ids'].tolist()):raise ValueError('Source L and OE training roles overlap')
        labeled=CachedL(arrays,meta)
        augmentor,no_pa,augmentation_state=native._augmentor(model._phase3_parent_args)
        torch.manual_seed(c['L_seed']);np.random.seed(c['L_seed']);random.seed(c['L_seed'])
        if torch.device(device).type=='cuda':torch.cuda.manual_seed(c['L_seed'])
        scenes=torch.Generator().manual_seed(c['scene_seed']);sampler=native.DomainSampler(oe_meta['domain'],c['OE_seed'])
        histogram={}
        def next_oe(count):
            idx=sampler.sample(count)
            scene=np.asarray(torch.randint(3,(count,),generator=scenes).tolist(),np.int64)
            clean=_tensor(np.ascontiguousarray(oe_arrays['clean'][idx],dtype=np.float32),device)
            extra=_tensor(select_view(oe_arrays,idx,scene,c['arm']),device)
            _histogram(histogram,'additional_OE',scene,c['arm'])
            return torch.cat((clean,extra),dim=0)
        from experiments.cric_dura_comparison.f_data import FixedEpochLoader
        from experiments.cric_dura_comparison.f_optimizer import build,clip_physical
        batches=FixedEpochLoader(labeled,128,c['L_seed'],steps=222,drop_last=True)
        weight_decay=float(model._phase3_parent_args['weight_decay'])
        optimizer=build(model,list(model.parameters()),'CENTER_UNITS',lr=c['lr'],weight_decay=weight_decay,betas=(.9,.999),eps=1e-8,foreach=False)
        if optimizer.state:raise ValueError('Fresh optimizer moments required')
        out.mkdir(parents=True,exist_ok=False);runner._write(out/'resolved_config.json',c)
        before_head=dict(model.named_parameters())[native.HEAD_NAME].detach().clone()
        before_sinc={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(native.SINC_PREFIX)}
        started=time.perf_counter()
        with (out/'steps.jsonl').open('x',encoding='utf-8') as stream:
            def on_step(row):
                row.update(cumulative_updates=PARENT_UPDATES+row['step'],epoch=52,native_L_packet_views=768,
                    additional_L_packet_views=128,L_packet_views=896,OE_physical_samples=64,OE_packet_views=128,arm=c['arm'])
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print('SOURCE_PRACTICAL_ENCODER_PROGRESS',c['arm'],row['step'],222,'loss',row['loss'],'extra_CE',row['native_metrics']['extra_CE_loss'],flush=True)
            trace=native._train_steps(model,batches,optimizer,augmentor,no_pa,device,222,next_oe,
                lambda opt,params:clip_physical(opt,params,5.),
                labeled_loss=labeled_loss(arrays,c['arm'],scenes,histogram),on_step=on_step)
        if (len(trace)!=222 or optimizer._sinc_unit_steps!=222 or optimizer._sinc_physical_clip_calls!=222
                or model._cd_calls!=222 or model._feature_l_steps!=222 or model._cd_U_seen or model._cd_U_accepted
                or batches.actual_examples!=28416):raise ValueError('Complete matched native E52 trajectory required')
        if torch.equal(before_head,dict(model.named_parameters())[native.HEAD_NAME]):raise ValueError('Original default identity head was not jointly updated')
        if not any(not torch.equal(p,dict(model.named_parameters())[n]) for n,p in before_sinc.items()):raise ValueError('Native Sinc was not updated')
        model.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
        identity=dict(run_id=RUN,arm=c['arm'],code_commit=c['code_commit'],checkpoint=str(out/'final.pt'),
            updates=222,cumulative_updates=11544,parent_encoder_identity=model._phase3_E51_identity)
        summary=dict(status=STATUS,schema=SCHEMA,config=c,encoder_identity=identity,source_data_handle=model._phase3_source_handle,
            source_OE_handle=cache['source_OE_handle'],dura_identity=cache['dura_identity'],cache_root=c['cache_root'],
            native_config=model._phase3_native_config,updates=222,parent_updates=11322,cumulative_updates=11544,epoch=52,
            optimizer_reset=True,optimizer='AdamW_CENTER_UNITS',lr=c['lr'],weight_decay=weight_decay,physical_clip=5.,
            source_only=True,U_reads=0,validation_reads=0,novel_validation_reads=0,target_reads=0,OE_truth_reads=0,
            parameters=232993,default_head_joint=True,identity_head_replaced=False,Sinc_updated=True,head_updated=True,
            L_sampled_exposures=28416,native_L_packet_views=170496,additional_L_packet_views=28416,L_packet_views=198912,
            OE_sampled_exposures=14208,OE_packet_views=28416,scene_histogram=histogram,
            augmentation_state=augmentation_state,additional_CE_weight=.68,OE_weight=.1,
            smoke=dict(known=smoke,OE=oe_smoke,real_checkpoint=True,source_IQ_reads=0,query_reads=0),seconds=time.perf_counter()-started,
            final_Phase3_complete=False)
        with (out/'final.pt').open('xb') as stream:torch.save(dict(schema=SCHEMA,config=c,summary=summary,model={n:v.detach().cpu() for n,v in model.state_dict().items()}),stream)
        runner._write(out/'complete.json',summary);return summary
    finally:_close(arrays);_close(oe_arrays)


def load_trained(train_root,device='cuda:0'):
    root=Path(train_root)
    try:done=runner._read(root/'complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Complete frozen E52 checkpoint required') from exc
    if (done.get('status')!=STATUS or done.get('schema')!=SCHEMA or done.get('updates')!=222
            or done.get('cumulative_updates')!=11544 or done.get('default_head_joint') is not True
            or done.get('identity_head_replaced') is not False or done.get('optimizer_reset') is not True
            or any(done.get(k)!=0 for k in ('U_reads','validation_reads','novel_validation_reads','target_reads','OE_truth_reads'))):
        raise ValueError('Frozen source-only E52 identity/budget differs')
    c=validate_config(done['config'])
    if Path(c['output_root']).resolve()!=root.resolve():raise ValueError('Frozen E52 output path differs')
    checkpoint=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if checkpoint.get('schema')!=SCHEMA or checkpoint.get('config')!=c or checkpoint.get('summary')!=done:
        raise ValueError('Frozen E52 checkpoint and completion differ')
    model=initial_model(c,device)
    if done['encoder_identity']['parent_encoder_identity']!=model._phase3_E51_identity or done['source_data_handle']!=model._phase3_source_handle:
        raise ValueError('E51/source ancestry differs')
    model.load_state_dict(checkpoint['model'],strict=True);native._assert_trainable(model)
    model.eval().requires_grad_(False);model._phase3_encoder_identity=deepcopy(done['encoder_identity'])
    return model


def main(argv=None):
    print(json.dumps(dict(event='SOURCE_PRACTICAL_ENCODER_PROCESS',pid=os.getpid(),cwd=os.getcwd())),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);r=train(runner._read(a.config),a.device);print(r['status'],flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
