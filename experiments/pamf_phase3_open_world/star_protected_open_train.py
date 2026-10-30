"""Source-only protected160 adapters; original F0 is frozen, never continued."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from . import star_protected_open_cache as cache
from . import star_protected_open_model as ow
from . import star_protected_open_loss as losses

RUN_ID=cache.RUN_ID
ARMS=('N0','R0','R1')
SCHEMA='pamf_phase3_star_protected_open_train_v1'
STATUS='STAR_PROTECTED_SOURCE_FROZEN'
STEPS=2220
FIELDS={'run_id','code_commit','arm','native_config','feature_root','output_root','steps','lr','sampling_seed','slot_seed'}
load_features=cache.load_features
_read,_write,_tensor=cache._read,cache._write,cache._tensor


def validate_config(c):
    cache._base_validate(c,FIELDS)
    if c.get('arm') not in ARMS:raise ValueError('Only the fixed N0/R0/R1 arms are allowed')
    for k,v in dict(steps=STEPS,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018).items():
        if type(c[k]) is not type(v) or c[k]!=v:raise ValueError('Fixed 2220 adapter updates, LR and sampling seeds required')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in ('feature_root','output_root')):
        raise ValueError('Nonempty source feature and immutable output paths required')
    n=c['native_config']
    if n['row_id']!=c['arm']+'-s2026092701' or n['output_root']!=c['output_root']:
        raise ValueError('Constructor row/output must match the new adapter arm')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out)
            for p in (c['feature_root'],n['source_output'],n['source_cache'])):
        raise ValueError('Training output must be separate from frozen features and original inputs')
    return c


def _model(c,device):
    model=cache._construct(c['native_config'],device,c['arm']!='R0')
    p=ow.parameter_report(model)
    if (p['native_identity_parameters']!=232993 or p['open_world_parameters']!=66177 or p['active_parameters']!=299170
            or p['training_container_parameters']!=966899 or p['trainable_identity_parameters']!=0
            or p['trainable_open_world_parameters']!=(10305 if c['arm']=='R0' else 66177)):
        raise ValueError('Measured frozen-original/protected adapter parameter counts differ')
    return model


class EpisodeSampler:
    """Independent L/OE RNG streams; no OE inspection for the N0 control."""
    def __init__(self,L,OE=None,seed=2026103017,slot_seed=2026103018):
        if type(seed) is not int or type(slot_seed) is not int or seed<0 or slot_seed<0:
            raise ValueError('Nonnegative fixed sampling seeds required')
        self.seed=seed;self.slot_seed=slot_seed;self.L=self._cells(L,True);self.OE=None
        if OE is not None:self.OE=self._cells(OE,False)

    @staticmethod
    def _cells(role,labeled):
        ids=role['ids'];meta=role['metadata'];n=len(ids);allowed={'rx_i','day_i','domain'}|({'y'} if labeled else set())
        if (ids.ndim!=1 or ids.dtype.kind!='U' or len(set(ids.tolist()))!=n or np.any(np.char.strip(ids)=='')
                or set(meta)!=allowed or any(v.shape!=(n,) or v.dtype!=np.int64 for v in meta.values())
                or not np.isin(meta['rx_i'],cache.native.RX).all() or not np.isin(meta['day_i'],(1,2,3)).all()
                or not np.array_equal(meta['domain'],np.searchsorted(cache.native.RX,meta['rx_i'])*3+meta['day_i']-1)
                or (labeled and not np.isin(meta['y'],range(6)).all())):
            raise ValueError('Only unique physical IDs and public source domains, plus L old labels, are allowed')
        keys=[(c,d) for c in range(6) for d in range(15)] if labeled else list(range(15));result={}
        for key in keys:
            mask=(meta['y']==key[0])&(meta['domain']==key[1]) if labeled else meta['domain']==key
            rows=np.flatnonzero(mask).astype(np.int64)
            if len(rows)<(4 if labeled else 5):raise ValueError('Every public cell must supply disjoint no-replacement support/query or OE draws')
            result[key]=rows
        return result

    def sample(self,step):
        if type(step) is not int or step<1:raise ValueError('Positive integer step required')
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,0]));extra=(step-1)%15
        support=[];query=[]
        for c in range(6):
            for domain in range(15):
                count=1+int(domain==extra);chosen=rng.choice(self.L[c,domain],2*count,replace=False)
                support.extend(chosen[:count]);query.extend(chosen[count:])
        outside=[]
        if self.OE is not None:
            oe_rng=np.random.default_rng(np.random.SeedSequence([self.seed,step,1]));extras={(step-1+j)%15 for j in range(4)}
            for domain in range(15):outside.extend(oe_rng.choice(self.OE[domain],4+int(domain in extras),replace=False))
        slot_rng=np.random.default_rng(np.random.SeedSequence([self.slot_seed,step]));k=2+(step-1)%4
        return dict(support=np.asarray(support,np.int64),query=np.asarray(query,np.int64),OE=np.asarray(outside,np.int64),
            pseudo_known_classes=slot_rng.permutation(6)[:k].astype(np.int64))


def _objective(model,L,OE,sample,device):
    si=sample['support'];qi=sample['query'];ix=np.concatenate((si,qi));meta=L['metadata']
    raw=_tensor(L['raw_original_fused160'][ix],device)
    z=ow.encode(model,raw,_tensor(L['contributions'][ix],device))['protected160']
    rx=_tensor(meta['rx_i'][ix],device,torch.long);day=_tensor(meta['day_i'][ix],device,torch.long)
    labels=_tensor(meta['y'][si],device,torch.long);n=len(si)
    proto=losses.query_prototypes(z[:n],labels,rx[:n],day[:n],rx[n:],day[n:])
    member=losses.member_loss(ow.relation(model,z[n:],proto),_tensor(meta['y'][qi],device,torch.long),
        _tensor(sample['pseudo_known_classes'],device,torch.long))
    local_L=losses.soft_neighbor_loss(z,raw,rx,day,_tensor(ix,device,torch.long))
    outside=None;local_OE=None
    if OE is not None:
        oi=sample['OE'];om=OE['metadata'];old=_tensor(OE['raw_original_fused160'][oi],device)
        oz=ow.encode(model,old,_tensor(OE['contributions'][oi],device))['protected160']
        orx=_tensor(om['rx_i'][oi],device,torch.long);oday=_tensor(om['day_i'][oi],device,torch.long)
        op=losses.query_prototypes(z[:n],labels,rx[:n],day[:n],orx,oday)
        outside=losses.oe_member_loss(ow.relation(model,oz,op))
        local_OE=losses.soft_neighbor_loss(oz,old,orx,oday,_tensor(oi,device,torch.long))
    neighbor=local_L['loss'] if OE is None else .5*(local_L['loss']+local_OE['loss'])
    total=member['loss']+.1*neighbor+(outside['loss'] if outside is not None else 0.)
    def plain(row):return None if row is None else {k:float(v.detach().cpu()) if torch.is_tensor(v) else v for k,v in row.items()}
    return total,dict(member=plain(member),OE_member=plain(outside),neighbor_L=plain(local_L),neighbor_OE=plain(local_OE),
        soft_neighbor_weight=.1,neighbor_role_weights=[1.,0.] if OE is None else [.5,.5],
        member_weight=1.,OE_member_weight=0. if OE is None else 1.,loss=float(total.detach().cpu()))


def _synthetic_features(with_OE):
    rng=np.random.default_rng(2026103017);grid=[(r,d) for r in cache.native.RX for d in (1,2,3)]
    domains=[cell for _ in range(2) for c in range(6) for cell in grid+[grid[c]]]
    def role(name,domains,labels=None):
        f=rng.standard_normal((len(domains),160)).astype(np.float32)
        meta=dict(rx_i=np.asarray([r for r,d in domains],np.int64),day_i=np.asarray([d for r,d in domains],np.int64))
        meta['domain']=np.searchsorted(cache.native.RX,meta['rx_i'])*3+meta['day_i']-1
        if labels is not None:meta['y']=labels
        return dict(ids=np.asarray([name+'-'+str(i) for i in range(len(f))]),metadata=meta,raw_original_fused160=f,
            contributions=np.stack((.4*f,.3*f,.2*f,.1*f),1))
    L=role('synthetic-L',domains,np.tile(np.repeat(np.arange(6,dtype=np.int64),16),2))
    OE=role('synthetic-OE',grid*4+grid[:4]) if with_OE else None
    sample=dict(support=np.arange(96,dtype=np.int64),query=np.arange(96,192,dtype=np.int64),
        OE=np.arange(64,dtype=np.int64) if with_OE else np.empty(0,np.int64),pseudo_known_classes=np.asarray([0,2,4],np.int64))
    return L,OE,sample


def _gradient_norms(model):
    result={}
    for prefix in ('projector.','direction_projection.','residual_gain','relation_scorer.'):
        result[prefix]=sum(float(p.grad.detach().double().square().sum().cpu()) for n,p in model.open_world.named_parameters()
            if n.startswith(prefix) and p.grad is not None)**.5
    return result


def _smoke(model,c,device):
    before=cache._snapshot(model)
    try:
        first=cache._smoke(model,device);L,OE,sample=_synthetic_features(c['arm']!='N0')
        model.zero_grad(set_to_none=True);total,metrics=_objective(model,L,OE,sample,device);total.backward()
        gradients=_gradient_norms(model)
        if (not np.isfinite(list(gradients.values())).all() or gradients['relation_scorer.']<=0
                or (c['arm']!='R0' and gradients['residual_gain']<=0)
                or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('open_world.'))
                or (c['arm']=='R0' and any(p.grad is not None for n,p in model.open_world.named_parameters() if not n.startswith('relation_scorer.')))):
            raise ValueError('No-step protected adapter/relation gradients or frozen original F0 smoke failed')
        cache._unchanged(model,before)
        return dict(first,gradient_status='PASS',gradient_norms=gradients,synthetic_L_feature_rows=192,
            synthetic_OE_feature_rows=64 if OE is not None else 0,optimizer_updates=0,metrics=metrics)
    finally:model.zero_grad(set_to_none=True);ow.clear(model)


def _identity(c,model,manifest):
    p=ow.parameter_report(model)
    return dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],checkpoint=str(Path(c['output_root'])/'final.pt'),
        adapter_updates=STEPS,original_parent_updates=11100,native_updates=0,original_parent_identity=model._phase3_parent_identity,
        parent_F0_identity=manifest['parent_F0_identity'],dura_identity=manifest['dura_identity'],
        identifier_parameters=p['active_parameters'],open_world_parameters=p['open_world_parameters'],
        training_container_parameters=p['training_container_parameters'],trainable_parameters=p['trainable_open_world_parameters'],
        native_identity_frozen=True,identity_head_replaced=False,input_path='cached_P1_only',
        residual_enabled=c['arm']!='R0',OE_member_supervision=c['arm']!='N0')


def _check_feature_source(bundle,c,model):
    m=cache._manifest(bundle['manifest']);cache._parent(m['parent_F0_identity'],model._phase3_parent_identity)
    if m['code_commit']!=c['code_commit'] or m['config']['native_config']['source_cache']!=c['native_config']['source_cache']:
        raise ValueError('Feature cache and current original-F0 constructor source identity differ')


def _loop(model,L,OE,c,device,optimizer,on_step=None):
    if (c['arm']=='N0')!=(OE is None):raise ValueError('N0 must not open/sample OE; R0/R1 require source OE')
    sampler=EpisodeSampler(L,OE,c['sampling_seed'],c['slot_seed']);params=[p for p in model.open_world.parameters() if p.requires_grad]
    used={k:set() for k in ('support','query','OE')};trace=[]
    for step in range(1,c['steps']+1):
        start=time.perf_counter();sample=sampler.sample(step);optimizer.zero_grad(set_to_none=True)
        total,metrics=_objective(model,L,OE,sample,device);total.backward();gradients=_gradient_norms(model)
        if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('open_world.')):
            raise ValueError('Original F0 received a gradient')
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):raise FloatingPointError('Nonfinite adapter gradient')
        norm=float(torch.nn.utils.clip_grad_norm_(params,5.))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite adapter gradient norm')
        optimizer.step();ow.clear(model)
        for key in used:used[key].update(map(int,sample[key]))
        row=dict(step=step,adapter_updates=step,native_updates=0,original_parent_updates=11100,arm=c['arm'],
            support_indices=sample['support'].tolist(),query_indices=sample['query'].tolist(),OE_indices=sample['OE'].tolist(),
            pseudo_known_classes=sample['pseudo_known_classes'].tolist(),gradient_norms=gradients,gradient_norm_before_clip=norm,
            lr=float(optimizer.param_groups[0]['lr']),seconds=time.perf_counter()-start,**metrics)
        trace.append(row)
        if on_step:on_step(row)
        if step==1 or step%100==0 or step==c['steps']:print('STAR_PROTECTED_TRAIN_PROGRESS',c['arm'],step,c['steps'],metrics['loss'],flush=True)
    exposures=dict(L_support=96*c['steps'],L_query=96*c['steps'],OE=(64*c['steps'] if OE is not None else 0))
    return trace,dict(unique_support=len(used['support']),unique_query=len(used['query']),unique_L_union=len(used['support']|used['query']),
        unique_OE=len(used['OE']),exposures=exposures)


def train(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('A new immutable adapter training root is required')
    start=time.perf_counter();model=_model(c,device);smoke=_smoke(model,c,device)
    if smoke.get('status')!='PASS' or smoke.get('gradient_status')!='PASS':raise ValueError('Actual-checkpoint no-query gradient smoke must pass')
    output.mkdir(parents=True,exist_ok=False);_write(output/'resolved_config.json',c);_write(output/'smoke.json',smoke)
    L=load_features(c['feature_root'],'L');_check_feature_source(L,c,model);OE=None
    if c['arm']!='N0':
        OE=load_features(c['feature_root'],'OE');_check_feature_source(OE,c,model)
        if OE['manifest']!=L['manifest'] or set(L['ids'].tolist())&set(OE['ids'].tolist()):raise ValueError('Disjoint roles from one source feature cache required')
    before=cache._snapshot(model,native_only=True);params=[p for p in model.open_world.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=c['lr'],weight_decay=.0001)
    with (output/'steps.jsonl').open('x',encoding='utf-8') as log:
        def on_step(row):log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
        trace,usage=_loop(model,L,OE,c,device,optimizer,on_step)
    cache._unchanged(model,before)
    if len(trace)!=STEPS or trace[-1]['step']!=STEPS:raise ValueError('All fixed adapter updates must complete')
    identity=_identity(c,model,L['manifest']);zeros={k:0 for k in cache.ZEROS}
    checkpoint=dict(schema=SCHEMA,config=c,identity=identity,open_world=model.open_world.state_dict(),
        feature_manifest=L['manifest'],source_data_handle=model._phase3_source_handle,source_only=True,
        model_native_state_unchanged=True,**zeros)
    torch.save(checkpoint,output/'final.pt')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,identity=identity,
        updates=STEPS,adapter_updates=STEPS,original_parent_updates=11100,model_native_state_unchanged=True,
        source_only=True,parent_F0_identity=L['manifest']['parent_F0_identity'],dura_identity=L['manifest']['dura_identity'],
        feature_manifest=L['manifest'],input_path='cached_P1_only',OE_member_supervision=OE is not None,
        L_feature_rows_loaded=len(L['ids']),OE_feature_rows_loaded=0 if OE is None else len(OE['ids']),
        L_support_exposures=96*STEPS,L_query_exposures=96*STEPS,OE_exposures=64*STEPS if OE is not None else 0,
        F_forward_rows=0,synthetic_F_forward_rows=2,encoded_feature_rows=(192+(64 if OE is not None else 0))*STEPS,
        trainable_parameters=sum(p.numel() for p in params),parameter_report=ow.parameter_report(model),
        optimizer=dict(name='AdamW',lr=.0001,weight_decay=.0001,gradient_clip_L2=5.),
        actual_usage=usage,final_step=trace[-1],seconds=time.perf_counter()-start,final_phase3_complete=False,**zeros)
    _write(output/'training_summary.json',done);_write(output/'complete.json',done)
    print(STATUS,c['arm'],flush=True);return done


def load_trained_model(c,checkpoint,device='cuda:0'):
    validate_config(c)
    if Path(checkpoint).resolve()!=Path(c['output_root'],'final.pt').resolve():raise ValueError('Only the fixed final adapter checkpoint is allowed')
    done=_read(Path(c['output_root'])/'complete.json')
    required=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,updates=STEPS,
        adapter_updates=STEPS,original_parent_updates=11100,source_only=True,model_native_state_unchanged=True,**{k:0 for k in cache.ZEROS})
    if any(done.get(k)!=v for k,v in required.items()):raise ValueError('Complete frozen source-only adapter training required')
    value=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('config')!=c
            or value.get('identity')!=done.get('identity') or value.get('feature_manifest')!=done.get('feature_manifest')
            or value.get('source_only') is not True or value.get('model_native_state_unchanged') is not True
            or 'model' in value or any(value.get(k)!=0 for k in cache.ZEROS)):
        raise ValueError('Only matching new-parts weights and original-source ancestry may be loaded')
    manifest=cache._manifest(value['feature_manifest']);model=_model(c,device)
    _check_feature_source(dict(manifest=manifest),c,model)
    if (value['identity']!=_identity(c,model,manifest) or value.get('source_data_handle')!=model._phase3_source_handle):
        raise ValueError('Adapter checkpoint original F0 identity or source handle differs')
    try:model.open_world.load_state_dict(value['open_world'],strict=True)
    except (KeyError,RuntimeError) as exc:raise ValueError('Complete finite new adapter/relation state required') from exc
    if any(not bool(torch.isfinite(p).all()) for p in model.open_world.parameters()):raise ValueError('Finite frozen adapter parameters required')
    ow.clear(model);return model.eval().requires_grad_(False),value['identity']


def main(argv=None):
    print(json.dumps(dict(event='STAR_PROTECTED_TRAIN_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(argv)
    train(_read(a.config),a.device)


if __name__=='__main__':main()
