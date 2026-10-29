"""Four-state unlabeled development adaptation/discovery and frozen future Q."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import development_transfer as frozen,development_alignment as public
from . import star_relative_core as core
from . import unlabeled_registration_core as reg
from .source_open_export import DIMENSIONS,_tensor,extract_branches
METHODS=core.METHODS

RUN=core.RUN
STATES=('DA0_REG0','DA1_REG0','DA0_REG1','DA1_REG1')
SCHEMA='pamf_phase3_star_relative_target_v1'
STATUS='STAR_RELATIVE_PREDICTIONS_COMPLETE'
FIELDS={'run_id','code_commit','condition','view_family','source_root','U_public_root','U_capsule_id','U_split_id',
    'U_expected_count','Q_public_root','Q_capsule_id','Q_split_id','Q_expected_count','output_root','batch_size','dura_checkpoint','dura_release'}
PRED_FIELDS={'ids','U_ids','state_names','prediction','closed_prediction','baseline_closed_prediction','knownness',
    'diagnostic_reg0_prediction','thresholds','joint_margins_DA0','joint_margins_DA1'}
DIAG_FIELDS={'ids','assignments_DA0','assignments_DA1','candidate_DA0','candidate_DA1'}


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(public._jsonable(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS:raise ValueError('Public U/Q and frozen source only; no truth, unknown count or quota')
    if any(not isinstance(c[k],str) or not c[k].strip() for k in FIELDS-{'U_expected_count','Q_expected_count','batch_size','dura_checkpoint','dura_release'}):raise ValueError('Nonempty identities/paths required')
    if (c['run_id']!=RUN or len(c['code_commit'])!=40 or any(x not in '0123456789abcdef' for x in c['code_commit'])
            or type(c['U_expected_count']) is not int or c['U_expected_count']!=384
            or type(c['Q_expected_count']) is not int or c['Q_expected_count']!=864
            or type(c['batch_size']) is not int or c['batch_size']!=128):raise ValueError('Fixed development data and actual code commit required')
    family={'P0':'practical_cross_receiver','P1':'practical_cross_receiver'}
    if c['condition'] not in family or c['view_family']!=family[c['condition']]:raise ValueError('Condition/input family differs')
    u,q=Path(c['U_public_root']),Path(c['Q_public_root'])
    if u.name!='public' or u.parent.name!='discovery' or q.name!='public' or q.parent.name!='evaluation':raise ValueError('Distinct discovery U and future evaluation Q roles required')
    if any(Path(c['output_root']).resolve().is_relative_to(Path(c[k]).resolve()) for k in ('source_root','U_public_root','Q_public_root')):raise ValueError('Output cannot overwrite source/public inputs')
    if c['condition']=='P0':
        if c['dura_checkpoint'] is not None or c['dura_release'] is not None:raise ValueError('P0 is satellite received without G')
    elif any(not isinstance(c[k],str) or not c[k] for k in ('dura_checkpoint','dura_release')):raise ValueError('P1 requires frozen G')
    return c


def _source(c):
    from .star_relative_source import load_source
    s=load_source(c['source_root'],c['condition']);m=s['marker']
    if m['run_id']!=RUN or m['code_commit']!=c['code_commit'] or Path(m['config']['output_root']).resolve()!=Path(c['source_root']).resolve():raise ValueError('Frozen star source identity differs')
    if c['condition']=='P1' and any(c[k]!=m['config'][k] for k in ('dura_checkpoint','dura_release')):raise ValueError('Source/target G differs')
    return s


def _backend(s,c,device):
    from .star_relative_source import _backend as load
    b=load(s['marker']['config'],device,with_dura=c['condition']=='P1')
    if b['reader_identity']!=s['marker']['reader_identity']:raise ValueError('Loaded Phase1 F0 differs')
    if c['condition']=='P1' and b['dura_identity']!=s['marker']['dura_identity']:raise ValueError('Loaded frozen G differs')
    return b['reader'],b['dura']


def _metadata(c,role):
    import re
    from .data import PUBLIC_FIELDS,PUBLIC_FILES
    root=Path(c[role+'_public_root'])
    if {p.name for p in root.iterdir()}!=PUBLIC_FILES:raise ValueError('Exactly three public G10 files required')
    m=_read(root/'manifest.json');ids=np.load(root/'ids.npy',allow_pickle=False)
    expected=dict(schema='phase3_min_v1',data_status='VALIDATED_ONCE',capsule_id=c[role+'_capsule_id'],split_id=c[role+'_split_id'],
        count=c[role+'_expected_count'],registered_classes=['14-10','14-7','20-15','20-19','6-15','8-20'],
        processing_route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000,single_observation=True,
        iq_shape=[c[role+'_expected_count'],2,256])
    if (set(m)!=PUBLIC_FIELDS or any(m.get(k)!=v for k,v in expected.items())
            or ids.shape!=(c[role+'_expected_count'],) or ids.dtype.kind!='U' or len(set(ids.tolist()))!=len(ids)
            or any(not re.fullmatch(r'p3dev-star-g10-[0-9a-f]{24}',str(x)) for x in ids)):
        raise ValueError('Matching G10 capsule/split/count/opaque IDs required; old clean/three-scene data excluded')
    return m,ids


def _open(c,role):
    from .data import load_public
    m,ids=_metadata(c,role);actual,opened,iq=load_public(c[role+'_public_root'])
    if actual!=m or not np.array_equal(ids,opened):
        if hasattr(iq,'_mmap'):iq._mmap.close()
        raise ValueError('Public metadata/loaded rows differ')
    return m,ids,iq


def _encode(reader,x):
    branches,logits=extract_branches(reader,x)
    b={k:np.asarray(v.detach().cpu().tolist(),np.float32) for k,v in branches.items()}
    l=np.asarray(logits.detach().cpu().tolist(),np.float32)
    z=b['fused160'];z=(z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)).astype(np.float32)
    if set(b)!=set(DIMENSIONS) or l.shape!=(len(x),6) or z.shape!=(len(x),160) or not np.isfinite(l).all():raise ValueError('Complete single F0 outputs required')
    return b,l,z


def _scores(branches,logits,source):return core.all_scores(branches,logits,source['geometry'],source['metric'])


def _shift(reader,branches,logits,source,biases,applied,device):
    if not applied:return branches,logits
    import torch
    shifted=reg.apply_bias(branches,source['geometry'],biases)
    with torch.no_grad():out=reader.classify_features(_tensor(shifted['fused160'],device))
    return shifted,np.asarray(out.detach().cpu().tolist(),np.float32)


def _smoke(source,reader,g,c,device):
    from .star_channel import observe
    config=source['marker']['config'];mix=_read(config['mixture_config_path']);time=np.arange(256,dtype=np.float32)
    raw=[np.stack((np.cos(w*time),np.sin(w*time))) for w in (.11,.19)]
    values=[observe(x,physical_id='synthetic-star-relative-'+str(i),receiver='smoke',day='synthetic',
        episode=config['episode'],mixture=mix)[0] for i,x in enumerate(raw)]
    x=_current(np.asarray(values,np.float32),0,2,device,g);b,l,z=_encode(reader,x)
    if _scores(b,l,source).shape!=(2,7) or reg.embedding(b,source['geometry']).shape!=(2,688):raise ValueError('Single F0 star checkpoint smoke failed')
    print('STAR_RELATIVE_CHECKPOINT_SMOKE_PASS',c['condition'],2,flush=True)


def _current(iq,start,stop,device,g):
    import torch
    x=_tensor(iq[start:stop],device)
    if g is not None:
        with torch.no_grad():x=g[1](x)[0].detach().clone()
    return x


def _frozen(models,before,source,arrays):
    for m,snapshot in zip(models,before):frozen._check_model(m,snapshot)
    after=frozen._array_snapshot(source)
    if set(after)!=set(arrays) or any(not np.array_equal(arrays[k],after[k]) for k in arrays):raise ValueError('Frozen star source aggregate changed')


def _save_npz(path,values):
    with Path(path).open('xb') as stream:np.savez(stream,**values)


def predict(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable four-state prediction output required')
    source=_source(c)
    # Public identities may be checked before model work; neither input IQ is open.
    _,Uids=_metadata(c,'U');_,Qids=_metadata(c,'Q')
    if set(Uids.tolist())&set(Qids.tolist()):raise ValueError('U/Q physical roles overlap')
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reader,g=_backend(source,c,device);models=[reader]+([] if g is None else [g[0]])
    before=[frozen._snapshot_model(m) for m in models];arrays=frozen._array_snapshot(source)
    _smoke(source,reader,g,c,device);_frozen(models,before,source,arrays)
    out.mkdir(parents=True,exist_ok=False);adapt=out/'adaptation';adapt.mkdir()
    _,ids,iq=_open(c,'U')
    try:
        if not np.array_equal(ids,Uids):raise ValueError('U metadata/loaded IDs differ')
        values={k:[] for k in DIMENSIONS};rawlogits=[];identitylogits=[];identityunits=[]
        for start in range(0,len(ids),c['batch_size']):
            stop=min(start+c['batch_size'],len(ids));b,l0,z0=_encode(reader,_current(iq,start,stop,device,g))
            for k in DIMENSIONS:values[k].append(b[k])
            rawlogits.append(l0);identitylogits.append(l0);identityunits.append(z0)
        branches={k:np.concatenate(v) for k,v in values.items()};frozen_logits=np.concatenate(rawlogits);l0=np.concatenate(identitylogits);z0=np.concatenate(identityunits)
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()
    biases,bd=core.fit_bias_single(branches,l0,source['geometry'])
    b1,l1=_shift(reader,branches,frozen_logits,source,biases,bd['applied'],device)
    uscores=[_scores(branches,frozen_logits,source),_scores(b1,l1,source)]
    stats=source['statistics'];threshold=float(source['thresholds'][6]);sigma=float(stats['sigma'][6])
    anchor_q=(stats['anchor_scores'][:,6]-threshold)/sigma
    discoveries=[]
    for name,b,known in zip(('DA0','DA1'),(branches,b1),uscores):
        discoveries.append(reg.discover(reg.embedding(b,source['geometry']),z0,known[:,6],threshold,ids,
            float(stats['eps']),float(stats['radius']),stats['anchor_units'],anchor_q,c['condition']+'-'+name))
    state={**{'bias__'+k:v for k,v in biases.items()}}
    diag={'ids':ids.copy()}
    for name,d in zip(('DA0','DA1'),discoveries):
        state.update({'prototypes_'+name:d['prototypes'],'penalty_'+name:np.asarray(d['penalty'],np.float64),'anonymous_ids_'+name:d['anonymous_ids']})
        diag['assignments_'+name]=d['assignments'];diag['candidate_'+name]=uscores[0 if name=='DA0' else 1][:,6]<threshold
    _save_npz(adapt/'state.npz',state);_save_npz(adapt/'discovery_diagnostics.npz',diag)
    frozen_state=frozen._array_snapshot(state)
    _write(adapt/'complete.json',dict(status='STAR_RELATIVE_TARGET_REGISTRY_FROZEN',run_id=RUN,code_commit=c['code_commit'],condition=c['condition'],
        U_count=len(ids),bias_applied=bd['applied'],bias_diagnostics=bd,prototype_counts={n:len(d['prototypes']) for n,d in zip(('DA0','DA1'),discoveries)},
        discovery_diagnostics={n:d['diagnostics'] for n,d in zip(('DA0','DA1'),discoveries)},truth_read=False,Q_reads=0,query_fit=False))
    _frozen(models,before,source,arrays)
    # No row-wise U feature bank is used once the aggregate checkpoint is frozen.
    del branches,b1,frozen_logits,l1,l0,z0,values,rawlogits,identitylogits,identityunits,uscores
    _,idsQ,iq=_open(c,'Q')
    try:
        if not np.array_equal(idsQ,Qids):raise ValueError('Q metadata/loaded IDs differ')
        n=len(idsQ);knownness=np.empty((2,n,7),np.float64);closed=np.empty(n,np.int64)
        predictions=np.empty((4,n),np.int64)
        margins=[np.empty((n,6+len(d['prototypes'])),np.float64) for d in discoveries]
        for start in range(0,n,c['batch_size']):
            stop=min(start+c['batch_size'],n);b,l0,z0=_encode(reader,_current(iq,start,stop,device,g))
            shifted,shifted_logits=_shift(reader,b,l0,source,biases,bd['applied'],device)
            for index,(features,logits) in enumerate(((b,l0),(shifted,shifted_logits))):
                knownness[index,start:stop]=_scores(features,logits,source)
                predictions[index,start:stop]=np.where(knownness[index,start:stop,6]>=threshold,l0.argmax(1),-1)
                d=discoveries[index];m=reg.joint_margins(l0,z0,knownness[index,start:stop,6],threshold,sigma,
                    d['prototypes'],float(stats['radius']),d['penalty'])
                margins[index][start:stop]=m;predictions[2+index,start:stop]=np.where(m.max(1)>=0,m.argmax(1),-1)
            closed[start:stop]=l0.argmax(1)
            if start==0 or stop==n:print('STAR_RELATIVE_Q_PROGRESS',c['condition'],stop,n,flush=True)
        _frozen(models,before,source,arrays)
        after=frozen._array_snapshot(state)
        if set(after)!=set(frozen_state) or any(not np.array_equal(after[k],frozen_state[k]) for k in after):raise ValueError('U bias/prototypes changed on Q')
        diagnostic=np.where(knownness>=source['thresholds'][None,None,:],closed[None,:,None],-1).astype(np.int64)
        _save_npz(out/'predictions.npz',dict(ids=idsQ,U_ids=Uids,state_names=np.asarray(STATES),prediction=predictions,
            closed_prediction=closed,baseline_closed_prediction=closed.copy(),knownness=knownness,
            diagnostic_reg0_prediction=diagnostic,thresholds=source['thresholds'],joint_margins_DA0=margins[0],joint_margins_DA1=margins[1]))
        done=dict(status=STATUS,schema=SCHEMA,config=c,run_id=RUN,code_commit=c['code_commit'],condition=c['condition'],view_family=c['view_family'],
            count=n,U_count=len(Uids),states=list(STATES),state_names=list(STATES),method_names=list(METHODS),primary_state='DA1_REG1',primary_method=core.PRIMARY,
            bias_applied=bd['applied'],prototype_counts={k:len(d['prototypes']) for k,d in zip(('DA0','DA1'),discoveries)},
            source_frozen_before_U=True,registration_frozen_before_Q=True,frozen_parameters_buffers_unchanged=True,
            frozen_source_aggregates_unchanged=True,frozen_target_aggregates_unchanged=True,identity_F0_unchanged=True,
            query_fit=False,truth_read=False,target_labels_read=0,formal_target_reads=0,source_IQ_reads=0,
            new_encoder_updates=0,new_identity_head_updates=0,new_readout_updates=0,new_threshold_calibrations=0,
            real_checkpoint_synthetic_smoke_passed=True,all_new_IQ_satellite=True,encoder_models=1,auxiliary_E51_read=False,anonymous_registration_attempted=True,
            anonymous_registration_performed=any(len(d['prototypes']) for d in discoveries),final_phase3_complete=False)
        _write(out/'complete.json',done);return done
    finally:
        if hasattr(iq,'_mmap'):iq._mmap.close()


def _predictions(root):
    root=Path(root)
    try:done=_read(root/'complete.json');ad=_read(root/'adaptation/complete.json')
    except (OSError,ValueError) as exc:raise ValueError('Truth closed: complete four-state Q predictions and frozen U checkpoint required') from exc
    c=validate_config(done.get('config'))
    flags=('source_frozen_before_U','registration_frozen_before_Q','frozen_parameters_buffers_unchanged','frozen_source_aggregates_unchanged','frozen_target_aggregates_unchanged','identity_F0_unchanged')
    zero=('target_labels_read','formal_target_reads','source_IQ_reads','new_encoder_updates','new_identity_head_updates','new_readout_updates','new_threshold_calibrations')
    if (done.get('status')!=STATUS or done.get('schema')!=SCHEMA or done.get('encoder_models')!=1 or done.get('all_new_IQ_satellite') is not True or done.get('auxiliary_E51_read') is not False or done.get('states')!=list(STATES) or done.get('method_names')!=list(METHODS)
            or done.get('query_fit') is not False or done.get('truth_read') is not False or any(done.get(k) is not True for k in flags)
            or any(done.get(k)!=0 for k in zero) or done.get('count')!=c['Q_expected_count'] or done.get('U_count')!=c['U_expected_count']
            or ad.get('status')!='STAR_RELATIVE_TARGET_REGISTRY_FROZEN' or ad.get('truth_read') is not False or ad.get('Q_reads')!=0
            or ad.get('prototype_counts')!=done.get('prototype_counts')):raise ValueError('Truth closed: state completion/freeze/role boundary differs')
    with np.load(root/'predictions.npz',allow_pickle=False) as f:
        if set(f.files)!=PRED_FIELDS:raise ValueError('Truth closed: Q prediction fields differ')
        p={k:f[k].copy() for k in f.files}
    with np.load(root/'adaptation/discovery_diagnostics.npz',allow_pickle=False) as f:
        if set(f.files)!=DIAG_FIELDS:raise ValueError('Truth closed: U diagnostic fields differ')
        diag={k:f[k].copy() for k in f.files}
    with np.load(root/'adaptation/state.npz',allow_pickle=False) as f:state={k:f[k].copy() for k in f.files}
    n,nu=done['count'],done['U_count']
    for key,size in (('ids',n),('U_ids',nu)):
        if p[key].shape!=(size,) or p[key].dtype.kind!='U' or len(set(p[key].tolist()))!=size:raise ValueError('Truth closed: unique aligned public IDs required')
    if set(p['ids'].tolist())&set(p['U_ids'].tolist()) or not np.array_equal(diag['ids'],p['U_ids']):raise ValueError('Truth closed: U/Q IDs overlap or diagnostic identity differs')
    if (p['state_names'].tolist()!=list(STATES) or p['prediction'].shape!=(4,n) or p['prediction'].dtype.kind not in 'iu'
            or p['closed_prediction'].shape!=(n,) or p['baseline_closed_prediction'].shape!=(n,)
            or np.any(p['closed_prediction']<0) or np.any(p['closed_prediction']>=6)
            or not np.array_equal(p['closed_prediction'],p['baseline_closed_prediction'])
            or p['knownness'].shape!=(2,n,7) or not np.isfinite(p['knownness']).all()
            or p['thresholds'].shape!=(7,) or not np.isfinite(p['thresholds']).all()
            or not np.array_equal(p['diagnostic_reg0_prediction'],np.where(p['knownness']>=p['thresholds'][None,None,:],p['closed_prediction'][None,:,None],-1))
            or not np.array_equal(p['prediction'][:2],p['diagnostic_reg0_prediction'][:,:,6])):raise ValueError('Truth closed: complete original identity/knownness decisions required')
    expected={'bias__'+k for k in DIMENSIONS}|{k+'_'+name for name in ('DA0','DA1') for k in ('prototypes','penalty','anonymous_ids')}
    if set(state)!=expected:raise ValueError('Truth closed: target aggregate schema differs')
    for j,name in enumerate(('DA0','DA1')):
        k=done['prototype_counts'][name];m=p['joint_margins_'+name];assign=diag['assignments_'+name]
        if (type(k) is not int or k<0 or m.shape!=(n,6+k) or not np.isfinite(m).all()
                or state['prototypes_'+name].shape!=(k,160) or not np.isfinite(state['prototypes_'+name]).all()
                or state['anonymous_ids_'+name].shape!=(k,) or state['anonymous_ids_'+name].dtype.kind!='U'
                or assign.shape!=(nu,) or assign.dtype.kind not in 'iu' or np.any(assign<-1) or np.any(assign>=k)
                or diag['candidate_'+name].shape!=(nu,) or diag['candidate_'+name].dtype.kind!='b'
                or not np.array_equal(p['prediction'][2+j],np.where(m.max(1)>=0,m.argmax(1),-1))):raise ValueError('Truth closed: anonymous count/assignment/joint decisions differ')
        if any(np.sum(assign==i)<16 for i in range(k)) or np.any((assign>=0)&~diag['candidate_'+name]):raise ValueError('Truth closed: each registered cluster needs16 independent candidate IDs')
        old=(p['prediction'][2+j]>=0)&(p['prediction'][2+j]<6)
        if not np.array_equal(p['prediction'][2+j,old],p['closed_prediction'][old]):raise ValueError('Truth closed: old identity head was changed')
    return done,p,diag,state


def main(argv=None):
    print(json.dumps(dict(event='STAR_RELATIVE_TARGET_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args(argv);r=predict(_read(a.config),a.device);print(r['status'],flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
