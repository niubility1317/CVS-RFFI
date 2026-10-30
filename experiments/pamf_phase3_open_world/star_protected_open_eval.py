"""Frozen protected160 source-L aggregates, known-V risk and heldout prediction."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import time
import numpy as np
from . import star_protected_open_model as model_api
from . import star_class_risk as risk
from . import star_consensus_score as common
from .development_transfer import _snapshot_model, _check_model

RUN_ID='20261030-phase3-star-protected-open-s2026092701-r01'
ARMS=('N0','R0','R1');PRIMARY_ARM='R1';PRIMARY_METHOD='relation_max'
SCHEMA='pamf_phase3_star_protected_prediction_v1'
STATUS='STAR_PROTECTED_PREDICTIONS_COMPLETE'
CALIBRATION_STATUS='STAR_PROTECTED_THRESHOLDS_FROZEN'
FIELDS={'run_id','code_commit','arm','train_config','feature_root','cache_root','output_root','batch_size'}
COUNTS=dict(L=6300,V=27000,novel=3150)
RECEIVERS,SCENES=common.RECEIVERS,common.SCENES
AGGREGATE_FIELDS={'sums','counts','rx_values','day_values'}
BASE_FIELDS={'ids','default_logits','relation_logits','closed','scores'}
KNOWN_FIELDS=BASE_FIELDS|{'y','rx_i','day_i','scenario','baseline_closed'}
NOVEL_FIELDS=BASE_FIELDS|{'rx_i','day_i'}
FEATURE_FIELDS={'ids','rx_i','day_i','original_fused160','protected160'}
_read,_write=common._read,common._write


def _npz(path,data):
    with Path(path).open('xb') as stream:np.savez(stream,**data)


def _train_validate(c):
    from .star_protected_open_train import validate_config
    return validate_config(c)


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID or c.get('arm') not in ARMS
            or not isinstance(c.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',c['code_commit']) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or any(not isinstance(c[k],str) or not c[k].strip() for k in FIELDS-{'train_config','batch_size'})):
        raise ValueError('Exact eight-field source-only protected evaluation config required; no truth/U/Q/K/G')
    _train_validate(c['train_config']);t=c['train_config']
    if any(t.get(k)!=c[k] for k in ('run_id','code_commit','arm','feature_root')):
        raise ValueError('Matching immutable trained row and source feature handle required')
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out)
           for p in (c['feature_root'],c['cache_root'],t['output_root'])):
        raise ValueError('Evaluation output must be separate from source cache/features/checkpoint')
    return c


def _identity(identity,c):
    expected=dict(run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],checkpoint=str(Path(c['train_config']['output_root'])/'final.pt'),
        adapter_updates=2220,original_parent_updates=11100,native_updates=0,identifier_parameters=299170,
        open_world_parameters=66177,training_container_parameters=966899,native_identity_frozen=True,
        identity_head_replaced=False,input_path='cached_P1_only',OE_member_supervision=c['arm']!='N0',residual_enabled=c['arm']!='R0')
    if not isinstance(identity,dict) or any(identity.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen original-F0 protected checkpoint identity differs')
    native=c['train_config']['native_config'];p=identity.get('parent_F0_identity',{});old=identity.get('original_parent_identity',{})
    if (any(p.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items())
            or any(p.get(k)!=native.get(k) for k in ('source_output','source_contract','base_release'))
            or old.get('parent_epoch')!=50 or old.get('parent_updates')!=11100 or old.get('model_seed')!=2026092701
            or any(old.get(k)!=native.get(k) for k in ('source_output','source_contract','base_release'))):
        raise ValueError('Exact original E50 source ancestry required; no Phase3 encoder chain')
    if any(identity.get('dura_identity',{}).get(k)!=v for k,v in dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940).items()):
        raise ValueError('Frozen DURA v1 P1 cache ancestry required')
    return p


def _backend(c,device):
    from .star_protected_open_train import load_trained_model
    model,identity=load_trained_model(c['train_config'],Path(c['train_config']['output_root'])/'final.pt',device)
    parent=_identity(identity,c);m=_read(Path(c['train_config']['output_root'])/'complete.json')
    if (m.get('status')!='STAR_PROTECTED_SOURCE_FROZEN' or m.get('identity')!=identity
            or m.get('model_native_state_unchanged') is not True):
        raise ValueError('Complete protected training with unchanged original native state required')
    return dict(model=model,identity=identity,parent=parent,dura_identity=identity['dura_identity'])


def _load_features(root,role):
    if role!='L':raise ValueError('Evaluation consumes only L aggregate features, never OE')
    from .star_protected_open_cache import load_features
    return load_features(root,role)


def _load_role(root,role):
    if role not in ('V','novel'):raise ValueError('Only known V and frozen source novel IQ roles allowed')
    from .star_representation_cache import load_role
    return load_role(Path(root)/role,role)


def _close_role(role):
    from .star_representation_cache import close_role
    close_role(role)


def _domain(meta,n,known=False,scenario=False):
    fields={'rx_i','day_i','domain'}|({'y'} if known else set())|({'scenario'} if scenario else set())
    if (set(meta)!=fields or any(meta[k].shape!=(n,) or meta[k].dtype!=np.int64 for k in fields-{'scenario'})
            or not np.isin(meta['rx_i'],RECEIVERS).all() or not np.isin(meta['day_i'],(1,2,3)).all()
            or not np.array_equal(meta['domain'],np.searchsorted(RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)
            or (known and not np.isin(meta['y'],range(6)).all())
            or (scenario and (meta['scenario'].shape!=(n,) or meta['scenario'].dtype.kind!='U' or not np.isin(meta['scenario'],SCENES).all()))):
        raise ValueError('Exact public source domain/visible-role metadata required')


def _role(role,name,b):
    n=COUNTS[name];m=role['manifest']
    if (m.get('role')!=name or m.get('count')!=n or m.get('parent_F0_identity')!=b['parent'] or m.get('dura_identity')!=b['dura_identity']
            or m.get('source_only') is not True or m.get('all_new_IQ_satellite') is not True or m.get('restored_input_path')!='P1'
            or not common._ids(role['ids'],n) or role['restored'].shape!=(n,2,256) or role['restored'].dtype!=np.float32):
        raise ValueError('Only matching original-F0 source G10P1 cache is legal')
    _domain(role['metadata'],n,known=name=='V',scenario=name=='V')
    if name=='V' and (role['baseline_closed'].shape!=(n,2) or role['baseline_closed'].dtype!=np.int64
                     or not np.isin(role['baseline_closed'],range(6)).all()):
        raise ValueError('Independent original-F0 same-P1 cached baseline required')


def validate_aggregates(a):
    if (set(a)!=AGGREGATE_FIELDS or a['sums'].shape!=(6,5,3,160) or a['sums'].dtype!=np.float64 or not np.isfinite(a['sums']).all()
            or a['counts'].shape!=(6,5,3) or a['counts'].dtype!=np.int64 or np.any(a['counts']<=0)
            or a['rx_values'].dtype!=np.int64 or not np.array_equal(a['rx_values'],RECEIVERS)
            or a['day_values'].dtype!=np.int64 or not np.array_equal(a['day_values'],[1,2,3])):
        raise ValueError('Complete source-six x five-RX x three-day aggregate sums/counts required')
    return a


def aggregate_source(z,y,rx,day):
    n=len(z)
    if (z.shape!=(n,160) or z.dtype!=np.float32 or not np.isfinite(z).all()
            or any(x.shape!=(n,) or x.dtype!=np.int64 for x in (y,rx,day)) or not np.isin(y,range(6)).all()):
        raise ValueError('Aligned source L unit160 and visible six labels/domains required')
    a=dict(sums=np.zeros((6,5,3,160),np.float64),counts=np.zeros((6,5,3),np.int64),
           rx_values=np.asarray(RECEIVERS,np.int64),day_values=np.asarray([1,2,3],np.int64))
    for c in range(6):
        for r,rv in enumerate(RECEIVERS):
            for d,dv in enumerate((1,2,3)):
                mask=(y==c)&(rx==rv)&(day==dv);a['counts'][c,r,d]=int(mask.sum())
                a['sums'][c,r,d]=z[mask].sum(axis=0,dtype=np.float64)
    return validate_aggregates(a)


def query_prototypes(a,rx,day):
    valid=(rx[:,None,None]!=a['rx_values'][None,:,None])&(day[:,None,None]!=a['day_values'][None,None,:])
    counts=np.einsum('qrd,crd->qc',valid.astype(np.int64),a['counts'])
    if np.any(counts<=0):raise ValueError('Each query/class needs different-RX AND different-day source support')
    means=np.einsum('qrd,crdf->qcf',valid,a['sums'])/counts[:,:,None]
    norm=np.linalg.norm(means,axis=-1,keepdims=True)
    return np.divide(means,norm,out=np.zeros_like(means),where=norm>0).astype(np.float32)


def _tensor(x,device):
    import torch
    return torch.tensor(np.asarray(x).tolist(),dtype=torch.float32,device=device)


def _synthetic(device):
    from .star_open_feature_export import _synthetic as generate
    return generate(device)


def _probe(b,device):
    import torch
    model=b['model']
    try:
        with torch.no_grad():
            out=model_api.forward(model,_synthetic(device));n=len(out['logits'])
            prototype=torch.eye(160,device=device,dtype=torch.float32)[:6][None].expand(n,-1,-1)
            relation=model_api.relation(model,out['protected160'],prototype)
            same=model.id_backbone.encoder.classify_features(out['raw_original_fused160'])
            if not torch.equal(out['logits'],same) or not bool(torch.isfinite(relation).all()):
                raise ValueError('Actual checkpoint default identity or relation smoke failed')
    finally:model_api.clear(model)
    return dict(status='PASS',actual_checkpoint=True,synthetic_G10_P0_F_rows=2,synthetic_adapter_rows=2,
                real_source_reads=0,truth_reads=0,G_forward_rows=0,native_identity_unchanged=True)


def _encode_L(model,f,device):
    import torch
    rows=[]
    with torch.no_grad():
        for i in range(0,len(f['ids']),128):
            out=model_api.encode(model,_tensor(f['raw_original_fused160'][i:i+128],device),_tensor(f['contributions'][i:i+128],device))
            rows.append(np.asarray(out['protected160'].cpu().tolist(),np.float32))
    return np.concatenate(rows)


def _predict(model,values,rx,day,a,device,features=False):
    import torch
    logits=[];relations=[];original=[];protected=[]
    for i in range(0,len(values),128):
        try:
            with torch.no_grad():
                out=model_api.forward(model,_tensor(values[i:i+128],device))
                p=_tensor(query_prototypes(a,rx[i:i+128],day[i:i+128]),device)
                r=model_api.relation(model,out['protected160'],p)
                logits.append(np.asarray(out['logits'].cpu().tolist(),np.float32));relations.append(np.asarray(r.cpu().tolist(),np.float32))
                if features:
                    original.append(np.asarray(torch.nn.functional.normalize(out['raw_original_fused160'],dim=1,eps=1e-12).cpu().tolist(),np.float32))
                    protected.append(np.asarray(out['protected160'].cpu().tolist(),np.float32))
        finally:model_api.clear(model)
    return np.concatenate(logits),np.concatenate(relations),({
        'original_fused160':np.concatenate(original),'protected160':np.concatenate(protected)} if features else None)


def _rows(ids,z,r,tau):
    closed=z.argmax(1).astype(np.int64)
    return dict(ids=ids,default_logits=z,relation_logits=r,closed=closed,scores=r.max(1).astype(np.float64)-tau[closed])


def evaluate(c,device='cuda:0'):
    validate_config(c);output=Path(c['output_root'])
    if output.exists():raise FileExistsError('New immutable protected source evaluation output required')
    start=time.perf_counter();b=_backend(c,device);model=b['model'];snapshot=_snapshot_model(model)
    smoke=_probe(b,device);_check_model(model,snapshot)
    if smoke.get('status')!='PASS':raise ValueError('Actual-checkpoint no-query smoke must pass before source')
    output.mkdir(parents=True);_write(output/'smoke.json',smoke)
    f=_load_features(c['feature_root'],'L');m=f['manifest'];meta=f['metadata']
    if (set(f)!={'ids','metadata','raw_original_fused160','contributions','default_logits','manifest'}
            or m.get('status')!='STAR_PROTECTED_SOURCE_FEATURES_COMPLETE' or m.get('schema')!='pamf_phase3_protected_source_features_v1'
            or m.get('parent_F0_identity')!=b['parent'] or m.get('dura_identity')!=b['dura_identity']
            or m.get('counts',{}).get('L')!=COUNTS['L'] or not common._ids(f['ids'],COUNTS['L'])):
        raise ValueError('Matching source-only original-F0 L feature handle required')
    _domain(meta,COUNTS['L'],known=True)
    a=aggregate_source(_encode_L(model,f,device),meta['y'],meta['rx_i'],meta['day_i']);seen=set(f['ids'].tolist())
    _npz(output/'source_aggregates.npz',a);frozen={k:v.copy() for k,v in a.items()};_check_model(model,snapshot)
    def check():
        _check_model(model,snapshot)
        if any(not np.array_equal(v,frozen[k]) for k,v in a.items()):raise ValueError('Frozen source aggregates changed')
    role=_load_role(c['cache_root'],'V')
    try:
        _role(role,'V',b);meta=role['metadata']
        if seen&set(role['ids'].tolist()):raise ValueError('L/V/novel source physical IDs must be disjoint')
        seen.update(role['ids'].tolist());z,r,_=_predict(model,role['restored'],meta['rx_i'],meta['day_i'],a,device)
        closed=z.argmax(1).astype(np.int64);baseline=role['baseline_closed'][:,1].copy()
        if not np.array_equal(closed,baseline):raise ValueError('Frozen candidate identities differ from original same-P1 cached baseline')
        state,cal=risk.calibrate_known(role['ids'],meta['y'],closed,baseline,r.max(1).astype(np.float64),meta['rx_i'],meta['scenario'])
        tau=state['conditional_thresholds'].copy()
        cal.update(schema=SCHEMA,status=CALIBRATION_STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],
                   method=PRIMARY_METHOD,selected_calibration_method='predicted_class',aggregates_fit_role='L')
        (output/'calibration').mkdir();_write(output/'calibration/complete.json',cal)
        with (output/'thresholds.npy').open('xb') as stream:np.save(stream,tau,allow_pickle=False)
        _npz(output/'known_predictions.npz',dict(_rows(role['ids'],z,r,tau),y=meta['y'],rx_i=meta['rx_i'],day_i=meta['day_i'],scenario=meta['scenario'],baseline_closed=baseline))
    finally:_close_role(role)
    check();frozen_tau=tau.copy();role=_load_role(c['cache_root'],'novel')
    try:
        _role(role,'novel',b);meta=role['metadata']
        if seen&set(role['ids'].tolist()):raise ValueError('Source novel IDs must be disjoint from L/V')
        z,r,features=_predict(model,role['restored'],meta['rx_i'],meta['day_i'],a,device,features=True)
        public=dict(ids=role['ids'],rx_i=meta['rx_i'],day_i=meta['day_i'])
        _npz(output/'novel_predictions.npz',dict(_rows(role['ids'],z,r,tau),rx_i=meta['rx_i'],day_i=meta['day_i']))
        _npz(output/'novel_features.npz',dict(public,**features))
    finally:_close_role(role)
    check()
    if not np.array_equal(tau,frozen_tau):raise ValueError('Thresholds changed after calibration')
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,
        model_identity=b['identity'],parent_F0_identity=b['parent'],dura_identity=b['dura_identity'],method=PRIMARY_METHOD,primary_arm=PRIMARY_ARM,path='P1',
        known_count=COUNTS['V'],novel_count=COUNTS['novel'],source_L_count=COUNTS['L'],source_only=True,
        source_truth_reads=0,target_reads=0,source_OE_reads=0,original_U_s_reads=0,source_V_neural_fit=False,source_V_threshold_fit=True,
        native_identity_frozen=True,native_updates=0,eval_neural_updates=0,identity_predictions_unchanged=True,
        aggregates_frozen_before_V=True,thresholds_frozen_before_novel=True,frozen_models_verified=True,source_aggregates_unchanged=True,
        source_roles_disjoint=True,source_bank_deployed=False,per_record_source_rows_deployed=0,
        identifier_parameters=299170,open_world_parameters=66177,training_container_parameters=966899,unused_native_parameters=667729,
        real_F_forward_rows=COUNTS['V']+COUNTS['novel'],synthetic_F_forward_rows=2,adapter_L_feature_rows=COUNTS['L'],
        G_forward_rows=0,G_updates=0,received_cache_values_read=0,clean_IQ_reads=0,novel_features_same_forward=True,
        geometry_features_deployed=False,source_aggregate_bytes=sum(x.nbytes for x in a.values()),threshold_bytes=tau.nbytes,
        formal_target_accessed=False,target_DA=False,target_REG=False,A_N=None,H_OW=None,final_phase3_complete=False,seconds=time.perf_counter()-start)
    _write(output/'complete.json',done);print(STATUS,c['arm'],flush=True);return done


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    evaluate(_read(a.config),a.device)


if __name__=='__main__':main()
