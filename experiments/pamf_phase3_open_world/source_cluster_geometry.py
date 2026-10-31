"""Frozen source-L prototype receiver leaveout diagnostic, not unseen-RX evidence.

All twenty label-blind discovery/future prediction files precede label-side
alignment and evaluation. PCD retains its historical target-U576 ancestry.
"""
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
from . import pcd_cluster_registry_run as reg
from .star_hier_register_core import PARAMETERS

RUN_ID='20261031-phase3-source-cluster-geometry-s2026092701-r01'
SCHEMA='pamf_phase3_source_cluster_geometry_v1'
STATUS='SOURCE_CLUSTER_GEOMETRY_COMPLETE'
STATE_SCHEMA='pamf_phase3_source_cluster_geometry_support_v1'
REFERENCE_CODE=reg.REFERENCE_CODE
FIELDS={'run_id','code_commit','reference_config','output_root','batch_size'}
REPRESENTATIONS=('F0_unit','r03_unit','PCD_unit','PCD_white')
RECEIVERS=(1,3,4,6,8)
DAYS=(1,2,3)
SOURCE_COUNT=6300
CELL_COUNT=70
FLOOR=1e-6
_read,_write=reg._read,reg._write


def validate_config(c):
    if (not isinstance(c,dict) or set(c)!=FIELDS or c.get('run_id')!=RUN_ID
            or re.fullmatch('[0-9a-f]{40}',str(c.get('code_commit'))) is None
            or type(c.get('batch_size')) is not int or c['batch_size']!=128
            or not isinstance(c.get('output_root'),str) or not c['output_root'].strip()):
        raise ValueError('Exact five-field source-only geometry configuration required')
    r=reg.fresh.validate_config(c['reference_config'])
    if r['code_commit']!=REFERENCE_CODE:raise ValueError('Fixed fresh f4b5 reference required')
    roots=[r['output_root'],r['train_config']['output_root'],r['train_config']['U_public_root'],r['source_control_config']['output_root'],
        r['source_control_config']['reference_config']['train_config']['IQ_root']]+[v['public_root'] for v in r['Q_conditions'].values()]
    out=Path(c['output_root']).resolve()
    if any(out.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(out) for p in roots):
        raise ValueError('Immutable diagnostic output overlaps an existing input')
    return c


def _matrix(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=160 or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('Finite floating [N,160] coordinates required')
    return x.astype(np.float64,copy=False)


def _labels(y,n):return reg.core.base._closed(y,n)


def fit_support(x,y):
    """Only the other four source RXs supply labelled source support."""
    x=_matrix(x);y=_labels(y,len(x))
    if set(y.tolist())!=set(range(6)):raise ValueError('All six source classes required in support')
    centers=np.stack([x[y==c].mean(0) for c in range(6)])
    radius=max(float(np.quantile(np.linalg.norm(x-centers[y],axis=1),.95)),FLOOR)
    return dict(schema=np.asarray(STATE_SCHEMA),source_centers=centers,source_radius=np.asarray(radius,np.float64))


def _validate_state(state):
    if not isinstance(state,dict) or set(state)!={'schema','source_centers','source_radius','centers','radii'} or np.asarray(state['schema']).shape!=() or state['schema'].item()!=STATE_SCHEMA:
        raise ValueError('Frozen source and cluster aggregate state required')
    centers=_matrix(state['centers']);reg.core.base._float(state['source_centers'],(6,160),'six source means')
    radius=reg.core.base._float(state['source_radius'],(),'source support radius');radii=reg.core.base._float(state['radii'],(len(centers),),'cluster radii')
    if radius<FLOOR or np.any((radii<FLOOR)|(radii>radius)):raise ValueError('Positive source-capped cluster radii required')


def _cluster(x):
    from sklearn.cluster import HDBSCAN
    if len(x)<PARAMETERS['min_cluster_size']:return np.full(len(x),-1,np.int64)
    return HDBSCAN(**PARAMETERS).fit_predict(x)


def discover(x,ids,support):
    """No early/future labels or true class count enter discovery."""
    x=_matrix(x);ids=reg.core.base._ids(ids,len(x));order=np.argsort(ids,kind='stable');labels=_cluster(x[order])
    groups=[order[np.flatnonzero(labels==k)] for k in set(labels.tolist()) if k>=0]
    groups.sort(key=lambda rows:str(ids[rows[0]]));assignment=np.full(len(x),-1,np.int64);centers=[];radii=[];clusters=[]
    for j,rows in enumerate(groups):
        assignment[rows]=j;center=x[rows].mean(0);radius=max(min(float(support['source_radius']),float(np.quantile(np.linalg.norm(x[rows]-center,axis=1),.95))),FLOOR)
        centers.append(center);radii.append(radius);clusters.append(dict(cluster=j,size=len(rows),minimum_opaque_id=str(ids[rows[0]]),radius=radius))
    state={k:v.copy() for k,v in support.items()};state.update(centers=np.asarray(centers,np.float64).reshape(-1,160),radii=np.asarray(radii,np.float64));_validate_state(state)
    import sklearn
    return state,assignment,dict(algorithm='sklearn.cluster.HDBSCAN',parameters=PARAMETERS.copy(),sklearn_version=sklearn.__version__,cluster_count=len(groups),
        noise_count=int((assignment<0).sum()),clusters=clusters,discovery_labels_used=False,future_inputs_used=False,true_K_used=False)


def predict(x,state):
    """Row-local prototype decision; no fallback, fitting or query statistics."""
    _validate_state(state);x=_matrix(x);n=len(x);prediction=np.full(n,-1,np.int64);ratio=np.full(n,-1.,np.float64)
    if len(state['centers']) and n:
        distance=np.column_stack([np.linalg.norm(x-center,axis=1)/r for center,r in zip(state['centers'],state['radii'])]);nearest=distance.argmin(1);ratio=distance[np.arange(n),nearest]
        prediction=np.where(ratio<=1,nearest,-1).astype(np.int64)
    source=np.column_stack([np.linalg.norm(x-center,axis=1) for center in state['source_centers']]).argmin(1).astype(np.int64)
    return dict(cluster_prediction=prediction,normalized_distance=ratio,source_prediction=source)


def _metrics(y,prediction,baseline,roster=tuple(range(6))):
    n=len(y);per={}
    for c in roster:
        mask=y==c;count=int(mask.sum());accepted=int((mask&(prediction!=-1)).sum());correct=int((mask&(prediction==y)).sum());base=int((mask&(baseline==y)).sum()) if baseline is not None else None
        per[str(c)]=dict(count=count,accepted_count=accepted,correct_accept_count=correct,rejected_count=count-accepted,baseline_correct_count=base,
            acceptance=accepted/count if count else None,correct_acceptance=correct/count if count else None,F0_closed_accuracy=base/count if count and base is not None else None)
    covered=bool(per) and all(v['count']>0 for v in per.values());acc=float(np.mean([v['correct_acceptance'] for v in per.values()])) if covered else None
    closed=float(np.mean([v['F0_closed_accuracy'] for v in per.values()])) if covered and baseline is not None else None
    return dict(count=n,accepted_count=int((prediction!=-1).sum()),rejected_count=int((prediction==-1).sum()),unmatched_accepted_count=int((prediction==-2).sum()),
        correct_accept_count=int((prediction==y).sum()),macro_correct_acceptance=acc,macro_acceptance=float(np.mean([v['acceptance'] for v in per.values()])) if covered else None,
        F0_closed_macro_accuracy=closed,F0_closed_minus_correct_acceptance=closed-acc if closed is not None else None,coverage_complete=covered,per_TX=per)


def evaluate_case(early_y,assignment,future_y,future_cluster,baseline,future_day,source_prediction):
    """One early-only strict-majority Hungarian mapping; future labels only score."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score
    early_y=_labels(early_y,len(assignment));future_y=_labels(future_y,len(future_cluster));baseline=_labels(baseline,len(future_y));source_prediction=_labels(source_prediction,len(future_y))
    assignment=np.asarray(assignment);future_cluster=np.asarray(future_cluster);future_day=np.asarray(future_day)
    if (assignment.dtype!=np.int64 or assignment.shape!=early_y.shape or np.any(assignment<-1) or future_cluster.dtype!=np.int64 or future_cluster.shape!=future_y.shape
            or np.any(future_cluster<-1) or future_day.shape!=future_y.shape or not np.isin(future_day,(2,3)).all()):raise ValueError('Aligned fixed discovery and future predictions required')
    clusters=sorted(set(assignment[assignment>=0].tolist()));counts=np.asarray([np.bincount(early_y[assignment==j],minlength=6) for j in clusters],np.int64).reshape(-1,6)
    eligible=np.where(counts*2>counts.sum(1,keepdims=True),counts,0);mapping={}
    if len(clusters):
        rows,cols=linear_sum_assignment(-eligible)
        mapping={str(clusters[i]):int(c) for i,c in zip(rows,cols) if eligible[i,c]>0}
    def mapped(raw):return np.asarray([-1 if j<0 else mapping.get(str(int(j)),-2) for j in raw],np.int64)
    early=_metrics(early_y,mapped(assignment),None);early.update(cluster_count=len(clusters),matched_class_count=len(mapping),mapping=mapping,contingency=counts.tolist(),
        matched_all_six=bool(len(mapping)==6 and set(mapping.values())==set(range(6))),noise_count=int((assignment<0).sum()),ARI=float(adjusted_rand_score(early_y,assignment)) if len(early_y) else None,
        purity=float(counts.max(1).sum()/counts.sum()) if counts.sum() else None,cluster_sizes=counts.sum(1).tolist(),strict_majority_required=True,future_labels_used_for_matching=False)
    mapped_future=mapped(future_cluster);future=_metrics(future_y,mapped_future,baseline)
    future['by_day']={str(d):_metrics(future_y[future_day==d],mapped_future[future_day==d],baseline[future_day==d]) for d in (2,3)}
    source=_metrics(future_y,source_prediction,baseline)
    passed=bool(future['coverage_complete'] and future['macro_correct_acceptance']>=.8-1e-12 and future['macro_acceptance']>=.9-1e-12
        and future['F0_closed_minus_correct_acceptance']<=.02+1e-12 and early['matched_all_six'])
    return dict(early=early,future=future,source_centers_nearest_class=source,source_screen_passed=passed,A_N=None,R_U=None,H_OW=None)


def screen(cases):
    return set(cases)=={str(r) for r in RECEIVERS} and all(v['source_screen_passed'] for v in cases.values())


def _geometry(x,y):
    centers=np.stack([x[y==c].mean(0) for c in range(6)]);distance=np.linalg.norm(x-centers[y],axis=1)
    between=np.asarray([np.linalg.norm(centers[a]-centers[b]) for a in range(6) for b in range(a+1,6)])
    within=float(distance.mean());inter=float(between.mean())
    return dict(mean_distance_to_own_class_center=within,rms_distance_to_own_class_center=float(np.sqrt(np.square(distance).mean())),
        mean_distance_between_six_class_centers=inter,between_over_within=inter/within if within>0 else None,
        coordinate_mean_norm=float(np.linalg.norm(x,axis=1).mean()),definition='Euclidean distances in the declared coordinates; descriptive, not causal; white coordinates are not renormalized')


def _metadata(role):
    n=len(role['ids']);reg.core.base._ids(role['ids'],n);m=role['metadata']
    if n!=SOURCE_COUNT or set(m)!={'y','rx_i','day_i','domain'}:raise ValueError('Only full original source L metadata is allowed')
    y=_labels(m['y'],n)
    for name,roster in (('rx_i',RECEIVERS),('day_i',DAYS)):
        a=np.asarray(m[name])
        if a.shape!=(n,) or a.dtype.kind not in 'iu' or not np.isin(a,roster).all():raise ValueError('Original source RX/day metadata required')
    domain=np.searchsorted(RECEIVERS,m['rx_i'])*3+m['day_i']-1
    if not np.array_equal(domain,m['domain']):raise ValueError('Source RX/day domain mapping differs')
    if any(int(((y==c)&(m['rx_i']==r)&(m['day_i']==d)).sum())!=CELL_COUNT for c in range(6) for r in RECEIVERS for d in DAYS):raise ValueError('Original complete 90-cell source grid required')
    return {k:np.asarray(v).copy() for k,v in m.items()}


def _encode(c,b,model,pm,state,device):
    role=reg.fresh._iq_role(c['reference_config'],'L',pm['source_control_prediction_identity'])
    try:
        meta=_metadata(role);ids=role['ids'].copy();blocks=[];values=role['restored']
        if values.shape!=(SOURCE_COUNT,2,256) or values.dtype!=np.float32:raise ValueError('Only original cached source P1 FP32 IQ allowed')
        for start in range(0,SOURCE_COUNT,c['batch_size']):
            batch=np.asarray(values[start:start+c['batch_size']])
            if not np.isfinite(batch).all():raise ValueError('Finite source P1 input required')
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                iq=torch.tensor(batch.tolist(),device=device,dtype=torch.float32)
                raw=b['reader'].features(iq);logits=b['reader'].classify_features(raw);r03=b['model'](iq);pcd=model(iq)
            arrays=[np.asarray(v.detach().cpu().tolist(),np.float32) for v in (raw,logits,r03,pcd)]
            if any(a.dtype!=np.float32 or a.shape!=(len(batch),6 if j==1 else 160) or not np.isfinite(a).all() for j,a in enumerate(arrays)):raise ValueError('Finite same-pass F0 logits/raw and r03/PCD features required')
            blocks.append(arrays)
        raw,logits,r03,pcd=[np.concatenate([row[j] for row in blocks]) for j in range(4)]
        density=reg.core.density
        features=dict(F0_unit=density.unit(raw).astype(np.float32),r03_unit=density.unit(r03).astype(np.float32),PCD_unit=density.unit(pcd).astype(np.float32))
        features['PCD_white']=reg.core.base._white(state,pcd)
        return ids,meta,logits.argmax(1).astype(np.int64),features,role['manifest']
    finally:reg.fresh.source_eval._close_role(role)


def run(c,device='cuda:0'):
    validate_config(c);out=Path(c['output_root'])
    if out.exists():raise FileExistsError('New immutable source geometry output required')
    pm,state,_,_=reg._reference(c);b=reg._backend(c,device);model,identity=reg._load_model(c,device)
    if identity!=pm['model_identities']['TARGET_PCD'] or b['reader_identity']!=pm['parent_F0_identity'] or b['dura_identity']!=pm['loaded_dura_identity']:raise ValueError('Frozen F0/r03/PCD/G identity differs')
    models=(b['reader'],b['model'],b['dura'][0],model)
    if any(any(p.requires_grad or p.dtype!=torch.float32 for p in m.parameters()) or any(child.training for child in m.modules()) for m in models):raise ValueError('All original checkpoint models must be frozen FP32/eval')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    snapshots=[(m,reg.fresh.best._snapshot_model(m)) for m in models];before=deepcopy(state)
    def check():
        for m,snapshot in snapshots:reg.fresh.best._check_model(m,snapshot)
        if not reg._same(state,before):raise ValueError('Original PCD density changed')
    smoke=reg._smoke(b,model,device);check();print('SOURCE_CLUSTER_GEOMETRY_SMOKE_PASS',flush=True)
    out.mkdir(parents=True);_write(out/'resolved_config.json',c);_write(out/'smoke.json',smoke)
    ids,meta,baseline,features,manifest=_encode(c,b,model,pm,state,device);check()
    reg.fresh.best._save(out/'source_features.npz',dict(ids=ids,rx_i=meta['rx_i'],day_i=meta['day_i'],F0_closed=baseline,**features));print('SOURCE_CLUSTER_GEOMETRY_ENCODED',SOURCE_COUNT,flush=True)
    cases=[]
    for name in REPRESENTATIONS:
        x=features[name]
        for receiver in RECEIVERS:
            support=meta['rx_i']!=receiver;early=(meta['rx_i']==receiver)&(meta['day_i']==1);future=(meta['rx_i']==receiver)&(meta['day_i']!=1)
            source=fit_support(x[support],meta['y'][support]);cluster,assignment,diag=discover(x[early],ids[early],source);prediction=predict(x[future],cluster)
            path=out/'cases'/name/('RX'+str(receiver));reg.fresh.best._save(path/'state.npz',cluster)
            reg.fresh.best._save(path/'predictions.npz',dict(support_ids=ids[support],early_ids=ids[early],early_assignment=assignment,future_ids=ids[future],future_baseline=baseline[future],future_day=meta['day_i'][future],**prediction))
            _write(path/'discovery.json',diag);cases.append((name,receiver,path,early,future,assignment,prediction));check();print('SOURCE_CLUSTER_GEOMETRY_CASE_FROZEN',name,receiver,len(cluster['centers']),flush=True)
    _write(out/'predictions_complete.json',dict(schema=SCHEMA,run_id=RUN_ID,case_count=len(cases),all_predictions_frozen=True,evaluation_started=False))
    results={name:dict(cases={},geometry=_geometry(_matrix(features[name]),meta['y']),A_N=None,R_U=None,H_OW=None) for name in REPRESENTATIONS}
    for name,receiver,path,early,future,assignment,prediction in cases:
        result=evaluate_case(meta['y'][early],assignment,meta['y'][future],prediction['cluster_prediction'],baseline[future],meta['day_i'][future],prediction['source_prediction'])
        _write(path/'metrics.json',result);results[name]['cases'][str(receiver)]=result
    for row in results.values():row['source_screen_eligible']=screen(row['cases'])
    summary=dict(schema=SCHEMA,run_id=RUN_ID,code_commit=c['code_commit'],representations=results,representation_order=list(REPRESENTATIONS),source_manifest=manifest,
        checkpoint_identities=dict(F0=pm['parent_F0_identity'],r03=b['model_identity'],PCD=identity,DURA=pm['dura_identity']),
        source_L_count=SOURCE_COUNT,source_RX_seen_in_parameter_training=True,all_six_TX_seen_in_parameter_training=True,PCD_target_U_ancestor_count=576,
        prototype_receiver_leaveout_only=True,unseen_RX_generalization=False,independent_validation=False,diagnostic_only=True,source_screen_is_future_research_only=True,
        loaded_parameter_counts={name:sum(p.numel() for p in m.parameters()) for name,m in zip(('F0','r03','PCD','DURA'),models)},
        total_loaded_parameters=sum(sum(p.numel() for p in m.parameters()) for m in models),future_deployment_cost=None,
        screen=dict(all_five_RX=True,future_macro_correct_acceptance_min=.8,future_macro_acceptance_min=.9,F0_closed_loss_max=.02,early_all_six_matched=True),
        A_N=None,R_U=None,H_OW=None,eligible_primary=False,next_step='DIAGNOSTIC_ONLY',full_goal_status='ACTIVE',final_phase3_complete=False)
    check();_write(out/'summary.json',summary)
    done=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=c['code_commit'],config=c,summary='summary.json',case_count=len(cases),cluster_fits=len(cases),source_support_fits=len(cases),
        source_L_reads=SOURCE_COUNT,real_F_forward_rows=SOURCE_COUNT,real_signal_forward_rows=2*SOURCE_COUNT,G_forward_rows=0,
        source_V_reads=0,source_NV_reads=0,source_OE_reads=0,original_U_s_reads=0,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,
        optimizer_updates=0,new_GMM_fits=0,new_threshold_fits=0,original_models_unchanged=True,source_states_unchanged=True,source_only_inputs=True,model_source_only=False,
        PCD_target_U_ancestor_count=576,all_predictions_frozen_before_evaluation=True,source_screen_eligible={k:v['source_screen_eligible'] for k,v in results.items()},
        synthetic_F_forward_rows=smoke['synthetic_F_forward_rows'],synthetic_signal_forward_rows=smoke['synthetic_signal_forward_rows'],G_synthetic_forward_rows=smoke['G_synthetic_forward_rows'],
        diagnostic_only=True,A_N=None,R_U=None,H_OW=None,final_phase3_complete=False,full_goal_status='ACTIVE')
    _write(out/'complete.json',done);print(STATUS,flush=True);return done


evaluate=run


def main():
    print(json.dumps(dict(event='SOURCE_CLUSTER_GEOMETRY_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args();run(_read(args.config),args.device)


if __name__=='__main__':main()
