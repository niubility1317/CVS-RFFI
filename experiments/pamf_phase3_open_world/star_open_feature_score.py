"""Truth-last frozen source-novel geometry diagnostics; no models or target data.

Nearest neighbors are selected for every representation before private truth
opens. Source RX/day restrict candidates; TX labels only evaluate the fixed
choices and describe geometry. None of these statistics is future A_N.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from . import star_consensus_score as common


RUN_ID='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01'
PARENT_RUN='20261030-phase3-star-open-episodic-s2026092701-r02'
PARENT_COMMIT='1029bc4f4ab86022c75ae0f3332b5dfafc1fe909'
EXPORT_SCHEMA='pamf_phase3_open_feature_export_v1'
EXPORT_STATUS='STAR_OPEN_FEATURES_COMPLETE'
SCHEMA='pamf_phase3_open_feature_score_v1'
STATUS='STAR_OPEN_FEATURE_SCORES_COMPLETE'
SOURCE_COUNT=3150
BLOCK=128
CONDITIONS=('all_other','cross_RX','cross_day','cross_RX_and_day')
REPRESENTATIONS={'F0.fused160':dict(file='F0.npz',key='fused160',dimension=160)}|{
    arm+'.'+key:dict(file=arm+'.npz',key=key,dimension=dimension)
    for arm in ('S0','S1','M') for key,dimension in (('fused160',160),('open64',64))}
_read,_write=common._read,common._write


def _prediction(root):
    root=Path(root);done=_read(root/'complete.json')
    required=dict(schema=EXPORT_SCHEMA,status=EXPORT_STATUS,run_id=RUN_ID,source_count=SOURCE_COUNT,
        optimizer_updates=0,target_reads=0,private_truth_reads=0,model_states_unchanged=True,source_role='novel',
        model_updates=0,threshold_updates=0,source_L_reads=0,source_V_reads=0,source_OE_reads=0,original_U_s_reads=0,
        G_forward_rows=0,input_path='cached_P1_only',received_cache_values_read=0)
    if (any(done.get(k)!=v for k,v in required.items())
            or not isinstance(done.get('code_commit'),str) or re.fullmatch('[0-9a-f]{40}',done['code_commit']) is None
            or done.get('representations')!=REPRESENTATIONS):
        raise ValueError('All seven source-only frozen feature exports must complete before private truth')
    parents=done.get('parent_identities',{})
    if set(parents)!={'F0','S0','S1','M'}:raise ValueError('Original F0 and all three fixed source parent identities required')
    f0=parents['F0']
    if any(f0.get(k)!=v for k,v in dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701).items()):
        raise ValueError('The fixed original F0 E50 identity is required')
    for arm in ('S0','S1','M'):
        identity=parents[arm]
        if any(identity.get(k)!=v for k,v in dict(run_id=PARENT_RUN,code_commit=PARENT_COMMIT,arm=arm,
                updates=2220,cumulative_updates=13320,identifier_parameters=284834,open_world_parameters=51841,
                training_container_parameters=952563,identity_head_replaced=False,parent_F0_identity=f0).items()):
            raise ValueError('Every representation must use its predeclared frozen source-only r02 arm')
    m=done.get('source_manifest',{})
    if any(m.get(k)!=v for k,v in dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',
            role='novel',count=SOURCE_COUNT,iq_shape=[SOURCE_COUNT,2,256],source_only=True,all_new_IQ_satellite=True,
            restored_input_path='P1',parent_F0_identity=f0).items()):
        raise ValueError('Only the existing source-novel G10 restored cache may be exported')
    metadata=None;features={}
    for arm in ('F0','S0','S1','M'):
        fields={'ids','rx_i','day_i','fused160'}|({'open64'} if arm!='F0' else set())
        with np.load(root/(arm+'.npz'),allow_pickle=False) as saved:
            if set(saved.files)!=fields:raise ValueError('Exact feature/public metadata fields required; labels/truth forbidden')
            data={k:saved[k].copy() for k in saved.files}
        if (not common._ids(data['ids'],SOURCE_COUNT)
                or any(data[k].shape!=(SOURCE_COUNT,) or data[k].dtype!=np.int64 or not np.isin(data[k],roster).all()
                    for k,roster in (('rx_i',common.RECEIVERS),('day_i',(1,2,3))))):
            raise ValueError('Unique opaque IDs and legal aligned source RX/day metadata required')
        current={k:data[k] for k in ('ids','rx_i','day_i')}
        if metadata is not None and any(not np.array_equal(current[k],metadata[k]) for k in current):
            raise ValueError('All source representations must preserve exactly the same ID order and public metadata')
        metadata=current
        for name,spec in REPRESENTATIONS.items():
            if spec['file']!=arm+'.npz':continue
            x=data[spec['key']]
            if x.shape!=(SOURCE_COUNT,spec['dimension']) or x.dtype!=np.float32 or not np.isfinite(x).all():
                raise ValueError('Every source representation must be aligned finite FP32 with its declared dimension')
            norms=np.linalg.norm(x.astype(np.float64),axis=1)
            if not np.all((norms==0)|np.isclose(norms,1.,rtol=0.,atol=2e-6)):
                raise ValueError('Only already unit-normalized features or exact zero vectors are accepted')
            features[name]=x
    return done,metadata,features


def _unit(features):
    x=np.asarray(features,np.float64);norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)


def _nearest_indices(features,ids,rx,day):
    """Label-free exact cosine 1NN, blocked128; all candidates exclude self."""
    x=_unit(features);n=len(x);order=np.argsort(ids,kind='stable');reference=x[order]
    neighbors={mode:np.full(n,-1,np.int64) for mode in CONDITIONS}
    for start in range(0,n,BLOCK):
        end=min(start+BLOCK,n);similarity=x[start:end]@reference.T
        other=np.arange(start,end)[:,None]!=order[None,:]
        cross_rx=rx[start:end,None]!=rx[order][None,:];cross_day=day[start:end,None]!=day[order][None,:]
        masks=(other,other&cross_rx,other&cross_day,other&cross_rx&cross_day)
        for mode,valid in zip(CONDITIONS,masks):
            selected=np.where(valid,similarity,-np.inf).argmax(axis=1);has=valid.any(axis=1)
            neighbors[mode][start:end]=np.where(has,order[selected],-1)
    return neighbors


def _retrieval(neighbors,tx,rx,day,roster):
    result={};n=len(tx)
    for mode,indices in neighbors.items():
        valid=indices>=0;hit=np.zeros(n,bool);hit[valid]=tx[valid]==tx[indices[valid]]
        def row(mask,required):
            per={}
            for label in required:
                m=mask&(tx==label);count=int(m.sum());eligible=int((m&valid).sum());hits=int((m&hit).sum())
                per[str(label)]=dict(count=count,eligible_count=eligible,hits=hits,
                    hit_rate=float(hits/count) if count and eligible==count else None)
            values=[p['hit_rate'] for p in per.values()];count=int(mask.sum());eligible=int((mask&valid).sum())
            return dict(count=count,eligible_count=eligible,missing_candidate_count=count-eligible,hits=int((mask&hit).sum()),
                macro_hit_rate=float(np.mean(values)) if values and all(v is not None for v in values) else None,
                pooled_hit_rate=float(hit[mask].mean()) if count and eligible==count else None,
                coverage_complete=bool(values and all(v is not None for v in values)),per_TX=per)
        result[mode]=dict(overall=row(np.ones(n,bool),roster),
            by_RX={str(r):row(rx==r,roster) for r in common.RECEIVERS},
            by_day={str(d):row(day==d,roster) for d in (1,2,3)},
            by_TX={str(t):row(tx==t,(t,)) for t in roster},
            candidate_rule=mode+'; self always excluded; no TX labels or true K used',
            tie_rule='lexicographically smallest opaque ID among exact cosine maxima',
            averaging='equal weight per required true TX; absent candidate or TX is N/A',
            not_future_registration_accuracy=True)
    return result


def _cosine_gap(features,tx,rx,day):
    x=_unit(features);n=len(x);sums=np.zeros(2,np.float64);counts=np.zeros(2,np.int64)
    for start in range(0,n,BLOCK):
        end=min(start+BLOCK,n);similarity=np.clip(x[start:end]@x.T,-1.,1.)
        valid=(rx[start:end,None]!=rx[None,:])&(day[start:end,None]!=day[None,:])
        same=tx[start:end,None]==tx[None,:]
        for i,mask in enumerate((valid&same,valid&~same)):
            sums[i]+=similarity[mask].sum(dtype=np.float64);counts[i]+=int(mask.sum())
    means=[float(sums[i]/counts[i]) if counts[i] else None for i in (0,1)]
    return dict(same_TX_pair_count=int(counts[0]),different_TX_pair_count=int(counts[1]),
        same_TX_mean=means[0],different_TX_mean=means[1],gap=means[0]-means[1] if all(v is not None for v in means) else None,
        pair_scope='all ordered pairs with different source RX AND day; no self; pair-weighted descriptive means',block_size=BLOCK)


def _spectrum(features):
    x=_unit(features)
    # Center differences from one row to avoid roundoff inventing variance
    # when every unit feature has collapsed to the same nonzero vector.
    shifted=x-x[:1];centered=shifted-shifted.mean(axis=0)
    covariance=centered.T@centered/max(len(x)-1,1)
    eigenvalues=np.maximum(np.linalg.eigvalsh(covariance),0.);trace=float(eigenvalues.sum())
    if trace>0:
        probability=eigenvalues[eigenvalues>0]/trace
        effective=float(np.exp(-np.sum(probability*np.log(probability))))
        participation=float(trace*trace/np.square(eigenvalues).sum())
    else:effective=participation=0.
    return dict(dimension=x.shape[1],covariance_trace=trace,effective_rank=effective,participation_rank=participation,
        zero_vectors=int((np.linalg.norm(x,axis=1)==0).sum()),
        definition='centered sample covariance eigenvalue entropy exp(H) and trace(C)^2/trace(C^2); collapsed rank=0')


def _between_ratio(features,groups):
    unit=_unit(features);x=unit-unit[:1];mean=x.mean(axis=0);total=float(np.square(x-mean).sum())
    if total==0:return None
    between=sum(int((groups==g).sum())*float(np.square(x[groups==g].mean(axis=0)-mean).sum()) for g in np.unique(groups))
    return float(np.clip(between/total,0.,1.))


def score(feature_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable independent geometry score root required')
    marker,metadata,features=_prediction(feature_root)
    nearest={}
    for name in REPRESENTATIONS:
        nearest[name]=_nearest_indices(features[name],metadata['ids'],metadata['rx_i'],metadata['day_i'])
        print('STAR_OPEN_FEATURE_NEIGHBORS_FROZEN',name,SOURCE_COUNT,flush=True)
    truth=common._truth(novel_truth,metadata)  # Sole private read, after all seven label-blind choices.
    tx=truth['tx'];rx=metadata['rx_i'];day=metadata['day_i'];results={}
    for name in REPRESENTATIONS:
        x=features[name]
        results[name]=dict(dimension=REPRESENTATIONS[name]['dimension'],
            nearest_neighbor=_retrieval(nearest[name],tx,rx,day,common.UNKNOWN_TX),
            cosine_cross_RX_and_day=_cosine_gap(x,tx,rx,day),spectrum=_spectrum(x),
            between_group_variance_ratio=dict(TX=_between_ratio(x,tx),RX=_between_ratio(x,rx),day=_between_ratio(x,day)),
            variance_scope='one-factor between-group sum of squares / total centered sum of squares; not additive or causal; collapse N/A')
        print('STAR_OPEN_FEATURE_GEOMETRY_SCORED',name,flush=True)
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],source_count=SOURCE_COUNT,
        source_role='novel',representations=results,primary_arm='M',representation_order=list(REPRESENTATIONS),truth_last=True,
        all_neighbors_fixed_before_truth=True,neighbor_selection_TX_labels_used=False,true_K_used_for_neighbor_selection=False,
        source_only=True,source_RX_seen=True,source_novel_historically_exposed=True,new_blind_confirmation=False,
        required_source_heldout_TX=list(common.UNKNOWN_TX),source_RX=list(common.RECEIVERS),source_days=[1,2,3],
        target_reads=0,target_U_reads=0,target_Q_reads=0,G_forward_rows=0,optimizer_updates=0,trained_parameters=0,
        diagnostic_only=True,not_future_registration_accuracy=True,A_N=None,H_OW=None,
        final_phase3_complete=False,full_goal_passed=False,full_goal_status='ACTIVE',eligible_primary=False,next_step='DIAGNOSTIC_ONLY',
        claim='Frozen source-heldout same-pool geometry only; not unseen RX, blind new TX, unsupervised classification or future anonymous registration')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k!='representations'})
    print(STATUS,flush=True);return result


def main(argv=None):
    print(json.dumps(dict(event='STAR_OPEN_FEATURE_SCORE_PROCESS',pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv)),flush=True)
    p=argparse.ArgumentParser();p.add_argument('--feature-root',required=True);p.add_argument('--novel-truth',required=True)
    p.add_argument('--output',required=True);a=p.parse_args(argv);score(a.feature_root,a.novel_truth,a.output)


if __name__=='__main__':main()
