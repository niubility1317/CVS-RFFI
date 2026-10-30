"""Truth-last joint old/heldout source-neighborhood diagnostics, without fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import star_open_feature_score as geometry
from . import star_consensus_score as common

RUN_ID='20261030-phase3-star-joint-geometry-s2026092701-r01'
SCHEMA='pamf_phase3_joint_geometry_score_v1'
STATUS='STAR_JOINT_GEOMETRY_SCORES_COMPLETE'
COUNTS=dict(L=6300,novel=3150,union=9450)
REPRESENTATIONS=('original_fused160','protected160')
UNION_FIELDS={'ids','rx_i','day_i','pool','original_fused160','protected160'}
OLD_NAMES=('14-10','14-7','20-15','20-19','6-15','8-20')
_read,_write=common._read,common._write


def _prediction(root):
    from .star_joint_geometry_export import validate_metadata
    root=Path(root);marker=validate_metadata(_read(root/'complete.json'))
    if marker['counts']!=COUNTS:raise ValueError('Fixed L-first plus novel union counts required')
    with np.load(root/'union.npz',allow_pickle=False) as saved:
        if set(saved.files)!=UNION_FIELDS:raise ValueError('Exact public union fields required; no labels/TX/true K')
        data={k:saved[k].copy() for k in saved.files}
    n=COUNTS['union']
    if (not common._ids(data['ids'],n)
            or any(data[k].shape!=(n,) or data[k].dtype!=np.int64 for k in ('rx_i','day_i','pool'))
            or not np.isin(data['rx_i'],common.RECEIVERS).all() or not np.isin(data['day_i'],(1,2,3)).all()
            or not np.array_equal(data['pool'],np.r_[np.zeros(COUNTS['L'],np.int64),np.ones(COUNTS['novel'],np.int64)])):
        raise ValueError('Unique public IDs, source domains and fixed L-first/novel-last roles required')
    for key in REPRESENTATIONS:
        x=data[key]
        if x.shape!=(n,160) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Finite FP32 union unit160 required')
        norms=np.linalg.norm(x.astype(np.float64),axis=1)
        if not np.all((norms==0)|np.isclose(norms,1.,rtol=0.,atol=2e-6)):raise ValueError('Union features must be unit or zero')
    return marker,data


def _load_L_labels(root,marker,data):
    """Only these three public files, called after all 16 neighbor sets freeze."""
    root=Path(root);manifest=_read(root/'manifest.json')
    ids=np.load(root/'ids.npy',allow_pickle=False)
    with np.load(root/'metadata.npz',allow_pickle=False) as saved:
        if set(saved.files)!={'y','rx_i','day_i','domain'}:raise ValueError('Exact source-L public metadata fields required')
        meta={k:saved[k].copy() for k in saved.files}
    n=COUNTS['L']
    if (manifest!=marker['source_feature_manifest']['source_manifests']['L']
            or not np.array_equal(ids,data['ids'][:n])
            or any(a.shape!=(n,) or a.dtype!=np.int64 for a in meta.values())
            or not np.isin(meta['y'],range(6)).all()
            or any(not np.array_equal(meta[k],data[k][:n]) for k in ('rx_i','day_i'))
            or not np.array_equal(meta['domain'],np.searchsorted(common.RECEIVERS,meta['rx_i'])*3+meta['day_i']-1)):
        raise ValueError('Original L handle, opaque IDs, legal visible labels and public domains must align')
    return meta['y']


def _mean_required(per,labels):
    values=[per[str(c)]['hit_rate'] for c in labels]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _role_landing(indices,y,mask,novel):
    labels=tuple(range(6,9)) if novel else tuple(range(6));mask=mask&np.isin(y,labels)
    names=('to_known','same_novel_TX','other_novel_TX') if novel else ('to_novel','same_known_TX','other_known_TX')
    valid=indices>=0;neighbor=np.full(len(y),-1,np.int64);neighbor[valid]=y[indices[valid]]
    events=((neighbor<6)&valid,neighbor==y,(neighbor>=6)&(neighbor!=y)&valid) if novel else (
        neighbor>=6,neighbor==y,(neighbor<6)&(neighbor!=y)&valid)
    per={}
    for c in labels:
        m=mask&(y==c);count=int(m.sum());eligible=int((m&valid).sum());complete=count>0 and eligible==count
        per[str(c)]=dict(count=count,eligible_count=eligible,**{name+'_count':int((m&e).sum()) for name,e in zip(names,events)},
            **{name+'_rate':float((m&e).sum()/count) if complete else None for name,e in zip(names,events)})
    count=int(mask.sum());eligible=int((mask&valid).sum());complete=all(v['count']>0 and v['eligible_count']==v['count'] for v in per.values())
    return dict(count=count,eligible_count=eligible,missing_candidate_count=count-eligible,coverage_complete=complete,
        missing_TX=[str(c) for c in labels if per[str(c)]['count']==0],by_TX=per,
        **{name+'_count':int((mask&e).sum()) for name,e in zip(names,events)},
        **{'pooled_'+name+'_rate':float((mask&e).sum()/count) if complete else None for name,e in zip(names,events)},
        **{'macro_'+name+'_rate':float(np.mean([v[name+'_rate'] for v in per.values()])) if complete else None for name in names})


def condition_report(indices,y,rx,day,condition='fixed'):
    n=len(y)
    if indices.shape!=(n,) or indices.dtype!=np.int64 or np.any((indices<-1)|(indices>=n)) or np.any(indices==np.arange(n)):
        raise ValueError('Fixed neighbor indices must be aligned, in range and exclude self')
    retrieval=geometry._retrieval({condition:indices},y,rx,day,tuple(range(9)))[condition]
    valid=indices>=0;confusion=np.zeros((9,9),np.int64)
    np.add.at(confusion,(y[valid],y[indices[valid]]),1)
    allq=np.ones(n,bool)
    return dict(confusion_counts=confusion.tolist(),confusion_label_order=list(range(9)),
        macro9=retrieval['overall']['macro_hit_rate'],macro_known6=_mean_required(retrieval['overall']['per_TX'],range(6)),
        macro_novel3=_mean_required(retrieval['overall']['per_TX'],range(6,9)),retrieval=retrieval,
        known_queries=_role_landing(indices,y,allq,False),novel_queries=_role_landing(indices,y,allq,True),
        role_landings_by_RX={str(r):dict(known_queries=_role_landing(indices,y,rx==r,False),novel_queries=_role_landing(indices,y,rx==r,True)) for r in common.RECEIVERS},
        role_landings_by_day={str(d):dict(known_queries=_role_landing(indices,y,day==d,False),novel_queries=_role_landing(indices,y,day==d,True)) for d in (1,2,3)},
        count_scope='Full query denominators; a missing candidate or required TX gives N/A, never zero-filled or dropped')


def paired_correct(new,old,y,mask,required_roster=None):
    valid=mask&(new>=0)&(old>=0);n=int(mask.sum());complete=n>0 and int(valid.sum())==n
    if required_roster is not None:complete=complete and all(np.any(mask&(y==c)) for c in required_roster)
    a=np.zeros(len(y),bool);b=np.zeros(len(y),bool);a[valid]=y[new[valid]]==y[valid];b[valid]=y[old[valid]]==y[valid]
    gain=int((valid&a&~b).sum());lost=int((valid&b&~a).sum())
    return dict(query_count=n,valid_pair_count=int(valid.sum()),coverage_complete=complete,
        gained=gain if complete else None,lost=lost if complete else None,net=gain-lost if complete else None,
        counts_on_valid_pairs=dict(gained=gain,lost=lost,net=gain-lost))


def _cohort_cosines(x,pool,rx,day):
    x=geometry._unit(x);sums=np.zeros(3,np.float64);counts=np.zeros(3,np.int64)
    for start in range(0,len(x),geometry.BLOCK):
        end=min(start+geometry.BLOCK,len(x));similarity=np.clip(x[start:end]@x.T,-1.,1.)
        valid=(rx[start:end,None]!=rx[None])&(day[start:end,None]!=day[None])
        masks=(valid&(pool[start:end,None]==0)&(pool[None]==0),valid&(pool[start:end,None]==1)&(pool[None]==1),
               valid&(pool[start:end,None]!=pool[None]))
        for i,mask in enumerate(masks):sums[i]+=similarity[mask].sum(dtype=np.float64);counts[i]+=int(mask.sum())
    return {name:dict(ordered_pair_count=int(counts[i]),mean_cosine=float(sums[i]/counts[i]) if counts[i] else None)
            for i,name in enumerate(('L_L','novel_novel','L_novel_both_directions'))}


def score(feature_root,L_public_root,novel_truth,output):
    output=Path(output)
    if output.exists():raise FileExistsError('New immutable joint diagnostic score output required')
    marker,data=_prediction(feature_root);nL=COUNTS['L'];nv=np.flatnonzero(data['pool']==1)
    nearest={}
    for name in REPRESENTATIONS:
        nearest[name]=dict(union=geometry._nearest_indices(data[name],data['ids'],data['rx_i'],data['day_i']),
            novel=geometry._nearest_indices(data[name][nv],data['ids'][nv],data['rx_i'][nv],data['day_i'][nv]))
    # First per-record true-label reads: all 2x2x4 label-blind choices already fixed.
    visible=_load_L_labels(L_public_root,marker,data)
    novel_public={k:data[k][nv] for k in ('ids','rx_i','day_i')}
    truth=common._truth(novel_truth,novel_public)
    mapping={tx:i+6 for i,tx in enumerate(common.UNKNOWN_TX)}
    y=np.r_[visible,np.asarray([mapping[tx] for tx in truth['tx']],np.int64)];rx=data['rx_i'];day=data['day_i'];results={}
    for name in REPRESENTATIONS:
        union={};matched={}
        novel_result=geometry._retrieval(nearest[name]['novel'],y[nv],rx[nv],day[nv],(6,7,8))
        for mode in geometry.CONDITIONS:
            indices=nearest[name]['union'][mode];union[mode]=condition_report(indices,y,rx,day,mode)
            local=nearest[name]['novel'][mode];global_indices=np.full(len(y),-1,np.int64)
            valid=local>=0;global_indices[nv[valid]]=nv[local[valid]]
            a=union[mode]['macro_novel3'];b=novel_result[mode]['overall']['macro_hit_rate']
            landings=union[mode]['novel_queries']
            matched[mode]=dict(novel_only_macro3=b,union_novel_macro3=a,delta_macro3=None if a is None or b is None else a-b,
                paired_correct=paired_correct(indices,global_indices,y,data['pool']==1,(6,7,8)),
                unknown_to_known_invasion_count=landings['to_known_count'],unknown_to_known_invasion_rate=landings['pooled_to_known_rate'],
                scope='Same source novel queries; union adds old-label references, not a changed deployed rejector')
        x=data[name]
        results[name]=dict(union_reference=union,novel_reference=novel_result,novel_reference_vs_union=matched,
            spectrum=geometry._spectrum(x),cosine_cross_RX_and_day=geometry._cosine_gap(x,y,rx,day),
            between_group_variance_ratio=dict(TX=geometry._between_ratio(x,y),RX=geometry._between_ratio(x,rx),day=geometry._between_ratio(x,day)),
            cohort_cosine_means=_cohort_cosines(x,data['pool'],rx,day))
    comparisons=dict(union_reference={},novel_reference={})
    for mode in geometry.CONDITIONS:
        new=nearest['protected160']['union'][mode];old=nearest['original_fused160']['union'][mode]
        row={name:paired_correct(new,old,y,mask,roster) for name,mask,roster in (
            ('all9',np.ones(len(y),bool),tuple(range(9))),('known_queries',data['pool']==0,tuple(range(6))),('novel_queries',data['pool']==1,(6,7,8)))}
        valid=(data['pool']==1)&(new>=0)&(old>=0);a=np.zeros(len(y),bool);b=np.zeros(len(y),bool)
        a[valid]=y[new[valid]]<6;b[valid]=y[old[valid]]<6
        complete=int(valid.sum())==len(nv)
        row['novel_to_known_invasion_change']=dict(valid_pair_count=int(valid.sum()),gained=int((valid&a&~b).sum()) if complete else None,
            lost=int((valid&b&~a).sum()) if complete else None,net=int((a[valid].sum()-b[valid].sum())) if complete else None)
        comparisons['union_reference'][mode]=row
        comparisons['novel_reference'][mode]=paired_correct(nearest['protected160']['novel'][mode],nearest['original_fused160']['novel'][mode],
                                                          y[nv],np.ones(len(nv),bool),(6,7,8))
    result=dict(schema=SCHEMA,status=STATUS,run_id=RUN_ID,code_commit=marker['code_commit'],counts=COUNTS,representations=results,
        paired_representations=comparisons,primary='R1.protected160',representation_order=list(REPRESENTATIONS),
        true_label_order=dict(enumerate(OLD_NAMES+common.UNKNOWN_TX)),truth_last=True,all_16_neighbors_fixed_before_truth=True,
        neighbor_selection_TX_or_K_used=False,source_only=True,input_path='cached_P1_features_only',registered_old_class_count=6,
        source_RX_seen=True,source_novel_historically_exposed=True,known_L_used_in_parent_training=True,novel_reference_uses_benchmark_role=True,
        source_role_pool_is_model_prediction=False,diagnostic_only=True,eligible_primary=False,next_step='DIAGNOSTIC_ONLY',
        model_updates=0,optimizer_updates=0,threshold_updates=0,target_reads=0,target_U_reads=0,target_Q_reads=0,G_forward_rows=0,
        A_K=None,C_K=None,R_U=None,A_N=None,H_OW=None,final_phase3_complete=False,full_goal_status='ACTIVE',
        parent_model_identity=marker['parent_model_identity'],parent_F0_identity=marker['parent_F0_identity'],dura_identity=marker['dura_identity'],
        descriptive_scope='Same-pool source neighborhood and variance, not unseen-RX test, deployed rejection, future anonymous accuracy or causal decomposition')
    output.mkdir(parents=True);_write(output/'results.json',result)
    _write(output/'complete.json',{k:v for k,v in result.items() if k not in ('representations','paired_representations')})
    print(STATUS,flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--feature-root',required=True);p.add_argument('--L-public-root',required=True)
    p.add_argument('--novel-truth',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    score(a.feature_root,a.L_public_root,a.novel_truth,a.output)


if __name__=='__main__':main()
