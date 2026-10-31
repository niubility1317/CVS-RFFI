"""All-U cluster association and finite-support prototype/source decisions."""
from __future__ import annotations
import numpy as np
from . import star_signal_register_core as base
from .star_hier_register_core import PARAMETERS

density=base.density
SOURCE_SCHEMA='pamf_phase3_pcd_cluster_source_v1'
REGISTRY_SCHEMA='pamf_phase3_pcd_cluster_registry_v1'
METHODS=('pooled_REG0','predicted_REG0','pooled_REG1','predicted_REG1')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3
FLOOR=1e-6


def validate_source(source):
    if not isinstance(source,dict) or set(source)!={'schema','source_centers','source_radius'} or np.asarray(source['schema']).shape!=() or np.asarray(source['schema']).dtype.kind!='U' or source['schema'].item()!=SOURCE_SCHEMA:raise ValueError('Exact source support schema required')
    base._float(source['source_centers'],(6,160),'six source means');r=base._float(source['source_radius'],(),'source support radius')
    if r<FLOOR:raise ValueError('Positive source support radius required')
    return source


def source_statistics(state,x,y):
    white=base._white(state,x);y=base._closed(y,len(white))
    if set(y.tolist())!=set(range(6)):raise ValueError('All six labelled source classes required')
    means=np.stack([white[y==label].mean(0) for label in range(6)]);distance=np.linalg.norm(white-means[y],axis=1);radius=max(float(np.quantile(distance,.95)),FLOOR)
    source=dict(schema=np.asarray(SOURCE_SCHEMA),source_centers=means,source_radius=np.asarray(radius,np.float64));validate_source(source)
    return source,dict(source_count=len(white),class_counts=np.bincount(y,minlength=6).tolist(),quantile=.95,source_labels_used=True,source_only_statistics=True,source_GMM_refitted=False)


def validate_registry(registry):
    keys={'schema','source_centers','source_radius','centers','radii','labels','anonymous_ids'}
    if not isinstance(registry,dict) or set(registry)!=keys or np.asarray(registry['schema']).shape!=() or np.asarray(registry['schema']).dtype.kind!='U' or registry['schema'].item()!=REGISTRY_SCHEMA:raise ValueError('Exact aggregate cluster registry required')
    validate_source(dict(schema=np.asarray(SOURCE_SCHEMA),source_centers=registry['source_centers'],source_radius=registry['source_radius']))
    centers=np.asarray(registry['centers']);n=len(centers);base._float(centers,(n,160),'all cluster centers');r=base._float(registry['radii'],(n,),'cluster support radii');labels=np.asarray(registry['labels']);names=np.asarray(registry['anonymous_ids'])
    if (np.any((r<FLOOR)|(r>registry['source_radius'])) or labels.shape!=(n,) or labels.dtype!=np.int64 or np.any(labels<0) or names.ndim!=1 or names.dtype.kind!='U'
            or names.tolist()!=['anon-'+str(j+1).zfill(4) for j in range(len(names))] or labels[labels>=6].tolist()!=list(range(6,6+len(names)))):raise ValueError('Supported old/anonymous labels and canonical anonymous slots required')
    return registry


def _cluster(white):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(**PARAMETERS).fit_predict(white)


def discover(state,source,features,closed,base_margin,ids):
    import sklearn
    validate_source(source);white=base._white(state,features);n=len(white);closed=base._closed(closed,n);margin=base._float(base_margin,(n,2),'old margins');ids=base._ids(ids,n)
    order=np.argsort(ids,kind='stable');raw=_cluster(white[order]);groups=[order[np.flatnonzero(raw==j)] for j in set(raw.tolist()) if j>=0];groups.sort(key=lambda rows:str(ids[rows[0]]))
    assignment=np.full(n,-1,np.int64);means=[];radii=[];labels=[];anonymous=[];clusters=[]
    for index,rows in enumerate(groups):
        assignment[rows]=index;votes=np.bincount(closed[rows],minlength=6);winner=int(votes.argmax());fraction=float(votes[winner]/len(rows));median=float(np.median(margin[rows,1]));associated=fraction>=.8 and median>=0
        if associated:label=winner;slot=None
        else:slot=len(anonymous);label=6+slot;anonymous.append('anon-'+str(slot+1).zfill(4))
        center=white[rows].mean(0);radius=max(min(float(source['source_radius']),float(np.quantile(np.linalg.norm(white[rows]-center,axis=1),.95))),FLOOR)
        means.append(center);radii.append(radius);labels.append(label);clusters.append(dict(cluster_index=index,size=len(rows),minimum_opaque_id=str(ids[rows[0]]),head_vote_counts=votes.tolist(),head_vote_fraction=fraction,
            majority_closed=winner,median_predicted_margin=median,associated_old=associated,label=label,anonymous_index=slot,radius=radius))
    registry=dict(schema=np.asarray(REGISTRY_SCHEMA),source_centers=source['source_centers'].copy(),source_radius=source['source_radius'].copy(),centers=np.asarray(means,np.float64).reshape(-1,160),
        radii=np.asarray(radii,np.float64),labels=np.asarray(labels,np.int64),anonymous_ids=np.asarray(anonymous,dtype='U'));validate_registry(registry)
    return registry,assignment,dict(algorithm='sklearn.cluster.HDBSCAN',parameters=PARAMETERS.copy(),sklearn_version=sklearn.__version__,U_count=n,all_U_rows_used=True,
        cluster_count=len(groups),anonymous_count=len(anonymous),noise_count=int((assignment<0).sum()),clusters=clusters,association_head_fraction_min=.8,association_median_margin_min=0.,
        radius_quantile=.95,source_radius_caps_clusters=True,true_U_labels_used=False,true_unknown_count_used=False,source_GMM_refitted=False)


def predict(state,registry,features,closed,base_margin):
    validate_registry(registry);white=base._white(state,features);n=len(white);closed=base._closed(closed,n);margin=base._float(base_margin,(n,2),'old margins')
    source_distance=np.linalg.norm(white-registry['source_centers'][closed],axis=1);fallback=np.minimum(margin,(1-source_distance/registry['source_radius'])[:,None])
    reg0=np.where(margin>=0,closed[:,None],-1);reg1=np.where(fallback>=0,closed[:,None],-1);scores=fallback.copy();support=np.full(n,-1,np.int64);ratio=np.full(n,-1.,np.float64)
    if len(registry['centers']):
        distances=np.column_stack([np.linalg.norm(white-center,axis=1)/radius for center,radius in zip(registry['centers'],registry['radii'])]);nearest=distances.argmin(1);ratio=distances[np.arange(n),nearest];hit=ratio<=1
        support[hit]=nearest[hit];reg1[hit]=registry['labels'][nearest[hit],None];scores[hit]=(1-ratio[hit])[:,None]
    return dict(raw_prediction=np.column_stack((reg0,reg1)).astype(np.int64),decision_score=np.column_stack((margin,scores)),support_cluster=support,support_ratio=ratio,source_distance=source_distance)
