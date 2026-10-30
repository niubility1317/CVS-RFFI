"""HDBSCAN discovery policy; all Gaussian and known-risk arithmetic is shared."""
from __future__ import annotations
import numpy as np
from . import star_signal_register_core as base
from .star_signal_register_core import (SOURCE_SCHEMA,ANONYMOUS_SCHEMA,METHODS,PRIMARY_METHOD,PRIMARY_INDEX,
    RECEIVERS,SCENES,HARD_NAMES,DIAGNOSTIC_NAMES,DIMENSION,MIN_SAMPLES,MIN_CLUSTER,
    validate_source,validate_anonymous,source_radius,anonymous_ratios,predict,calibrate)

density=base.density
PARAMETERS=dict(min_cluster_size=16,min_samples=8,metric='euclidean',algorithm='brute',n_jobs=1,alpha=1,
    cluster_selection_epsilon=0.,cluster_selection_method='eom',allow_single_cluster=False,copy=True,max_cluster_size=None)


def discover(state,source,U_features,base_scores,ids):
    """All-U hierarchical clusters; no source-epsilon cut, quotas or hidden labels."""
    import sklearn
    from sklearn.cluster import HDBSCAN
    validate_source(source);white=base._white(state,U_features);n=len(white);ids=base._ids(ids,n)
    scores=base._float(base_scores,(n,2),'old pooled/predicted margins');order=np.argsort(ids,kind='stable')
    labels=HDBSCAN(**PARAMETERS).fit_predict(white[order])
    members=[np.flatnonzero(labels==label) for label in set(labels.tolist()) if label>=0]
    members.sort(key=lambda rows:ids[order[rows[0]]])
    canonical=np.full(n,-1,np.int64);assigned=np.full(n,-1,np.int64);rejected=scores[:,1]<0
    means=[];variances=[];clusters=[]
    for label,sorted_rows in enumerate(members):
        rows=order[sorted_rows];canonical[rows]=label;count=len(rows);rejects=int(rejected[rows].sum())
        qualified=count>=MIN_CLUSTER and rejects*2>count;slot=None
        if qualified:
            slot=len(means);assigned[rows]=slot;mean=white[rows].mean(axis=0)
            radial=float(np.square(white[rows]-mean).sum()/(count*DIMENSION));alpha=count/(count+DIMENSION)
            means.append(mean);variances.append(max(alpha*radial+(1-alpha)*float(state['pooled_L_variance']),density.VARIANCE_FLOOR))
        clusters.append(dict(dbscan_label=label,size=count,rejected_count=rejects,rejected_fraction=rejects/count,
            qualified=qualified,anonymous_index=slot,minimum_opaque_id=str(ids[rows[0]])))
    anonymous=dict(schema=np.asarray(ANONYMOUS_SCHEMA),means=np.asarray(means,np.float64).reshape(-1,160),
        variances=np.asarray(variances,np.float64),anonymous_ids=np.asarray(['anon-'+str(j+1).zfill(4) for j in range(len(means))],dtype='U'))
    validate_anonymous(anonymous)
    assignments=dict(ids=ids.copy(),assignments=assigned,dbscan_labels=canonical,rejected=rejected.copy())
    diag=dict(algorithm='sklearn.cluster.HDBSCAN',parameters=PARAMETERS.copy(),sklearn_version=sklearn.__version__,
        source_radius_used_for_clustering=False,label_field='dbscan_labels holds canonical HDBSCAN cluster labels for compatibility',
        U_count=n,all_U_rows_used=True,registered_count=len(means),clusters=clusters,noise_count=int((canonical<0).sum()),
        min_samples=MIN_SAMPLES,minimum_registered_size=MIN_CLUSTER,strict_rejected_majority=True,
        true_unknown_count_used=False,true_U_labels_used=False,source_density_refitted=False)
    return anonymous,assignments,diag
