import numpy as np
from . import design as d
from experiments.pamf_analytic_identity_probe.analysis import class_similarities,contrasts,permuted_queries,null_control
from tools.analyze_pamf_source_cross_record import reference_pool,summarize,metadata


def analyze(features,ids,labels,rx,day):
    expected={(v,r,g) for v in d.VIEWS for r in d.READOUTS for g in d.GROUPS}
    if set(features)!=expected:raise ValueError('Every candidate,view and analytic group required')
    ids,labels,rx,day=metadata(ids,labels,rx,day,len(ids));n=len(ids)
    for value in features.values():
        if value.ndim!=2 or len(value)!=n or not value.shape[1] or not np.isfinite(value).all():raise ValueError('Finite complete aligned analytic features required')
    permuted=permuted_queries(labels,rx,day);groups=[('overall','all',np.ones(n,bool))]
    for dimension,values in [('TX',labels),('RX',rx),('day',day)]:
        groups.extend((dimension,str(v),values==v) for v in np.unique(values))
    records=[];negative=[];comparisons=[];pools={}
    for group in d.GROUPS:
        clean=features[('clean','raw',group)];reference=reference_pool(clean,ids,labels,rx,day)
        pools[group]={k:reference[k] for k in ('cap','reference_count','empty_TX_cells','zero_reference_count')}
        clean_scores=class_similarities(clean,reference,rx,day);clean_contrast=contrasts(clean_scores,labels)
        for view in d.VIEWS:
            scores={name:class_similarities(features[(view,name,group)],reference,rx,day) for name in d.READOUTS}
            values={name:contrasts(score,labels) for name,score in scores.items()};before=values['raw']
            reducer=np.sum if group=='combined' else np.mean
            error={name:reducer((features[(view,name,group)]-clean)**2,axis=1) for name in d.READOUTS}
            for name in d.CANDIDATES:
                after=values[name];common=before['valid']&after['valid']&clean_contrast['valid']
                for dimension,value,mask in groups:
                    records.append(dict(candidate=name,group=group,view=view,dimension=dimension,value=value,
                        **summarize(before,after,clean_contrast,mask),paired_descriptor_error=dict(count=int(mask.sum()),
                        raw=float(error['raw'][mask].mean()),restored=float(error[name][mask].mean()))))
                for readout,score in [('raw',scores['raw']),('restored',scores[name]),('clean',clean_scores)]:
                    negative.append(dict(candidate=name,group=group,view=view,readout=readout,**null_control(score,labels,permuted,common)))
            common=clean_contrast['valid'].copy()
            for value in values.values():common&=value['valid']
            count=int(common.sum())
            for a,b in d.COMPARISONS:
                comparisons.append(dict(candidate=a,control=b,group=group,view=view,common_count=count,
                    margin_change=float((values[a]['margin'][common]-values[b]['margin'][common]).mean()) if count else None,
                    descriptor_error_change=float((error[a][common]-error[b][common]).mean()) if count else None))
    return dict(records=records,query_label_negative_controls=negative,comparisons=comparisons,reference_pools=pools,
        selected_candidate=None,source_selection_changed=False,new_source_thresholds=False,target_decisions_changed=False,goal_complete=False)
