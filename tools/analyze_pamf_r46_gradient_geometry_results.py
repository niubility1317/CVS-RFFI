"""Audit stored source batch summaries; never read IQ, model weights or truth."""
import math
import numpy as np
from experiments.pamf_r46_gradient_geometry import design as d

METHOD='375bf471fa5eb30a9ad36eee0411771fdace9e8a'


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-16)


def stats(values):
    if not values:return dict(mean=None,median=None,p90=None,p95=None,minimum=None,maximum=None)
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Nonfinite source statistics')
    return dict(mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()))


def audit_batches(records):
    if len(records)!=16:raise ValueError('Complete fixed sixteen-batch source panel required')
    for i,row in enumerate(records):
        if any(row.get(k)!=v for k,v in dict(batch=i,count=16,L_s=8,U_s=8).items()):raise ValueError('Actual mixed-role batches differ')
        if set(row['losses'])!=set(d.OBJECTIVES) or any(not math.isfinite(v) or v<0 for v in row['losses'].values()):raise ValueError('Finite three source losses required')
        if set(row['geometry'])!=set(d.GROUPS):raise ValueError('Complete unique parameter groups required')
        for group,item in row['geometry'].items():
            norms=item['norms']
            if set(norms)!=set(d.OBJECTIVES) or any(not math.isfinite(v) or v<0 for v in norms.values()):raise ValueError('Finite source gradient norms required')
            if set(item['pairs'])!={a+'_vs_'+b for a,b in d.PAIRS}:raise ValueError('All objective pairs required')
            for a,b in d.PAIRS:
                pair=item['pairs'][a+'_vs_'+b];valid=norms[a]>d.NORM_FLOOR and norms[b]>d.NORM_FLOOR
                if type(pair['valid']) is not bool or pair['valid']!=valid or not math.isfinite(pair['dot']):raise ValueError('Invalid source gradient pair')
                if valid:
                    expected=max(-1.,min(1.,pair['dot']/(norms[a]*norms[b])))
                    if pair['cosine'] is None or not close(pair['cosine'],expected):raise ValueError('Cosine does not match stored dot/norms')
                elif pair['cosine'] is not None:raise ValueError('Undefined cosine must stay null')
                ratio=norms[b]/norms[a] if norms[a]>d.NORM_FLOOR else None
                if (ratio is None)!=(pair['norm_ratio'] is None) or (ratio is not None and not close(ratio,pair['norm_ratio'])):raise ValueError('Norm ratio does not close')
        for objective in d.OBJECTIVES:
            total=sum(row['geometry'][g]['norms'][objective]**2 for g in ('coarse','action','prior'))
            if not close(total,row['geometry']['whole']['norms'][objective]**2):raise ValueError('Unique module gradient norms do not partition whole model')
        for a,b in d.PAIRS:
            key=a+'_vs_'+b;total=sum(row['geometry'][g]['pairs'][key]['dot'] for g in ('coarse','action','prior'))
            if not close(total,row['geometry']['whole']['pairs'][key]['dot']):raise ValueError('Module dot products do not partition whole model')
    rows=[]
    for name in d.OBJECTIVES:
        rows.append(dict(kind='loss',group='whole',metric=name,count=16,valid_count=16,invalid_count=0,negative_fraction=None,**stats([r['losses'][name] for r in records])))
    for group in d.GROUPS:
        for name in d.OBJECTIVES:
            rows.append(dict(kind='norm',group=group,metric=name,count=16,valid_count=16,invalid_count=0,negative_fraction=None,**stats([r['geometry'][group]['norms'][name] for r in records])))
        for first,second in d.PAIRS:
            name=first+'_vs_'+second;pairs=[r['geometry'][group]['pairs'][name] for r in records]
            x=[v['cosine'] for v in pairs if v['valid']];ratios=[v['norm_ratio'] for v in pairs if v['norm_ratio'] is not None]
            rows.append(dict(kind='cosine',group=group,metric=name,count=16,valid_count=len(x),invalid_count=16-len(x),negative_fraction=sum(v<0 for v in x)/len(x) if x else None,
                norm_ratio_valid_count=len(ratios),norm_ratio_mean=float(np.mean(ratios)) if ratios else None,**stats(x)))
    return rows


def compare_rows(actual,expected):
    if len(actual)!=len(expected):raise ValueError('Source summary length differs')
    for a,b in zip(actual,expected):
        if set(a)!=set(b):raise ValueError('Source summary fields differ')
        for key,value in b.items():
            if isinstance(value,float):
                if not isinstance(a[key],(int,float)) or not close(a[key],value):raise ValueError('Independent source aggregate does not match')
            elif a[key]!=value:raise ValueError('Source summary key/count/null differs')


def audit_collection(collection):
    result=collection['result']
    if (collection.get('actual_owned_handle_live') or collection.get('failure') is not None or result.get('run')!=d.RUN
        or result.get('method_commit')!=METHOD or result.get('status')!='SOURCE_GRADIENT_GEOMETRY_COMPLETE'
        or result.get('configuration')!=d.configuration() or not result.get('all_checkpoint_states_unchanged')):raise ValueError('Completed actual fixed source diagnostic required')
    if any(result.get(k)!=0 for k in ('F_weight_reads','F_output_reads','TX_label_reads','V_reads','target_reads','optimizer_updates','new_network_versions')):raise ValueError('Source-only no-update permissions differ')
    if (result.get('teacher_cache_IQ_inputs'),result.get('total_actual_G_IQ_inputs'),result.get('batch_records'))!=(770,6930,192):raise ValueError('Actual source exposure differs')
    if set(result['rows'])!=set(d.ARMS) or set(collection['batch_records'])!=set(d.ARMS):raise ValueError('All four fixed source models required')
    recomputed=[]
    for arm,row in result['rows'].items():
        if row['parameters']!=d.PARAMETERS[arm] or row['source_final']!=list(d.SOURCE_BUDGET[arm]) or not row['checkpoint_unchanged'] or row['optimizer_updates']!=0:raise ValueError('Frozen source checkpoint budget differs')
        if row['actual_student_IQ_inputs']!=dict(smoke_received=2,smoke_clean=2,received=768,clean_identity=768) or (row['L_s_diagnostic_inputs'],row['U_s_diagnostic_inputs'],row['L_s_smoke_inputs'])!=(768,768,4):raise ValueError('Actual source roles/G calls do not close')
        if set(row['views'])!=set(d.VIEWS) or set(collection['batch_records'][arm])!=set(d.VIEWS):raise ValueError('All three source views required')
        for view in d.VIEWS:
            cell=collection['batch_records'][arm][view]
            if (cell['arm'],cell['view'])!=(arm,view):raise ValueError('Batch cell identity differs')
            expected=audit_batches(cell['records']);compare_rows(row['views'][view],expected)
            recomputed.extend(dict(arm=arm,source_view=view,**v) for v in expected)
    if len(recomputed)!=324:raise ValueError('Complete 324-row source matrix required')
    compare_rows(result['summary_rows'],recomputed)
    return result
