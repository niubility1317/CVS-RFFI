import numpy as np
from . import design as d
from tools.analyze_pamf_source_cross_record import unit_features

SIZES=dict(mirror=128,spectral=4,reference=28)
SLICES=dict(mirror=slice(64,128),spectral=slice(None),reference=slice(0,24))


def fit_moments(batches):
    sums={k:np.zeros(n) for k,n in SIZES.items()};squares={k:np.zeros(n) for k,n in SIZES.items()};count=0
    for batch in batches:
        if set(batch)!=set(SIZES):raise ValueError('All fixed analytic groups required')
        size=None
        for key,n in SIZES.items():
            value=np.asarray(batch[key],dtype=np.float64)
            if value.ndim!=2 or value.shape[1]!=n or not len(value) or not np.isfinite(value).all():raise ValueError('Finite aligned calibration features required')
            if size is not None and len(value)!=size:raise ValueError('Aligned calibration sample counts required')
            size=len(value);sums[key]+=value.sum(0);squares[key]+=(value*value).sum(0)
        count+=size
    if not count:raise ValueError('Nonempty training calibration required')
    moments={k:dict(mean=(sums[k]/count).tolist(),std=np.maximum(np.sqrt(np.maximum(squares[k]/count-(sums[k]/count)**2,0)),.05).tolist()) for k in SIZES}
    return dict(count=count,moments=moments,label_reads=0,F_access=False,V_fit=False)


def representations(raw,moments):
    result={};count=None
    for key,n in SIZES.items():
        x=np.asarray(raw[key],dtype=np.float64);mean=np.asarray(moments[key]['mean']);std=np.asarray(moments[key]['std'])
        if (x.ndim!=2 or x.shape[1]!=n or not len(x) or mean.shape!=(n,) or std.shape!=(n,)
                or not all(np.isfinite(a).all() for a in (x,mean,std)) or (std<.05-1e-12).any()):raise ValueError('Fixed finite training moments required')
        if count is not None and len(x)!=count:raise ValueError('Aligned analytic representations required')
        count=len(x);result[key]=((x-mean)/std)[:,SLICES[key]]
    result['combined']=np.concatenate([result[k]/np.sqrt(3*result[k].shape[1]) for k in SIZES],axis=1)
    return result


def class_similarities(features,reference,rx,day):
    unit,nonzero=unit_features(features);rx=np.asarray(rx).astype(str);day=np.asarray(day).astype(str)
    if rx.shape!=(len(unit),) or day.shape!=rx.shape or unit.shape[1]!=reference['unit'].shape[1]:raise ValueError('Aligned reference dimensions/RX/day required')
    scores=unit@reference['unit'].T;classes=np.zeros((len(unit),6));cells=np.zeros(len(unit),np.int64);missing=np.zeros_like(cells)
    for r in sorted(set(rx)):
        for date in sorted(set(day)):
            query=np.flatnonzero((rx==r)&(day==date))
            if not len(query):continue
            for cr,cd in reference['cells']:
                if cr==r or cd==date:continue
                mask=(reference['rx']==cr)&(reference['day']==cd)&reference['nonzero']
                indices=[np.flatnonzero(mask&(reference['labels']==tx)) for tx in range(6)]
                if any(not len(index) for index in indices):missing[query]+=1;continue
                classes[query]+=np.stack([scores[np.ix_(query,index)].mean(1) for index in indices],1);cells[query]+=1
    classes/=np.maximum(cells[:,None],1)
    return dict(classes=classes,valid=(cells>0)&nonzero,zero_query=~nonzero,valid_cells=cells,missing_cells=missing)


def contrasts(scores,labels):
    labels=np.asarray(labels)
    if labels.shape!=(len(scores['classes']),) or labels.dtype.kind not in 'iu' or not set(labels.tolist())<=set(range(6)):raise ValueError('Aligned six TX group labels required')
    positive=scores['classes'][np.arange(len(labels)),labels];negative=(scores['classes'].sum(1)-positive)/5
    return dict(positive=positive,negative=negative,margin=positive-negative,
        **{k:scores[k] for k in ('valid','zero_query','valid_cells','missing_cells')})


def permuted_queries(labels,rx,day):
    y=np.asarray(labels);rx=np.asarray(rx);day=np.asarray(day)
    if y.ndim!=1 or rx.shape!=y.shape or day.shape!=y.shape:raise ValueError('Aligned query strata required')
    rng=np.random.default_rng(d.SEED);result=np.tile(y,(d.PERMUTATIONS,1))
    for r in np.unique(rx):
        for date in np.unique(day):
            index=np.flatnonzero((rx==r)&(day==date))
            for row in result:row[index]=rng.permutation(y[index])
    return result


def null_control(scores,labels,permuted,mask):
    mask=np.asarray(mask,dtype=bool)&scores['valid'];n=int(mask.sum())
    result=dict(count=n,permutations=d.PERMUTATIONS,seed=d.SEED,reference_labels_fixed=True,is_p_value=False)
    if not n:return dict(result,observed_mean=None,null_mean=None,null_p05=None,null_p95=None,above_null_p95=None)
    values=scores['classes'][mask];p=permuted[:,mask];index=np.arange(n)[None,:]
    own=values[index,p];null=((6*own-values.sum(1)[None,:])/5).mean(1)
    observed=float(contrasts(scores,labels)['margin'][mask].mean());upper=float(np.quantile(null,.95))
    return dict(result,observed_mean=observed,null_mean=float(null.mean()),null_p05=float(np.quantile(null,.05)),null_p95=upper,
        above_null_p95=bool(observed>upper+1e-12))
