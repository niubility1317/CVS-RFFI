"""No prototypes, fitted transformations, classifiers or classification decisions."""
import collections
import math
import numpy as np

def validate_metadata(y,rx,day,fit):
    n=144 if fit else 1260
    if any(np.asarray(v).shape!=(n,) or not np.issubdtype(np.asarray(v).dtype,np.integer) for v in (y,rx,day)):
        raise ValueError('Fixed integer visible-L metadata required')
    expected={(yy,rr,dd) for yy in range(6) for rr in ((3,4,6,8) if fit else (1,)) for dd in (1,2,3)}
    cells=collections.Counter(zip(map(int,y),map(int,rx),map(int,day)))
    if set(cells)!=expected or set(cells.values())!={2 if fit else 70}:raise ValueError('Balanced fit/held source physical cells required')

def aggregate(records):
    if not records:raise ValueError('Nonempty relationship records required')
    positive=sum(r['same_pairs'] for r in records);negative=sum(r['other_pairs'] for r in records)
    triplets=sum(r['triplets'] for r in records);wins=sum(r['strict_wins'] for r in records);ties=sum(r['ties'] for r in records)
    same=sum(r['same_distance_sum'] for r in records)/positive;other=sum(r['other_distance_sum'] for r in records)/negative
    result=dict(anchors=len(records),same_pairs=positive,other_pairs=negative,triplets=triplets,strict_wins=wins,ties=ties,
        relation_rate=(wins+.5*ties)/triplets,mean_same_distance=same,mean_other_distance=other,mean_distance_gap=other-same,
        anchor_relation_minimum=min((r['strict_wins']+.5*r['ties'])/r['triplets'] for r in records),
        anchor_relation_maximum=max((r['strict_wins']+.5*r['ties'])/r['triplets'] for r in records))
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in result.values()):raise ValueError('Finite source geometry summaries required')
    return result

def relations(z,fit_y,fit_rx,fit_day,held_y,held_rx,held_day):
    validate_metadata(fit_y,fit_rx,fit_day,True);validate_metadata(held_y,held_rx,held_day,False)
    z=np.asarray(z)
    if z.shape!=(1404,160) or z.dtype not in (np.float32,np.float64) or not np.isfinite(z).all():raise ValueError('Finite actual1404x160 joint features required')
    raw_norm=np.linalg.norm(z.astype(np.float64),axis=1)
    if not (raw_norm>1e-4).all():raise ValueError('Nondegenerate native feature norms required')
    u=z.astype(np.float64)/raw_norm[:,None]
    distances=np.clip(2.-2.*(u[:144]@u[144:].T),0.,4.)
    records=[]
    for i,(y,rx,day) in enumerate(zip(fit_y,fit_rx,fit_day)):
        same=distances[i,(held_y==y)&(held_day==day)]
        other=distances[i,(held_y!=y)&(held_day==day)]
        if same.shape!=(70,) or other.shape!=(350,):raise ValueError('Same-day identity pair census differs')
        comparison=same[:,None]-other[None,:]
        records.append(dict(anchor_index=i,fit_RX=int(rx),day=int(day),TX=int(y),same_pairs=70,other_pairs=350,triplets=24500,
            strict_wins=int((comparison< -1e-12).sum()),ties=int((np.abs(comparison)<=1e-12).sum()),same_distance_sum=float(same.sum()),other_distance_sum=float(other.sum())))
    groups=[dict(group='overall',value=None,**aggregate(records))]
    for name,values in (('fit_RX',(3,4,6,8)),('day',(1,2,3)),('TX',range(6))):
        for value in values:groups.append(dict(group=name,value=value,**aggregate([r for r in records if r[name]==value])))
    for y in range(6):
        for rx in (3,4,6,8):
            for day in (1,2,3):
                groups.append(dict(group='cell',value=[y,rx,day],**aggregate([r for r in records if (r['TX'],r['fit_RX'],r['day'])==(y,rx,day)])))
    assert len(records)==144 and len(groups)==86
    return dict(records=records,groups=groups,raw_norm_summary=dict(minimum=float(raw_norm.min()),maximum=float(raw_norm.max()),mean=float(raw_norm.mean())),
        squared_distance='2-2*cosine,FP64 unit features',strict_tie_rule='difference<-1e-12=1,abs difference<=1e-12=.5,otherwise0',classifier_fit_updates=0)
