"""Source/independent post-prediction metrics; never consumed by a training update."""
import math
import numpy as np

def classification(truth,prediction,classes):
    y=np.asarray(truth,dtype=int);p=np.asarray(prediction,dtype=int)
    if y.shape!=p.shape or y.ndim!=1 or not len(y):raise ValueError('nonempty aligned predictions required')
    if np.any(y<0) or np.any(y>=classes) or np.any(p<0) or np.any(p>=classes):raise ValueError('class outside declared universe')
    cm=np.bincount(y*classes+p,minlength=classes*classes).reshape(classes,classes)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(classes,dtype=float),where=den!=0)
    return dict(n=len(y),accuracy=float((y==p).mean()),macro_f1=float(f1.mean()),
        confusion=cm.tolist(),per_class_f1=f1.tolist())

def task_metrics(truth,prediction,rx,day,scene,classes=6,cvar_fraction=.2,reference=None):
    y,p,r,d,s=map(np.asarray,(truth,prediction,rx,day,scene))
    if not all(a.shape==y.shape for a in (p,r,d,s)):raise ValueError('metadata alignment')
    result=dict(overall=classification(y,p,classes),groups={})
    for name,axes in [('scene',[s]),('RX',[r]),('TX',[y]),('day',[d]),('RX_scene',[r,s]),('TX_scene',[y,s]),('RX_day_TX_scene',[r,d,y,s])]:
        cells={}
        keys=sorted(set(zip(*(a.tolist() for a in axes))),key=str)
        for key in keys:
            mask=np.ones(len(y),dtype=bool)
            for a,v in zip(axes,key):mask &= a==v
            cells['|'.join(map(str,key))]=classification(y[mask],p[mask],classes)
        result['groups'][name]=cells
    leo=np.isin(s,['leo_clear_weak','leo_low_elev_weak','leo_rain_weak'])
    for axis,name in [(r,'RX'),(y,'TX')]:
        means={}
        for val in np.unique(axis):
            metrics=[]
            for sc in ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
                m=(axis==val)&(s==sc)
                if m.any():metrics.append(float((y[m]==p[m]).mean()))
            if len(metrics)==3:means[str(val)]=float(np.mean(metrics))
        result['worst_'+name+'_leo_mean']=min(means.values()) if means else None
    risks=[1-v['accuracy'] for k,v in result['groups']['RX_scene'].items() if 'leo_' in k]
    result['worst_RX_scene_accuracy']=1-max(risks) if risks else None
    n=max(1,math.ceil(len(risks)*cvar_fraction))
    result['group_risk_cvar']=float(np.mean(sorted(risks,reverse=True)[:n])) if risks else None
    result['cvar_definition']=dict(unit='equally_weighted_RX_x_LEO_scene',worst_fraction=cvar_fraction,rounding='ceil')
    if reference is not None:
        ref=np.asarray(reference)
        if ref.shape!=y.shape:raise ValueError('reference alignment')
        rescue=int(((ref!=y)&(p==y)).sum());harm=int(((ref==y)&(p!=y)).sum())
        result['paired']=dict(rescue=rescue,harm=harm,net=rescue-harm)
    return result

def seed_contrasts(rows):
    """rows maps seed to same-metric values 00/10/01/11; no pseudo-replicates."""
    per_seed={str(seed):dict(joint_minus_DR=v['11']-v['10'],joint_minus_EG=v['11']-v['01'],
        interaction=v['11']-v['10']-v['01']+v['00']) for seed,v in rows.items()}
    return dict(per_seed=per_seed,summary={k:dict(mean=float(np.mean([v[k] for v in per_seed.values()])),
        sd=float(np.std([v[k] for v in per_seed.values()],ddof=1)) if len(per_seed)>1 else None)
        for k in ('joint_minus_DR','joint_minus_EG','interaction')},unit='paired_training_seed')

def wilson(success,total,z=1.959963984540054):
    if total==0:return [None,None]
    p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def route_quality(truth,route):
    # Caller may only use a legal source audit; never pass training U truth.
    y=np.asarray(truth);pseudo=route.pseudo.detach().cpu().numpy()
    candidate=route.candidate_mask.detach().cpu().numpy()
    answer={}
    for name,mask in [('H',route.hard),('P',route.partial),('N',route.negative)]:
        m=mask.detach().cpu().numpy();n=int(m.sum())
        success=int(((pseudo==y)&m).sum()) if name=='H' else int((candidate[np.arange(len(y)),y]&m).sum())
        if name=='N':success=n-success
        answer[name]=dict(selected=n,coverage=n/len(y),rate=success/n if n else None,
            confidence_interval=wilson(success,n),event='false_exclusion' if name=='N' else 'correct_or_set_contains_truth')
    answer['P']['mean_set_size']=float(candidate[route.partial.cpu().numpy()].sum(1).mean()) if route.partial.any() else None
    return answer
