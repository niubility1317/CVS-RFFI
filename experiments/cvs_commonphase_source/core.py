"""Input scope, unit phase operation and fixed accuracy-gap decision."""
import math
import torch
from . import design as d

def rotate(x,angle):
    if angle not in d.VIEWS.values() or x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError('Fixed finite IQ256/phase required')
    if angle==0:return x.clone()
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((x[:,0]*c-x[:,1]*s,x[:,0]*s+x[:,1]*c),dim=1)

def select_fit(L,carrier,trained):
    ids=L['ids'];roles=carrier['role_ids']
    if len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(roles['L_s']) or set(ids)&(set(roles['U_s'])|set(roles['V'])):raise ValueError('Original disjoint L6300 handles required')
    if L['x'].shape!=(6300,2,256) or L['x'].dtype!=torch.float32 or any(len(L[k])!=6300 for k in ('y','rx','day')):raise ValueError('Native float32 source cache dimensions differ')
    mask=L['rx']!=1
    selected={k:L[k][mask].clone() for k in ('y','rx','day')}
    selected['ids']=[value for value,keep in zip(ids,mask.tolist()) if keep]
    if len(selected['ids'])!=5040 or len(set(selected['ids']))!=5040 or set(selected['ids'])!=set(trained['role_ids']['L_s']) or set(selected['rx'].tolist())!=set(d.FIT):raise ValueError('Exact fourRX model-fit L5040 required')
    for r in d.FIT:
        for day in (1,2,3):
            for tx in range(6):
                if int(((selected['rx']==r)&(selected['day']==day)&(selected['y']==tx)).sum())!=70:raise ValueError('Original visible L cell coverage differs')
    selected['x']=L['x'][mask].clone()
    if not bool(torch.isfinite(selected['x']).all()):raise ValueError('Finite fit-L IQ required')
    return selected

def score(labels,predictions):
    if len(labels)!=len(predictions) or not labels or any(v not in range(6) for v in labels+predictions):raise ValueError('Complete six-class decisions required')
    cm=[[0]*6 for _ in range(6)]
    for y,p in zip(labels,predictions):cm[y][p]+=1
    f1=[]
    for i in range(6):
        tp=cm[i][i];den=sum(cm[i])+sum(row[i] for row in cm);f1.append(2*tp/den if den else 0.)
    return dict(count=len(labels),accuracy=sum(cm[i][i] for i in range(6))/len(labels),macro_F1=sum(f1)/6,confusion=cm)

def qualify(scores):
    if set(scores)!=set(d.VIEWS):raise ValueError('All three fixed source views required')
    base=scores['identity'];changes={};conditions={}
    for view in ('phase_plus','phase_minus'):
        v=scores[view];drop=100*(base['accuracy']-v['accuracy'])
        byrx={str(r):100*(base['by_RX'][str(r)]['accuracy']-v['by_RX'][str(r)]['accuracy']) for r in d.FIT}
        changes[view]=dict(accuracy_drop_pp=drop,RX_drop_pp=byrx,RX_at_least1pp=sum(x>=1. for x in byrx.values()))
        conditions[view+'_overall']=drop>=2.
        conditions[view+'_RX']=changes[view]['RX_at_least1pp']>=3
    return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,changes=changes,Ag_efficacy_proved=False,automatic_training=False,joint_goal_complete=False)
