import sys
import json
from pathlib import Path
import numpy as np
root=Path('E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt')
sys.path.insert(0,str(root/'code'))
from cvsrffi import d92_prototype_transport_local_ridge as pt
rng=np.random.default_rng(81);labels=np.repeat(np.arange(3),3)
raw={key:rng.normal(size=(9,d))+.35*rng.normal(size=(3,d))[labels] for key,d in zip(pt._NAMES,(160,96,160,160,160))}
args=dict(**raw,support_labels=labels,support_ids=tuple(f'physical-{c}-{i:02}' for c in range(3) for i in range(3)),classes=('c0','c1','c2'),old_classes=('c0','c1','c2'))
p=pt.prepare_prototype_transport_training(**args)
f=p.full_problem;table=f.prototypes;theta=np.array([.1,-.08,.06,.04,-.03,.02,-.03,.04,-.02,-.01])
whole=pt._map_context(f.original.train_b,f.original.train_a,table)
parts=[pt._map_context(f.original.train_b[i:i+1],f.original.train_a[i:i+1],table) for i in range(9)]
def cmp(a,b):return dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0)
result={'fixture':'synthetic seed81; no optimization or real data','theta':theta.tolist(),'context':{k:cmp(whole[k],np.concatenate([part[k] for part in parts])) for k in whole}}
def attention(ctx):
    weighted=np.sum(ctx['d']*np.exp(theta[5:]),axis=2)
    nearest=np.min(np.where(ctx['neighbors'],weighted,np.inf),axis=1)
    logits=np.where(ctx['neighbors'],-(weighted-nearest[:,None])/table.nu,-np.inf)
    att=np.exp(logits);att/=att.sum(axis=1)[:,None]
    return weighted,att
weighted,att=attention(whole);pieces=[attention(c) for c in parts]
result['weighted_distance']=cmp(weighted,np.concatenate([a[0] for a in pieces]))
result['attention']=cmp(att,np.concatenate([a[1] for a in pieces]))
old_batch_mu=att@table.m
old_single_mu=np.concatenate([a[1]@table.m for a in pieces])
result['original_GEMM_vs_single_GEMV_mu']=cmp(old_batch_mu,old_single_mu)
mb,ma,cache=pt._adapt(whole,table,theta,derivative_cache=True)
singles=[pt._adapt(c,table,theta,derivative_cache=True) for c in parts]
result['patched_mu']=cmp(cache['mu'],np.concatenate([r[2]['mu'] for r in singles]))
result['patched_background']=cmp(mb,np.concatenate([r[0] for r in singles]))
result['patched_auxiliary']=cmp(ma,np.concatenate([r[1] for r in singles]))
print(json.dumps(result,ensure_ascii=False,indent=2))
