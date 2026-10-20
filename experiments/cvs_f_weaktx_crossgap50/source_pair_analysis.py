import numpy as np,json,itertools,sys
from pathlib import Path
root=Path(sys.argv[1]);classes=['14-10','14-7','20-15','20-19','6-15','8-20'];rows=[]
def cos(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if min(np.linalg.norm(a),np.linalg.norm(b))>1e-10 else None
def stats(v):return dict(mean=float(np.mean(v)),min=float(np.min(v)),max=float(np.max(v)))
for seed in [2026092701,2026092702,2026092703]:
 p=root/('BASE-s'+str(seed));meta=json.loads((p/'source_geometry.json').read_text())
 if meta.get('model_seed')!=seed or meta.get('source_code_commit')!='31467086c3ced30237f3ac86e91eae68645425d7' or meta['source_role']!='V' or meta['count']!=27000 or not meta['model_state_unchanged'] or meta['target_reads'] or meta['head_fit_updates']:raise ValueError('Existing validated sourceV cache only')
 with np.load(p/'source_features.npz') as a:z=a['features'].astype(np.float64);y=a['y'];rx=a['rx'];day=a['day']
 if z.shape!=(27000,160) or not np.isfinite(z).all() or set(np.unique(y))!=set(range(6)) or set(np.unique(rx))!={1,3,4,6,8} or set(np.unique(day))!={1,2,3}:raise ValueError('Expected full cached sourceV')
 z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);rs=[1,3,4,6,8];ds=[1,2,3];mu={(c,r):z[(y==c)&(rx==r)].mean(0) for c in range(6) for r in rs};mud={(c,r,d):z[(y==c)&(rx==r)&(day==d)].mean(0) for c in range(6) for r in rs for d in ds};pairs=[]
 for a,b in itertools.combinations(range(6),2):
  delta={r:mu[a,r]-mu[b,r] for r in rs};cross=[cos(delta[r],delta[s]) for r,s in itertools.combinations(rs,2)];daywithin=[cos(mud[a,r,d]-mud[b,r,d],mud[a,r,e]-mud[b,r,e]) for r in rs for d,e in itertools.combinations(ds,2)];bydaycross=[cos(mud[a,r,d]-mud[b,r,d],mud[a,s,d]-mud[b,s,d]) for d in ds for r,s in itertools.combinations(rs,2)]
  pairs.append(dict(indices=[a,b],TX=[classes[a],classes[b]],cross_RX_direction_cosine=stats(cross),same_RX_cross_day_direction_cosine=stats(daywithin),same_day_cross_RX_direction_cosine=stats(bydaycross),per_RX_gap=[dict(RX=r,norm=float(np.linalg.norm(delta[r]))) for r in rs]))
 ranked=sorted(pairs,key=lambda v:v['cross_RX_direction_cosine']['mean'])
 for i,v in enumerate(ranked):v['ascending_crossRX_consistency_rank']=i+1
 rows.append(dict(seed=seed,pairs=ranked,weak_pair=next(v for v in pairs if v['indices']==[1,3])))
print(json.dumps(dict(status='SOURCE_CACHED_FEATURE_ANALYSIS',source_role='V',model_seeds=[2026092701,2026092702,2026092703],unique_sourceV=27000,source_feature_rows=81000,model_forwards=0,optimizer_updates=0,head_fit_updates=0,U_truth_reads=0,target_reads=0,rows=rows,interpretation='Relative class-centroid directions in actual unit joint160, descriptive seen-sourceRX geometry; no fitted classifier, no unknownRX accuracy or causal improvement claim.'),allow_nan=False))
