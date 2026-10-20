"""Source-only prospective cross-RX contrast-direction premise; cached BASE3 only."""
from pathlib import Path
import itertools,json
import numpy as np
RUN='20261020-diagnostic-FcrossRX-weakPair-direction-sourceV-m3-local-r01'
CACHE_RUN='20261019-diagnostic-FcrossRX-feature-geometry-sourceV-m3-local-r01'
REFERENCE_RUN='20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01'
REFERENCE_COMMIT='31467086c3ced30237f3ac86e91eae68645425d7'
SEEDS=(2026092701,2026092702,2026092703)
CLASSES=('14-10','14-7','20-15','20-19','6-15','8-20')
RX=(1,3,4,6,8)
WEAK=(1,3)
MAX_WEAK_MEAN_COS=.90
MIN_RELATIVE_COS_DEFICIT=.05
EPS=1e-6

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def scope(role,seed,cache_name):
 if role!='V' or seed not in SEEDS or cache_name!=CACHE_RUN:raise ValueError('Only fixed all3 BASE sourceV cache')
def metadata(m,seed):
 expected=dict(row_id='BASE-s'+str(seed),model_seed=seed,status='SOURCE_GEOMETRY_COMPLETE',source_role='V',count=27000,head_fit_updates=0,optimizer_updates=0,prototype_classifier_decisions=0,target_reads=0,truth_reads=0,U_TX_reads=0,model_state_unchanged=True,source_code_commit=REFERENCE_COMMIT)
 if any(m.get(k)!=v for k,v in expected.items()) or m['geometry']['class_names']!=list(CLASSES):raise ValueError('Existing validated cache provenance differs')
def centers(features,y,rx,day):
 f=np.asarray(features,dtype=np.float64)
 if f.shape!=(27000,160) or y.shape!=rx.shape or y.shape!=day.shape or y.shape!=(27000,) or not np.isfinite(f).all() or set(y)!=set(range(6)) or set(rx)!=set(RX) or set(day)!={1,2,3}:raise ValueError('Existing source feature array interface differs')
 norm=np.linalg.norm(f,axis=1)
 if np.any(norm<=EPS):raise ValueError('Zero source feature cannot define a contrast direction')
 z=f/norm[:,None];out=np.empty((6,5,160),dtype=np.float64)
 for c in range(6):
  for r,k in enumerate(RX):
   mask=(y==c)&(rx==k)
   if mask.sum()!=900 or any(np.sum(mask&(day==d))!=300 for d in (1,2,3)):raise ValueError('Fixed class/RX/day source cache groups differ')
   out[c,r]=z[mask].mean(0)
 return out

def pair_stats(mu,c,d):
 if mu.ndim!=3 or mu.shape[:2]!=(6,5) or not np.isfinite(mu).all():raise ValueError('Finite 6class x5RX class means required')
 delta=mu[c]-mu[d];norm=np.linalg.norm(delta,axis=1);valid=norm>EPS;angles=[];dots=[]
 for i,j in itertools.combinations(range(5),2):
  if valid[i] and valid[j]:angles.append(dict(RX_i=RX[i],RX_j=RX[j],cosine=float(np.clip(delta[i]@delta[j]/(norm[i]*norm[j]),-1.,1.))))
  dots.append(float(delta[i]@delta[j]))
 self_energy=float(np.mean(norm**2));cross=float(np.mean(dots))
 return dict(pair=[CLASSES[c],CLASSES[d]],indices=[c,d],valid_RX_directions=int(valid.sum()),valid_RX_pairs=len(angles),mean_direction_cosine=float(np.mean([v['cosine'] for v in angles])) if angles else None,min_direction_cosine=min((v['cosine'] for v in angles),default=None),self_energy=self_energy,cross_RX_dot=cross,cross_to_self=cross/self_energy if self_energy>EPS**2 else None,direction_norm_by_RX={str(r):float(v) for r,v in zip(RX,norm)},RX_pair_cosines=angles)

def analyze(mu,seed):
 stats=[pair_stats(mu,c,d) for c,d in itertools.combinations(range(6),2)];weak=next(v for v in stats if v['indices']==list(WEAK));others=[v['mean_direction_cosine'] for v in stats if v['indices']!=list(WEAK) and v['valid_RX_pairs']==10];med=float(np.median(others)) if len(others)==14 else None
 weak_cos=weak['mean_direction_cosine'];deficit=med-weak_cos if med is not None and weak_cos is not None else None
 qualified=weak['valid_RX_pairs']==10 and len(others)==14 and weak_cos<=MAX_WEAK_MEAN_COS and deficit>=MIN_RELATIVE_COS_DEFICIT
 return dict(model_seed=seed,pairs=stats,weak_pair=weak,median_other_pair_direction_cosine=med,weak_cosine_deficit_to_other_pair_median=deficit,prospective_direction_route_qualified=bool(qualified))
def decision(records):
 if [r['model_seed'] for r in records]!=list(SEEDS):raise ValueError('All fixed3 source seeds required')
 return dict(PASS=all(r['prospective_direction_route_qualified'] for r in records),rule='All3 BASE seeds: weakpair mean crossRX direction cosine<=.90 and >=.05 below median other14 pairs; all10 RXpairs valid in all15 classpairs. Fixed before this analysis; do not relax after result.',training_run_launched=False,target_performance_measured=False)
