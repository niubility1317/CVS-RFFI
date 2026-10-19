"""Cached source-only mathematical probe; no model, IQ, optimizer or target access."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'local_artifacts/F_sameIQ_sensitivity_20261019/complete_analysis/source_derivatives.npz'
data=np.load(p,allow_pickle=False);y=data['TX'];eps=.01;alpha=(eps/6)/(1-eps+2*eps/6)
rows={}
for name in ('clean','d0_mid','d0_low','d1_mid','d1_low'):
 z=np.asarray(data[name+'_student'],dtype=np.float64);mask=np.arange(6)[None,:]!=y[:,None];gap=z-z[np.arange(len(y)),y,None]
 pz=np.exp(z-z.max(1,keepdims=True));pz/=pz.sum(1,keepdims=True)
 target=np.full_like(z,eps/6);target[np.arange(len(y)),y]+=1-eps
 cegrad=pz-target
 pairgrad=(1/(1+np.exp(-np.clip(gap,-700,700)))-alpha)/5;pairgrad[~mask]=0;pairgrad[np.arange(len(y)),y]=-pairgrad.sum(1)
 mixed=.5*(cegrad+pairgrad);cn=np.linalg.norm(cegrad,axis=1);mn=np.linalg.norm(mixed,axis=1)
 cos=(mixed*cegrad).sum(1)/(cn*mn);wrong=z.argmax(1)!=y;impostors=((gap>0)&mask).sum(1)
 rows[name]=dict(n=len(y),accuracy=float((~wrong).mean()),wrong_count=int(wrong.sum()),wrong_two_or_more_impostors=int((wrong&(impostors>=2)).sum()),impostor_count_histogram=np.bincount(impostors,minlength=6).tolist(),mixed_vs_CE_cosine_percentiles=np.percentile(cos,[0,10,50,90,100]).tolist(),wrong_cosine_percentiles=np.percentile(cos[wrong],[0,10,50,90,100]).tolist(),matched_CE_scale_percentiles=np.percentile(mn/cn,[0,10,50,90,100]).tolist(),wrong_mean_angle_degrees=float(np.rad2deg(np.arccos(np.clip(cos[wrong],-1,1))).mean()),gradient_direction_identical=bool(np.allclose(mixed,cegrad*(mn/cn)[:,None],atol=1e-10,rtol=0)))
out=dict(status='CACHED_SOURCE_MATHEMATICAL_PROBE',source='fixed old B1 FINAL20 V360 two-draw logits; no new source inference',proposal='As ordinary CE plus conditional-smoothed pairwise logistic CE; preserve positive ordinary multiclass CE; compare direction against per-packet gradient-norm-matched multiclass CE',conditional_binary_false_probability=alpha,rows=rows,new_NN=0,new_IQ=0,new_updates=0,target_access=False,teacher_outputs_used=False,performance_claim=False)
destination=ROOT/'local_artifacts/F_cleanLookahead_source_20261019/pair_direction_evidence.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],wrong_mean_angle_degrees={k:v['wrong_mean_angle_degrees'] for k,v in rows.items()},new_NN=0,new_IQ=0,new_updates=0,target_access=False,performance_claim=False),ensure_ascii=False))
