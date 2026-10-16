"""Saved scalar dot-product interpretation; never reads IQ, labels or models."""
import math,statistics

def measure(p2,s2,dot):
 vals=(p2,s2,dot)
 if any(not math.isfinite(v) for v in vals) or p2<0 or s2<0:raise ValueError('Finite nonnegative squared gradient norms required')
 bound=math.sqrt(p2*s2)
 if abs(dot)>bound+1e-8*max(1.,bound):raise ValueError('Gradient dot exceeds Cauchy bound')
 return dict(primary_norm=math.sqrt(p2),secondary_norm=math.sqrt(s2),norm_ratio=math.sqrt(s2/p2) if p2>0 else None,
  cosine=max(-1.,min(1.,dot/bound)) if bound>0 else None,opposed=dot<0,primary_direction_reversed=p2+dot<0,
  joint_norm=math.sqrt(max(0.,p2+s2+2*dot)),primary_squared=p2,secondary_squared=s2,dot=dot)

def decision(rows):
 whole=[r for r in rows if r['group']=='whole'];groups={}
 for model in ('C0','C1'):
  for rep in (0,1):
   subset=[r for r in whole if r['checkpoint']==model and r['rep']==rep]
   ctr=[r for r in subset if r['secondary']=='same_ordinary'];sat=[r for r in subset if r['secondary']=='As']
   if len(ctr)!=40 or len(sat)!=40 or {r['batch'] for r in ctr}!={r['batch'] for r in sat}:raise ValueError('Both complete matched 40batch controls required')
   ratios=[r['norm_ratio'] for r in sat]
   if any(v is None for v in ratios):raise ValueError('Primary zero norm prevents stated dominance test')
   c=sum(r['opposed'] for r in ctr)/40.;s=sum(r['opposed'] for r in sat)/40.;ratio=statistics.median(ratios)
   groups[f'{model}/r{rep}']=dict(median_As_primary_norm_ratio=ratio,As_opposed_fraction=s,control_opposed_fraction=c,opposed_excess=s-c,pass_rule=ratio>1 and s-c>=.20-1e-12)
 return dict(status='SUPPORTED_FOR_SEPARATE_SOURCE_EXPERIMENT_ONLY' if all(r['pass_rule'] for r in groups.values()) else 'RATIONALE_REJECTED',groups=groups,
  gradient_geometry_is_not_accuracy=True,actual_AdamW_motion_measured=False,automatic_training=False,goal_completed=False)
