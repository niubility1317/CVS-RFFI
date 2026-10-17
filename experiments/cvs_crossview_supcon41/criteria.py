"""Source-only fixed criterion;cannot substitute other gradient or target metrics."""
import math
MODES=('ORD128','DUP192','SAT192')
def decide(clean,urban):
 if set(clean)!=set(MODES) or set(urban)!=set(MODES):raise ValueError('Allthree frozen source rows required')
 for a in MODES:
  if type(clean[a])not in(int,float) or not math.isfinite(clean[a]) or not 0<=clean[a]<=1 or len(urban[a])!=2 or any(len(v)!=3 for v in urban[a]) or any(not math.isfinite(x) or not 0<=x<=1 for v in urban[a] for x in v):raise ValueError('Finite fullV and two originalthree-scene source draws required')
 deltas={a:[[100*(urban['SAT192'][draw][scene]-urban[a][draw][scene]) for scene in (1,2)] for draw in (0,1)] for a in MODES[:2]}
 checks=dict(clean_preserved_ORD=clean['SAT192']>=clean['ORD128']-.0025-1e-12,clean_preserved_DUP=clean['SAT192']>=clean['DUP192']-.0025-1e-12,allfour_urban_over_ORD=all(v>=1-1e-12 for row in deltas['ORD128'] for v in row),allfour_urban_over_DUP=all(v>=1-1e-12 for row in deltas['DUP192'] for v in row))
 return dict(status='SOURCE_DECIDED',source_qualified=all(checks.values()),checks=checks,clean_V=clean,urban_delta_pp=deltas,source_only=True,formal_scores_read=False,automatic_seed_extension=False,Ag_proved=False,goal_complete=False)
