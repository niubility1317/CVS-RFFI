"""Preregistered source protection;not target selection or a joint-success claim."""
import math
def decide(clean,urban,reference):
 if set(clean)!={'C0','R25'} or any(not math.isfinite(v) or not 0<=v<=1 for v in clean.values()) or len(urban)!=2 or len(reference)!=2 or any(len(x)!=2 for x in urban+reference) or any(not math.isfinite(v) or not 0<=v<=1 for a in urban+reference for v in a):raise ValueError('Full V and both original source urban draws required')
 delta=[[100*(x-y) for x,y in zip(a,b)] for a,b in zip(urban,reference)];cv=100*(clean['R25']-clean['C0']);checks=dict(clean_preserved=cv>=-.25-1e-10,each_urban_draw_preserved=all(v>=-2.-1e-10 for a in delta for v in a))
 return dict(status='SOURCE_DECIDED',source_only=True,formal_scores_read=False,clean_V=clean,clean_net_pp=cv,urban_delta_vs_AS50_pp=delta,checks=checks,source_protection_evidence=all(checks.values()),automatic_seed_extension=False,Ag_proved=False,goal_complete=False)
