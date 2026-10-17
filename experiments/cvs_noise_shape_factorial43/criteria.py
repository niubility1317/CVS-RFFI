"""Prospective source criterion,never reads target or frozen-head Ag invariance."""
import math
ARMS=('B00','B10_WHITE','B10_BG','B01','B11_WHITE','B11_BG')
def decide(clean,urban,rx):
 if set(clean)!=set(ARMS) or set(urban)!=set(ARMS) or set(rx)!=set(ARMS):raise ValueError('All6 source rows required')
 for a in ARMS:
  vals=[clean[a],*rx[a].values(),*(x for row in urban[a] for x in row)]
  if set(rx[a])!={'1','3','4','6','8'} or len(urban[a])!=2 or any(len(row)!=3 for row in urban[a]) or any(type(x)not in(int,float) or not math.isfinite(x) or not 0<=x<=1 for x in vals):raise ValueError('Finite allsource scalar metrics and fixed2x3 panel required')
 def allgain(a,b,delta):return all(urban[a][j][s]-urban[b][j][s]>=delta-1e-12 for j in (0,1) for s in (1,2))
 checks=dict(Ag_clean_gain=clean['B10_BG']>=clean['B00']+.002-1e-12,joint_clean_net_gain=clean['B11_BG']>=clean['B00']+.002-1e-12,joint_clean_over_As=clean['B11_BG']>=clean['B01']+.002-1e-12,As_independent=allgain('B01','B00',.03),As_with_Ag=allgain('B11_BG','B10_BG',.03),Ag_urban_joint=allgain('B11_BG','B01',.01),shape_over_WHITE=allgain('B11_BG','B11_WHITE',.005),shape_clean_preserved=clean['B11_BG']>=clean['B11_WHITE']-.001-1e-12,all_RX_clean_preserved=all(rx[a][r]>=rx[b][r]-.0025-1e-12 for a,b in [('B10_BG','B00'),('B11_BG','B00'),('B11_BG','B01')] for r in rx[a]),suburban_source_preserved=all(urban['B11_BG'][j][0]>=urban['B01'][j][0]-.0025-1e-12 for j in (0,1)))
 pairs=[('B10_BG','B00'),('B10_WHITE','B00'),('B11_BG','B01'),('B11_WHITE','B01'),('B11_BG','B11_WHITE')]
 return dict(status='SOURCE_DECIDED',source_qualified=all(checks.values()),checks=checks,clean_V=clean,clean_RX=rx,urban_delta_pp={a+'-'+b:[[100*(urban[a][j][s]-urban[b][j][s]) for s in (1,2)] for j in (0,1)] for a,b in pairs},source_interaction_pp=[[100*(urban['B11_BG'][j][s]-urban['B10_BG'][j][s]-urban['B01'][j][s]+urban['B00'][j][s]) for s in (1,2)] for j in (0,1)],source_only=True,formal_scores_read=False,automatic_seed_extension=False,Ag_proved=False,goal_complete=False)
