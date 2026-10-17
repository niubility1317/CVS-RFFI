"""Fixed source-only loss-ablation conditions; not IQ-Ag or target proof."""
import math
ARMS=('B00','H00','B01','H01')
def decide(clean,urban,rx):
 if any(set(x)!=set(ARMS) for x in (clean,urban,rx)):raise ValueError('Four fixed source controls/candidates required')
 for arm in ARMS:
  vals=[clean[arm]]
  if len(urban[arm])!=2 or any(len(x)!=3 for x in urban[arm]) or set(map(str,rx[arm]))!={'1','3','4','6','8'}:raise ValueError('All source draws/scenes/RX required')
  vals += [v for draw in urban[arm] for v in draw]+list(rx[arm].values())
  if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in vals):raise ValueError('Finite source accuracies required')
 deltas=[[urban['H01'][draw][s]-urban['B01'][draw][s] for s in (1,2)] for draw in range(2)]
 checks=dict(hard_clean_independent=clean['H00']-clean['B00']>=.002,
             joint_clean_net=clean['H01']>clean['B00'],
             joint_clean_preserves_As=clean['H01']>=clean['B01'],
             both_urban_each_draw=all(v>=.005 for draw in deltas for v in draw),
             suburban_preservation=all(urban['H01'][draw][0]>=urban['B01'][draw][0]-.0025 for draw in range(2)),
             each_source_RX_preserved=all(rx['H01'][str(r)]>=rx['B01'][str(r)]-.0025 for r in (1,3,4,6,8)))
 return dict(status='SOURCE_QUALIFIED' if all(checks.values()) else 'SOURCE_REJECTED',source_qualified=all(checks.values()),checks=checks,urban_differences=deltas,clean=clean,source_accuracies=urban,target_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False)
