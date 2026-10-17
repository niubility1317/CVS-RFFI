"""Fixed source test for C4; never accepts target metrics."""
from math import isfinite
def decide(clean,urban,teacher):
 if set(clean)!={'B0','URAW','UAS'} or set(urban)!=set(clean) or set(teacher)!=set(clean):raise ValueError('All3 mandatory source rows required')
 if any(len(urban[k])!=2 or any(len(row)!=3 or any(not isfinite(v) or not 0<=v<=1 for v in row) for row in urban[k]) or not isfinite(clean[k]) or not 0<=clean[k]<=1 for k in clean):raise ValueError('Exactly2draws/3scenes/finite source accuracies required')
 cells=[100*(a-b) for aa,bb in zip(urban['UAS'],urban['URAW']) for a,b in zip(aa[1:],bb[1:])]
 checks=dict(all_own_teachers_qualified=all(teacher.values()),UAS_clean_kept_vs_base=clean['UAS']>=clean['B0']-.0025,UAS_clean_kept_vs_raw=clean['UAS']>=clean['URAW']-.0025,each_urban_draw_gain_over_raw=all(v>=1.-1e-10 for v in cells))
 return dict(status='SOURCE_DECIDED',checks=checks,source_qualified=all(checks.values()),urban_deltas_pp=cells,clean=clean,automatic_seed_extension=False,formal_feedback=False,goal_complete=False)
