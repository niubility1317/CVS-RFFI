"""Fixed source identification criteria,no formal performance input."""
import numpy as np
def decide(clean,timing_reference,timing_Ag,urban_reference,urban_joint):
 if set(clean)!=set(('B00','B10','B01','B11')):raise ValueError('All four source clean controls required')
 t0=np.asarray(timing_reference,float);t1=np.asarray(timing_Ag,float);u0=np.asarray(urban_reference,float);u1=np.asarray(urban_joint,float);cv=np.asarray(list(clean.values()),float)
 if t0.shape!=(2,) or t1.shape!=(2,) or u0.shape!=(2,2) or u1.shape!=(2,2) or not all(np.isfinite(v).all() and ((v>=0)&(v<=1)).all() for v in (t0,t1,u0,u1,cv)):raise ValueError('Finite paired source accuracy cells only')
 delta_t=100*(t1-t0);delta_u=100*(u1-u0)
 checks=dict(Ag_clean_V_preserved=100*(clean['B10']-clean['B00'])>=-.25-1e-10,joint_clean_V_preserved=100*(clean['B11']-clean['B00'])>=-.25-1e-10,both_private_timing_draws_gain=bool((delta_t>=5.-1e-10).all()),all_original_urban_draws_gain=bool((delta_u>=1.-1e-10).all()))
 return dict(status='SOURCE_DECIDED',checks=checks,source_replication_evidence=all(checks.values()),clean_V=clean,timing_delta_pp=delta_t.tolist(),urban_draw_delta_pp=delta_u.tolist(),source_only=True,formal_scores_read=False,automatic_seed_extension=False,goal_complete=False)
