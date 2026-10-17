"""Fixed pre-receiver source guard and native replay checks; no model fitting."""
import numpy as np
def replay_guard(actual,cached,kernel0,kernel1,g0,g1):
 if actual.shape!=(256,) or cached.shape!=(256,) or not np.isfinite(actual).all() or not np.isfinite(cached).all():raise ValueError('Full finite original256 replay required')
 if not np.allclose(actual,cached,atol=2e-6,rtol=2e-6):raise ValueError('Native original-input replay differs from cached source IQ')
 if kernel0.shape!=kernel1.shape or not np.allclose(kernel0,kernel1,atol=1e-12,rtol=1e-12):raise ValueError('Same latent propagation kernel required')
 if g0.shape!=(1,) or g1.shape!=(1,) or not np.array_equal(g0,np.ones(1)) or not np.array_equal(g1,np.ones(1)):raise ValueError('EQ is off, G must equal1')
 return float(np.max(np.abs(actual-cached)))
def source_rule(clean_ok,urban_delta,arm):
 x=np.asarray(urban_delta)
 if arm not in ('B00','B01') or not isinstance(clean_ok,bool) or x.shape!=(2,2) or not np.isfinite(x).all():raise ValueError('Fixed complete source cells required')
 accepted=x>=-.5-1e-10
 return dict(status='PASS_SOURCE_PLACEMENT_ONLY' if clean_ok and (arm=='B00' or accepted.all()) else 'FAIL_FIXED_PRE_RECEIVER_CYCLE_ORDER',prior_clean_guard=clean_ok,urban_rule_applies=arm=='B01',urban_delta_pp=x.tolist(),protected_cells=int(accepted.sum()),cells_total=4,target_access=False,training_updates=0,Ag_proved=False,goal_completed=False)
