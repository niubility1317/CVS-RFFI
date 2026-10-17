"""Non-fitted six-class metrics and a fixed source-only falsification rule."""
import numpy as np
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def metric(y,z):
 y=np.asarray(y);z=np.asarray(z)
 if y.ndim!=1 or z.shape!=(len(y),6) or not len(y) or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=6)) or not np.isfinite(z).all():raise ValueError('Finite original six-class logits and legal visible labels required')
 p=z.argmax(1);cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 margins=(z[np.arange(len(y)),y]-np.max(np.where(np.arange(6)[None,:]==y[:,None],-np.inf,z),axis=1))/30.
 return dict(accuracy=float((p==y).mean()),macro_F1=float(f.mean()),CM=cm.tolist(),margin_scale30=float(margins.mean()))
def verdict(delta_clean,rx_delta,urban_deltas):
 x=np.asarray(urban_deltas);r=np.asarray(rx_delta)
 if x.shape!=(2,2) or r.shape!=(5,) or not np.isfinite(x).all() or not np.isfinite(r).all() or not np.isfinite(delta_clean):raise ValueError('All fixed source cells required')
 eps=1e-10 # Numerical tolerance in percentage points, not a scientific threshold change.
 accepted=x>=1.-eps
 checks=dict(clean_preserved=bool(delta_clean>=-eps),all_RX_clean_preserved=bool(np.all(r>=-.5-eps)),urban_all_four_gain=bool(np.all(accepted)))
 return dict(status='PASS_SOURCE_READOUT_ONLY' if all(checks.values()) else 'FAIL_FIXED_FINAL30_EMA_READOUT',checks=checks,delta_clean_pp=float(delta_clean),RX_clean_delta_pp=r.tolist(),urban_delta_pp=x.tolist(),cells_passed=int(accepted.sum()),cells_total=4,automatic_target_access=False,automatic_seed_extension=False,Ag_proved=False,goal_completed=False)
