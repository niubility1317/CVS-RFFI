"""Non-fitted six-class metrics and a fixed source-only falsification rule."""
import numpy as np
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
def metric(y,z):
 y=np.asarray(y);z=np.asarray(z)
 if y.ndim!=1 or z.shape!=(len(y),6) or not len(y) or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=6)) or not np.isfinite(z).all():raise ValueError('Finite original six-class logits and legal visible labels required')
 p=z.argmax(1);cm=np.bincount(y*6+p,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0)
 margins=(z[np.arange(len(y)),y]-np.max(np.where(np.arange(6)[None,:]==y[:,None],-np.inf,z),axis=1))/30.
 return dict(accuracy=float((p==y).mean()),macro_F1=float(f.mean()),CM=cm.tolist(),margin_scale30=float(margins.mean()))
def verdict(clean_delta,rx_delta,tx_delta,retention,urban_deltas,coherence_q10,relative_l2_median,undefined,require_urban=True):
 x=np.asarray(urban_deltas);r=np.asarray(rx_delta);t=np.asarray(tx_delta);v=np.asarray([clean_delta,retention,coherence_q10,relative_l2_median],float)
 if x.shape!=(2,2) or r.shape!=(5,) or t.shape!=(6,) or not np.isfinite(x).all() or not np.isfinite(r).all() or not np.isfinite(t).all() or not np.isfinite(v).all() or not isinstance(undefined,int) or undefined<0:raise ValueError('All fixed source cells and finite IQ summaries required')
 eps=1e-10;checks=dict(clean_protected=bool(clean_delta>=-.5-eps),every_RX_protected=bool(np.all(r>=-1-eps)),every_TX_protected=bool(np.all(t>=-1-eps)),clean_correct_retained=bool(retention>=.99-eps),periodic_reference_supported=bool(coherence_q10>=.8-eps),nontrivial_view=bool(relative_l2_median>=.01-eps),clean_phase_all_defined=undefined==0,urban_protected=bool(not require_urban or np.all(x>=-.5-eps)))
 return dict(status='PASS_SOURCE_FALSIFICATION_ONLY' if all(checks.values()) else 'FAIL_FIXED_CYCLE_ORDER',checks=checks,clean_delta_pp=float(clean_delta),RX_clean_delta_pp=r.tolist(),TX_clean_delta_pp=t.tolist(),clean_correct_retention=float(retention),urban_delta_pp=x.tolist(),coherence_q10=float(coherence_q10),relative_L2_median=float(relative_l2_median),clean_phase_undefined=undefined,urban_rule_applies=require_urban,training_updates=0,target_access=False,Ag_proved=False,goal_completed=False)
