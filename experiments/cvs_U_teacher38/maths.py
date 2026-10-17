"""No fitted classifier, calibration or learned diagnostic state."""
import numpy as np

def probabilities(logits):
 z=np.asarray(logits,dtype=np.float64)
 if z.ndim!=2 or z.shape[1]!=6 or not np.isfinite(z).all():raise ValueError('Finite original six-class cosine logits required')
 z=z-z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def teacher_summary(teacher,student,satellite,y,rx,rule):
 p=probabilities(teacher);s=probabilities(student);a=np.asarray(satellite)
 y=np.asarray(y);rx=np.asarray(rx);n=len(y)
 if y.shape!=(n,) or not np.issubdtype(y.dtype,np.integer) or not ((y>=0)&(y<6)).all() or rx.shape!=(n,) or set(rx.tolist())!={1,3,4,6,8} or s.shape!=(n,6) or a.shape!=(2,3,n,6) or not np.isfinite(a).all():raise ValueError('Only fixed legal source V panel/2draw/3scenes')
 conf=p.max(1);pred=p.argmax(1);q=np.clip((conf-rule['tau'])/(1-rule['tau']),0,1);mask=q>0;correct=pred==y
 precision=float(correct[mask].mean()) if mask.any() else None
 groups={}
 for r in (1,3,4,6,8):
  m=mask&(rx==r);groups[str(r)]=dict(n=int((rx==r).sum()),selected=int(m.sum()),precision=float(correct[m].mean()) if m.any() else None)
 opportunity=np.empty((2,3));contradiction=np.empty((2,3));acc=np.empty((2,3))
 for d in range(2):
  for scene in range(3):
   right=a[d,scene].argmax(1)==y;acc[d,scene]=right.mean();den=q.sum()
   opportunity[d,scene]=np.sum(q*correct*~right)/den if den else 0.
   contradiction[d,scene]=np.sum(q*~correct*right)/den if den else 0.
 checks=dict(teacher_precision=precision is not None and precision>=rule['min_precision'],teacher_coverage=float(mask.mean())>=rule['min_coverage'],each_RX_precision=all(v['precision'] is not None and v['precision']>=rule['min_RX_precision'] for v in groups.values()),both_draws_mid_low_opportunity=bool((opportunity[:,1:]>=rule['min_transfer_opportunity']).all()))
 return dict(teacher_accuracy=float(correct.mean()),student_clean_accuracy=float((s.argmax(1)==y).mean()),student_satellite_accuracy=acc.tolist(),coverage=float(mask.mean()),confidence_precision=precision,q_weighted_precision=float(np.sum(q*correct)/q.sum()) if q.sum() else None,q_mean=float(q.mean()),selected=int(mask.sum()),per_RX=groups,transfer_opportunity=opportunity.tolist(),harmful_teacher_opportunity=contradiction.tolist(),checks=checks,source_qualified=all(checks.values()),thresholds=rule),q
