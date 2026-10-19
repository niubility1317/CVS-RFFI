"""Label-equivariant bounded weights from frozen source-validation counts."""
import math

def calibrate(class_counts,receiver_counts,beta=.5):
 n=len(class_counts)
 if n!=6 or len(receiver_counts)!=5 or not 0<=beta<=.5:raise ValueError('Original sixTX/fiveRX source calibration and bounded beta required')
 def err(pair):
  hit,total=pair
  if not isinstance(hit,int) or not isinstance(total,int) or total<1 or not 0<=hit<=total:raise ValueError('Positive source-only class counts required')
  return 1.-hit/total
 if any(len(r)!=n for r in receiver_counts):raise ValueError('Complete all-class source RX cells required')
 global_error=[err(v) for v in class_counts]
 worst_error=[max(err(r[c]) for r in receiver_counts) for c in range(n)]
 risk=[.5*(a+b) for a,b in zip(global_error,worst_error)]
 center=sum(risk)/n;spread=max(abs(v-center) for v in risk)
 weights=[1.+beta*(v-center)/spread for v in risk] if spread>1e-12 else [1.]*n
 if any(not math.isfinite(v) or v<1-beta-1e-12 or v>1+beta+1e-12 for v in weights) or abs(sum(weights)-n)>1e-10:raise ValueError('Bounded mean-one class weights required')
 return dict(global_error=global_error,worst_source_RX_error=worst_error,risk=risk,weights=weights,beta=beta,formula='r=.5*global_error+.5*worst_source_RX_error; w=1+beta*(r-mean(r))/max(abs(r-mean(r))); equal-risk ->1',calibration_role='sourceV only',target_feedback=False)
