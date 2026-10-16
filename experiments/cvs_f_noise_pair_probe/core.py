"""Unbiased trace estimators; independent reference reuses marginal samples."""
import math
GROUPS=('default_head','SIG_TIME','Sinc_Hz','time_path','remaining')
def group(name):
 if '.cls_head.' in name:return 'default_head'
 if '.sequence_pool.' in name:return 'SIG_TIME'
 if '.sinc.' in name:return 'Sinc_Hz'
 if any('.'+s+'.' in name for s in ('time_stem','time_down','t1','t2')):return 'time_path'
 return 'remaining'
def statistics(plus,minus):
 import torch
 if plus.ndim!=2 or minus.shape!=plus.shape or len(plus)<2 or plus.shape[1]<1 or not bool(torch.isfinite(plus).all() and torch.isfinite(minus).all()):raise ValueError('Finite paired derivative matrices required')
 a=plus.double();b=minus.double();a=a-a.mean(0);b=b-b.mean(0);den=len(a)-1
 vp=float(a.square().sum()/den);vm=float(b.square().sum()/den);cov=float((a*b).sum()/den);vi=(vp+vm)/4;va=float(((a+b)/2).square().sum()/den)
 if abs(va-(vi+cov/2))>1e-10*max(1.,va,vi):raise ArithmeticError('Variance/covariance identity failed')
 raw=torch.cat((plus.double(),minus.double()))
 gram=(raw@raw.T).tolist()
 return dict(draws=len(a),Vplus=vp,Vminus=vm,noise_covariance=cov,independent_pair_trace=vi,antithetic_pair_trace=va,trace_ratio=va/vi if vi>0 else None,raw_derivative_gram=gram)
def aggregate(rows):
 if not rows or any(r['group']!='whole' for r in rows):raise ValueError('All whole-model batch records required')
 va=sum(r['antithetic_pair_trace'] for r in rows);vi=sum(r['independent_pair_trace'] for r in rows)
 if not math.isfinite(va+vi) or va<0 or vi<=0:raise ValueError('Positive finite reference variance required')
 negative=sum(r['noise_covariance']<0 for r in rows)/len(rows)
 return dict(batches=len(rows),summed_antithetic_trace=va,summed_independent_trace=vi,trace_ratio=va/vi,negative_covariance_fraction=negative,qualified_rationale=va/vi<=.80 and negative>=.60,performance_improvement_proved=False)
