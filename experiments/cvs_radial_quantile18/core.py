"""Centered log-amplitude quantiles and fixed L-only transfer controls."""
import numpy as np
from . import design as d
Q=np.array([.1,.2,.3,.7,.8,.9],dtype=np.float64)
def observe(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=3 or x.shape[1:]!=(2,256) or not np.isfinite(x).all():raise ValueError('Finite real IQ[N,2,256] required')
 power=np.sum(x*x,axis=1);rms=np.sqrt(power.mean(1));zero=rms==0;amplitude=np.sqrt(power)/np.where(zero,1.,rms)[:,None];floor=amplitude<1e-6;log=np.log(np.maximum(amplitude,1e-6));center=np.quantile(log,.5,axis=1);shape=np.quantile(log,Q,axis=1).T-center[:,None]
 return dict(shape=shape,amplitude_floor_fraction=floor.mean(1),zero_RMS=zero)
def cube(shape,meta,per_cell):
 x=np.asarray(shape,dtype=np.float64)
 if x.shape!=(6*5*3*per_cell,6) or not np.isfinite(x).all():raise ValueError('Complete source quantile matrix required')
 y,rx,day=(np.asarray(meta[k]) for k in ('y','rx','day'));out=np.empty((6,5,3,6));cells=[]
 for t in range(6):
  for j,r in enumerate(d.RX):
   for k,z in enumerate((1,2,3)):
    mask=(y==t)&(rx==r)&(day==z)
    if int(mask.sum())!=per_cell:raise ValueError('All90original source cells required')
    out[t,j,k]=x[mask].mean(0);cells.append(dict(TX=t,RX=r,day=z,count=per_cell))
 return out,cells
def slope(x,y):
 den=float(np.sum(x*x))
 if den<=1e-20 or not np.isfinite(den):raise ValueError('Nondegenerate L-only reference shape required')
 return float(np.sum(x*y))/den
def evaluate(L,V,quality=None):
 L,V=(np.asarray(x,dtype=np.float64) for x in (L,V))
 if L.shape!=(6,5,3,6) or V.shape!=L.shape or not np.isfinite(L).all() or not np.isfinite(V).all():raise ValueError('Fixed allTX/RX/day cells required')
 pred={k:V.copy() for k in ('shared','identity','additive','TXconditional')};a=np.ones((6,5,3));ac=a.copy();b=np.zeros_like(L);folds=[]
 for t in range(6):
  tx=[i for i in range(6) if i!=t]
  for day in range(3):
   days=[i for i in range(3) if i!=day]
   for r in range(1,5):
    ref=L[np.ix_(tx,[0],days,range(6))].reshape(-1,6);dst=L[np.ix_(tx,[r],days,range(6))].reshape(-1,6);a[t,r,day]=slope(ref,dst);ac[t,r,day]=slope(L[t,0,days],L[t,r,days]);b[t,r,day]=(dst-ref).mean(0)
    pred['shared'][t,r,day]=a[t,r,day]*V[t,0,day];pred['identity'][t,r,day]=V[t,0,day];pred['additive'][t,r,day]=V[t,0,day]+b[t,r,day];pred['TXconditional'][t,r,day]=ac[t,r,day]*V[t,0,day]
   errors={key:float(np.sum((value[t,1:,day]-V[t,1:,day])**2)) for key,value in pred.items()};folds.append(dict(TX=t,day=day+1,SSE=errors))
 errors={key:float(np.sum((value[:,1:]-V[:,1:])**2)) for key,value in pred.items()};per_rx={str(d.RX[r]):{key:float(np.sum((value[:,r]-V[:,r])**2)) for key,value in pred.items()} for r in range(1,5)};n=6*4*3*6;tol=1e-20;positive=sum(f['SSE']['shared']<f['SSE']['identity'] for f in folds);params=a[:,1:];q=quality or dict(max_floor_fraction_each_cell=0.,zero_RMS_packets=0)
 conditions=dict(nonzero_identity_error=bool(np.sqrt(errors['identity']/n)>=.01),shared_vs_identity=errors['shared']<=.8*errors['identity']+tol,shared_vs_additive=errors['shared']<=.9*errors['additive']+tol,positive_folds=positive>=15,each_nonreference_RX=all(v['shared']<=min(v['identity'],v['additive'])+tol for v in per_rx.values()),not_worse_TXconditional=errors['shared']<=1.05*errors['TXconditional']+tol,common_slope_mild=bool(np.all((params>=.9)&(params<=1.1))),valid_log_shape=q['max_floor_fraction_each_cell']<=.01 and q['zero_RMS_packets']==0)
 passed=all(conditions.values());result=dict(decision='SOURCE_RADIAL_RELATION_ONLY' if passed else 'CLOSED_FIXED_RADIAL_QUANTILE_HYPOTHESIS',source_pass=passed,conditions=conditions,SSE=errors,baseline_RMS=float(np.sqrt(errors['identity']/n)),shared_relative_identity_gain=1-errors['shared']/errors['identity'] if errors['identity']>tol else None,shared_relative_additive_gain=1-errors['shared']/errors['additive'] if errors['additive']>tol else None,positive_folds=positive,folds=folds,per_RX_SSE=per_rx,reference_RX=dict(RX=1,descriptive_only=True,prediction_error=0.,not_counted_as_positive_domain=True),slope_min=float(params.min()),slope_max=float(params.max()),L_only_scalar_fits=144,L_only_additive_vectors=72,predicted_nonreference_cells=72,predicted_nonreference_coordinates=432,source_RX_holdout=None,IQ_operator_implemented=False,label_preservation_proved=False,Ag_proved=False,automatic_training=False,joint_goal_complete=False)
 return result,dict(shared_slope=a,TXconditional_slope=ac,additive_offset=b,**{'prediction_'+k:v for k,v in pred.items()})
