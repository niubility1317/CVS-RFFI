"""Positive affine PSD relations only;no IQ/noise generation/model/identity classifier."""
import math
import numpy as np
RX=(3,4,6,8);WINDOWS=('rectangular','hann')

def probabilities(x):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=2 or a.shape[1]!=256 or not np.isfinite(a).all() or np.any(a<0) or np.any(np.abs(a.sum(-1)-1)>1e-10):raise ValueError('Complete finite normalized 256-bin spectra required')
 return a

def simplex(v,mass):
 v=np.asarray(v,dtype=np.float64)
 if v.shape!=(256,) or not np.isfinite(v).all() or not math.isfinite(mass) or not 0<=mass<=1:raise ValueError('Finite vector and simplex mass in[0,1]')
 if mass==0:return np.zeros_like(v)
 u=np.sort(v)[::-1];css=np.cumsum(u)-mass;active=np.flatnonzero(u-css/np.arange(1,257)>0)
 if not len(active):raise ValueError('Empty simplex solution')
 k=int(active[-1]);return np.maximum(v-css[k]/(k+1),0.)

def fit(x,y):
 x,y=probabilities(x),probabilities(y)
 if x.shape!=y.shape or len(x)!=10:raise ValueError('Original5TX x2days training cells required')
 mx,my=x.mean(0),y.mean(0)
 def evaluate(a):
  b=simplex(my-a*mx,1-a);error=float(np.square(y-a*x-b).sum(1).mean());return error,a,b
 lo,hi=0.,1.;rho=(math.sqrt(5)-1)/2;left=hi-rho*(hi-lo);right=lo+rho*(hi-lo);fl,fr=evaluate(left),evaluate(right)
 for _ in range(64):
  if fl[0]<fr[0]:hi=right;right=left;fr=fl;left=hi-rho*(hi-lo);fl=evaluate(left)
  else:lo=left;left=right;fl=fr;right=lo+rho*(hi-lo);fr=evaluate(right)
 candidates=[evaluate(0.),evaluate(1.),fl,fr,evaluate((lo+hi)/2)];best=min(v[0] for v in candidates);answer=max((v for v in candidates if v[0]<=best+1e-18),key=lambda v:v[1]);error,a,b=answer
 return dict(a=float(a),b=b,fit_MSE=error)

def criteria(folds):
 e=math.fsum(z['model_SSE'] for z in folds);identity=math.fsum(z['identity_SSE'] for z in folds);mean=math.fsum(z['mean_RX_SSE'] for z in folds);wrong=[math.fsum(z['wrong_TX_SSE'][j] for z in folds) for j in range(5)]
 gain=lambda numerator,denominator:None if denominator<=1e-12 else 1-numerator/denominator
 gi,gm=gain(e,identity),gain(e,mean);margin=None if mean<=1e-12 else (min(wrong)-e)/mean
 per_tx=[gain(math.fsum(z['model_SSE'] for z in folds if z['held_TX']==t),math.fsum(z['identity_SSE'] for z in folds if z['held_TX']==t)) for t in range(6)]
 per_day=[gain(math.fsum(z['model_SSE'] for z in folds if z['held_day']==d),math.fsum(z['identity_SSE'] for z in folds if z['held_day']==d)) for d in (1,2,3)]
 conditions=dict(nonzero_references=identity>1e-12 and mean>1e-12,retains_TX_contrast=all(.5-1e-10<=z['a']<=.995+1e-10 for z in folds),identity_risk_reduction=gi is not None and gi>=.25-1e-10,mean_RX_risk_reduction=gm is not None and gm>=.20-1e-10,positive_folds=sum(z['model_SSE']<z['identity_SSE'] for z in folds)>=15,all_TX_nonnegative=all(v is not None and v>=-1e-10 for v in per_tx),all_day_nonnegative=all(v is not None and v>=-1e-10 for v in per_day),beats_every_wrong_TX=margin is not None and margin>=.10-1e-10)
 return dict(qualified=all(conditions.values()),conditions=conditions,model_SSE=e,identity_SSE=identity,mean_RX_SSE=mean,wrong_TX_SSE=wrong,identity_risk_reduction=gi,mean_RX_risk_reduction=gm,wrong_TX_margin_over_mean_risk=margin,per_TX_identity_risk_reductions=per_tx,per_day_identity_risk_reductions=per_day,minimum_a=min(z['a'] for z in folds),maximum_a=max(z['a'] for z in folds),positive_folds=sum(z['model_SSE']<z['identity_SSE'] for z in folds))

def analyze(psd):
 p=np.asarray(psd,dtype=np.float64)
 if p.shape!=(2,6,3,4,256) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),256,rtol=1e-10,atol=1e-10):raise ValueError('Exact original2windows/6TX/3days/4RX/all256bins unit-mean cell PSD required')
 p=p/256.;folds=[];directions={}
 for src in range(4):
  for dst in range(4):
   if src==dst:continue
   name=str(RX[src])+'->'+str(RX[dst]);directions[name]={}
   for w,window in enumerate(WINDOWS):
    values=[]
    for tx in range(6):
     for day in range(3):
      txs=[t for t in range(6) if t!=tx];days=[d for d in range(3) if d!=day];x=np.stack([p[w,t,d,src] for t in txs for d in days]);y=np.stack([p[w,t,d,dst] for t in txs for d in days]);f=fit(x,y);a,b=f['a'],f['b'];xx,yy=p[w,tx,day,src],p[w,tx,day,dst];prediction=a*xx+b
      if np.any(b<0) or abs(b.sum()-(1-a))>1e-10 or abs(prediction.sum()-1)>1e-10:raise ValueError('Nonnegative mass/PSD conservation failed')
      sse=lambda v:float(np.square(v-yy).sum())
      rec=dict(source_RX=RX[src],destination_RX=RX[dst],window=window,held_TX=tx,held_day=day+1,a=a,b=b.tolist(),fit_MSE=f['fit_MSE'],model_SSE=sse(prediction),identity_SSE=sse(xx),mean_RX_SSE=sse(y.mean(0)),wrong_TX_SSE=[sse(a*p[w,(tx+k)%6,day,src]+b) for k in range(1,6)])
      values.append(rec);folds.append(rec)
    directions[name][window]=criteria(values)
   directions[name]['qualified_both_windows']=all(directions[name][w]['qualified'] for w in WINDOWS)
 pairs=[]
 for i in range(4):
  for j in range(i+1,4):
   names=[str(RX[i])+'->'+str(RX[j]),str(RX[j])+'->'+str(RX[i])];passing=[n for n in names if directions[n]['qualified_both_windows']];pairs.append(dict(RX_pair=[RX[i],RX[j]],passing_directions=passing,qualified=bool(passing)))
 qualified=all(v['qualified'] for v in pairs)
 return dict(status='ANALYZED',verdict='PASS_FIXED_PSD_RELATION_ONLY' if qualified else 'FAIL_FIXED_PSD_AFFINE_BASIS_CLOSED',qualified=qualified,unordered_pairs=pairs,directions=directions,fit_count=len(folds),source_only=True,source_IQ_reads=0,checkpoint_loads=0,new_NN=0,optimizer_updates=0,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,target_access=False),folds
