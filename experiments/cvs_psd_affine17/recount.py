"""Independent arithmetic/convex-KKT recount from saved spectra,not candidate fitter."""
import gzip,itertools,json,math,time
import numpy as np
from . import design as d
from .data import load
RX=(3,4,6,8);WINDOWS=('rectangular','hann')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sse(a,b):return math.fsum((float(x)-float(y))**2 for x,y in zip(a,b))
def same(a,b):return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-12)
def run():
 start=time.perf_counter();psd,origin=load();p=psd/256;s=json.loads((d.BASE/'summary.json').read_text(encoding='utf8'))
 with gzip.open(d.BASE/'folds.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(x) for x in f if x.strip()]
 expected={(r,z,w,t,day) for r in RX for z in RX if r!=z for w in WINDOWS for t in range(6) for day in (1,2,3)}
 observed=[(v['source_RX'],v['destination_RX'],v['window'],v['held_TX'],v['held_day']) for v in rows];require(len(rows)==432 and len(set(observed))==432 and set(observed)==expected,'Complete432 distinct folds')
 groups={};max_KKT=0.;max_SSE=0.
 for v in rows:
  r,z=RX.index(v['source_RX']),RX.index(v['destination_RX']);w=WINDOWS.index(v['window']);t,day=v['held_TX'],v['held_day']-1;a=v['a'];b=np.asarray(v['b'],dtype=float)
  require(0<=a<=1 and b.shape==(256,) and np.isfinite(b).all() and np.min(b)>=0 and abs(float(b.sum())-(1-a))<=1e-10,'Physical PSD simplex')
  index=list(itertools.product([x for x in range(6) if x!=t],[x for x in range(3) if x!=day]));x=np.array([p[w,u,e,r] for u,e in index]);y=np.array([p[w,u,e,z] for u,e in index]);mx=np.mean(x,axis=0);my=np.mean(y,axis=0);q=my-a*mx
  active=b>1e-11;tau=float(np.mean(q[active]-b[active])) if np.any(active) else float(np.max(q));residual=q-b
  stationarity=max(float(np.max(np.abs(residual[active]-tau))) if np.any(active) else 0.,float(np.max(np.maximum(residual[~active]-tau,0.))) if np.any(~active) else 0.)
  grad=-2*math.fsum(float(xx)*float(yy-a*xx-bb) for xxrow,yyrow in zip(x,y) for xx,yy,bb in zip(xxrow,yyrow,b))/10+2*tau
  violation=max(-grad,0.) if a<=1e-9 else max(grad,0.) if a>=1-1e-9 else abs(grad);max_KKT=max(max_KKT,stationarity,violation);require(max(stationarity,violation)<1e-7,'Convex fitter is not a valid constrained minimum')
  target=p[w,t,day,z];predict=a*p[w,t,day,r]+b
  values=dict(model_SSE=sse(target,predict),identity_SSE=sse(target,p[w,t,day,r]),mean_RX_SSE=sse(target,my),wrong_TX_SSE=[sse(target,a*p[w,(t+k)%6,day,r]+b) for k in range(1,6)],fit_MSE=math.fsum(sse(yy,a*xx+b) for xx,yy in zip(x,y))/10)
  for key,value in values.items():
   if isinstance(value,list):require(len(v[key])==5 and all(same(j,k) for j,k in zip(value,v[key])),'WrongTX control SSE')
   else:require(same(value,v[key]),'Saved scalar '+key);max_SSE=max(max_SSE,abs(value-v[key]))
  groups.setdefault((str(RX[r])+'->'+str(RX[z]),WINDOWS[w]),[]).append(dict(v,**values))
 independent={}
 for (direction,w),values in groups.items():
  model=math.fsum(z['model_SSE'] for z in values);identity=math.fsum(z['identity_SSE'] for z in values);mean=math.fsum(z['mean_RX_SSE'] for z in values);wrong=[math.fsum(z['wrong_TX_SSE'][k] for z in values) for k in range(5)]
  gain=lambda e,b:None if b<=1e-12 else 1-e/b
  gi,gm=gain(model,identity),gain(model,mean);margin=None if mean<=1e-12 else (min(wrong)-model)/mean
  tx=[gain(math.fsum(z['model_SSE'] for z in values if z['held_TX']==t),math.fsum(z['identity_SSE'] for z in values if z['held_TX']==t)) for t in range(6)];days=[gain(math.fsum(z['model_SSE'] for z in values if z['held_day']==e),math.fsum(z['identity_SSE'] for z in values if z['held_day']==e)) for e in (1,2,3)]
  c=dict(nonzero_references=identity>1e-12 and mean>1e-12,retains_TX_contrast=all(.5-1e-10<=z['a']<=.995+1e-10 for z in values),identity_risk_reduction=gi is not None and gi>=.25-1e-10,mean_RX_risk_reduction=gm is not None and gm>=.20-1e-10,positive_folds=sum(z['model_SSE']<z['identity_SSE'] for z in values)>=15,all_TX_nonnegative=all(v is not None and v>=-1e-10 for v in tx),all_day_nonnegative=all(v is not None and v>=-1e-10 for v in days),beats_every_wrong_TX=margin is not None and margin>=.10-1e-10)
  saved=s['directions'][direction][w];require(c==saved['conditions'] and all(c.values())==saved['qualified'],'Fixed criteria differed')
  for key,v in dict(model_SSE=model,identity_SSE=identity,mean_RX_SSE=mean,identity_risk_reduction=gi,mean_RX_risk_reduction=gm,wrong_TX_margin_over_mean_risk=margin).items():require(v==saved[key] if v is None else same(v,saved[key]),'Window summary scalar')
  require(all(same(j,k) for j,k in zip(wrong,saved['wrong_TX_SSE'])),'Window wrongTX sums');independent[(direction,w)]=all(c.values())
 for name,direction in s['directions'].items():require(direction['qualified_both_windows']==all(independent[(name,w)] for w in WINDOWS),'Window direction promotion differs')
 qualified=True
 for record in s['unordered_pairs']:
  r,z=record['RX_pair'];passing=[name for name in (str(r)+'->'+str(z),str(z)+'->'+str(r)) if all(independent[(name,w)] for w in WINDOWS)];require(passing==record['passing_directions'] and bool(passing)==record['qualified'],'RX pair decision differs');qualified=qualified and bool(passing)
 require(s['fit_count']==432 and s['qualified']==qualified and s['target_access'] is False and s['source_IQ_reads']==s['new_NN']==s['optimizer_updates']==0,'Full budget/scope/decision differs')
 value=dict(status='VERIFIED',all_folds_recounted=432,all144_source_spectra_reused=True,all256_bins=True,all_masses_SSE_wrongTX_and_fixed_criteria_recounted=True,convex_KKT_max_violation=max_KKT,max_scalar_recount_error=max_SSE,qualified=qualified,verdict=s['verdict'],elapsed_CPU_wall_seconds=time.perf_counter()-start,additional_aggregate_NPZ_reads=1,source_IQ_reads=0,checkpoint_loads=0,NN=0,updates=0,target_access=False)
 output=d.BASE/'independent_recount.json';require(not output.exists(),'Preserve existing independent recount');d.write(output,value);return value
if __name__=='__main__':print(json.dumps(run()))
