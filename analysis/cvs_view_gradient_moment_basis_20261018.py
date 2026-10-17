"""Recount full saved source gradient scalars; no IQ/checkpoint/target access."""
import argparse,gzip,json,math,statistics
from pathlib import Path
def analyze(path):
 with gzip.open(path,'rt',encoding='utf8') as f:rows=[json.loads(s) for s in f]
 if len(rows)!=2220 or [r['step'] for r in rows]!=list(range(1,2221)) or any(r['mode']!='MEASURE' or r['cumulative_step']!=r['step']+4440 for r in rows):raise ValueError('Entire fixed source MEASURE trajectory required')
 D=[];M=[];ratios=[];epochs=[]
 for r in rows:
  o=r['ordinary_norm']**2;s=r['satellite_norm']**2;cross=r['dot'];dif=o+s-2*cross;mix=o+s+2*cross
  if not all(math.isfinite(x) for x in (o,s,cross,dif,mix)) or min(dif,mix)<-1e-12:raise ValueError('Invalid complete gradient geometry')
  D.append(max(dif,0));M.append(max(mix,0));ratios.append(dif/mix if mix>1e-30 else None)
 for i in range(10):
  q=rows[i*222:(i+1)*222];lo=i*222;hi=(i+1)*222;epochs.append(dict(cumulative_epoch=21+i,steps=222,variance_trace_over_squared_mean=sum(D[lo:hi])/sum(M[lo:hi]),ordinary_norm_mean=statistics.fmean(r['ordinary_norm'] for r in q),satellite_norm_mean=statistics.fmean(r['satellite_norm'] for r in q),negative_dot_count=sum(r['dot']<0 for r in q)))
 return dict(status='VERIFIED_COMPLETE_SOURCE_GRADIENT_SCALARS',source_run='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02',source_row='MEASURE-s2026092701',steps=2220,read_all_rows=True,weighted_half_view_gradients=True,coordinate='Sinc physicalHz gradient;all other parameters native',pooled_variance_trace_over_squared_mean=sum(D)/sum(M),per_step_ratio_quantiles=[statistics.quantiles([x for x in ratios if x is not None],n=10)[i] for i in (0,4,8)],epochs=epochs,limitation='Saved scalar norms/dot only:does not recover coordinatewise Adam v or total SupCon/U gradient,clipping,trajectory or recognition improvement',IQ=0,CP=0,NN=0,updates=0,target_access=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-log',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 r=analyze(a.source_log);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf8')
