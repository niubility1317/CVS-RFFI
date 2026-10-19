"""Independently replay saved source scatter using pairwise class contrasts; no NN/fit/query."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cvs_prepool_source import design as d
from experiments.cvs_prepool_source.publish import remote
SCRIPT=r'''
import json
from pathlib import Path
import numpy as np
c=CONFIG;run=Path(c['project'])/'runs'/c['run_id'];result=json.loads((run/'results.json').read_text());done=json.loads((run/'complete.json').read_text());progress=json.loads((run/'progress.json').read_text())
if result['NN']!=18000 or done['NN']!=18000 or progress['NN']!=18000 or result['fit_updates'] or result['target_access'] or result['U_used'] or result['V_used'] or result['formal_scores_read'] or result['new_physical_simulations']:raise ValueError('Fixed complete no-fit/source-only evidence differs')
records=0;maxerror=0.;models={}
for model,row in result['rows'].items():
 with np.load(run/(model+'_descriptors.npz'),allow_pickle=False) as f:
  y,rx,day,ids=(f[k] for k in ('y','rx','day','ids'));rng=np.random.default_rng(20261019821);groups=[]
  if len(ids)!=1800 or len(set(ids))!=1800:raise ValueError('Complete unique source diagnostic panel required')
  for r in sorted(set(rx)):
   for t in sorted(set(day)):
    halves=[]
    for label in range(6):
     ix=np.flatnonzero((rx==r)&(day==t)&(y==label));ix=np.asarray(sorted(ix,key=lambda i:ids[i]));shuffled=rng.permutation(ix)
     if len(ix)!=20:raise ValueError('Same fixed six-class20-per-cell panel required')
     halves.append((shuffled[:10],shuffled[10:]))
    groups.append((int(r),int(t),halves))
  models[model]={}
  for view,metrics in row['views'].items():
   models[model][view]={}
   for kind in ('pooled','removed'):
    x=f[view+'_'+kind].astype(np.float64);ss=0.;tt=0.
    if x.shape!=(1800,144) or not np.isfinite(x).all():raise ValueError('Actual descriptor shape/finiteness differs')
    for r,t,halves in groups:
     ma=np.stack([x[a].mean(0) for a,b in halves]);mb=np.stack([x[b].mean(0) for a,b in halves]);da=ma[:,None]-ma[None,:];db=mb[:,None]-mb[None,:];s=float(np.sum(da*db)/(2*6*6));ix=np.concatenate([np.concatenate([a,b]) for a,b in halves]);z=x[ix];total=float(np.mean(np.sum(z*z,axis=1))-np.sum(np.mean(z,axis=0)**2));old=next(v for v in metrics[kind]['strata'] if v['RX']==r and v['day']==t)
     if not np.allclose([s,total],[old['cross_half_TX_scatter'],old['total_scatter']],rtol=1e-10,atol=1e-10):raise ValueError('Independent pairwise-contrast scatter recount differs')
     maxerror=max(maxerror,abs(s-old['cross_half_TX_scatter']),abs(total-old['total_scatter']));ss+=s;tt+=total;records+=1
    ratio=ss/tt if tt>0 else 0.
    if not np.isclose(ratio,metrics[kind]['ratio'],rtol=1e-10,atol=1e-10):raise ValueError('Aggregate source scatter ratio differs')
    models[model][view][kind]=ratio
support=all(models[m][v]['removed']>0 for m in models for v in ('low_draw0','low_draw1'))
if records!=300 or support!=result['supports_local_removed_TX_variation']:raise ValueError('Declared local diagnostic decision differs')
print(json.dumps(dict(status='VERIFIED',source_result=result,independent_pairwise_contrast_strata=records,max_abs_recount_error=maxerror,extra_NN=0,fit_updates=0,query_read=False,formal_scores_read=False,goal_complete=False)))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);a=p.parse_args();rt=d.read(a.runtime)
 if rt['run_id']!=d.RUN:raise ValueError('Wrong diagnostic runtime')
 compile(SCRIPT.replace('CONFIG','{}'),'<source-analysis>','exec');x=remote(SCRIPT.replace('CONFIG',repr(dict(project=rt['project'],run_id=d.RUN))),rt);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);d.write(out/'analysis.json',x);print(json.dumps(dict(status=x['status'],strata=x['independent_pairwise_contrast_strata'],max_abs_recount_error=x['max_abs_recount_error'],extra_NN=0,support=x['source_result']['supports_local_removed_TX_variation'])))
if __name__=='__main__':main()
