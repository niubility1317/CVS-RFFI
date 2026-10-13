"""Build small, inspectable metric/curve tables and static source figures."""
import argparse,csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def write_csv(path,rows):
 if not rows:return
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def aggregate(values):return dict(mean=st.mean(values),sd=st.stdev(values) if len(values)>1 else None,n=len(values))
def render(input_file,out):
 v=json.loads(Path(input_file).read_text(encoding='utf-8'));out=Path(out);out.mkdir(parents=True,exist_ok=True)
 sources=[r for r in v['rows'] if r['source_complete']];scored=[r for r in v['rows'] if r['row_complete']];seeds=[];groups=[];curves=[]
 for r in sources:
  a=r['source_audit']
  for e in a['curve']:curves.append(dict(run_id=v['run_id'],row_id=r['row_id'],arm=r['arm'],seed=r['model_seed'],**e))
 for r in scored:
  records=r['metric_records'];overall=next(x for x in records if x['dimension']=='overall');a=r['source_audit']
  rr=[x for x in records if x['dimension']=='receiver'];dd=[x for x in records if x['dimension']=='day']
  if sum(x['query_count'] for x in rr)!=168000 or sum(x['query_count'] for x in dd)!=168000:raise ValueError('Group count/overall mismatch')
  seeds.append(dict(run_id=v['run_id'],row_id=r['row_id'],arm=r['arm'],seed=r['model_seed'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=min(x['accuracy'] for x in rr),worst_day=min(x['accuracy'] for x in dd),final_L_CE=a['final_L_CE'],source_V=a['final_source']['source_val_accuracy'],worst_source_RX=a['final_source']['source_val_worst_rx'],source_training_seconds=a['training_seconds'],U_observation_draws=a['U_observation_draws'],U_accepted=a['U_accepted'],U_acceptance=a['U_accept_rate'],historical_backbone=r['config']['baseline_reuse']))
  for metric in records:groups.append({k:metric[k] for k in ('row_id','arm','model_seed','view','readout','dimension','stratum','accuracy','macro_f1','query_count')})
 arms=list(dict.fromkeys(r['arm'] for r in v['rows']));summaries=[];paired=[]
 for arm in arms:
  rr=[r for r in seeds if r['arm']==arm]
  if not rr:continue
  record=dict(arm=arm,seed_count=len(rr),complete_three_seed=len(rr)==3)
  for metric in ('accuracy','macro_f1','worst_RX','worst_day','final_L_CE','source_V','worst_source_RX','source_training_seconds','U_acceptance'):
   a=aggregate([r[metric] for r in rr]);record[metric+'_mean']=a['mean'];record[metric+'_sd']=a['sd']
  summaries.append(record)
 index={(r['arm'],r['seed']):r for r in seeds}
 pair_list=[(a,'BASE_REF') for a in arms[1:]]
 if 'RFF' in arms:pair_list+=[(a,'RFF') for a in ('MEAN','MEAN_VAR','NONLINEAR_MEAN')]+[('SIG','COV'),('SIG_TIME','COV'),('SIG_TIME','SIG')]
 else:pair_list+=[('LONG_LINEAR','COMPACT_LINEAR'),('COMPACT_SILU','COMPACT_LINEAR'),('LONG_SILU','LONG_LINEAR'),('LONG_SILU','COMPACT_SILU')]
 for t,c in pair_list:
  matched=sorted({r['seed'] for r in seeds if r['arm']==t}&{r['seed'] for r in seeds if r['arm']==c});ds=[index[t,s]['accuracy']-index[c,s]['accuracy'] for s in matched]
  if ds:paired.append(dict(treatment=t,control=c,seed_count=len(ds),complete_three_seed=len(ds)==3,seed_differences=dict(zip(map(str,matched),ds)),mean_difference=st.mean(ds),sd_difference=st.stdev(ds) if len(ds)>1 else None,positive_seeds=sum(x>0 for x in ds)))
 result=dict(run_id=v['run_id'],source_rows=len(sources),scored_rows=len(scored),expected_rows=len(v['rows']),all_rows_complete=v['all_rows_complete'],summary=summaries,paired=paired,default_classifier_only=True,head_fit_updates=0,truth_last=True,training_curve_scope='all completed-source epochs/steps; native source V in percent; U one observed sample per weak/strong pair',release_commit=v['release_commit'])
 (out/'result_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');write_csv(out/'per_seed_results.csv',seeds);write_csv(out/'group_metrics.csv',groups);write_csv(out/'source_curves.csv',curves)
 colors=plt.get_cmap('tab10');fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 plot_fields=[('mean_L_CE','Labeled source CE'),('source_V_percent','Source V accuracy (%)'),('U_accept_rate','Hidden U acceptance (%)'),('native_unlabeled_loss','Native unlabeled loss')]
 for i,(field,label) in enumerate(plot_fields):
  a=ax.flat[i]
  for j,arm in enumerate(arms):
   rr=[r for r in sources if r['arm']==arm];rr=[r for r in rr if all(e.get(field) is not None for e in r['source_audit']['curve'][33:] if field in ('U_accept_rate','native_unlabeled_loss'))]
   if not rr:continue
   start=33 if field in ('U_accept_rate','native_unlabeled_loss') else 0;array=np.array([[e.get(field) if e.get(field) is not None else np.nan for e in r['source_audit']['curve'][start:]] for r in rr],dtype=float)
   if field=='U_accept_rate':array*=100
   mean=np.nanmean(array,0);sd=np.nanstd(array,0,ddof=1) if len(rr)>1 else np.zeros_like(mean);x=np.arange(start+1,51)
   if arm=='BASE_REF':color='black'
   else:color=colors(j%10)
   a.plot(x,mean,label=f'{arm} (n={len(rr)})',color=color,linewidth=1.5);a.fill_between(x,mean-sd,mean+sd,color=color,alpha=.12)
  a.axvline(33.5,color='.5',linestyle='--',linewidth=.8);a.set(xlabel='Epoch',ylabel=label);a.grid(alpha=.2)
 ax[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Fixed source training: mean and sample SD; default classifier jointly learned')
 fig.savefig(out/'source_training_curves.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(source_rows=len(sources),scored_rows=len(scored),all_rows_complete=v['all_rows_complete'],summary=summaries,paired=paired)))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.input,a.output)
