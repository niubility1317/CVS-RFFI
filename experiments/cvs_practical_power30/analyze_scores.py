"""Recount fixed confusion matrices and paired correctness; no raw data or inference."""
from collections import Counter
import json,math
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')

def recount(data,pairs):
 records={};cm_count=0;max_error=0.
 for name,row in data['rows'].items():
  score=row['score']
  if score['status']!='SCORED' or not score['truth_last'] or score['head_fit_updates']!=0 or len(score['records'])!=126:raise ValueError('Complete independent default-head scores required')
  indexed={}
  for r in score['records']:
   key=(r['view'],r['dimension'],str(r['stratum']))
   if key in indexed:raise ValueError('Duplicate score stratum')
   indexed[key]=r;c=r['confusion']
   if len(c)!=6 or any(len(x)!=6 or any(not isinstance(v,int) or v<0 for v in x) for x in c):raise ValueError('Nonnegative 6-class counts required')
   rows=[sum(x) for x in c];cols=[sum(c[i][j] for i in range(6)) for j in range(6)];n=sum(rows);diag=sum(c[i][i] for i in range(6))
   if n!=r['query_count'] or n<=0:raise ValueError('Score size mismatch')
   acc=diag/n;f1=sum(2*c[i][i]/(rows[i]+cols[i]) if rows[i]+cols[i] else 0. for i in range(6))/6
   present=[c[i][i]/rows[i] for i in range(6) if rows[i]];macro_acc=sum(present)/len(present)
   for k,v in (('accuracy',acc),('macro_f1',f1),('macro_accuracy',macro_acc)):
    error=abs(v-r[k]);max_error=max(error,max_error)
    if not math.isfinite(r[k]) or error>1e-12:raise ValueError('CM metric mismatch:'+str(key)+k)
   cm_count+=1
  for view in VIEWS:
   overall=indexed[(view,'overall','ALL')]
   if overall['query_count']!=168000:raise ValueError('Fixed full view size required')
   for dim,count in (('receiver',7),('transmitter',6),('day',4)):
    strata=[r for (v,d,s),r in indexed.items() if v==view and d==dim]
    if len(strata)!=count:raise ValueError('Missing partition')
    if any(sum(r['confusion'][i][j] for r in strata)!=overall['confusion'][i][j] for i in range(6) for j in range(6)):raise ValueError('Partitions do not sum to overall')
   summary=data['result']['summary'][name][view]
   if abs(overall['accuracy']-summary['accuracy'])>1e-12 or abs(overall['macro_f1']-summary['macro_F1'])>1e-12:raise ValueError('Published overall summary mismatch')
   worst=min(r['accuracy'] for (v,d,s),r in indexed.items() if v==view and d=='receiver')
   if abs(worst-summary['worst_RX_accuracy'])>1e-12:raise ValueError('Worst RX mismatch')
  records[name]=indexed
 s=data['result']['summary'];p=pairs['pairs'];differences={}
 for view in VIEWS:
  x=p[view];delta=(x['left_correct']-x['right_correct'])/168000*100
  if x['left_correct']!=round(s['QUALITY_POWER'][view]['accuracy']*168000) or x['right_correct']!=round(s['NOMINAL_AS'][view]['accuracy']*168000) or x['repairs']-x['new_errors']!=x['left_correct']-x['right_correct']:raise ValueError('Paired count mismatch')
  if abs(delta-x['delta_accuracy_pp'])>1e-12 or abs(delta-data['result']['paired_differences'][view]['delta_accuracy_pp'])>1e-12:raise ValueError('Paired delta mismatch')
  if x['left_clean_and_scene']>x['left_correct'] or x['right_clean_and_scene']>x['right_correct']:raise ValueError('Invalid joint count')
  if abs((x['left_clean_and_scene']-x['right_clean_and_scene'])/168000*100-x['joint_delta_pp'])>1e-12:raise ValueError('Joint delta mismatch')
  differences[view]=delta
 criteria={'low_urban_gain_at_least_0_5pp':differences['practical_low_urban']>=.5,'clean_not_lower':differences['clean']>=0,'mid_urban_not_lower':differences['practical_mid_urban']>=0,'each_suburban_no_more_than_0_25pp_lower':all(differences[v]>=-.25 for v in VIEWS[1:4])}
 gaps={name:{v:100*(sv['clean']['accuracy']-sv[v]['accuracy']) for v in VIEWS[1:4]} for name,sv in s.items()}
 strata=[]
 for key in records['QUALITY_POWER']:
  v,d,k=key;a=records['QUALITY_POWER'][key];b=records['NOMINAL_AS'][key]
  strata.append(dict(view=v,dimension=d,stratum=k,count=a['query_count'],quality_accuracy_pct=100*a['accuracy'],nominal_accuracy_pct=100*b['accuracy'],delta_pp=100*(a['accuracy']-b['accuracy']),quality_macro_F1_pct=100*a['macro_f1'],nominal_macro_F1_pct=100*b['macro_f1']))
 return dict(status='VERIFIED',CM_recounts=cm_count,partition_sums_checked=42,max_metric_error=max_error,views=list(VIEWS),paired_recounts=7,differences_pp=differences,suburban_gap_pp=gaps,preregistered_criteria=criteria,scientific_pass=all(criteria.values()),fixed_candidate='CLOSED_FAILED_PREREGISTERED_IDENTITY_BENEFIT',strata=strata,IQ=0,CP=0,raw_truth_reads=0,NN=0,new_updates=0)

if __name__=='__main__':
 import sys
 data=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));pairs=json.loads(Path(sys.argv[2]).read_text(encoding='utf8'));out=recount(data,pairs)
 Path(sys.argv[3]).write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 print(json.dumps({k:v for k,v in out.items() if k!='strata'},ensure_ascii=False,allow_nan=False))
