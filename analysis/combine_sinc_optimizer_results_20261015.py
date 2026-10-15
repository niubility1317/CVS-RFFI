"""Independently combine the two completed, frozen default-head Sinc runs.

Reads collected artifacts only. No IQ, model fitting, training or new predictions.
"""
import argparse,json,math,statistics
from pathlib import Path
SEEDS=(2026092701,2026092702,2026092703)
ARMS=('OLD_GATE_COS50','SIG_TIME','SINC_UNITS')
PAIRS=(('SINC_UNITS','SIG_TIME'),('SINC_UNITS','OLD_GATE_COS50'),('SIG_TIME','OLD_GATE_COS50'))
def close(a,b):
 if a is None or b is None:return a is b
 return math.isclose(a,b,abs_tol=1e-13,rel_tol=1e-13)
def require(test,message):
 if not test:raise ValueError(message)
def key(r):return (r['row_id'],r['readout'],r['view'],r['dimension'],r['stratum'])
def combine(first,remaining):
 require(first['status']==remaining['status']=='COLLECTED','Completed collections required')
 require(first['completion']['models']==3 and remaining['completion']['models']==6,'Full 3+6 rows required')
 blocks=[first,remaining];records=sum([b['analysis']['records'] for b in blocks],[])
 require(len(records)==162 and len({key(r) for r in records})==162,'162 distinct registered metrics required')
 remote=remaining['combined_analysis'];require(remote['model_seeds']==list(SEEDS) and remote['models']==9,'Three predefined model seeds required')
 registered={key(r):r for r in remote['records']};require(len(registered)==162,'Combined registered record count')
 for block in blocks:
  require(block['audit']['status']=='VERIFIED' and block['audit']['audit']['truth_last_recount'] is True,'Full raw-prediction truth-last recount required')
  native={key(r):r for r in block['audit']['audit']['metrics']}
  for r in block['analysis']['records']:
   require(r['view']=='clean' and r['readout']=='ORIGINAL_HEAD','Only registered clean/default head')
   require(r==native[key(r)]==registered[key(r)],'Saved score, full recount and combined score differ')
   C=r['confusion'];require(len(C)==6 and all(len(x)==6 for x in C),'Six-class confusion required')
   row=[sum(x) for x in C];col=[sum(x[j] for x in C) for j in range(6)];N=sum(row)
   require(N==r['query_count'],'Confusion sample count differs')
   accuracy=sum(C[i][i] for i in range(6))/N
   f1=sum(2*C[i][i]/(row[i]+col[i]) if row[i]+col[i] else 0 for i in range(6))/6
   require(close(accuracy,r['accuracy']) and close(f1,r['macro_f1']),'Independent confusion arithmetic differs')
 summary=[]
 for arm in ARMS:
  model=[]
  for seed in SEEDS:
   rs=[r for r in records if r['arm']==arm and r['model_seed']==seed]
   require(len(rs)==18,'Each seed/arm requires all18 metrics')
   overall=next(r for r in rs if r['dimension']=='overall')
   require(overall['query_count']==168000,'Full fixed clean population required')
   model.append(dict(seed=seed,accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],
    worst_RX=min(r['accuracy'] for r in rs if r['dimension']=='receiver'),
    worst_day=min(r['accuracy'] for r in rs if r['dimension']=='day')))
  s=dict(arm=arm,seed_count=3,accuracy_mean=statistics.mean(x['accuracy'] for x in model),
   accuracy_sd=statistics.stdev(x['accuracy'] for x in model),seed_accuracy={str(x['seed']):x['accuracy'] for x in model},
   macro_f1_mean=statistics.mean(x['macro_f1'] for x in model),worst_RX_mean=statistics.mean(x['worst_RX'] for x in model),
   worst_day_mean=statistics.mean(x['worst_day'] for x in model))
  expected=next(x for x in remote['summary'] if x['arm']==arm)
  require(all(close(s[k],expected[k]) for k in ('accuracy_mean','accuracy_sd','macro_f1_mean','worst_RX_mean','worst_day_mean')),'Independent three-seed summary differs')
  summary.append(s)
 paired=[];errors=sum([b['audit']['audit']['error_transitions'] for b in blocks],[])
 require(len(errors)==9,'Three pairs per each model seed required')
 for arm,control in PAIRS:
  A=next(s for s in summary if s['arm']==arm);B=next(s for s in summary if s['arm']==control)
  diffs=[A['seed_accuracy'][str(seed)]-B['seed_accuracy'][str(seed)] for seed in SEEDS]
  e=[next(r for r in errors if (r['arm'],r['control'],r['seed'])==(arm,control,seed)) for seed in SEEDS]
  for seed,delta,row in zip(SEEDS,diffs,e):
   require(row['count']==168000 and sum(row[k] for k in ('corrected_errors','introduced_errors','both_correct','both_wrong'))==168000,'Paired error partition differs')
   require(close((row['corrected_errors']-row['introduced_errors'])/168000,delta),'Same-ID error net differs from paired accuracy')
  p=dict(arm=arm,control=control,seeds=list(SEEDS),seed_differences=diffs,
   mean_difference=statistics.mean(diffs),sd_difference=statistics.stdev(diffs),positive_seeds=sum(x>0 for x in diffs))
  expected=next(r for r in remote['paired'] if (r['arm'],r['control'])==(arm,control))
  require(all(close(p[k],expected[k]) for k in ('mean_difference','sd_difference','positive_seeds')),'Independent paired statistics differ')
  p['same_ID_errors']=dict(decisions=504000,corrected_errors=sum(r['corrected_errors'] for r in e),introduced_errors=sum(r['introduced_errors'] for r in e),net_correct=sum(r['corrected_errors']-r['introduced_errors'] for r in e))
  paired.append(p)
 updates=sum([b['source_updates'] for b in blocks],[])
 require(len(updates)==6 and len({r['row_id'] for r in updates})==6,'Six actual Sinc update witnesses required')
 for u in updates:
  require(u['status']=='VERIFIED' and u['actual_steps']==11100 and u['actual_U_packets']==963900 and u['actual_Sinc_parameter_records']==104 and u['actual_snapshot_batches']==52,'Incomplete Sinc actual-training observations')
  require(u['head_fit_updates']==u['extra_model_forwards']==0 and u['U_truth_read'] is False and u['query_read'] is False,'Source witness boundary differs')
 curves=sum([b['source_curves'] for b in blocks],[])
 require(len(curves)==9 and all([r['epoch'] for r in c['records']]==list(range(1,51)) for c in curves),'All450 source epoch records required')
 new_updates=sum(b['audit']['audit']['new_step_records_parsed'] for b in blocks)
 layers=sum(b['audit']['gradient_layer_records_recounted'] for b in blocks)
 require(new_updates==66600 and layers==11988,'Full actual-training audit totals differ')
 costs=[]
 for block in blocks:
  for row in block['audit']['audit']['rows']:
   costs.append(dict(row_id=row['row_id'],arm=row['arm'],seed=row['seed'],historical_reuse=row['source_reused'],epochs=row['epochs'],updates=row['updates'],U_packets=row['U_packets'],U_accepted=row['U_accepted'],epoch_seconds=sum(r['epoch_time_s'] for r in row['source_epoch_curve'])))
 return dict(status='VERIFIED',run_ids=[b['run_id'] for b in blocks],model_seeds=list(SEEDS),model_seed_count=3,models=9,
  newly_trained_models=6,baseline_reused_models=3,default_decision_count=1512000,metric_records=162,summary=summary,paired=paired,
  records=records,paired_error_transitions=errors,training_costs=costs,
  audit_counts=dict(source_epoch_records=450,new_epochs=300,historical_epochs_reused=150,new_optimizer_updates=new_updates,
   actual_U_packets=5783400,L_U_gradient_layer_records=layers,saved_L_U_gradient_input_vectors=layers*3,
   Sinc_snapshot_batches=312,Sinc_parameter_records=624,head_fit_updates=0,extra_training_model_forwards=0),
  goal_84=next(s for s in summary if s['arm']=='SINC_UNITS')['accuracy_mean']>=.84,
  independent_confusion_and_paired_statistics=True,truth_last=True,target_feedback_to_training=False,head_fit=False,
  first_seed_retrained=False,Phase2_claim=False)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('first');p.add_argument('remaining');p.add_argument('output')
 a=p.parse_args();value=combine(json.loads(Path(a.first).read_text(encoding='utf-8')),json.loads(Path(a.remaining).read_text(encoding='utf-8')))
 with Path(a.output).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:value[k] for k in ('status','models','metric_records','summary','paired','goal_84')}))
