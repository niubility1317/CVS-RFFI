"""Preserve one completed NoAs row and the technical E21 failure; no new inference/truth."""
import argparse,csv,gzip,json,math
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text())
def write(p,q):p.write_text(json.dumps(q,indent=2,ensure_ascii=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.run);out=Path(a.output)
 if out.exists():raise FileExistsError('Preserve existing partial analysis')
 b=root/'B00-s2026092701';f=root/'B01-s2026092701'
 if not (root/'failure.json').is_file() or (f/'source/final_ssdg.pth').exists():raise ValueError('Expected unrecoverable technical As failure only')
 pred=read(b/'prediction/complete.json');score=read(b/'scoring/complete.json');saved=read(b/'scoring/scores.json')
 if pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(VIEWS) or pred['truth_read'] or score['status']!='SCORED_COMPLETE' or not score['truth_last'] or not saved['truth_last'] or not saved['target_feedback_forbidden']:raise ValueError('All7 frozen independent truthlast scores required')
 curves=[];cost={};total_steps=0
 for arm,expected in [('B00',30),('B01',20)]:
  p=root/(arm+'-s2026092701')/'source';epochs=[json.loads(l) for l in (p/'metrics_epoch.jsonl').read_text().splitlines() if l.strip()]
  if [e['epoch'] for e in epochs]!=list(range(1,expected+1)):raise ValueError('Full completed epoch log required')
  for e in epochs:
   for k in ['lr','train_loss','train_grad_total','val_tx_acc']:
    if not math.isfinite(e[k]):raise ValueError('Nonfinite actual training metric')
   if e['train_pseudo_truth_available']:raise ValueError('U truth exposed')
   curves.append(dict(arm=arm,epoch=e['epoch'],lr=e['lr'],loss=e['train_loss'],source_V_accuracy_pct=e['val_tx_acc'],gradient=e['train_grad_total'],U_selected=e['train_pseudo_selected'],U_total=e['train_pseudo_total'],epoch_seconds=e['epoch_time_s']))
  with gzip.open(p/'step_metrics.jsonl.gz','rt') as h:steps=[json.loads(l) for l in h if l.strip()]
  if [q['step'] for q in steps]!=list(range(1,expected*222+1)):raise ValueError('Full successful-step log required')
  total_steps+=len(steps);cost[arm]=dict(completed_epochs=expected,successful_updates=len(steps),L_NN_on_completed_steps=len(steps)*768,As_packets=len(steps)*64 if arm=='B01' else 0,completed_epoch_wall_seconds=sum(e['epoch_time_s'] for e in epochs))
  counts={};unique={};samples={}
  with gzip.open(p/'training_sequence.jsonl.gz','rt') as h:
   for l in h:
    q=json.loads(l);r=q['role'];counts[r]=counts.get(r,0)+1;samples[r]=samples.get(r,0)+len(q['ids']);unique.setdefault(r,set()).update(json.dumps(v,sort_keys=True) for v in q['ids'])
  cost[arm].update(sequence_records=counts,sequence_samples=samples,unique_opaque_training_ids={k:len(v) for k,v in unique.items()})
 if cost['B01']['sequence_records']['L']!=4441:raise ValueError('Failed step input record not preserved')
 result=read(b/'source/result.json');source_NN=sum(result[k] for k in ['student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN']);readout=read(b/'source_qualification/complete.json')
 cost['B00'].update(source_counters={k:result[k] for k in ['student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN']},native_source_NN=source_NN,source_readout_NN=readout['NN'],prediction_NN=pred['actual_NN'],total_NN=source_NN+readout['NN']+pred['actual_NN'],native_source_wall_seconds=result['wall_seconds'],prediction_wall_seconds=pred['wall_seconds'])
 cost['B01'].update(total_NN=None,partial_failed_step_L_input_count=128,failed_step_forward_count='not completely recorded;excluded from known lower bound',final_checkpoint=False,formal_metrics=None)
 metrics=[]
 if len(saved['results'])!=126:raise ValueError('All126 scored strata required')
 for v in VIEWS:
  z=next(q for q in saved['results'] if (q['view'],q['dimension'],q['stratum'])==(v,'overall','ALL'));rx=[q['accuracy'] for q in saved['results'] if q['view']==v and q['dimension']=='receiver']
  if z['query_count']!=168000 or len(rx)!=7:raise ValueError('Fixed target population or RX strata differ')
  metrics.append(dict(row='B00-s2026092701',view=v,accuracy_pct=100*z['accuracy'],macro_F1_pct=100*z['macro_f1'],worst_RX_accuracy_pct=100*min(rx)))
 out.mkdir();write(out/'summary.json',dict(status='PARTIAL_TECHNICAL_FAILURE',model_seed_count=1,rows_complete=1,rows_failed=1,successful_updates=total_steps,completed_model_epochs=50,metrics=metrics,cost=cost,shared_launch_smoke_NN=1538,known_total_NN_lower_bound=cost['B00']['total_NN']+cost['B01']['L_NN_on_completed_steps']+1538,new_As_effect=None,clean_0_1pp_stage=None,source_promotion=None,IQ_reads=0,checkpoint_reads=0,truth_reads=0,NN=0,training_updates=0,target_feedback=False,no_automatic_retry=True))
 for name,rows in [('full_completed_epoch_curves.csv',curves),('completed_NoAs_views.csv',metrics)]:
  with (out/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,3,figsize=(13,7));panels=[('loss','Training loss'),('lr','Learning rate'),('source_V_accuracy_pct','Source V accuracy (%)'),('gradient','Gradient norm'),('U_selected','Original U accepted count'),('epoch_seconds','Epoch wall time (s)')]
 for ax,(key,label) in zip(axs.flat,panels):
  for arm in ['B00','B01']:
   rs=[q for q in curves if q['arm']==arm];ax.plot([q['epoch'] for q in rs],[q[key] for q in rs],label=arm+(' completed30' if arm=='B00' else ' failed after20'))
  ax.axvline(20.5,color='red',ls=':',alpha=.5);ax.set(xlabel='Epoch',ylabel=label,xlim=(1,30));ax.grid(alpha=.2)
 axs[0,0].legend();fig.suptitle('Continuous E30: NoAs complete; As stopped by old E20 adapter guard');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/('partial_training_curves.'+ext),dpi=180)
 plt.close(fig);write(out/'complete.json',dict(status='PARTIAL_TECHNICAL_FAILURE_REPORTED',NN=0,truth_reads=0,new_As_effect=None));print(json.dumps(dict(status='PARTIAL_TECHNICAL_FAILURE_REPORTED',epochs=50,updates=total_steps,completed_views=7,NN=0,truth_reads=0)))
if __name__=='__main__':main()
