from pathlib import Path
import json,csv,statistics as st,math
from collections import defaultdict
P=Path('E:/type10-7');OUT=P/'local_artifacts/phase1_transfer_analysis_20261008'
d=json.loads((OUT/'metadata_epochs.json').read_text(encoding='utf-8'))
scorepath=P/'automation_reports/CV-SincNet/20261007-phase1-reference-completed-sixscene-manysig-m48-r01/evidence/combined80/scores.csv'
scores=list(csv.DictReader(scorepath.open(encoding='utf-8-sig')))
overall={(r['row_id'],r['view']):float(r['accuracy'])*100 for r in scores if r['dimension']=='overall'}
groups=defaultdict(list)
for r in d['rows']:
 if r['completion']:groups[(r['config']['stage'],r['config']['arm'])].append(r)
def mean(v):
 v=[x for x in v if x is not None and isinstance(x,(int,float)) and math.isfinite(x)]
 return st.mean(v) if v else None
summary=[];curves=[];paired=[]
for (stage,arm),rows in groups.items():
 v=[r['completion']['final_source_metrics']['source_val_accuracy']*100 for r in rows]
 w=[r['completion']['final_source_metrics']['source_val_worst_rx']*100 for r in rows]
 item=dict(stage=stage,arm=arm,source_v=mean(v),source_worst=mean(w),source_score=mean([(a+b)/2 for a,b in zip(v,w)]),clean=mean([overall[r['row_id'],'clean'] for r in rows]),lowurban=mean([overall[r['row_id'],'practical_low_urban'] for r in rows]),
   val_best=mean([max(e['val_tx_acc'] for e in r['epochs']) for r in rows]),val_best_epoch=[max(r['epochs'],key=lambda e:e['val_tx_acc'])['epoch'] for r in rows],
   lr_min=min(e['lr'] for r in rows for e in r['epochs']),lr_max=max(e['lr'] for r in rows for e in r['epochs']),
   duration_hours=mean([r['completion']['elapsed_seconds']/3600 for r in rows]))
 summary.append(item)
 for epoch in range(1,201):
  eps=[r['epochs'][epoch-1] for r in rows]
  curves.append(dict(stage=stage,arm=arm,epoch=epoch,**{k:mean([e.get(k) for e in eps]) for k in eps[0] if k in ['lr','val_tx_acc','val_dom_acc','val_probe_dom_acc','train_tx_acc','train_reliable_ratio','train_pseudo_conf','train_pseudo_selected','train_pseudo_total','train_grad_before_clip','train_grad_clip_limit','train_grad_clip_active','train_grad_total'] or k.startswith(('train_w_loss_','train_loss_','loss_weight_'))}))
 base='bridge' if stage=='r2' else 'base'
 if (stage,base) in groups and arm!=base:
  br={r['config']['model_seed']:r for r in groups[stage,base]}
  for view in ['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban']:
   dv=[overall[r['row_id'],view]-overall[br[r['config']['model_seed']]['row_id'],view] for r in rows]
   paired.append(dict(stage=stage,arm=arm,base=base,view=view,delta=mean(dv),sd=st.stdev(dv),ci95_low=mean(dv)-3.182446*st.stdev(dv)/2,ci95_high=mean(dv)+3.182446*st.stdev(dv)/2,wins=sum(x>0 for x in dv),deltas=dv,leave_one_out=[mean(dv[:i]+dv[i+1:]) for i in range(4)]))
def csvwrite(name,data):
 fields=list(dict.fromkeys(k for r in data for k in r))
 with (OUT/name).open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(data)
csvwrite('source_target_summary.csv',summary);csvwrite('epoch_curves.csv',curves);csvwrite('paired_differences.csv',paired)
(OUT/'analysis_stats.json').write_text(json.dumps(dict(summary=summary,paired=paired),indent=2),encoding='utf-8')
print('GROUPS')
for r in summary:print(json.dumps(r))
print('PHASE_METRICS')
keys=['val_tx_acc','train_tx_acc','train_reliable_ratio','train_pseudo_conf','train_grad_before_clip','train_grad_clip_limit','train_w_loss_tx_labeled','train_w_loss_sat_cls_labeled','train_w_loss_domain_labeled','train_w_loss_adv_labeled','train_w_loss_cons_labeled','train_w_loss_fishr_labeled','train_w_loss_orth_labeled','train_w_loss_proto_labeled','train_w_loss_source_episode','train_loss_unlabeled','train_pseudo_selected','train_pseudo_total']
for stage,arm in [('r2','bridge'),('r2','pseudo'),('r3','base'),('r3','domain'),('r3','cons'),('r3','fishr'),('r3','all_dg'),('r4','base'),('r4','proto'),('r4','episode')]:
 rows=[r for r in curves if r['stage']==stage and r['arm']==arm and r['epoch']>=181]
 print(stage,arm,{k:mean([r.get(k) for r in rows]) for k in keys})
print('PAIRED_EPISODE', [r for r in paired if r['stage']=='r4' and r['arm']=='episode' and r['view'] in ['clean','practical_low_urban']])
