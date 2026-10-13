"""Aggregate already-frozen Sinc evidence; no model fitting or remote launch."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_csv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect(result,features,audit,root):
 v=json.loads(result.read_text(encoding='utf-8'));f=json.loads(features.read_text(encoding='utf-8'));a=json.loads(audit.read_text(encoding='utf-8'))
 assert v['completion']['status']=='EXPERIMENT_COMPLETE' and len(v['rows'])==9
 assert a['all_workers_terminal'] and a['prediction_count']==1512000 and a['metric_records_independently_recounted']==162 and a['new_step_records_parsed']==66600
 assert f['status']=='VERIFIED' and f['models']==6 and not f['missing'] and not f['head_fit']
 arms=('OLD_GATE_COS50','RAW_HZ','CENTER_25K');seeds=(2026092701,2026092702,2026092703);records=[];curves=[];rows=[]
 for r in v['rows']:
  assert len(r['epoch_curve'])==50 and [e['epoch'] for e in r['epoch_curve']]==list(range(1,51))
  ms=r['score']['records'];assert len(ms)==18;records.extend(ms)
  for m in ms:
   cm=np.asarray(m['confusion'],dtype=int);n=int(cm.sum());den=cm.sum(0)+cm.sum(1)
   assert n==m['query_count'] and float(cm.trace()/n)==m['accuracy']
   assert abs(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()-m['macro_f1'])<1e-14
  overall=next(m for m in ms if m['dimension']=='overall')
  s=r['source'];k=r['Sinc_diagnostic'];feat=next((x for x in f['rows'] if x['row_id']==r['row_id']),None)
  rows.append(dict(row_id=r['row_id'],arm=r['arm'],seed=r['seed'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=min(m['accuracy'] for m in ms if m['dimension']=='receiver'),worst_day=min(m['accuracy'] for m in ms if m['dimension']=='day'),source_L_CE=s['final_training_L_CE'],source_V=s['final_source_metrics']['source_val_accuracy'],source_worst_RX=s['final_source_metrics']['source_val_worst_rx'],kernel_relative_L2=k['records'][-1]['Sinc']['kernel_relative_L2'] if k else None,unchanged_kernel_channels=k['records'][-1]['Sinc']['kernel_exactly_unchanged_channels'] if k else None,source_U_packets=sum(e['train_pseudo_total'] for e in r['epoch_curve']),source_U_accepted=sum(e['train_pseudo_selected'] for e in r['epoch_curve']),source_seconds=s['elapsed_seconds'],feature_boundary_verified=bool(feat and feat['feature_boundary']['export_exact']),Sinc_restored_z_delta=feat['Sinc_restoration']['records'][0]['final_to_initial_Sinc_joint160_L2_mean'] if feat else None))
  for e in r['epoch_curve']:
   assert e['test_tx_total']==0 and e['test_tx_acc'] is None and e['val_tx_total']==27000
   kr=next(x['Sinc']['kernel_relative_L2'] for x in k['records'] if x['epoch']==e['epoch']) if k else None
   curves.append(dict(row_id=r['row_id'],arm=r['arm'],seed=r['seed'],epoch=e['epoch'],L_CE=e['train_loss_tx_labeled'],source_V_percent=e['val_tx_acc'],U_packets=e['train_pseudo_total'],U_accepted=e['train_pseudo_selected'],U_acceptance=(e['train_pseudo_selected']/e['train_pseudo_total'] if e['train_pseudo_total'] else None),Sinc_kernel_relative_L2=kr))
 summaries=[]
 for arm in arms:
  group=[r for r in rows if r['arm']==arm];assert len(group)==3
  summaries.append(dict(arm=arm,seed_count=3,accuracy_mean=statistics.mean(r['accuracy'] for r in group),accuracy_sd=statistics.stdev(r['accuracy'] for r in group),macro_f1_mean=statistics.mean(r['macro_f1'] for r in group),worst_RX_mean=statistics.mean(r['worst_RX'] for r in group),worst_day_mean=statistics.mean(r['worst_day'] for r in group)))

 table=root/'analysis/phase1_feature_evidence_seed_table_20261014.csv'
 with table.open(encoding='utf-8-sig',newline='') as stream:
  reader=csv.DictReader(stream);fields=reader.fieldnames;historical=list(reader)
 historical=[r for r in historical if r['run_id']!=a['run_id']]
 for r in rows:
  s=next(s for s in summaries if s['arm']==r['arm'])
  historical.append(dict(run_id=a['run_id'],recipe='A33_label17_pseudo',arm=r['arm'],readout='NATIVE_HEAD',head_recipe='joint11100steps',seed=r['seed'],accuracy=r['accuracy'],accuracy_mean=s['accuracy_mean'],accuracy_sd=s['accuracy_sd'],macro_f1_mean=s['macro_f1_mean'],worst_RX_mean=s['worst_RX_mean'],status='VERIFIED',seed_macro_f1=r['macro_f1'],seed_worst_RX=r['worst_RX'],seed_worst_day=r['worst_day'],worst_day_mean=s['worst_day_mean']))
 with table.open('w',encoding='utf-8',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(historical)

 out=root/'analysis';out.mkdir(exist_ok=True)
 write_csv(out/'sinc_coordinate_seeds_20261014.csv',rows);write_csv(out/'sinc_coordinate_source_curves_20261014.csv',curves)
 write_csv(out/'sinc_coordinate_groups_20261014.csv',[{k:m[k] for k in ('row_id','arm','model_seed','dimension','stratum','accuracy','macro_f1','query_count')} for m in records])
 write_csv(out/'sinc_coordinate_error_transitions_20261014.csv',a['error_transitions'])
 report=dict(status='VERIFIED',run_id=a['run_id'],execution_commit=a['executing_code'],rows=9,new_backbones=6,historical_references=3,epoch_records=450,new_step_records=66600,prediction_count=1512000,independent_CM_recount=162,head_fit_updates=0,default_head_joint_only=True,summary=summaries,paired=v['analysis']['paired'],actual_initial_equality=a['actual_initial_equality'],kernel_movement=[{k:r[k] for k in ('row_id','kernel_relative_L2','unchanged_kernel_channels','Sinc_restored_z_delta')} for r in rows if r['arm']!='OLD_GATE_COS50'],actual_source_feature_scope=f['source_IQ'],source_process_seconds_sum=sum(r['source_seconds'] for r in rows if r['arm']!='OLD_GATE_COS50'),process_time_not_exclusive_GPU_cost=True,goal84=False,goal_gap_pp=100*(.84-next(s['accuracy_mean'] for s in summaries if s['arm']=='CENTER_25K')),baseline_epoch_counts_are_totals=True,source_features_changed_but_new_TX_transfer_unmeasured=True)
 (out/'sinc_coordinate_summary_20261014.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10});fig,ax=plt.subplots(2,2,figsize=(11,6.7),layout='constrained');colors={'OLD_GATE_COS50':'#7b8794','RAW_HZ':'#3b82b8','CENTER_25K':'#e07935'}
 for arm in arms:
  for panel,key in [(ax[0,0],'L_CE'),(ax[0,1],'source_V_percent'),(ax[1,0],'U_acceptance'),(ax[1,1],'Sinc_kernel_relative_L2')]:
   if key=='Sinc_kernel_relative_L2' and arm=='OLD_GATE_COS50':continue
   epoch=range(34,51) if key=='U_acceptance' else range(1,51);x=[];means=[];sd=[]
   for e in epoch:
    values=[r[key] for r in curves if r['arm']==arm and r['epoch']==e and r[key] is not None];assert len(values)==3
    values=np.asarray(values)*(100 if key=='U_acceptance' else 1);x.append(e);means.append(values.mean());sd.append(values.std(ddof=1))
   panel.plot(x,means,color=colors[arm],label=arm,lw=1.8);panel.fill_between(x,np.maximum(0,np.asarray(means)-sd),np.asarray(means)+sd,color=colors[arm],alpha=.12)
 for panel,title in zip(ax.flat,['Labeled-source CE','Source V accuracy (%)','Hidden U acceptance (%)','Actual Sinc kernel relative L2']):
  panel.set_title(title,loc='left');panel.set_xlabel('Epoch');panel.grid(alpha=.2);panel.spines[['top','right']].set_visible(False)
 ax[0,0].set_yscale('log');ax[1,1].set_yscale('symlog',linthresh=1e-8);ax[0,1].legend(frameon=False)
 fig.suptitle('Sinc coordinate comparison: fixed 50 epochs, three model seeds\nBands: sample SD; source evidence only',fontsize=13)
 fig.savefig(out/'sinc_coordinate_source_curves_20261014.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(status='VERIFIED',rows=len(rows),epochs=len(curves),metrics=len(records),summary=summaries,source_hours=report['source_process_seconds_sum']/3600)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();collect(a.results,a.features,a.audit,a.root)
