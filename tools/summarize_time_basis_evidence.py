"""Report the completed, fixed temporal-basis matrix and historical controls."""
import argparse,csv,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def csv_out(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(source,pairs,compact,operations,root):
 v=read(source)
 if not v['all_rows_complete'] or v['complete_scored_rows']!=6 or (v.get('completion') or {}).get('status')!='EXPERIMENT_COMPLETE':raise ValueError('Complete registered three-seed results required')
 p=read(pairs);h=read(compact);ops=read(operations)
 assert len(p['trained_feature_evidence'])==3 and len(p['actual_initial_matches'])==3
 assert p['status']=='VERIFIED' and p['independent_confusions_recounted']==162 and p['all_prediction_sets_fixed_before_truth'] and p['head_fit_updates']==0
 refs=[r for r in h['rows'] if r['arm']=='COMPACT_LINEAR'];assert len(refs)==3 and all(r['row_complete'] for r in refs)
 entries=v['rows']+refs;assert len(entries)==9
 arms=('BASE_REF','COMPACT_LINEAR','SINGLE_BASIS');seeds=(2026092701,2026092702,2026092703)
 rows=[];curves=[];metrics=[];geometries=[]
 for r in entries:
  audit=r['source_audit'];assert audit['status']=='PASS' and audit['epochs']==50 and audit['steps']==11100 and audit['nonfinite_or_skipped']==0 and audit['U_truth_available']==0
  assert audit['U_observation_draws']==963900 and len(audit['curve'])==50
  ms=r['metric_records'];assert len(ms)==18;metrics.extend(ms)
  overall=next(m for m in ms if m['dimension']=='overall')
  diag=r['source_diagnostics'];geom=diag['geometry'];assert geom['role']=='L_s' and geom['count']==6300
  rows.append(dict(run_id=r['config']['run_id'],arm=r['arm'],seed=r['model_seed'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_RX=min(m['accuracy'] for m in ms if m['dimension']=='receiver'),worst_day=min(m['accuracy'] for m in ms if m['dimension']=='day'),source_L_CE=audit['final_L_CE'],source_V=audit['final_source']['source_val_accuracy'],source_worst_RX=audit['final_source']['source_val_worst_rx'],U_observations=audit['U_observation_draws'],U_accepted=audit['U_accepted'],source_process_seconds=audit['training_seconds']))
  for c in audit['curve']:curves.append(dict(arm=r['arm'],seed=r['model_seed'],**c))
  geometries.append(dict(arm=r['arm'],seed=r['model_seed'],role='L_s',count=6300,effective_rank=geom['global_spectrum']['effective_rank'],participation_rank=geom['global_spectrum']['participation_rank'],between_TX_fraction=geom['variance']['between_TX_fraction'],RX_within_TX_fraction=geom['variance']['RX_within_TX_fraction'],within_TX_RX_fraction=geom['variance']['within_TX_RX_fraction']))
 assert len(metrics)==162 and len(curves)==450
 summary=[];paired=[]
 for arm in arms:
  g=[r for r in rows if r['arm']==arm];assert len(g)==3 and {r['seed'] for r in g}==set(seeds)
  summary.append(dict(arm=arm,seed_count=3,accuracy_mean=statistics.mean(r['accuracy'] for r in g),accuracy_sd=statistics.stdev(r['accuracy'] for r in g),macro_f1_mean=statistics.mean(r['macro_f1'] for r in g),worst_RX_mean=statistics.mean(r['worst_RX'] for r in g),worst_day_mean=statistics.mean(r['worst_day'] for r in g)))
 index={(r['arm'],r['seed']):r for r in rows}
 for control in ('BASE_REF','COMPACT_LINEAR'):
  differences=[index['SINGLE_BASIS',s]['accuracy']-index[control,s]['accuracy'] for s in seeds]
  paired.append(dict(treatment='SINGLE_BASIS',control=control,seeds=list(seeds),differences=differences,mean_difference=statistics.mean(differences),sample_sd=statistics.stdev(differences),positive_seeds=sum(x>0 for x in differences)))
 stages=[];feedback=[];feature=[]
 for r in p['trained_feature_evidence']:
  seed=int(r['row_id'].split('-s')[1]);assert not r['U_truth_read'] and all(r['export'].values())
  for e in r['source_stages']:
   delta=e['feature_change_vs_epoch33_L32']
   stages.append(dict(seed=seed,epoch=e['epoch'],source_V=e['source']['source_val_accuracy'],source_worst_RX=e['source']['source_val_worst_rx'],joint160_delta_vs33=json.dumps(delta)))
  for e in r['source_U_feedback']:
   for name,g in e['groups'].items():feedback.append(dict(seed=seed,epoch=e['epoch'],RX_day_group=name,total=g['total'],accepted=g['accepted'],agreement=g['agreement'],predicted_class=json.dumps(g['predicted_class']),accepted_class=json.dumps(g['accepted_class'])))
  for layer,values in r['parameter_updates'].items():
   assert all(x['changed_values']>0 for x in values.values());ranks=r['kernels'][layer]['preGN_rank_per_input']
   feature.append(dict(seed=seed,layer=layer,parameter_tensors=len(values),all_tensors_updated=True,changed_parameter_values=sum(x['changed_values'] for x in values.values()),preGN_rank_min=min(ranks),preGN_rank_max=max(ranks),actual_export_equals_feature_only=True,W_modified_deleted_invariant=True,not_feature_rank=True))
 new=[r for r in rows if r['arm']=='SINGLE_BASIS'];out=root/'analysis';out.mkdir(exist_ok=True)
 for name,values in [('seeds',rows),('source_curves',curves),('source_geometry',geometries),('source_stages',stages),('U_feedback',feedback),('trained_parameters',feature),('error_transitions',p['error_transitions'])]:csv_out(out/f'time_basis_{name}_20261014.csv',values)
 csv_out(out/'time_basis_groups_20261014.csv',[{k:m[k] for k in ('row_id','arm','model_seed','dimension','stratum','accuracy','macro_f1','query_count')} for m in metrics])
 result=dict(status='VERIFIED',run_id=v['run_id'],execution_commit=v['release_commit'],registered_rows=6,new_backbones=3,historical_original_refs=3,external_compact_refs=3,new_model_epochs=150,new_updates=33300,historical_baseline_epochs=150,historical_baseline_updates=33300,historical_compact_epochs=150,historical_compact_updates=33300,new_prediction_count=504000,reused_baseline_count=504000,external_compact_decisions=504000,recounted_decisions=p['total_recounted_decisions'],recounted_CM=162,head_fit_updates=0,summary=summary,paired=paired,actual_initial_matches=p['actual_initial_matches'],source_U_truth_read=False,new_source_process_seconds=sum(r['source_process_seconds'] for r in new),process_time_not_exclusive_GPU=True,goal84=next(s['accuracy_mean'] for s in summary if s['arm']=='SINGLE_BASIS')>=.84,goal_gap_pp=100*(.84-next(s['accuracy_mean'] for s in summary if s['arm']=='SINGLE_BASIS')),operation_count_source='time_basis_operation_counts_20261014.json',single_to_compact_time_MAC_ratio=ops['single_to_compact_MAC_ratio'],limitations=['Different time functions and optimizer reparameterization; parameter/MAC equality does not equal function capacity','Kernel rank is not joint160 feature rank','Default-head outcome and measured source feature activity; no new-TX transfer, new headfit or satellite deployment evidence','Native standalone entropy component was not logged; complete CE/total/U count curves retained without reconstruction'])
 (out/'time_basis_summary_20261014.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10});fig,ax=plt.subplots(2,2,figsize=(11,6.7),layout='constrained');colors={'BASE_REF':'#7b8794','COMPACT_LINEAR':'#3b82b8','SINGLE_BASIS':'#e07935'}
 for arm in arms:
  for panel,key in [(ax[0,0],'mean_L_CE'),(ax[0,1],'source_V_percent'),(ax[1,0],'mean_total_loss'),(ax[1,1],'U_accept_rate')]:
   epochs=range(34,51) if key=='U_accept_rate' else range(1,51);x=[];mean=[];sd=[]
   for e in epochs:
    values=np.asarray([r[key] for r in curves if r['arm']==arm and r['epoch']==e])*(100 if key=='U_accept_rate' else 1);assert len(values)==3
    x.append(e);mean.append(values.mean());sd.append(values.std(ddof=1))
   panel.plot(x,mean,color=colors[arm],label=arm,lw=1.8);panel.fill_between(x,np.maximum(0,np.asarray(mean)-sd),np.asarray(mean)+sd,color=colors[arm],alpha=.12)
 for panel,title in zip(ax.flat,['Labeled-source CE','Source V accuracy (%)','Weighted total training loss','Hidden U acceptance (%)']):
  panel.set_title(title,loc='left');panel.set_xlabel('Epoch');panel.grid(alpha=.2);panel.spines[['top','right']].set_visible(False)
 ax[0,0].set_yscale('log');ax[1,0].set_yscale('log');ax[0,1].legend(frameon=False)
 fig.suptitle('Temporal basis comparison: fixed 50 epochs, three model seeds\nBands: sample SD; source evidence only',fontsize=13);fig.savefig(out/'time_basis_source_curves_20261014.png',dpi=180);plt.close(fig)
 print(json.dumps(dict(status='VERIFIED',rows=9,metrics=162,summary=summary,paired=paired,new_source_hours=result['new_source_process_seconds']/3600)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--compact',type=Path,required=True);p.add_argument('--operations',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();summarize(a.source,a.pairs,a.compact,a.operations,a.root)
