"""Summarize only independently recounted complete fixed-matrix artifacts."""
import argparse,copy,json,math,statistics
from pathlib import Path
from . import design as d

def stat(values):return dict(mean=statistics.mean(values),sd=statistics.stdev(values),n=len(values))
def summarize(audit,features):
 if audit['status']!='VERIFIED' or audit['run_id']!=d.RUN or audit['models']!=9 or audit['missing'] or audit['new_step_records_parsed']!=66600 or audit['epoch_records_parsed']!=450 or audit['prediction_count']!=1512000 or audit['metric_records_independently_recounted']!=162 or audit['head_fit']:raise ValueError('Complete fixed9row audit required')
 if features['status']!='VERIFIED' or features['run_id']!=d.RUN or features['models']!=6 or features['missing'] or len(features['paired'])!=3 or features['head_fit'] or features['head_updates'] or features['query_read'] or features['U_truth_read']:raise ValueError('Complete fixed6model feature analysis required')
 if {(r['arm'],r['seed']) for r in features['rows']}!={(r['arm'],r['model_seed']) for r in d.training_rows()}:raise ValueError('Actual feature row matrix differs')
 for r in features['rows']:
  f=r['feature_boundary']
  if not f['export_exact'] or not f['class_W_zeroing_leaves_features_identical'] or not f['old_class_W_removed_from_export'] or not f['feature_parameters_buffers_unchanged'] or f['head_fit'] or f['query_read'] or f['feature_parameters']!=176065:raise ValueError('Actual deployed-feature boundary not verified')
 metrics=audit['metrics'];summaries=[];per_seed={}
 for arm in d.ARMS:
  rows=[r for r in audit['rows'] if r['arm']==arm];seeds=sorted(r['seed'] for r in rows)
  if seeds!=list(d.SEEDS):raise ValueError('Three model seeds required per arm')
  overall=[next(m for m in metrics if m['row_id']==r['row_id'] and m['dimension']=='overall') for r in rows]
  for r,m in zip(rows,overall):per_seed[arm,r['seed']]=m['accuracy']
  summaries.append(dict(arm=arm,readout='ORIGINAL_HEAD',view='clean',seed_count=3,accuracy=stat([m['accuracy'] for m in overall]),macro_f1=stat([m['macro_f1'] for m in overall]),seed_accuracy={str(r['seed']):m['accuracy'] for r,m in zip(rows,overall)},worst_RX=stat([min(m['accuracy'] for m in metrics if m['row_id']==r['row_id'] and m['dimension']=='receiver') for r in rows]),worst_day=stat([min(m['accuracy'] for m in metrics if m['row_id']==r['row_id'] and m['dimension']=='day') for r in rows]),U_acceptance=stat([r['U_accepted']/r['U_packets'] for r in rows]),epoch_seconds_sum=sum(e['epoch_time_s'] for r in rows for e in r['source_epoch_curve'])))
 paired=[]
 for arm,control in d.PAIRS:
  difference=[per_seed[arm,s]-per_seed[control,s] for s in d.SEEDS];errors=[next(e for e in audit['error_transitions'] if e['arm']==arm and e['control']==control and e['seed']==s) for s in d.SEEDS]
  if any(abs(delta-(e['corrected_errors']-e['introduced_errors'])/168000)>1e-12 for delta,e in zip(difference,errors)):raise ValueError('Paired differences and actual error transitions disagree')
  paired.append(dict(arm=arm,control=control,seeds=list(d.SEEDS),seed_differences=difference,difference=stat(difference),positive_seeds=sum(v>0 for v in difference),error_transitions=errors))
 primary=next(s for s in summaries if s['arm']==d.PRIMARY)
 result=dict(status='ANALYZED',run_id=d.RUN,actual_training_code='c5e67c46f2a04ec33ff3235d392037f38c000eff',summary=summaries,paired=paired,primary_arm=d.PRIMARY,primary_readout='ORIGINAL_HEAD',goal_84=primary['accuracy']['mean']>=.84,gap_to_84_pp=100*(.84-primary['accuracy']['mean']),models=9,new_models=6,baseline_reused_models=3,views=['clean'],head_fit=False,truth_last=True,target_feedback_to_training=False,feature_claim='Actual deployed feature changes and class-W-independent export, old-six-TX default-head DG only; no newTX/Phase2 proof',cost=dict(new_model_epochs=300,new_updates=66600,historical_model_epochs_reused=150,extra_source_V_passes=12,extra_source_V_packets=324000,extra_U_forwards=0,separate_head_updates=0,source_feature_models=6,source_packets_per_feature_model=720),complete_audit=copy.deepcopy(audit),source_features=copy.deepcopy(features))
 return result

def plot(data,path):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors={'OLD_GATE_COS50':'#777777','RANDOM_ORDER':'#3274A1','PAIRED_ORDER':'#D66E32'}
 fig,ax=plt.subplots(2,2,figsize=(12.8,8.4),constrained_layout=True)
 for i,s in enumerate(data['summary']):
  color=colors[s['arm']];values=100*np.array(list(s['seed_accuracy'].values()));ax[0,0].errorbar(i,100*s['accuracy']['mean'],yerr=100*s['accuracy']['sd'],fmt='o',color=color,capsize=5);ax[0,0].scatter(i+np.linspace(-.07,.07,3),values,color=color,alpha=.7,s=22)
 ax[0,0].axhline(84,color='#A33',ls='--');ax[0,0].set(xticks=range(3),xticklabels=['Historical old gates','Matched random','Neighbor pairs'],ylabel='Clean accuracy (%)',title='Fixed default head: model mean ± sample SD, n=3')
 rows=data['complete_audit']['rows']
 for arm in d.ARMS:
  subset=[r for r in rows if r['arm']==arm];color=colors[arm]
  ce=np.array([[e['train_loss_tx_labeled'] for e in r['source_epoch_curve']] for r in subset]);ax[0,1].plot(range(1,51),ce.mean(0),color=color,label=arm);ax[0,1].fill_between(range(1,51),ce.min(0),ce.max(0),color=color,alpha=.12)
  coverage=np.array([[100*e['train_pseudo_selected']/e['train_pseudo_total'] for e in r['source_epoch_curve'][33:]] for r in subset]);ax[1,0].plot(range(34,51),coverage.mean(0),color=color,label=arm);ax[1,0].fill_between(range(34,51),coverage.min(0),coverage.max(0),color=color,alpha=.12)
 ax[0,1].axvline(33.5,color='#888',ls=':');ax[0,1].set(xlabel='Epoch',ylabel='Source labeled CE',title='All 50 epochs; shading is seed min–max');ax[0,1].legend(fontsize=8)
 ax[1,0].set(xlabel='Pseudo epoch',ylabel='Accepted U packets (%)',ylim=(0,100),title='Original gates retained; U truth hidden')
 fp=data['source_features']['paired'];v=[p['source_features']['records'][0]['joint160_L2_mean'] for p in fp];ax[1,1].scatter(range(1,4),v,color=colors[d.PRIMARY],s=40);ax[1,1].set(xticks=[1,2,3],xticklabels=['Seed01','Seed02','Seed03'],ylabel='Mean source joint160 L2 difference',title='Final PAIRS vs RANDOM on fixed L720');ax[1,1].text(.02,.03,'Feature difference is not proof of new-TX transfer',transform=ax[1,1].transAxes,fontsize=8)
 for a in ax.flat:a.grid(alpha=.2)
 fig.suptitle('Neighbor-pair sampling — original backbone, old gates, EMA and default classifier',fontsize=13);fig.savefig(path,dpi=160);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=summarize(json.loads(a.audit.read_text(encoding='utf8')),json.loads(a.features.read_text(encoding='utf8')));a.output.mkdir(parents=True,exist_ok=True);(a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');plot(result,a.output/'results.png');print(json.dumps(dict(status='ANALYZED',run_id=d.RUN,primary_mean=next(s['accuracy']['mean'] for s in result['summary'] if s['arm']==d.PRIMARY),goal_84=result['goal_84'],head_fit=False)))
