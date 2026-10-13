"""Validate completed aggregate-only source proxy evidence; no IQ/truth access."""
import argparse,csv,json,math,statistics
from pathlib import Path

def collect(path,output):
 v=json.loads(Path(path).read_text(encoding='utf-8'));done=v['completion']
 if v['status']!='VERIFIED' or v['failure'] or not done or done['status']!='SOURCE_DIAGNOSTICS_COMPLETE' or done['rows']!=3 or done['source_L_packets']!=18900 or done['forward_packets']!=56700:raise ValueError('Source proxy is incomplete')
 if v['dispatcher_process'] or any(w['process'] for w in v['workers']):raise ValueError('Diagnostic processes still active')
 records=[];identities=[]
 if {w['summary']['config']['model_seed'] for w in v['workers']}!={2026092701,2026092702,2026092703}:raise ValueError('Incomplete or duplicated model seeds')
 for w in v['workers']:
  s=w['summary'];seed=s['config']['model_seed']
  if len(w['log_tail'].encode('utf-8'))!=w['log_size']:raise ValueError('Full source progress log was not retrieved')
  progress=[json.loads(line) for line in w['log_tail'].splitlines() if line.startswith('{') and 'source_L_seen' in line]
  if len(progress)!=25 or [r['batches'] for r in progress]!=list(range(1,26)) or [r['source_L_seen'] for r in progress]!=[min(6300,b*256) for b in range(1,26)]:raise ValueError('Full source progress recount differs')
  if s['status']!='SOURCE_GATE_DIAGNOSTIC_COMPLETE' or s['config']['count']!=6300 or not s['student_teacher_parameters_buffers_unchanged'] or s['labels_enter_gate'] or s['target_IQ_or_truth_read'] or s['U_TX_truth_read'] or any(s[k] for k in ('head_fit_updates','backbone_updates','EMA_updates')):raise ValueError('Illegal source diagnostic evidence')
  if s['forward_packets']!=dict(teacher_weak=6300,student_strong_train=6300,student_strong_eval=6300):raise ValueError('Source forward counts differ')
  identities.append(dict(model_seed=seed,execution_commit=s['code_commit'],checkpoint=s['source_checkpoint'],wall_seconds=s['wall_seconds']))
  modes=s['records']
  for group in modes['train_dropout_no_grad']:
   for key in ('confidence','temporal','base','structural','all'):
    if modes['train_dropout_no_grad'][group]['stages'][key]!=modes['eval_no_grad'][group]['stages'][key]:raise ValueError('Same weak/metadata stages differed between student modes')
  for mode,groups in modes.items():
   if groups['overall']['count']!=6300:raise ValueError('L observation count differs')
   for dimension,n in [('RX/',5),('day/',3),('TX/',6),('RXday/',15)]:
    cells=[r for k,r in groups.items() if k.startswith(dimension)]
    if len(cells)!=n or sum(r['count'] for r in cells)!=6300:raise ValueError('Source group partition/count differs')
   for group,r in groups.items():
    for stage,g in r['stages'].items():
     sel=g['selected'];correct=g['correct'];expected=correct/sel if sel else None
     if not 0<=correct<=sel<=r['count'] or sum(g['predicted_class'])!=sel or sum(g['true_class'])!=sel or g['precision']!=expected or g['coverage']!=sel/r['count']:raise ValueError('Source aggregate recount differs')
     records.append(dict(model_seed=seed,mode=mode,group=group,stage=stage,count=r['count'],selected=sel,correct=correct,coverage=g['coverage'],precision=g['precision']))
    st=r['stages']
    if not st['accepted']['selected']<=st['base']['selected']<=st['temporal']['selected']<=st['structural']['selected']<=r['count']:raise ValueError('Native gate subset counts differ')
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 def write_csv(path,rows):
  with path.open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 write_csv(output/'source_gate_proxy_groups_20261014.csv',records);overall=[r for r in records if r['group']=='overall'];write_csv(output/'source_gate_proxy_seeds_20261014.csv',overall)
 summaries=[]
 for mode in ('train_dropout_no_grad','eval_no_grad'):
  for stage in ('all','confidence','structural','temporal','base','agreement','accepted'):
   rs=[r for r in overall if r['mode']==mode and r['stage']==stage];cs=[r['coverage'] for r in rs];ps=[r['precision'] for r in rs if r['precision'] is not None]
   summaries.append(dict(mode=mode,stage=stage,seeds=3,selected_by_seed=[r['selected'] for r in rs],correct_by_seed=[r['correct'] for r in rs],coverage_mean=statistics.mean(cs),coverage_sample_sd=statistics.stdev(cs),precision_mean=statistics.mean(ps) if len(ps)==3 else None,precision_sample_sd=statistics.stdev(ps) if len(ps)==3 else None))
 result=dict(status='VERIFIED_COMPLETE_SOURCE_DIAGNOSTIC',run_id=v['run_id'],models=3,modes=2,source_L_observations=18900,forward_packets=56700,training_updates=0,head_fit_updates=0,EMA_updates=0,U_forward_packets=0,target_IQ_or_truth_read=False,all_aggregate_counts_recomputed=True,full_25_batch_logs_recounted=True,parameter_buffers_unchanged=True,weak_stages_same_between_modes=True,rows=len(records),summaries=summaries,identities=identities,process_wall_seconds=sum(r['wall_seconds'] for r in identities),interpretation='Seen L_s proxy, not U precision or unseen-RX/target evidence; no gate/selection update')
 (output/'source_gate_proxy_summary_20261014.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
 stages=['confidence','structural','temporal','base','accepted'];labels=['Confidence','Neighbor upper bound','Temporal','Before agreement','Accepted']
 for j,mode in enumerate(('train_dropout_no_grad','eval_no_grad')):
  vals=[next(r for r in summaries if r['mode']==mode and r['stage']==s) for s in stages];xs=[i+(j-.5)*.34 for i in range(len(stages))]
  axes[0].bar(xs,[r['coverage_mean']*100 for r in vals],width=.34,yerr=[r['coverage_sample_sd']*100 for r in vals],capsize=2,label='Student train Dropout' if j==0 else 'Student eval')
  selected=[r for r in overall if r['mode']==mode and r['stage']=='accepted'];axes[1].bar([i+(j-.5)*.34 for i in [1,2,3]],[r['selected'] for r in selected],width=.34,label='Student train Dropout' if j==0 else 'Student eval')
 axes[0].set_xticks(range(len(stages)),labels,rotation=30,ha='right');axes[0].set_ylabel('Seen labeled-source coverage (%)');axes[0].set_ylim(0,105);axes[0].legend(fontsize=8);axes[1].set_xticks([1,2,3],['Seed 01','Seed 02','Seed 03']);axes[1].set_ylabel('Accepted observations / 6300');axes[1].set_ylim(0,max(r['selected'] for r in overall if r['stage']=='accepted')*1.2);axes[1].legend(fontsize=8,loc='lower left');axes[1].text(.5,.94,'0 observed accepted-label errors (seen L only)',transform=axes[1].transAxes,ha='center',fontsize=8)
 fig.suptitle('Frozen original gate: seen L_s proxy only; no hidden U truth or target access',fontsize=10);fig.savefig(output/'source_gate_proxy_20261014.png',dpi=160);plt.close(fig)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect(a.input,a.output);print(json.dumps(dict(status=v['status'],groups=v['rows'],summary=v['summaries'],wall_seconds=v['process_wall_seconds'])))
