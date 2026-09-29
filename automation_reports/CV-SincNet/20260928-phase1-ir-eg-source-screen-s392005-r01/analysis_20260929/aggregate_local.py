import gzip,json,csv,statistics,math
from pathlib import Path
base=Path('E:/type10-7/code/worktrees/ir_eg_v1_20260927/automation_reports/CV-SincNet/20260928-phase1-ir-eg-source-screen-s392005-r01/analysis_20260929')
d=json.loads(gzip.decompress((base/'full_scan.json.gz').read_bytes()))
def mean(s,k):return s.get(k,{}).get('mean')
def combine(epochs,k,start,end):
 a=[e[k] for i,e in epochs.items() if start<=int(i)<=end and k in e and 'n' in e[k]]
 return sum(x['mean']*x['n'] for x in a)/sum(x['n'] for x in a) if a else None
keys=['loss','grad_norm','derived.clipped','derived.step_wall_delta','terms.tx','terms.adv','terms.dom','terms.sat_cls','terms.daot_labeled','terms.daot_unlabeled','terms.rc4_identity','terms.rc4_total','daot_rc4.weighted_identity','daot_rc4.hard_count','daot_rc4.partial_count','daot_rc4.route_funnel.before_budget.P.count','derived.P_starved','derived.HP_coverage','derived.U_max_domain_fraction','derived.U_max_pseudoTX_fraction','derived.identity_abs_below_1e-6','daot_rc4.teacher_student.mean_kl','daot_rc4.teacher_student.mean_angle_rad','daot_rc4.route_switch_rate','solver_telemetry.response_norm','solver_telemetry.e_norm','derived.response_over_full_update_norm','solver_telemetry.formal_clip_coefficient','solver_telemetry.cg_iterations','solver_telemetry.linear_relative_residual','solver_telemetry.clip_attribution.native_clip_comparison.difference_norm','solver_telemetry.clip_attribution.native_clip_comparison.cosine','solver_telemetry.heavy_diagnostics.recovery.ce_reduction','solver_telemetry.heavy_diagnostics.recovery.accuracy_gain','learning_rates.backbone','cuda_peak_memory_bytes']
allrows=[];summaries=[]
for row in d['rows']:
 name=row['row'];stats=row['files']['actions.jsonl']['stats'];epochs=row['epochs']['actions.jsonl'];logs=row['records']['logs.jsonl'];last=logs[-1]
 summary=dict(row=name,records={k:{j:v[j] for j in ['records','bytes','invalid','partial_lines']} for k,v in row['files'].items()},sequence_errors=row['sequence_errors'],stdout_anomalies=row['stdout']['anomalies'],U_ID_exposure=row['U_ID_exposure'],last_epoch=last['epoch'],last_elapsed_seconds=last['elapsed_seconds'],source_best=max([(r['source_validation']['accuracy'],r['epoch']) for r in logs]),source_last=last['source_validation']['accuracy'],source_e160=next((r['source_validation']['accuracy'] for r in logs if r['epoch']==160),None),source_e163=next((r['source_validation']['accuracy'] for r in logs if r['epoch']==163),None),source_last_ce=last['source_validation']['ce'],last_loss=last['mean_loss'],means={k:mean(stats,k) for k in keys},phases={})
 for start,end in [(1,20),(21,40),(41,79),(80,90),(91,110),(111,140),(141,160),(161,200)]:summary['phases'][f'{start}-{end}']={k:combine(epochs,k,start,end) for k in keys}
 summary['cg_status']=stats.get('solver_telemetry.cg_status')
 summary['fallback']=stats.get('solver_telemetry.fallback');summary['applied']=stats.get('solver_telemetry.applied')
 summary['timing_e160']=next((r['elapsed_seconds'] for r in logs if r['epoch']==160),None)
 summary['peak_gib']=stats['cuda_peak_memory_bytes']['max']/1024**3
 summary['cumulative_forward_counts']={k.split('.')[-1]:v['last'] for k,v in stats.items() if k.startswith('backbone_forward_counts.')}
 summary['probe_stats']=row['files']['joint_probes.jsonl']['stats']
 summary['ir_diagnostics']={k:v for k,v in stats.items() if ('clip_attribution' in k or 'heavy_diagnostics' in k) and not k.endswith('physical_ids')}
 summary['nonfinite_scalar_paths']={k:v for k,v in stats.items() if v.get('bad')}
 previous=0.
 for log in logs:
  ep=log['epoch'];e=epochs[str(ep)];entry=dict(row=name,epoch=ep,source_accuracy=log['source_validation']['accuracy'],source_ce=log['source_validation']['ce'],elapsed_seconds=log['elapsed_seconds'],epoch_seconds=log['elapsed_seconds']-previous,mean_loss=log['mean_loss']);previous=log['elapsed_seconds']
  entry.update({k:mean(e,k) for k in keys});allrows.append(entry)
 summaries.append(summary)
 (base/(name+'.stdout.log')).write_text(row['stdout_full'],encoding='utf-8')
result=dict(scope=d['scope'],observed_at=d['observed_at'],active_training_processes=d['active_training_processes'],rows=summaries)
(base/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (base/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
with (base/'epoch_metrics.jsonl').open('w',encoding='utf-8') as f:
 for r in allrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for r in summaries:
 print(r['row'],'epoch',r['last_epoch'],'hours',round(r['last_elapsed_seconds']/3600,2),'Vlast',r['source_last'],'best',r['source_best'],'V160',r['source_e160'],'E160h',r['timing_e160']/3600,'peakGiB',r['peak_gib'])
 print('active E141-160',json.dumps(r['phases']['141-160'],ensure_ascii=False))
 print('CG',r['cg_status'],'fallback',r['fallback'],'badlines',{k:(v['invalid'],v['partial_lines']) for k,v in r['records'].items()})
