"""Read all current source logs;no IQ,checkpoint,truth or target-score inputs."""
import argparse,collections,gzip,json,math,re,time
from pathlib import Path
RUN='20261018-phase1-F-crossRX-batchStratify-e20plus10-s2026092701-m3-r01'
MODES=('ORIGINAL','RANDOM','TX_RX_BALANCED')

def jsonl(path,compressed=False):
 rows=[];error=None
 if not path.exists():return rows,dict(exists=False,records=0,partial=False,error=None)
 try:
  with (gzip.open(path,'rt',encoding='utf8') if compressed else path.open(encoding='utf8')) as stream:
   for line in stream:
    if not line.endswith('\n'):error='partial_record';break
    rows.append(json.loads(line))
 except (EOFError,OSError) as e:error=type(e).__name__
 return rows,dict(exists=True,records=len(rows),partial=error is not None,error=error)
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def summary(values):
 values=[v for v in values if isinstance(v,(int,float)) and not isinstance(v,bool)];finite=[float(v) for v in values if math.isfinite(v)]
 return dict(count=len(values),nonfinite=len(values)-len(finite),mean=sum(finite)/len(finite) if finite else None,minimum=min(finite) if finite else None,maximum=max(finite) if finite else None,first=finite[0] if finite else None,last=finite[-1] if finite else None)
def geometry(y,ids,indices):
 tx=collections.Counter(y[i] for i in indices);rx=collections.Counter(ids[i][1] for i in indices);cell=collections.Counter((y[i],ids[i][1]) for i in indices);positives=[tx[y[i]]-cell[(y[i],ids[i][1])] for i in indices]
 return dict(cell_squared_imbalance=sum((cell[(t,r)]-128/30)**2 for t in range(6) for r in (1,3,4,6,8))/30,TX_squared_imbalance=sum((tx[t]-128/6)**2 for t in range(6))/6,RX_squared_imbalance=sum((rx[r]-128/5)**2 for r in (1,3,4,6,8))/5,empty_TX_RX_cells=sum(cell[(t,r)]==0 for t in range(6) for r in (1,3,4,6,8)),valid_anchors=sum(v>0 for v in positives),positive_min=min(positives),positive_max=max(positives),positive_sum=sum(positives))
def run(project,complete=False):
 B=Path(project)/'runs'/RUN;result={};inputs={};sequences={};args_all={};launch=read(B/'launch.json');root_complete=(B/'completion.json').exists();began=time.perf_counter()
 if complete and not root_complete:raise ValueError('Final analysis requires all registered rows complete')
 for mode in MODES:
  rid=mode+'-s2026092701';S=B/rid/'source';a=read(S/'actual_native_args.json');a=a.get('args',a);args_all[mode]=a
  steps,step_info=jsonl(S/'step_metrics.jsonl.gz',True);epochs,epoch_info=jsonl(S/'epoch_metrics.jsonl');pools,pool_info=jsonl(S/'sampler_pools.jsonl.gz',True);sequence,seq_info=jsonl(S/'training_sequence.jsonl.gz',True);reference,ref_info=jsonl(S/'reference_steps.jsonl.gz',True);units,unit_info=jsonl(S/'unit_steps.jsonl.gz',True)
  if [r['step'] for r in steps]!=list(range(1,len(steps)+1)):raise ValueError('Nonconsecutive current steps')
  bad_steps=[];allkeys=set(k for r in steps for k in r['metrics']);numeric={k:summary([r['metrics'].get(k) for r in steps]) for k in sorted(allkeys)}
  for r in steps:
   m=r['metrics']
   if m.get('concat/As_packets')!=128 or m.get('concat/L_forwards')!=4 or m.get('train/pseudo_truth_available',0)!=0 or m.get('train/skipped_nonfinite_loss',0) or m.get('train/skipped_nonfinite_grad',0):bad_steps.append(r['step'])
  original_inputs=[];expected_L=[];original_stats=[];output_stats=[];max_geom_error=0.
  for index,p in enumerate(pools):
   if p['pool']!=index+1 or p['mode']!=mode or len(p['original_L_ids'])!=768 or len(p['visible_y'])!=768 or sorted(p['permutation'])!=list(range(768)):raise ValueError('Pool/exposure/permutation mismatch')
   original_inputs.append((p['original_L_ids'],p['visible_y']));ordered=[p['original_L_ids'][j] for j in p['permutation']];expected_L.extend([ordered[j:j+128] for j in range(0,768,128)])
   for group in range(6):
    for name,ix,stats in [('original_geometry',list(range(group*128,(group+1)*128)),original_stats),('output_geometry',p['permutation'][group*128:(group+1)*128],output_stats)]:
     measured=geometry(p['visible_y'],p['original_L_ids'],ix);saved=p[name][group]
     for k,v in measured.items():max_geom_error=max(max_geom_error,abs(v-saved[k]))
     stats.append(measured)
  inputs[mode]=original_inputs;L=[r['ids'] for r in sequence if r['role']=='L'];U=[r['ids'] for r in sequence if r['role']=='U'];n=min(len(L),len(expected_L));L_alignment=L[:n]==expected_L[:n];sequences[mode]=dict(L=L,U=U)
  if not L_alignment or max_geom_error>1e-12 or bad_steps:raise ValueError('Actual execution violated fixed source objective/alignment')
  log=(B/'logs'/(rid+'.stdout.log')).read_text();errors={v:log.count(v) for v in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','Killed','skipped_nonfinite')};warnings=[line for line in log.splitlines() if re.search(r'warning|warn:',line,re.I)];progress=read(S/'reference_progress.json') if (S/'reference_progress.json').exists() else None
  value=dict(step_log=step_info,epoch_log=epoch_info,pool_log=pool_info,sequence_log=seq_info,reference_log=ref_info,unit_log=unit_info,progress=progress,all_current_steps_As128_L4_hiddenU=True,pool_L_alignment_verified_batches=n,pool_geometry_max_error=max_geom_error,all_stdout_scanned=True,error_markers=errors,warning_count=len(warnings),numeric_totals=numeric,epoch_curves=epochs,observed_L_packets=sum(len(r) for r in L),observed_U_packets=sum(len(r) for r in U),observed_unique_L=len({tuple(z) for row in L for z in row}),observed_unique_U=len({tuple(z) for row in U for z in row}),source_RX=sorted({z[1] for row in L for z in row}),source_days=sorted({z[2] for row in L for z in row}),source_RX_holdout=None,original_batch_geometry={k:summary([s[k] for s in original_stats]) for k in original_stats[0]} if original_stats else {},output_batch_geometry={k:summary([s[k] for s in output_stats]) for k in output_stats[0]} if output_stats else {})
  reference_epochs=[]
  for epoch in sorted({r['epoch'] for r in reference}):
   group=[r for r in reference if r['epoch']==epoch];reference_epochs.append(dict(epoch=epoch,cumulative_epoch=20+epoch,steps=len(group),**{k:summary([r.get(k) for r in group]) for k in ['total_L','common_CE','common_KL','extra_CE','extra_KL','SupCon']}))
  value['reference_epoch_curves']=reference_epochs
  value['supervised_objective_recount_max_error']=max((abs(r['total_L']-(.5*(r['common_CE']+r['common_KL'])+.5*(r['extra_CE']+r['extra_KL'])+r['SupCon_weight']*r['SupCon'])) for r in reference),default=0.)
  if value['supervised_objective_recount_max_error']>2e-6:raise ValueError('Original supervised L reduction differs')
  if (S/'training_result.json').exists():value['training_cost']=read(S/'training_result.json')
  if complete:
   if any(v.get('partial') or not v['exists'] for v in (step_info,epoch_info,pool_info,seq_info,ref_info,unit_info)) or [len(steps),len(epochs),len(pools),len(sequence),len(reference),len(units)]!=[2220,10,370,4440,2220,2220] or len(L)!=2220 or len(U)!=2220 or n!=2220 or value['observed_L_packets']!=284160 or value['observed_U_packets']!=567000 or value['observed_unique_L']!=6300 or value['observed_unique_U']!=56700:raise ValueError('Incomplete source budget/membership logs')
   if any(errors.values()):raise ValueError('Technical marker in final stdout')
  result[mode]=value
 comparisons={};base=MODES[0]
 for mode in MODES[1:]:
  n=min(len(inputs[base]),len(inputs[mode]));u=min(len(sequences[base]['U']),len(sequences[mode]['U']));differences=[k for k in set(args_all[base])|set(args_all[mode]) if args_all[base].get(k)!=args_all[mode].get(k)]
  comparisons[mode]=dict(complete_input_pools_compared=n,input_pools_and_visible_L_labels_identical=inputs[base][:n]==inputs[mode][:n],U_batches_compared=u,U_sequence_identical=sequences[base]['U'][:u]==sequences[mode]['U'][:u],actual_args_keys=len(args_all[mode]),different_actual_arg_keys=sorted(differences))
  if not comparisons[mode]['input_pools_and_visible_L_labels_identical'] or not comparisons[mode]['U_sequence_identical'] or set(differences)-{'candidate_id','output_dir'}:raise ValueError('Fixed matched inputs/U/args differ across rows')
 return dict(status='COMPLETE_SOURCE_LOGS_VERIFIED' if complete else 'LIVE_SOURCE_LOGS_PARTIAL',run_id=RUN,complete=root_complete,rows=result,matched_comparisons=comparisons,IQ=0,CP=0,NN=0,training_updates=0,target_scores_read=False,wall_seconds=time.perf_counter()-began)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--complete',action='store_true');a=p.parse_args();print(json.dumps(run(a.project,a.complete),allow_nan=False))
