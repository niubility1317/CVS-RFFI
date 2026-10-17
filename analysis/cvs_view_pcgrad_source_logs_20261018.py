"""Read all current source logs;no IQ,checkpoint,truth or target-score inputs."""
import argparse,collections,gzip,json,math,re,time
from pathlib import Path
RUN='20261018-phase1-F-crossRX-viewPCGrad-e20plus10-s2026092701-m3-r02'
MODES=('ORIGINAL','MEASURE','PROJECT')

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
def run(project,complete=False,expected_As_packets=128):
 B=Path(project)/'runs'/RUN;result={};inputs={};sequences={};args_all={};launch=read(B/'launch.json');root_complete=all((B/(m+'-s2026092701')/'source/training_result.json').exists() for m in MODES);began=time.perf_counter()
 if complete and not root_complete:raise ValueError('Final analysis requires all registered rows complete')
 for mode in MODES:
  rid=mode+'-s2026092701';S=B/rid/'source';a=read(S/'actual_native_args.json');a=a.get('args',a);args_all[mode]=a
  steps,step_info=jsonl(S/'step_metrics.jsonl.gz',True);epochs,epoch_info=jsonl(S/'epoch_metrics.jsonl');pools,pool_info=jsonl(S/'sampler_pools.jsonl.gz',True);sequence,seq_info=jsonl(S/'training_sequence.jsonl.gz',True);reference,ref_info=jsonl(S/'reference_steps.jsonl.gz',True);units,unit_info=jsonl(S/'unit_steps.jsonl.gz',True)
  if [r['step'] for r in steps]!=list(range(1,len(steps)+1)):raise ValueError('Nonconsecutive current steps')
  bad_steps=[];allkeys=set(k for r in steps for k in r['metrics']);numeric={k:summary([r['metrics'].get(k) for r in steps]) for k in sorted(allkeys)}
  for r in steps:
   m=r['metrics']
   if m.get('concat/As_packets')!=expected_As_packets or m.get('concat/L_forwards')!=4 or m.get('train/pseudo_truth_available',0)!=0 or m.get('train/skipped_nonfinite_loss',0) or m.get('train/skipped_nonfinite_grad',0):bad_steps.append(r['step'])
  original_inputs=[];expected_L=[];original_stats=[];output_stats=[];max_geom_error=0.
  for index,p in enumerate(pools):
   if p['pool']!=index+1 or p['mode']!='ORIGINAL' or p['permutation']!=list(range(768)) or len(p['original_L_ids'])!=768 or len(p['visible_y'])!=768 or sorted(p['permutation'])!=list(range(768)):raise ValueError('Pool/exposure/permutation mismatch')
   original_inputs.append((p['original_L_ids'],p['visible_y']));ordered=[p['original_L_ids'][j] for j in p['permutation']];expected_L.extend([ordered[j:j+128] for j in range(0,768,128)])
   for group in range(6):
    for name,ix,stats in [('original_geometry',list(range(group*128,(group+1)*128)),original_stats),('output_geometry',p['permutation'][group*128:(group+1)*128],output_stats)]:
     measured=geometry(p['visible_y'],p['original_L_ids'],ix);saved=p[name][group]
     for k,v in measured.items():max_geom_error=max(max_geom_error,abs(v-saved[k]))
     stats.append(measured)
  inputs[mode]=original_inputs;L=[r['ids'] for r in sequence if r['role']=='L'];U=[r['ids'] for r in sequence if r['role']=='U'];n=min(len(L),len(expected_L));L_alignment=L[:n]==expected_L[:n];sequences[mode]=dict(L=L,U=U)
  if not L_alignment or max_geom_error>1e-12 or bad_steps:raise ValueError('Actual execution violated fixed source objective/alignment')
  log=(B/'logs'/(rid+'.stdout.log')).read_text();errors={v:log.count(v) for v in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','Killed','skipped_nonfinite')};warnings=[line for line in log.splitlines() if re.search(r'warning|warn:',line,re.I)];progress=read(S/'reference_progress.json') if (S/'reference_progress.json').exists() else None
  value=dict(step_log=step_info,epoch_log=epoch_info,pool_log=pool_info,sequence_log=seq_info,reference_log=ref_info,unit_log=unit_info,progress=progress,all_current_steps_expected_As_L4_hiddenU=True,expected_As_packets_per_step=expected_As_packets,pool_L_alignment_verified_batches=n,pool_geometry_max_error=max_geom_error,all_stdout_scanned=True,error_markers=errors,warning_count=len(warnings),numeric_totals=numeric,epoch_curves=epochs,observed_L_packets=sum(len(r) for r in L),observed_U_packets=sum(len(r) for r in U),observed_unique_L=len({tuple(z) for row in L for z in row}),observed_unique_U=len({tuple(z) for row in U for z in row}),source_RX=sorted({z[1] for row in L for z in row}),source_days=sorted({z[2] for row in L for z in row}),source_RX_holdout=None,original_batch_geometry={k:summary([s[k] for s in original_stats]) for k in original_stats[0]} if original_stats else {},output_batch_geometry={k:summary([s[k] for s in output_stats]) for k in output_stats[0]} if output_stats else {})
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

def gradient_logs(source,mode,complete):
 records,info=jsonl(source/'view_gradient_steps.jsonl.gz',True);numeric={};epochs=[];max_errors=dict(alpha=0.,cosine=0.,zeroed_dot=0.,delta_norm=0.);applied=conflicts=VJP=0;correction2=signal2=0.;before2=after2=0.;multipliers=[]
 if records and [r['step'] for r in records]!=list(range(1,len(records)+1)):raise ValueError('Nonconsecutive view gradients')
 for r in records:
  if r['mode']!=mode or r['cumulative_step']!=r['step']+4440 or r['extra_VJP_calls']!=(0 if mode=='ORIGINAL' else 2):raise ValueError('Wrong view gradient mode or budget')
  applied+=r['applied'];VJP+=r['extra_VJP_calls']
  if mode=='ORIGINAL':
   if r['applied'] or r['coefficient']!=0.:raise ValueError('Original gradient changed')
   continue
  o,s,dot=r['ordinary_norm'],r['satellite_norm'],r['dot'];oo=o*o;alpha=-dot/oo if dot<0 and oo>0 else 0.;cosine=dot/(o*s) if o>0 and s>0 else None
  if r['conflict']!=(dot<0) or min(o,s)<0 or not all(math.isfinite(v) for v in (o,s,dot,alpha,r['capture_host_enqueue_seconds'])):raise ValueError('Invalid scalar view geometry')
  conflicts+=r['conflict'];actual=alpha if mode=='PROJECT' else 0.
  if r['applied']!=bool(actual) or abs(r['applied_coefficient']-actual)>1e-10*max(1.,abs(actual)):raise ValueError('Actual versus theoretical projection differs')
  max_errors['alpha']=max(max_errors['alpha'],abs(r['coefficient']-alpha)/max(1.,abs(alpha)))
  max_errors['zeroed_dot']=max(max_errors['zeroed_dot'],abs(r['projected_satellite_dot']-(dot+alpha*oo))/max(1.,abs(dot)))
  max_errors['delta_norm']=max(max_errors['delta_norm'],abs(r['projection_delta_norm']-alpha*o)/max(1.,alpha*o))
  if cosine is None:
   if r['cosine'] is not None:raise ValueError('Zero norm must have null cosine')
  else:
   max_errors['cosine']=max(max_errors['cosine'],abs(r['cosine']-cosine))
   if abs(cosine)>1.+1e-10:raise ValueError('Invalid Cauchy-Schwarz geometry')
  correction2+=(actual*o)**2;signal2+=s*s;before2+=max(0.,oo+s*s+2*dot);after2+=max(0.,oo+s*s+2*dot+2*actual*(oo+dot)+actual*actual*oo);multipliers.append(1.+actual)
 for epoch in sorted({(r['step']-1)//222+1 for r in records}):
  group=[r for r in records if (r['step']-1)//222+1==epoch];epochs.append(dict(epoch=epoch,cumulative_epoch=20+epoch,steps=len(group),conflicts=sum(r.get('conflict',False) for r in group),applied=sum(r['applied'] for r in group),extra_VJP_calls=sum(r['extra_VJP_calls'] for r in group),**{k:summary([r.get(k) for r in group]) for k in ('dot','ordinary_norm','satellite_norm','cosine','coefficient','projection_delta_norm','capture_host_enqueue_seconds')}))
 if any(v>1e-10 for v in max_errors.values()):raise ValueError('Scalar projection recount differs')
 if complete and (info['partial'] or not info['exists'] or len(records)!=2220 or VJP!=(0 if mode=='ORIGINAL' else 4440)):raise ValueError('Incomplete view-gradient records')
 cost=read(source/'training_result.json') if (source/'training_result.json').exists() else None
 if cost is not None and (cost['gradient_mode']!=mode or cost['extra_supervised_VJP_calls']!=VJP or cost['gradient_projection_calls']!=len(records) or cost['gradient_projection_nonzero']!=applied):raise ValueError('View gradient actual counters disagree')
 return dict(log=info,scalar_recount_max_errors=max_errors,conflicts=conflicts,actual_projection_updates=applied,extra_VJP_calls=VJP,extra_L_backward_packet_derivatives=VJP*256,actual_projection_L2_RMS_fraction_of_As_gradient=math.sqrt(correction2/signal2) if signal2 else 0.,capture_wall_seconds=sum(r.get('capture_host_enqueue_seconds',0.) for r in records),effective_ordinary_gradient_multiplier=summary(multipliers),two_supervised_view_gradient_RMS_norm_before=math.sqrt(before2/len(records)) if records and mode!='ORIGINAL' else None,two_supervised_view_gradient_RMS_norm_after=math.sqrt(after2/len(records)) if records and mode!='ORIGINAL' else None,excludes_SupCon_U_clip_AdamW=True,epochs=epochs,scalar_logs_do_not_reconstruct_actual_tensor_gradients=True)

_base_run=run
def run(project,complete=False):
 value=_base_run(project,complete);B=Path(project)/'runs'/RUN
 for mode in MODES:value['rows'][mode]['view_gradient']=gradient_logs(B/(mode+'-s2026092701')/'source',mode,complete)
 a,ai=jsonl(B/'ORIGINAL-s2026092701/source/reference_steps.jsonl.gz',True);b,bi=jsonl(B/'MEASURE-s2026092701/source/reference_steps.jsonl.gz',True);n=min(len(a),len(b));stats={}
 for key in ('total_L','common_RD','common_CE','common_KL','extra_RD','extra_CE','extra_KL','SupCon','ordinary_raw_MSE','second_raw_MSE'):
  differences=[abs(x[key]-y[key]) for x,y in zip(a[:n],b[:n])];stats[key]=dict(paired_records=n,max_absolute_error=max(differences,default=0.),nonzero_records=sum(v>0 for v in differences),first_over_1e_minus7=next((i+1 for i,v in enumerate(differences) if v>1e-7),None))
 value['ORIGINAL_MEASURE_reference_comparison']=stats
 parent_init=read(B/'ORIGINAL-s2026092701/source/initialization.json')
 parent_path=Path(parent_init['ancestors'][0])/'training_result.json'
 value['parent_training_result']=read(parent_path)
 for mode in MODES:
  row=value['rows'][mode];truth=row['numeric_totals'].get('train/pseudo_truth_available',{})
  row['hidden_U_truth_records']=truth.get('count',0)
  if complete and (truth.get('count')!=2220 or truth.get('maximum')!=0):raise ValueError('Hidden U truth logs incomplete or nonzero')
 value['new_IQ_Ag']=False;value['joint_goal_complete']=False
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--complete',action='store_true');a=p.parse_args();print(json.dumps(run(a.project,a.complete),allow_nan=False))
