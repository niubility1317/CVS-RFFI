"""Read all three splitAdam R02 source logs, using only the standard library.

Observational analysis only: no checkpoint, IQ, prediction, scoring, or truth
files are opened. Missing/partial telemetry is reported and never launches,
stops, selects, or changes an experiment. Physical IDs stay inside comparisons.
"""
import argparse
from collections import Counter,defaultdict
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import tempfile

RUN='20261019-phase1-FcrossRX-splitAdam-e30-s2026092701-m3-r02'
ARMS=('N01','C01','S01')
SEED=2026092701
STEPS=2220
EPOCHS=10
HEAD='id_backbone.encoder.core.id_backbone.cls_head.weight'
STREAMS=('reference_steps','shape_steps','unit_steps','split_steps','split_components','split_native','step_metrics')
NATIVE_FIELDS=('train/loss','train/loss_labeled','train/loss_tx_labeled','train/loss_unlabeled','train/loss_entropy','train/pseudo_total','train/pseudo_selected','train/pseudo_truth_available','train/a1_ema_successful_updates','train/a1_ema_effective_decay','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad','train/os_gradient_surgery_active','train/os_budget_controller_active')


def number(value):
 return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _clean(value):
 if isinstance(value,float) and not math.isfinite(value):return None
 if isinstance(value,dict):return {key:_clean(child) for key,child in value.items()}
 if isinstance(value,(tuple,list)):return [_clean(child) for child in value]
 return value


class Moments:
 def __init__(self):self.count=0;self.null=0;self.nonfinite=0;self.total=0.;self.low=None;self.high=None
 def add(self,value):
  if value is None:self.null+=1;return
  if not number(value):self.nonfinite+=1;return
  value=float(value);self.count+=1;self.total+=value;self.low=value if self.low is None else min(self.low,value);self.high=value if self.high is None else max(self.high,value)
 def result(self):return dict(count=self.count,null_count=self.null,nonfinite_count=self.nonfinite,mean=self.total/self.count if self.count else None,sum=self.total,min=self.low,max=self.high)


class Checks:
 def __init__(self):self.values=defaultdict(Counter)
 def add(self,key,value):self.values[key]['unknown' if value is None else 'match' if value else 'mismatch']+=1
 def result(self):return {key:dict(match=v['match'],mismatch=v['mismatch'],unknown=v['unknown'],nonblocking=True) for key,v in sorted(self.values.items())}


def _finite_counts(value):
 counts=Counter()
 def walk(v):
  if v is None:counts['null']+=1
  elif isinstance(v,float) and not math.isfinite(v):counts['nonfinite']+=1
  elif isinstance(v,(int,float)) and not isinstance(v,bool):counts['finite_numeric']+=1
  elif isinstance(v,dict):
   for child in v.values():walk(child)
  elif isinstance(v,list):
   for child in v:walk(child)
 walk(value);return counts


def _read_json(path,files,label):
 meta=files.setdefault(label,dict(present=path.is_file(),records=0,parse_errors=0,nonfinite_values=0,null_values=0))
 if not path.is_file():return {}
 try:
  value=json.loads(path.read_text(encoding='utf-8-sig'));counts=_finite_counts(value);meta.update(records=1,nonfinite_values=counts['nonfinite'],null_values=counts['null']);return value
 except (OSError,ValueError,TypeError):meta['parse_errors']+=1;return {}


def _records(path,files,label):
 meta=files.setdefault(label,dict(present=path.is_file(),records=0,parse_errors=0,nonfinite_values=0,null_values=0,ordered_step_mismatches=0,cumulative_step_mismatches=0))
 if not path.is_file():return
 opener=gzip.open if path.suffix=='.gz' else open
 try:
  with opener(path,'rt',encoding='utf-8-sig') as stream:
   for index,line in enumerate(stream,1):
    meta['records']+=1
    try:
     value=json.loads(line)
     if not isinstance(value,dict):raise ValueError('Mapping required')
    except (ValueError,TypeError):meta['parse_errors']+=1;yield None;continue
    counts=_finite_counts(value);meta['nonfinite_values']+=counts['nonfinite'];meta['null_values']+=counts['null']
    if 'step' in value and value['step']!=index:meta['ordered_step_mismatches']+=1
    if 'cumulative_step' in value and value['cumulative_step']!=4440+index:meta['cumulative_step_mismatches']+=1
    yield value
 except (OSError,EOFError,UnicodeError):meta['parse_errors']+=1


def _difference(container,key,actual,expected):
 container[key].add(abs(actual-expected) if number(actual) and number(expected) else None)


def _head_expected(witness,mode):
 """One default-head scalar, from post-step moments; no fitted parameters."""
 if not isinstance(witness,dict):return None
 required=('parameter_before_step','parameter_after_step','lr','weight_decay','beta1','beta2','eps')
 if not all(number(witness.get(key)) for key in required):return None
 before=witness['parameter_before_step'];lr=witness['lr'];wd=witness['weight_decay'];b1=witness['beta1'];b2=witness['beta2'];eps=witness['eps']
 if witness.get('native_active') is False:return before
 def direction(first,second,step):
  if not all(number(v) for v in (first,second,step)) or step<=0 or second<0:return None
  return first/(1-b1**step)/(math.sqrt(second/(1-b2**step))+eps)
 if mode=='SPLIT':
  parts=[]
  for part in ('rest','as'):
   step=witness.get(part+'_step')
   if step==0:parts.append(0.);continue
   value=direction(witness.get(part+'_first'),witness.get(part+'_second'),step)
   if value is None:return None
   parts.append(value)
  adaptive=.8*parts[0]+.2*parts[1]
 elif mode=='MEASURE':
  adaptive=direction(witness.get('native_first'),witness.get('native_second'),witness.get('optimizer_state_step'))
  if adaptive is None:return None
 else:return None
 return before*(1-lr*wd)-lr*adaptive


def _flags(args):
 if isinstance(args.get('args'),dict):args=args['args']
 keys=('amp','seed','epochs','label_epochs','pseudo_epochs','batch_size','label_smoothing','lr','weight_decay','lambda_u','lambda_ent','use_ema_teacher','use_concat_sat_channel_aug','use_aug','a1_runtime_fast','os_gradient_surgery','os_budget_controller','source_fold_held_rx')
 return {key:args[key] for key in keys if key in args and (args[key] is None or isinstance(args[key],(bool,int,float)) or isinstance(args[key],str) and len(args[key])<48 and '/' not in args[key] and '\\' not in args[key])}


def _stdout(path):
 result=dict(present=path.is_file(),lines=0,traceback_headers=0,OOM_error_lines=0,nonfinite_error_lines=0,exception_terminal_lines=0,scalar_tokens={},read_errors=0)
 flags=defaultdict(Counter)
 if not path.is_file():return result
 terminal=re.compile(r'^\s*(?:[A-Za-z_][\w.]*Error|Exception):')
 try:
  with path.open('rt',encoding='utf8',errors='replace') as stream:
   for line in stream:
    result['lines']+=1
    if 'Traceback (most recent call last):' in line:result['traceback_headers']+=1
    is_error=bool(terminal.search(line) or re.match(r'^\s*(?:ERROR|FATAL)\b',line))
    if terminal.search(line):result['exception_terminal_lines']+=1
    if re.search(r'CUDA out of memory|(?:torch\.)?OutOfMemoryError:',line,re.I):result['OOM_error_lines']+=1
    if is_error and re.search(r'nonfinite|non-finite|\bnan\b|\binfinity\b',line,re.I):result['nonfinite_error_lines']+=1
    for key,value in re.findall(r'\b(amp|os_gradient_surgery|os_budget_controller|a1_runtime_fast)=(0|1|True|False|true|false)\b',line):flags[key][value]+=1
 except OSError:result['read_errors']+=1
 result['scalar_tokens']={key:dict(value) for key,value in sorted(flags.items())};return result


def _row(base,arm):
 source=base/(arm+'-s'+str(SEED))/'source';files={};checks=Checks();moments=defaultdict(Moments);diffs=defaultdict(Moments);epochs=defaultdict(lambda:defaultdict(Moments));totals=Counter();phases=Counter();modes=Counter()
 result=_read_json(source/'result.json',files,'result.json');completion=_read_json(source/'completion.json',files,'completion.json');args=_read_json(source/'actual_native_args.json',files,'actual_native_args.json');ordinary=_read_json(source/'ordinary_augmentation_actual.json',files,'ordinary_augmentation_actual.json');config=result.get('config',completion.get('config',{}));sat=arm!='N01';expected_mode='SPLIT' if arm=='S01' else 'MEASURE'
 streams=[_records(source/(name+'.jsonl.gz'),files,name+'.jsonl.gz') for name in STREAMS]
 for index,values in enumerate(zip_longest(*streams),1):
  mapping={key:value or {} for key,value in zip(STREAMS,values)};r=mapping['reference_steps'];s=mapping['shape_steps'];u=mapping['unit_steps'];g=mapping['split_steps'];c=mapping['split_components'];n=mapping['split_native'];m=mapping['step_metrics'].get('metrics',{});ep=(index-1)//222+1
  if not isinstance(m,dict):m={}
  for name,value in mapping.items():checks.add('record_step_'+name,value.get('step')==index if value else None)
  if r:
   checks.add('L6_same_IQ_supervised',r.get('L_forwards')==6 and r.get('same_IQ_each_pair') is True and r.get('all_satellite_L_supervised') is True and r.get('ordinary_redraws')==0)
   checks.add('L_As64_or_NoAs0',r.get('As_packets')==(64 if sat else 0));checks.add('SupCon_original_coefficient',r.get('SupCon_weight')==.1)
   for key in ('common_RD','common_CE','common_KL','extra_RD','extra_CE','extra_KL','SupCon','total_L'):
    moments[key].add(r.get(key));epochs[ep][key].add(r.get(key))
   if all(number(r.get(k)) for k in ('common_RD','extra_RD','SupCon')) and number(s.get('shape_RD')):
    original=.5*r['common_RD']+.5*r['extra_RD']+.25*s['shape_RD']+.1*r['SupCon'];_difference(diffs,'L_total_vs_original_1p25',r.get('total_L'),original)
    _difference(diffs,'native_total_vs_original_L_plus_U',n.get('total_loss'),original+n['unlabeled_loss'] if number(n.get('unlabeled_loss')) else None)
   for prefix in ('common','extra'):_difference(diffs,prefix+'_RD_vs_CE_plus_KL',r.get(prefix+'_RD'),r[prefix+'_CE']+r[prefix+'_KL'] if number(r.get(prefix+'_CE')) and number(r.get(prefix+'_KL')) else None)
  if s:
   checks.add('raw_control_same_IQ_weight025',s.get('same_IQ_each_pair') is True and s.get('weight')==.25 and s.get('NN')==256)
   moments['raw_RD'].add(s.get('shape_RD'));epochs[ep]['raw_RD'].add(s.get('shape_RD'));_difference(diffs,'raw_RD_vs_CE_plus_KL',s.get('shape_RD'),s['shape_CE']+s['shape_KL'] if number(s.get('shape_CE')) and number(s.get('shape_KL')) else None)
  if u:checks.add('one_physical_clip_per_native_step',u.get('physical_clip_calls')==index)
  if g:
   modes[str(g.get('mode'))]+=1;checks.add('optimizer_mode',g.get('mode')==expected_mode);checks.add('two_VJP_per_native_step',g.get('extra_VJP_calls')==2);totals['extra_VJP_calls']+=g.get('extra_VJP_calls',0) if number(g.get('extra_VJP_calls')) else 0
   for key in ('capture_host_enqueue_seconds','clip_factor','physical_global_norm','component_sum_max_abs_error'):
    moments[key].add(g.get(key));epochs[ep][key].add(g.get(key))
   comparison=g.get('component_sum_comparison',{});pointwise=comparison.get('old_pointwise_failure_coordinates');ratio=comparison.get('max_error_bound_ratio')
   moments['RMS_error_bound_ratio'].add(ratio);epochs[ep]['RMS_error_bound_ratio'].add(ratio);moments['old_pointwise_failure_coordinates'].add(pointwise)
   if number(pointwise):totals['old_pointwise_failure_coordinates']+=pointwise;totals['steps_with_old_pointwise_failures']+=pointwise>0
   totals['RMS_ratio_over_one_steps']+=number(ratio) and ratio>1;checks.add('RMS_nonblocking_pointwise_telemetry',comparison.get('rule')=='per_parameter_RMS' and comparison.get('old_pointwise_nonblocking') is True)
   for witness in g.get('witness',[]):
    if witness.get('parameter')!=HEAD:continue
    expected=_head_expected(witness,expected_mode);_difference(diffs,'default_head_first_coordinate_after_vs_AdamW_moments',witness.get('parameter_after_step'),expected);checks.add('default_head_native_optimizer_counter',witness.get('optimizer_state_step')==index)
  if c:
   checks.add('original_three_RD_terms_no_extra_NN',c.get('native_RD_calls')==3 and c.get('masked_RD_recomputations')==int(sat) and c.get('extra_NN')==0 and c.get('U_changed') is False and c.get('L_forwards')==6 and c.get('As')==sat)
   moments['As_RD'].add(c.get('As_RD'));epochs[ep]['As_RD'].add(c.get('As_RD'));checks.add('NoAs_component_zero',c.get('As_RD')==0. if not sat else True)
  if n:
   phases[str(n.get('phase'))]+=1;checks.add('native_original_U_included',n.get('includes_original_U') is True and n.get('extra_NN')==0 and n.get('labeled_count')==128)
   for key in ('labeled_count','unlabeled_count'):
    if number(n.get(key)):totals[key]+=n[key]
   for key,out in [('total_loss','native_total_loss'),('unlabeled_loss','weighted_U_loss')]:moments[out].add(n.get(key));epochs[ep][out].add(n.get(key))
   _difference(diffs,'native_total_vs_recorded_L_plus_U',n.get('total_loss'),r['total_L']+n['unlabeled_loss'] if number(r.get('total_L')) and number(n.get('unlabeled_loss')) else None)
   _difference(diffs,'captured_As_component_vs_native_probe',c.get('As_RD'),n.get('As_RD'))
  if m:
   for key in NATIVE_FIELDS:epochs[ep][key].add(m.get(key))
   checks.add('native_successful_EMA_counter',m.get('train/a1_ema_successful_updates')==index);checks.add('U_TX_truth_hidden',m.get('train/pseudo_truth_available')==0)
   for key in ('train/pseudo_total','train/pseudo_selected','train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad'):
    if number(m.get(key)):totals[key]+=m[key]
   _difference(diffs,'native_metrics_total_vs_probe',m.get('train/loss'),n.get('total_loss'))
   _difference(diffs,'native_metrics_full_L_vs_record',m.get('concat/total_L'),r.get('total_L'))
   for key in ('train/os_gradient_surgery_active','train/os_budget_controller_active'):checks.add(key+'_zero',m.get(key)==0)
   checks.add('native_required_total_finite',number(m.get('train/loss')))
 checks.add('all_result_2220_native_updates',result.get('optimizer_steps')==STEPS and completion.get('optimizer_steps')==STEPS);checks.add('all_result_EMA_2220',result.get('EMA_local_successful_updates')==STEPS and result.get('EMA_cumulative_updates')==6660)
 checks.add('U_physical_total567000',totals['unlabeled_count']==567000 and result.get('U_physical_samples')==567000 and result.get('U_steps')==STEPS)
 checks.add('student_train_L_plus_U',result.get('student_train_NN')==STEPS*768+totals['unlabeled_count']);checks.add('teacher_eval_equals_U',result.get('teacher_eval_NN')==totals['unlabeled_count'] and result.get('teacher_train_NN')==0)
 checks.add('native_U_unmodified',result.get('native_U_unchanged') is True and result.get('U_As')==0 and result.get('extra_U_NN')==0 and result.get('U_truth_read') is False)
 checks.add('ordinary_schedule13',ordinary.get('schedule_frozen_at')==13 and ordinary.get('schedule_args_epochs')==20 and ordinary.get('no_ordinary_before_practical') is True and ordinary.get('RDrop_same_IQ') is True)
 checks.add('ordinary_CROSS_RX_128',config.get('crossview',{}).get('mode')=='ORD128');checks.add('all_source_RX_no_holdout',result.get('source_RX')==[1,3,4,6,8] and result.get('source_RX_holdout') is None);checks.add('default_head_no_separate_fit',result.get('head_fit_updates')==0)
 epoch_file=list(_records(source/'epoch_metrics.jsonl',files,'epoch_metrics.jsonl'));epoch_native=[]
 for index,value in enumerate(epoch_file,1):
  if not isinstance(value,dict):continue
  selected={key:value[key] for key in NATIVE_FIELDS if key in value};epoch_native.append(dict(local_epoch=index,metrics=selected))
  actual=epochs[index].get('train/loss');_difference(diffs,'native_epoch_loss_vs_complete_step_mean',value.get('train/loss'),actual.total/actual.count if actual and actual.count else None)
 ep_summary=[dict(local_epoch=ep,cumulative_epoch=ep+20,metrics={key:value.result() for key,value in sorted(epochs[ep].items())}) for ep in sorted(epochs)]
 changes={}
 if epochs:
  first,last=min(epochs),max(epochs)
  for key in sorted(set(epochs[first])&set(epochs[last])):
   a=epochs[first][key];b=epochs[last][key]
   if a.count and b.count:
    aa=a.total/a.count;bb=b.total/b.count;changes[key]=dict(first_epoch=first,last_epoch=last,first_mean=aa,last_mean=bb,change=bb-aa,relative_change=(bb-aa)/abs(aa) if aa else None)
 cost_keys=('L_NN','L_As','U_steps','U_physical_samples','U_student_NN','U_teacher_NN','student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','extra_VJP_calls','optimizer_steps','EMA_local_successful_updates','EMA_cumulative_updates','wall_seconds')
 cost={key:result.get(key) for key in cost_keys};cost['native_training_and_validation_NN']=sum(result[key] for key in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')) if all(number(result.get(key)) for key in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN')) else None
 ordinary_args=ordinary.get('args',{});ordinary_scalars={k:v for k,v in ordinary_args.items() if v is None or isinstance(v,(bool,int,float))}
 return dict(arm=arm,row_id=arm+'-s'+str(SEED),expected_steps=STEPS,files=files,missing=[key for key,value in files.items() if not value['present']],observational_checks=checks.result(),cost=cost,actual_native_args_scalars=_flags(args),ordinary_augmentation=dict(schedule_frozen_at=ordinary.get('schedule_frozen_at'),schedule_args_epochs=ordinary.get('schedule_args_epochs'),RDrop_same_IQ=ordinary.get('RDrop_same_IQ'),no_ordinary_before_practical=ordinary.get('no_ordinary_before_practical'),args_scalars=ordinary_scalars),step_totals=dict(totals),optimizer_modes=dict(modes),native_phases=dict(phases),step_statistics={key:value.result() for key,value in sorted(moments.items())},absolute_deviations={key:value.result() for key,value in sorted(diffs.items())},epoch_step_means=ep_summary,native_epoch_metrics=epoch_native,loss_change_first_to_last=changes,stdout=_stdout(base/'logs'/(arm+'-s'+str(SEED)+'.log')),head_witness_scope='First default-head coordinate only;float64 scalar reconstruction of actual stored post-step moments;Sinc origin not inferred',nonblocking=True)


def _sequences(base,rows):
 files={};streams=[];stats={arm:dict(records=0,roles={},batch_size_histogram={},unknown_role_records=0,malformed_id_lists=0) for arm in ARMS};hist={arm:defaultdict(Counter) for arm in ARMS};pairs={(a,b):dict(compared_records=0,equal_records=0,missing_record_positions=0,role_mismatch_records=0,ID_order_mismatch_records=0) for i,a in enumerate(ARMS) for b in ARMS[i+1:]}
 for arm in ARMS:streams.append(_records(base/(arm+'-s'+str(SEED))/'source/training_sequence.jsonl.gz',files,arm))
 for values in zip_longest(*streams):
  for arm,value in zip(ARMS,values):
   if not isinstance(value,dict):continue
   stats[arm]['records']+=1;role=value.get('role');ids=value.get('ids')
   if role not in ('L','U'):stats[arm]['unknown_role_records']+=1;continue
   if not isinstance(ids,list):stats[arm]['malformed_id_lists']+=1;continue
   count=len(ids);record=stats[arm]['roles'].setdefault(role,dict(batches=0,physical_packets=0));record['batches']+=1;record['physical_packets']+=count;hist[arm][role][count]+=1
  for (a,b),summary in pairs.items():
   left=values[ARMS.index(a)];right=values[ARMS.index(b)]
   if not isinstance(left,dict) or not isinstance(right,dict):summary['missing_record_positions']+=1;continue
   summary['compared_records']+=1;same_role=left.get('role')==right.get('role');same_ids=left.get('ids')==right.get('ids');summary['equal_records']+=same_role and same_ids;summary['role_mismatch_records']+=not same_role;summary['ID_order_mismatch_records']+=not same_ids
 for arm in ARMS:
  stats[arm]['batch_size_histogram']={role:{str(k):v for k,v in sorted(counts.items())} for role,counts in hist[arm].items()};stats[arm]['file']=files[arm];rows[arm]['training_sequence']=stats[arm]
  roles=stats[arm]['roles'];stats[arm]['observational_counts']=dict(L_batches_equal2220=roles.get('L',{}).get('batches')==STEPS,U_batches_equal2220=roles.get('U',{}).get('batches')==STEPS,L_physical_packets_equal284160=roles.get('L',{}).get('physical_packets')==STEPS*128,U_physical_packets_equal567000=roles.get('U',{}).get('physical_packets')==567000,U_packets_equal_native_probe=roles.get('U',{}).get('physical_packets')==rows[arm]['step_totals'].get('unlabeled_count'),nonblocking=True)
 for summary in pairs.values():summary['all_record_roles_and_ID_order_equal']=bool(summary['compared_records'] and summary['equal_records']==summary['compared_records'] and not summary['missing_record_positions'])
 return {a+'_vs_'+b:summary for (a,b),summary in pairs.items()}


def summarize(base:Path):
 base=Path(base);rows={arm:_row(base,arm) for arm in ARMS};sequences=_sequences(base,rows);all_missing={arm:value['missing']+([] if value['training_sequence']['file']['present'] else ['training_sequence.jsonl.gz'])+([] if value['stdout']['present'] else ['stdout.log']) for arm,value in rows.items()}
 complete=all(not all_missing[arm] and all(rows[arm]['files'][name+'.jsonl.gz']['records']==STEPS for name in STREAMS) and rows[arm]['files']['epoch_metrics.jsonl']['records']==EPOCHS and not any(meta.get('parse_errors',0) for meta in rows[arm]['files'].values()) and not rows[arm]['training_sequence']['file'].get('parse_errors') for arm in ARMS)
 observed={}
 for source,target in [('native_training_and_validation_NN','native_training_and_validation_NN'),('optimizer_steps','successful_optimizer_updates'),('EMA_local_successful_updates','EMA_updates'),('extra_VJP_calls','extra_VJP_calls'),('U_student_NN','U_student_NN'),('U_teacher_NN','U_teacher_NN'),('wall_seconds','sum_native_worker_wall_seconds')]:
  values=[rows[arm]['cost'].get(source) for arm in ARMS];observed[target]=sum(values) if all(number(value) for value in values) else None
 return _clean(dict(status='SOURCE_LOGS_PARSED_COMPLETE' if complete else 'SOURCE_LOGS_PARSED_WITH_MISSING_OR_PARTIAL_ARTIFACTS',run_id=RUN,expected_model_seeds=1,expected_rows=3,rows=rows,observed_training=observed,training_sequence_pairwise=sequences,missing=all_missing,analysis_only=True,nonblocking=True,all_lines_processed=True,NN=0,checkpoint_reads=0,IQ_reads=0,prediction_reads=0,truth_reads=0,updates=0,no_training_or_selection_gate=True))


def fixture():
 """Small arithmetic/empty-file fixture, with no model, IQ, or real run input."""
 count=0
 def check(value):
  nonlocal count
  if not value:raise AssertionError('Source log fixture failed')
  count+=1
 m=Moments();m.add(2);m.add(4);m.add(None);check(m.result()['mean']==3. and m.result()['count']==2 and m.result()['null_count']==1);check(Moments().result()['mean'] is None);check(_clean(dict(value=float('inf')))['value'] is None)
 witness=dict(parameter_before_step=.2,parameter_after_step=.0,lr=.01,weight_decay=.1,beta1=.9,beta2=.999,eps=1e-8,native_active=True,optimizer_state_step=1,native_first=.2,native_second=.004)
 check(abs(_head_expected(witness,'MEASURE')-(.2*(1-.01*.1)-.01*2/(2+1e-8)))<1e-12)
 split=dict(witness,rest_step=1,rest_first=.2,rest_second=.004,as_step=1,as_first=-.1,as_second=.001);check(abs(_head_expected(split,'SPLIT')-(.2*(1-.01*.1)-.01*(.8*2/(2+1e-8)-.2/(1+1e-8))))<1e-12)
 with tempfile.TemporaryDirectory(prefix='splitadam_log_fixture_') as temp:
  base=Path(temp)
  for arm in ARMS:
   source=base/(arm+'-s'+str(SEED))/'source';source.mkdir(parents=True)
   with gzip.open(source/'training_sequence.jsonl.gz','wt',encoding='utf8') as stream:
    stream.write(json.dumps(dict(role='L',ids=['mockL']))+'\n');stream.write(json.dumps(dict(role='U',ids=['mockU','mockU2']))+'\n')
   with gzip.open(source/'split_native.jsonl.gz','wt',encoding='utf8') as stream:stream.write(json.dumps(dict(step=1,labeled_count=128,unlabeled_count=2,total_loss=3.,unlabeled_loss=1.,includes_original_U=True,extra_NN=0))+'\n')
  value=summarize(base);check(value['status']=='SOURCE_LOGS_PARSED_WITH_MISSING_OR_PARTIAL_ARTIFACTS');check(value['rows']['N01']['step_totals']['unlabeled_count']==2);check(value['rows']['N01']['training_sequence']['roles']['U']['physical_packets']==2);check(all(x['all_record_roles_and_ID_order_equal'] for x in value['training_sequence_pairwise'].values()));text=json.dumps(value,allow_nan=False);check('mockL' not in text and 'mockU' not in text)
 return dict(status='PASS',checks=count,NN=0,checkpoint_reads=0,IQ_reads=0,truth_reads=0,updates=0)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('base',nargs='?');parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
 if args.fixture:output=fixture()
 elif args.base:output=summarize(Path(args.base))
 else:parser.error('Supply a source run base or --fixture')
 print(json.dumps(output,ensure_ascii=False,allow_nan=False))
