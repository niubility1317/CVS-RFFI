"""Read saved complete epoch-replay training and frozen seven-view decisions.

The remote script performs no model forwards, predictions, optimizer updates or
scoring jobs. Collection requires the final run and all original scores first.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = '20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01'
REFERENCE_RUN = '20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
RELEASE = 'cvs_F_epochReplay_E20_20261019_r01'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')

SCRIPT = r'''
import csv,gzip,importlib.util,itertools,json,math,re
from collections import Counter
from pathlib import Path
import numpy as np
CHECK_CONTEXT={}

def read(path):
 return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
 path=Path(path)
 with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as stream:
  return [json.loads(line) for line in stream if line.strip()]

def number(record,key):
 value=record[key]
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
  raise ValueError('Required finite scalar absent: '+key+'; location='+json.dumps(CHECK_CONTEXT,sort_keys=True))
 return float(value)

def require(condition,message):
 if not condition:raise ValueError(message+'; location='+json.dumps(CHECK_CONTEXT,sort_keys=True))

def metrics(y,pred):
 cm=np.bincount(y*6+pred,minlength=36).reshape(6,6);den=cm.sum(0)+cm.sum(1)
 return dict(n=len(y),accuracy=float(np.trace(cm)/len(y)),
   macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den>0).mean()),confusion=cm.tolist())

def paired(y,before,after):
 old,new=before==y,after==y;both=int((old&new).sum());broken=int((old&~new).sum())
 delta=float(100*(new.mean()-old.mean()))
 return dict(n=len(y),corrected=int((~old&new).sum()),new_errors=broken,damaged=broken,
  both_correct=both,both_wrong=int((~old&~new).sum()),net_pp=delta,mean_pp=delta,
  baseline_correct_retention=None if both+broken==0 else both/(both+broken),
  model_seed_count=1,sample_SD=None,positive_model_seeds=int(delta>0),tied_model_seeds=int(delta==0))

def sequence(path,role=None):
 with gzip.open(path,'rt',encoding='utf8') as stream:
  for line in stream:
   row=json.loads(line)
   require(row['role'] in ('L','U'),'Unknown recorded training role')
   if role is None or row['role']==role:yield row

def sequence_budget(path):
 batches=Counter();packets=Counter()
 for row in sequence(path):batches[row['role']]+=1;packets[row['role']]+=len(row['ids'])
 require(dict(batches)=={'L':4440,'U':1554} and dict(packets)=={'L':568320,'U':396900},
         'Complete original L/U recorded sequence budget differs')
 return dict(batches=dict(batches),packets=dict(packets))

def sequence_comparisons(paths):
 result={}
 for a,b in itertools.combinations(paths,2):
  entry={}
  for role in (None,'L','U'):
   compared=mismatches=0;first=None
   for index,(left,right) in enumerate(itertools.zip_longest(
      sequence(paths[a]/'source/training_sequence.jsonl.gz',role),
      sequence(paths[b]/'source/training_sequence.jsonl.gz',role)),1):
    compared+=1
    if left!=right:
     mismatches+=1
     if first is None:first=index
   entry['interleaved' if role is None else role]=dict(equal=mismatches==0,
     compared_batches=compared,mismatch_batches=mismatches,first_mismatch_batch=first)
  result[a+'__'+b]=entry
 return result

def state_comparisons(initials,torch):
 result={}
 for a,b in itertools.combinations(initials,2):
  left,right=initials[a],initials[b];common=set(left)&set(right)
  changed=[key for key in sorted(common) if not torch.equal(left[key],right[key])]
  result[a+'__'+b]=dict(exact_equal=set(left)==set(right) and not changed,
    left_only=sorted(set(left)-set(right)),right_only=sorted(set(right)-set(left)),
    unequal_keys=changed,common_state_entries=len(common))
 return result

def motion(source,initial,actual,torch):
 cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 require(cp['epoch']==20 and cp['a1_ema_successful_updates']==4440
   and cp['checkpoint_selection']=='final_only' and cp.get('ema_model') is not None
   and not cp['args']['baseline_ckpt'] and not cp['args']['teacher_ckpt'],
   'Own scratch E20 checkpoint/EMA provenance differs')
 state=cp['model'];head='id_backbone.encoder.core.id_backbone.cls_head.weight'
 require(set(state)==set(initial) and tuple(state[head].shape)==(6,160)
   and not any('epochreplay' in k.lower() for k in state),'Default-head/cache-free state layout differs')
 changed={}
 for name,witness in actual['changed_parameters'].items():
  delta=float((state[name].float()-initial[name].float()).norm())
  require(delta>0 and math.isfinite(delta)
    and math.isclose(delta,float(witness['l2']),rel_tol=1e-6,abs_tol=1e-10),
    'Saved parameter-motion witness differs: '+name)
  changed[name]=dict(l2=delta,initial_l2=float(initial[name].float().norm()))
 groups=dict(default_head=[head] if head in changed else [],Sinc=[k for k in changed if '.sinc.' in k],
   SIG_TIME=[k for k in changed if '.sequence_pool.' in k],
   other_identity_backbone=[k for k in changed if '.core.id_backbone.' in k
                           and '.cls_head.' not in k and '.sinc.' not in k])
 require(all(groups.values()),'Required backbone, Sinc, signature or default-head motion absent')
 joint=actual['joint160_actual']
 require(joint['calls']>0 and joint['packets']==actual['student_train_packets']+actual['student_eval_packets']
   and joint['last_shape'][1]==160 and joint['finite_checked_each_forward']
   and joint['extra_verification_NN']==0,'Actual encoder joint160 observation differs')
 return dict(checkpoint_epoch=20,optimizer_updates=4440,default_head_shape=list(state[head].shape),
   joint160_actual=joint,changed_parameters=changed,groups={k:dict(count=len(v),parameters=v) for k,v in groups.items()},
   trainable_parameters=actual['trainable_parameters'],cache_in_model_state=False,analysis_NN=0)

def online_counts(packet_rows,step):
 result={key:dict(n=0,correct=0) for key in ('all','unique','hit','cross_step_hit','within_step_hit')}
 for p in packet_rows:
  CHECK_CONTEXT['slot']=p['slot']
  correct=int(p['prediction']==p['label'])
  require(p['correct']==correct and 0<=p['prediction']<6 and 0<=p['label']<6,'Online saved As decision differs')
  groups=['all','hit' if p['cache_hit'] else 'unique']
  if p['cache_hit']:groups.append('cross_step_hit' if p['source_step']<step else 'within_step_hit')
  for group in groups:result[group]['n']+=1;result[group]['correct']+=correct
 return result

def audit_new(path,root,initial,torch):
 source=path/'source';mechanism=rows(source/'mechanism_steps.jsonl.gz')
 replay=rows(source/'replay_steps.jsonl.gz');ordinary=rows(source/'ordinary_draw_steps.jsonl.gz')
 unit=rows(source/'unit_steps.jsonl.gz');native=rows(source/'step_metrics.jsonl.gz')
 epochs=rows(source/'metrics_epoch.jsonl');companion=rows(source/'epoch_metrics.jsonl')
 require(all(len(x)==4440 for x in (mechanism,replay,ordinary,unit,native))
   and len(epochs)==20 and len(companion)==20,'Complete five4440-step/20epoch streams required')
 csv_counts={}
 for name in ('metrics_epoch.csv','epoch_metrics.csv'):
  with (source/name).open(encoding='utf-8-sig',newline='') as f:csv_counts[name]=sum(1 for row in csv.DictReader(f))
 require(all(v==20 for v in csv_counts.values()),'Complete native and companion CSVs required')
 maximum={key:0. for key in ('CE_formula','native_ordinary','native_As','native_L','native_first_CE',
                            'labeled_phase_total','Sinc_lr','Sinc_decay')}
 curves=[];checked_packets=0;pid_table=[];pid_lookup={};epoch_cache={};epoch_online=None
 totals=dict(unique=0,hit=0,cross_step_hit=0,within_step_hit=0,hit_changed=0)
 metadata_total={side:{key:Counter() for key in ('scenes','states','locks','RX','days')} for side in ('fresh','used')}
 hit_squared=0.;hit_max=0.;ordinary_mse=0.;ordinary_changed=0;hidden_U=0
 scene_names=['practical_low_suburban','practical_mid_urban','practical_low_urban']
 scene_code={name:i for i,name in enumerate(scene_names)}
 recorded_L=sequence(source/'training_sequence.jsonl.gz','L')
 for index,(r,p,o,u,n,lbatch) in enumerate(zip(mechanism,replay,ordinary,unit,native,recorded_L),1):
  epoch=(index-1)//222+1;local=(index-1)%222
  CHECK_CONTEXT.update(phase='complete_training_recount',step=index,epoch=epoch,slot=None)
  require(all(v['step']==index for v in (r,p,o,u,n)) and all(v['epoch']==epoch for v in (r,p,n)),
          'Continuous ordered training steps/epochs differ')
  if local==0:
   epoch_cache={};epoch_online={key:dict(n=0,correct=0) for key in ('all','unique','hit','cross_step_hit','within_step_hit')}
   ep_ce=[];ep_grad=[];ep_total=[];ep_unique=ep_hits=ep_changed=0
   metadata_epoch={side:{key:Counter() for key in ('scenes','states','locks','RX','days')} for side in ('fresh','used')}
  require(r['L_forwards']==2 and r['As_packets']==r['fresh_As_packets']==r['used_As_packets']==128
   and r['teacher_packets']==0 and r['first_input']=='native ordinary L'
   and r['ordinary_first_forward_verified'] is True and r['encoder_ordinary_MSE']==0
   and r['hook_raw_MSE']==0 and r['encoder_used_As_MSE']==0
   and r['first_encoder_calls']==r['extra_encoder_calls']==1,
   'Real first ordinary and second used-As encoder witnesses differ')
  require(o['count']==128 and o['full_native_draws']==1 and o['returned_ordinary_MSE']==0
   and o['ordinary_output_used_by_student'] and o['ordinary_output_returned_unmodified']
   and o['original_input_unchanged'],'Original unique native ordinary call/output differs')
  first,second,total=number(r,'ordinary_CE'),number(r,'As_CE'),number(r,'total_L');m=n['metrics']
  for key,a,b in (('CE_formula',.5*(first+second),total),
    ('native_ordinary',number(m,'epochReplay/ordinary_CE'),first),
    ('native_As',number(m,'epochReplay/As_CE'),second),('native_L',number(m,'epochReplay/total_L'),total),
    ('native_first_CE',number(m,'train/loss_tx_labeled'),first)):
   maximum[key]=max(maximum[key],abs(a-b))
  require(number(r,'extra_CE')==second and number(m,'train/a1_ema_successful_updates')==index,
          'Native As CE or ownEMA successful-update sequence differs')
  if epoch<=13:maximum['labeled_phase_total']=max(maximum['labeled_phase_total'],abs(number(m,'train/loss')-total))
  else:
   require(number(m,'train/pseudo_truth_available')==0,'Hidden U TX became available')
   hidden_U+=1
  require(all(float(m.get(k,0) or 0)==0 for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')),
          'Nonfinite native update skipped')
  for key in ('train/loss','train/grad_before_clip','train/grad_total'):number(m,key)
  lr=1e-6+(2e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/19))/2
  require(u['physical_clip_calls']==index and u['sinc_weight_decay']==0 and number(u,'sinc_eps')>0,
          'Physical Sinc global clipping or unit optimizer differs')
  maximum['Sinc_lr']=max(maximum['Sinc_lr'],abs(number(u,'base_lr')-lr),abs(number(u,'sinc_lr')-lr*1000))
  maximum['Sinc_decay']=max(maximum['Sinc_decay'],abs(number(u,'physical_decay_alpha_lambda')-lr*1e-4))
  require(p['fresh_generated_packets']==p['used_packets']==128 and p['reset_count']==epoch-1
   and p['epoch_cache_reset']==(local==0) and p['cumulative_generated']==p['cumulative_used']==index*128,
   'Epoch reset or cumulative fresh/used budget differs')
  require(len(r['packets'])==128 and len(lbatch['ids'])==128,'Complete L128 packet provenance required')
  counters=online_counts(r['packets'],index)
  require(counters==r['online_As'],'Online unique/hit correct counts differ from saved same-forward decisions')
  for name,count in counters.items():
   epoch_online[name]['n']+=count['n'];epoch_online[name]['correct']+=count['correct']
  for key,group,field in (('epochReplay/As_correct','all','correct'),('epochReplay/As_n','all','n'),
    ('epochReplay/As_hit_correct','hit','correct'),('epochReplay/As_hit_n','hit','n'),
    ('epochReplay/As_unique_correct','unique','correct'),('epochReplay/As_unique_n','unique','n')):
   require(number(m,key)==counters[group][field],'Native online As counters differ')
  compact=dict(step=index,epoch=epoch,pid_index=[],source_steps=[],source_slots=[],hit=[],
               fresh_scene=[],used_scene=[],prediction=[],correct=[])
  fresh_counts=Counter();used_counts=Counter();hits=misses=0
  for slot,packet in enumerate(r['packets']):
   CHECK_CONTEXT['slot']=slot
   pid=tuple(packet['sample_id']);origin=(int(packet['source_step']),int(packet['source_slot']))
   hit=bool(packet['cache_hit']);fresh_scene=packet['fresh_scene'];used_scene=packet['used_scene']
   require(packet['slot']==slot and tuple(lbatch['ids'][slot])==pid
     and packet['first_observation']==(not hit),'Actual L sampler and cached provenance differ')
   require(fresh_scene in scene_code and used_scene in scene_code
     and packet['fresh_namespace']==f'FcrossRX/train/e{epoch}/step{index}/L_S'
     and packet['used_namespace']==f'FcrossRX/train/e{epoch}/step{origin[0]}/L_S',
     'Fresh and used complete channel namespaces/scenes differ')
   if hit:
    require(pid in epoch_cache and epoch_cache[pid]==(*origin,used_scene),
            'Hit is not the first same-epoch physical observation')
    hits+=1
   else:
    require(pid not in epoch_cache and origin==(index,slot) and fresh_scene==used_scene,
            'First full As observation did not use its actual current IQ slot')
    epoch_cache[pid]=(*origin,used_scene);misses+=1
   if pid not in pid_lookup:
    pid_lookup[pid]=len(pid_table);pid_table.append(dict(index=len(pid_table),sample_id=list(pid),label=packet['label']))
   require(pid_table[pid_lookup[pid]]['label']==packet['label'],'Physical L identity label changed')
   compact['pid_index'].append(pid_lookup[pid]);compact['source_steps'].append(origin[0]);compact['source_slots'].append(origin[1])
   compact['hit'].append(hit);compact['fresh_scene'].append(scene_code[fresh_scene]);compact['used_scene'].append(scene_code[used_scene])
   compact['prediction'].append(packet['prediction']);compact['correct'].append(packet['correct'])
   fresh_counts[fresh_scene]+=1;used_counts[used_scene]+=1
  require(p['source_steps']==compact['source_steps'] and p['source_slots']==compact['source_slots']
   and p['cacheflags']==compact['hit'] and p['fresh_scenes']==[scene_names[i] for i in compact['fresh_scene']]
   and p['used_scenes']==[scene_names[i] for i in compact['used_scene']]
   and p['used_namespaces']==[v['used_namespace'] for v in r['packets']]
   and p['miss_packets']==misses and p['hit_packets']==hits and p['current_cache_entries']==len(epoch_cache)
   and misses==counters['unique']['n'] and hits==counters['hit']['n'],
   'Complete cached source coordinates and unique/hit counters differ')
  require(r['fresh_meta_summary']['packets']==r['used_meta_summary']['packets']==128
   and r['fresh_meta_summary']['scenes']==dict(fresh_counts) and r['used_meta_summary']['scenes']==dict(used_counts),
   'Fresh versus used metadata summaries differ')
  physical_RX=dict(Counter(str(v['sample_id'][1]) for v in r['packets']))
  physical_days=dict(Counter(str(v['sample_id'][2]) for v in r['packets']))
  require(len(p['used_states'])==len(p['used_locks'])==128
   and dict(Counter(p['used_states']))==r['used_meta_summary']['states']
   and dict(Counter(p['used_locks']))==r['used_meta_summary']['locks'],
   'Used cache-state/lock summaries differ from complete replay records')
  for side in ('fresh','used'):
   summary=r[side+'_meta_summary']
   require(summary['RX']==physical_RX and summary['days']==physical_days
    and all(sum(summary[key].values())==128 for key in ('scenes','states','locks','RX','days')),
    'Fresh/used source metadata marginals differ')
   for key in ('scenes','states','locks','RX','days'):
    metadata_total[side][key].update(summary[key]);metadata_epoch[side][key].update(summary[key])
  checked_packets+=128
  CHECK_CONTEXT['slot']=None
  totals['unique']+=misses;totals['hit']+=hits
  totals['cross_step_hit']+=counters['cross_step_hit']['n'];totals['within_step_hit']+=counters['within_step_hit']['n']
  changed=int(p['hit_changed_packets']);require(0<=changed<=hits,'Fresh/used changed-hit count differs')
  error=number(p,'hit_fresh_used_MSE');require(error>=0,'Negative fresh/used IQ error')
  totals['hit_changed']+=changed;hit_squared+=error*hits*512;hit_max=max(hit_max,number(p,'hit_fresh_used_max_abs'))
  ordinary_mse+=number(o,'ordinary_raw_MSE');ordinary_changed+=int(o['ordinary_changed_packets'])
  ep_unique+=misses;ep_hits+=hits;ep_changed+=changed
  ep_ce.append((first,second,total));ep_grad.append(number(m,'train/grad_before_clip'));ep_total.append(number(m,'train/loss'))
  if local==221:
   for count in epoch_online.values():count['accuracy']=None if count['n']==0 else count['correct']/count['n']
   ce=np.asarray(ep_ce)
   curves.append(dict(epoch=epoch,ordinary_CE=float(ce[:,0].mean()),As_CE=float(ce[:,1].mean()),
    total_L=float(ce[:,2].mean()),native_total_loss=float(np.mean(ep_total)),grad_before_clip=float(np.mean(ep_grad)),
    online_As=epoch_online,unique_epoch_PID=len(epoch_cache),unique_packets=ep_unique,hit_packets=ep_hits,
    hit_changed_packets=ep_changed,hit_fraction=ep_hits/(222*128),base_lr=number(u,'base_lr'),
    physical_meta_summary={side:{key:dict(count) for key,count in values.items()}
                           for side,values in metadata_epoch.items()}))
 require(max(maximum.values())<=2e-5 and hidden_U==1554,'Full CE/native/Sinc formula or hidden-U recount differs')
 CHECK_CONTEXT.update(phase='training_terminal',step=4440,epoch=20,slot=None)
 require(totals['unique']+totals['hit']==568320 and checked_packets==568320,'Complete cached observation accounting differs')
 actual=read(source/'training_result.json');done=read(source/'completion.json');cached=actual['replay']
 require(done['epoch']==20 and done['optimizer_steps']==4440 and done['label_epochs']==13 and done['pseudo_epochs']==7
  and not done['target_access'] and done['head_fit_updates']==0 and done['final_source_metrics']['source_val_count']==27000,
  'Actual full-V/scratch20/native training terminal differs')
 require(actual['L_packets']==1136640 and actual['student_train_packets']==1533540
  and actual['teacher_packets']==actual['own_EMA_train_packets']==0 and actual['own_EMA_eval_packets']==396900
  and actual['own_EMA_successful_updates']==4440 and actual['U_packets']==396900 and actual['U_hidden_steps']==1554
  and actual['ordinary_packets']==568320 and actual['ordinary_first_forward_verified_steps']==4440
  and actual['fresh_As_packets']==actual['used_As_packets']==actual['channel']['packets']==568320
  and not any(actual[k] for k in ('CROSS_RX','RDrop','external_teacher','KD','head_fit_updates','target_access','cache_in_state_dict')),
  'Actual pure F/default-head/U/EMA/fresh/used budget differs')
 require(cached['steps']==4440 and cached['epoch']==cached['epochs_started']==20 and cached['reset_count']==19
  and cached['cumulative_generated']==cached['cumulative_used']==568320
  and cached['miss_packets']==totals['unique'] and cached['hit_packets']==totals['hit']
  and cached['hit_changed_packets']==totals['hit_changed'] and cached['current_cache_entries']==curves[-1]['unique_epoch_PID'],
  'Final cumulative replay counters differ')
 require(math.isclose(cached['hit_fresh_used_MSE'],hit_squared/(totals['hit']*512) if totals['hit'] else 0.,rel_tol=1e-10,abs_tol=1e-12)
  and math.isclose(cached['hit_fresh_used_max_abs'],hit_max,rel_tol=1e-12,abs_tol=1e-12),'Fresh/used IQ-difference summary differs')
 saved_epochs=read(source/'replay_epochs.json')
 require(len(saved_epochs)==len(actual['replay_epochs'])==20,'Complete saved epoch replay records required')
 for i,(curve,saved,duplicate) in enumerate(zip(curves,saved_epochs,actual['replay_epochs']),1):
  require(saved==duplicate and saved['epoch']==i and saved['steps']==222 and saved['cumulative_steps']==i*222
   and saved['fresh_packets']==saved['used_packets']==28416 and saved['expected_reset_count']==i-1
   and saved['unique_packets']==curve['unique_packets'] and saved['hit_packets']==curve['hit_packets']
   and saved['cross_step_hit_packets']==curve['online_As']['cross_step_hit']['n']
   and saved['unique_physical_IDs']==curve['unique_epoch_PID'] and saved['epoch_provenance_reconstructed'],
   'Saved observer epoch provenance differs from full independent recount')
 ordinary_actual=read(source/'ordinary_augmentation_actual.json')
 require(ordinary_actual['use_aug'] and ordinary_actual['ordinary_output_used_by_student']
  and ordinary_actual['ordinary_output_returned_unmodified'] and ordinary_actual['ordinary_calls']==4440
  and not ordinary_actual['U_augmentor_wrapped'] and ordinary_actual['practical_input_is_raw'],
  'Resolved ordinary switches or U wrapping differ')
 log=(root/'logs'/'EPOCH_REPLAY-s2026092701.log').read_text(encoding='utf8',errors='replace')
 errors=[line for line in log.splitlines() if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',line)]
 online_total={key:dict(n=sum(v['online_As'][key]['n'] for v in curves),correct=sum(v['online_As'][key]['correct'] for v in curves))
               for key in curves[0]['online_As']}
 for v in online_total.values():v['accuracy']=None if v['n']==0 else v['correct']/v['n']
 return dict(complete_records=dict(mechanism=len(mechanism),replay=len(replay),ordinary=len(ordinary),unit=len(unit),
    native=len(native),native_epochs=len(epochs),companion_epochs=len(companion),CSV_rows=csv_counts),
   all4440_steps_checked=True,max_formula_errors=maximum,hidden_U_steps=hidden_U,ownEMA_all4440_updates_exact=True,
   ordinary=dict(calls=4440,packets=568320,mean_raw_MSE=ordinary_mse/4440,changed_packet_draws=ordinary_changed,
                 first_and_extra_actual_encoder_witness_steps=4440,returned_ordinary_MSE_max=0.),
   replay=dict(saved=cached,recount=totals,online_As=online_total,
     hit_fresh_used_MSE=hit_squared/(totals['hit']*512) if totals['hit'] else 0.,hit_fresh_used_max_abs=hit_max,
     unique_physical_IDs_over_training=len(pid_table),IQ_difference_evidence='Saved original fresh/used equality and MSE counters; no IQ reread or new channel'),
   physical_meta_summary={side:{key:dict(count) for key,count in values.items()}
                          for side,values in metadata_total.items()},
   cached_origin_validation=dict(checked_steps=4440,checked_packets=checked_packets,
     origin_coordinate_matches=checked_packets,sampler_PID_matches=checked_packets,
     namespace_matches=checked_packets,used_scene_matches=checked_packets,
     first_contradiction=None,original_complete_artifacts=dict(
       mechanism=str(source/'mechanism_steps.jsonl.gz'),replay=str(source/'replay_steps.jsonl.gz'),
       L_and_U_sequence=str(source/'training_sequence.jsonl.gz'),epoch_replay=str(source/'replay_epochs.json')),
     complete_source_sequence_preserved_in='Original mechanism/replay/training_sequence artifacts; no duplicate trace emitted'),curves=curves,
   training_sequence=sequence_budget(source/'training_sequence.jsonl.gz'),motion=motion(source,initial,actual,torch),
   native_epochs=epochs,companion_epochs=companion,ordinary_actual=ordinary_actual,
   Sinc=dict(mode='CENTER_UNITS',coordinate_scale_Hz=25000.,sample_rate_Hz=25000000.,LR_multiplier=1000.,
     actual_eps_values=sorted(set(number(v,'sinc_eps') for v in unit)),physical_global_clip_calls=4440,
     physical_global_clip_max_norm=5.,explicit_decay_weight=1e-4),
   costs=dict(L_student_NN=actual['L_packets'],U_student_NN=actual['U_packets'],ownEMA_NN=actual['own_EMA_eval_packets'],
     external_teacher_NN=actual['teacher_packets'],source_training_eval_NN_actual=actual['student_eval_packets'],
     ordinary_generated_and_used=568320,ordinary_shadow=0,fresh_As_generated=568320,used_As_NN=568320,
     unique_epoch_As_realizations=totals['unique'],fresh_As_discarded_as_hit_shadow=totals['hit'],
     fresh_As_changed_on_hit=totals['hit_changed'],source_qualification_NN=12600,formal_prediction_NN=1176002,
     training_wall_seconds=done['elapsed_seconds'],extra_analysis_NN=0),
   native_log_lines=len(log.splitlines()),technical_error_lines=errors,technical_error_line_count=len(errors))

def collect(config):
 root=Path(config['project'])/'runs'/config['run_id'];reference=Path(config['project'])/'runs'/config['reference_run']
 CHECK_CONTEXT.update(phase='terminal_preconditions')
 terminal=read(root/'complete.json');decision=read(root/'source_matrix_decision.json')
 require(terminal['status']=='ANALYZED' and decision['status']=='SOURCE_DECISION_FIXED'
   and decision['formal_scores_read'] is False,'Complete run and source-first frozen decision required')
 paths={a:(root if a=='EPOCH_REPLAY' else reference)/(a+'-s2026092701') for a in ('B0','B1','EPOCH_REPLAY')}
 # Only terminal metadata is read until all three seven-view score sets exist.
 for arm,path in paths.items():
  pred=read(path/'prediction/complete.json');score=read(path/'scoring/complete.json')
  require(pred['status']=='PREDICTIONS_COMPLETE' and pred['count']==168000 and pred['views']==config['views']
   and pred['truth_read'] is False and pred['query_fit'] is False
   and score['status']=='SCORED_COMPLETE' and score['truth_last'] is True
   and score['views']==config['views'],'All frozen seven-view predictions and independent scores required')
  if arm=='EPOCH_REPLAY':require(decision['timestamp']<(path/'scoring/complete.json').stat().st_mtime,
                               'Source decision must precede independent new-row scoring')
 runner=Path(config['project'])/'releases'/config['release']/'experiments/cvs_F_epochReplay20/runner.py'
 spec=importlib.util.spec_from_file_location('_epochReplay_saved_analysis',runner)
 boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.bootstrap()
 import torch
 torch.set_num_threads(2)
 initials={a:torch.load(path/'source/initial_smoke.pt',map_location='cpu',weights_only=False)['model'] for a,path in paths.items()}
 initial_pairs=state_comparisons(initials,torch)
 result=dict(status='ANALYZED',run_id=config['run_id'],model_seed=2026092701,model_seed_count=1,sample_SD=None,
  source_decision=decision,initial_state_comparisons=initial_pairs,
  initial_all_three_exact_equal=all(v['exact_equal'] for v in initial_pairs.values()),
  source={},source_pairs={},formal={},paired={},extra_model_NN=0,new_predictions=0,new_scoring_jobs=0,
  target_feedback=False,goal_complete=False)
 result['logs']=audit_new(paths['EPOCH_REPLAY'],root,initials['EPOCH_REPLAY'],torch)
 result['training_sequence_pairs']=sequence_comparisons(paths)
 result['training_sequences_all_three_equal']=all(v['interleaved']['equal'] for v in result['training_sequence_pairs'].values())
 result['training_RNG_equality_assumed']=False
 result['reference_costs']={}
 for arm in ('B0','B1'):
  actual=read(paths[arm]/'source/training_result.json');done=read(paths[arm]/'source/completion.json')
  result['reference_costs'][arm]=dict(L_student_NN=actual['L_packets'],student_train_NN=actual['student_train_packets'],
    student_eval_NN_actual=actual['student_eval_packets'],external_teacher_shadow_NN=actual['teacher_packets'],
    source_qualification_NN=12600,formal_prediction_NN=1176002,training_wall_seconds=done['elapsed_seconds'],reused=True,
    sequence=sequence_budget(paths[arm]/'source/training_sequence.jsonl.gz'))
 source_ids=source_y=None;source_predictions={}
 CHECK_CONTEXT.clear();CHECK_CONTEXT.update(phase='source_readout_recount')
 for arm,path in paths.items():
  CHECK_CONTEXT['arm']=arm
  ready=read(path/'source_qualification/complete.json');done=read(path/'source/completion.json')
  require(ready['status']=='SOURCE_READOUT_COMPLETE' and ready['target_access'] is False
    and ready['training_updates']==ready['head_fit_updates']==0 and ready['NN']==12600,
    'Existing legal frozen source-panel readout required')
  with np.load(path/'source_qualification/predictions.npz',allow_pickle=False) as f:
   ids,y,clean,sat=(f[key].copy() for key in ('ids','TX','clean','satellite'))
   require(ids.shape==y.shape==(1800,) and len(set(ids.tolist()))==1800 and y.dtype.kind in 'iu'
    and y.min()>=0 and y.max()<6 and clean.shape==(1800,6) and sat.shape==(2,3,1800,6)
    and np.isfinite(clean).all() and np.isfinite(sat).all(),'Complete legal source-V panel shape differs')
   require(set(f['RX'].tolist())<=set((1,3,4,6,8)) and set(f['day'].tolist())<=set((1,2,3)),
           'Source readout RX/day roles differ')
  if source_ids is None:source_ids,source_y=ids,y
  else:require(np.array_equal(ids,source_ids) and np.array_equal(y,source_y),'Source paired physical IDs or labels differ')
  rates=(sat.argmax(-1)==y).mean(-1)
  require(done['final_source_metrics']['source_val_count']==27000
   and np.allclose(rates,decision['readouts'][arm]['source_accuracies'],atol=1e-12,rtol=0)
   and np.allclose(rates,ready['source_accuracies'],atol=1e-12,rtol=0),'Saved source decision/readout differs')
  source_predictions[arm]=dict(clean=clean.argmax(1),satellite=sat.argmax(-1))
  result['source'][arm]=dict(fullV_count=27000,fullVclean=done['final_source_metrics']['source_val_accuracy'],
   fullVworst_RX=done['final_source_metrics']['source_val_worst_rx'],panel_count=1800,panel_draws=2,
   panel_clean=metrics(y,clean.argmax(1)),source_accuracies=rates.tolist(),midlow=float(rates[:,1:].mean()))
 gain=100*(result['source']['EPOCH_REPLAY']['midlow']-result['source']['B1']['midlow'])
 net=100*(result['source']['EPOCH_REPLAY']['fullVclean']-result['source']['B0']['fullVclean'])
 passed=gain>=.5 and net>=-.2
 require(abs(gain-decision['midlow_increment_pp'])<1e-9 and abs(net-decision['fullVclean_net_pp'])<1e-9
   and passed==decision['source_qualified'],'Fixed source promotion recount differs')
 result['source_criterion_recount']=dict(midlow_increment_pp=gain,fullVclean_net_pp=net,passed=passed,
   thresholds=dict(midlow_min_pp=.5,fullVclean_min_pp=-.2),independently_recomputed=True,
   distinction='Promotion clean uses all27000 sourceV; clean paired errors below use saved V1800 panel')
 for baseline in ('B0','B1'):
  result['source_pairs'][baseline]=dict(panel_clean=paired(source_y,source_predictions[baseline]['clean'],source_predictions['EPOCH_REPLAY']['clean']),
   satellite={name:paired(np.tile(source_y,2),source_predictions[baseline]['satellite'][:,j].reshape(-1),
     source_predictions['EPOCH_REPLAY']['satellite'][:,j].reshape(-1))
     for j,name in enumerate(('practical_low_suburban','practical_mid_urban','practical_low_urban'))})
 # No target predictions/truth have been loaded above this point.
 predictions={};ids=None
 CHECK_CONTEXT.clear();CHECK_CONTEXT.update(phase='existing_formal_prediction_recount')
 for arm,path in paths.items():
  with np.load(path/'prediction/predictions.npz',allow_pickle=False) as f:
   current=f['ids'].copy()
   require(current.shape==(168000,) and len(set(current.tolist()))==168000,'Formal physical ID vector differs')
   if ids is None:ids=current
   else:require(np.array_equal(ids,current),'Formal paired physical IDs differ')
   predictions[arm]={view:f[view].copy() for view in config['views']}
   require(all(p.shape==(168000,) and p.dtype.kind in 'iu' and p.min()>=0 and p.max()<6
               for p in predictions[arm].values()),'Saved formal class decisions differ')
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 truth=read(Path(config['project'])/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
 y,rx,days=truth_arrays(truth,ids)
 for arm,path in paths.items():
  saved=read(path/'scoring/scores.json');data=saved['results'];overall={r['view']:r for r in data if r['dimension']=='overall'}
  require(saved['truth_last'] and set(overall)==set(config['views']),'Existing all7 scores incomplete')
  values={};strata={}
  for view in config['views']:
   predicted=predictions[arm][view];values[view]=metrics(y,predicted);strata[view]={}
   for dimension,labels in (('receiver',rx),('day',days),('transmitter',y)):
    strata[view][dimension]={str(value):metrics(y[labels==value],predicted[labels==value]) for value in sorted(set(labels))}
   values[view]['worst_RX']=min(v['accuracy'] for v in strata[view]['receiver'].values())
   for old in [r for r in data if r['view']==view]:
    recount=values[view] if old['dimension']=='overall' else strata[view][old['dimension']][str(old['stratum'])]
    require(all(abs(recount[key]-old[key])<1e-12 for key in ('accuracy','macro_f1')),
            'Existing independent overall/stratum score differs from frozen decisions')
  result['formal'][arm]=dict(overall=values,strata=strata,reused=arm!='EPOCH_REPLAY')
 for baseline in ('B0','B1'):
  result['paired']['EPOCH_REPLAY_minus_'+baseline]={view:paired(y,predictions[baseline][view],predictions['EPOCH_REPLAY'][view]) for view in config['views']}
 result['suburban_gap_pp']={arm:{view:100*(result['formal'][arm]['overall']['clean']['accuracy']-result['formal'][arm]['overall'][view]['accuracy'])
    for view in config['views'][1:4]} for arm in paths}
 fixed=predictions['B0']['clean']==y;n=int(fixed.sum())
 result['fixed_B0_clean_correct_cohort']=dict(n=n,definition='One fixed frozen B0-clean-correct set for every row/view',
   rows={arm:{view:dict(retained_correct=int((predictions[arm][view][fixed]==y[fixed]).sum()),
    retention=None if n==0 else float((predictions[arm][view][fixed]==y[fixed]).mean())) for view in config['views']} for arm in paths})
 result['budget_comparability']=dict(epochs_each=20,updates_each=4440,data='L6300/U56700/V27000',
  B0_L_student_NN=568320,B1_and_new_L_student_NN=1136640,B1_external_teacher_shadow_NN=568320,
  new_external_teacher_NN=0,new_full_fresh_As_generation=568320,new_used_As_NN=568320,
  extra_analysis_NN=0,note='No whole-training RNG or wall-time equality assumed; unique epoch realizations and discarded hit shadows are measured')
 return result

print(json.dumps(collect(CONFIG),allow_nan=False,separators=(',',':')))
'''


def configuration(project):
    return dict(project=project, run_id=RUN, reference_run=REFERENCE_RUN,
                release=RELEASE, views=list(VIEWS))


def remote_helper(runtime):
    explicit = runtime.get('analysis_remote_helper')
    candidates = ([Path(explicit)] if explicit else []) + [
        ROOT/'experiments/cvs_sameiq_experts/publish.py',
        ROOT.parent/'phase1_sameiq_experts_20261019/experiments/cvs_sameiq_experts/publish.py',
    ]
    path = next((path for path in candidates if path.is_file()), None)
    if path is None:
        raise FileNotFoundError('Existing cvs_sameiq_experts.publish.remote helper unavailable')
    if path.parent.parent.name == 'experiments':
        # Publish helpers use relative imports and must keep their package name.
        sys.path.insert(0, str(path.parents[2]))
        module = importlib.import_module('experiments.'+path.parent.name+'.'+path.stem)
    else:
        spec = importlib.util.spec_from_file_location('_existing_epochReplay_analysis_remote', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.remote


def main(runtime, output, render_figures=False):
    rt = json.loads(Path(runtime).read_text(encoding='utf-8-sig'))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = remote_helper(rt)(SCRIPT.replace('CONFIG', repr(configuration(rt['project']))), rt)
    (out/'complete_analysis.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if render_figures:
        render(result, out)
    print(json.dumps(dict(status=result['status'], all4440_steps_checked=True,
        initial_all_three_exact_equal=result['initial_all_three_exact_equal'],
        training_sequences_all_three_equal=result['training_sequences_all_three_equal'],
        extra_model_NN=0, source_decision=result['source_decision']['decision'])))
    return result


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    stems = ('seven_views_and_online_As', 'paired_errors_and_B0_clean_retention')
    if any((out/(stem+suffix)).exists() for stem in stems for suffix in ('.png', '.pdf')):
        raise FileExistsError('Existing figures are preserved; select a new output directory')
    colors = dict(B0='#929292', B1='#356d9b', EPOCH_REPLAY='#19856b')
    names = dict(B0='Original NoAs', B1='Fresh As each step', EPOCH_REPLAY='Epoch-first As replay')
    short = ['clean', 'high', 'mid', 'low sub', 'high urb', 'mid urb', 'low urb']
    curve = result['logs']['curves'];epoch = [r['epoch'] for r in curve]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout='constrained')
    for arm, color in colors.items():
        values = result['formal'][arm]['overall']
        axes[0, 0].plot(range(7), [100*values[v]['accuracy'] for v in VIEWS], '-o', color=color, label=names[arm], markersize=4)
    for key, color in (('all', '#19856b'), ('unique', '#b77d31'), ('hit', '#356d9b')):
        axes[0, 1].plot(epoch, [None if r['online_As'][key]['accuracy'] is None else 100*r['online_As'][key]['accuracy'] for r in curve],
                        color=color, label=key)
    axes[1, 0].plot(epoch, [r['ordinary_CE'] for r in curve], color='#356d9b', label='Ordinary CE')
    axes[1, 0].plot(epoch, [r['As_CE'] for r in curve], color='#19856b', label='Used As CE')
    axes[1, 1].plot(epoch, [100*r['hit_fraction'] for r in curve], color='#356d9b', label='Cache hits')
    axes[1, 1].plot(epoch, [100*r['unique_packets']/28416 for r in curve], color='#b77d31', label='First visits')
    axes[0, 0].set(xticks=range(7), xticklabels=short, ylabel='Frozen test accuracy (%)')
    axes[0, 1].set(xlabel='Epoch', ylabel='Online used-As accuracy (%)')
    axes[1, 0].set(xlabel='Epoch', ylabel='True-label CE (all128, smoothing0.01)')
    axes[1, 1].set(xlabel='Epoch', ylabel='Share of used As packets (%)')
    for index, ax in enumerate(axes.flat):
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.15)
        ax.legend(frameon=False, fontsize=8)
        if index: ax.axvline(13.5, color='#777777', linestyle=':', linewidth=.8)
    fig.suptitle('Pure F scratch E20; one model seed; no new analysis forwards')
    for suffix in ('.png', '.pdf'): fig.savefig(out/(stems[0]+suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for baseline in ('B0', 'B1'):
        pair = result['paired']['EPOCH_REPLAY_minus_'+baseline]
        axes[0].plot(range(7), [pair[v]['net_pp'] for v in VIEWS], '-o', color=colors[baseline], label='Replay minus '+names[baseline])
    cohort = result['fixed_B0_clean_correct_cohort']
    for arm, color in colors.items():
        axes[1].plot(range(7), [None if cohort['rows'][arm][v]['retention'] is None else 100*cohort['rows'][arm][v]['retention'] for v in VIEWS],
                     '-o', color=color, label=names[arm])
    for ax in axes:
        ax.set(xticks=range(7), xticklabels=short); ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15); ax.legend(frameon=False, fontsize=8)
    axes[0].axhline(0, color='#777777', linewidth=.8)
    axes[0].set_ylabel('Paired accuracy difference (percentage points)')
    axes[1].set_ylabel('Correct retained on fixed B0-clean-correct set (%)')
    fig.suptitle('Frozen seven-view decisions; model-seed n=1, SD unavailable')
    for suffix in ('.png', '.pdf'): fig.savefig(out/(stems[1]+suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime')
    parser.add_argument('--output', required=True)
    parser.add_argument('--render-cache')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.render_cache:
        render(json.loads(Path(args.render_cache).read_text(encoding='utf-8-sig')), args.output)
    else:
        if not args.runtime:
            parser.error('--runtime is required for collection')
        main(args.runtime, args.output, args.render)
