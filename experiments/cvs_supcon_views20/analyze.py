"""Independent full source prediction/step/cost/scene schedule recount; no model import."""
import csv,gzip,json,math,sys
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.cvs_iq_timing20.harvest import score
OUT=ROOT/'local_artifacts/F_supcon_view_routing_E20_20261017/final'
CONTROL=ROOT/'local_artifacts/F_physical_scene_cycle_E20_20261017/final'
PATHS={'Q0':CONTROL/'B00-RX1-s2026092701/source','Q_ord':CONTROL/'C_IID-RX1-s2026092701/source','Q_sat':OUT/'Q_sat-RX1-s2026092701/source','Q_mix':OUT/'Q_mix-RX1-s2026092701/source'}
POLICIES={'Q0':'ordinary','Q_ord':'ordinary','Q_sat':'satellite','Q_mix':'mixed'}
RUN='20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01'
SEED=2026092701
ARMS=('Q0','Q_ord','Q_sat','Q_mix')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def records(p):
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
def prediction(p):
 labels=p.get('labels');preds=p.get('predictions');ids=p.get('physical_ids')
 if not isinstance(labels,list) or not isinstance(preds,list) or not isinstance(ids,list) or len(labels)!=len(preds) or len(ids)!=1260 or len(labels)!=1260:raise ValueError('All1260 source decisions required')
 if any(not isinstance(z,list) or len(z)!=5 or any(type(v) is not int for v in z) or z[0]!=1 or z[1]!=1 or z[2] not in (1,2,3) for z in ids) or len({tuple(z) for z in ids})!=1260:raise ValueError('Unique declared sourceRX1 EQ1 physical IDs required')
 s=score(labels,preds)
 if any(sum(y==t and z[2]==dd for y,z in zip(labels,ids))!=70 for t in range(6) for dd in (1,2,3)):raise ValueError('Complete6TX/3day/70sourcecell decisions required')
 return s

def routing(arm,src,mix):
 policy=POLICIES[arm];selected_count=ordinary_count=terms=0;active=0;error=0.
 for v in mix:
  step=v['step'];on=step>1110;count=128 if policy=='satellite' and on else 64 if policy=='mixed' and on else 0
  expected=list(range(128)) if policy=='satellite' and on else sorted(np.random.default_rng(202610176001+step).permutation(128)[:64].tolist()) if policy=='mixed' and on else []
  if arm in ('Q_sat','Q_mix'):
   assert v['route_policy']==policy and v['route_active']==on and v['route_selected_As_positions']==expected and len(expected)==count
   assert v['route_As_appearances']==count and v['route_ordinary_appearances']==(128-count if on else 0) and v['route_new_NN_forwards']==v['route_new_loss_terms']==0
  detail=v['supcon_details'];assert len(detail)==(4 if on else 0)
  if on:
   assert {(z['phase'],z['level']) for z in detail}=={(ph,le) for ph in ('first','second') for le in ('time_down','t2.pool')}
   for z in detail:assert len(z['shape'])==3 and z['shape'][0]==128 and z['total_anchors']==128 and 0<=z['valid_anchors']<=128 and 0<=z['positive_min']<=z['positive_max']<=127
   error=max(error,abs(sum(z['loss'] for z in detail)/4-v['supcon']));active+=1;selected_count+=count;ordinary_count+=128-count;terms+=4
 assert active==3330 and terms==13320 and error<1e-5
 if arm in ('Q_sat','Q_mix'):
  audit=read(src/'supcon_views_audit.json');assert audit['calls']==4440 and audit['applied']==active and audit['As_appearances']==selected_count and audit['ordinary_appearances']==ordinary_count and audit['phase_terms']==terms and audit['teacher_calls']==audit['reader_calls']==audit['extra_NN_forwards']==audit['new_loss_terms']==0 and audit['default_head_joint'] and not audit['target_access']
 return dict(status='VERIFIED',policy=policy,all4440_route_records_checked=True,selected_As_appearances=selected_count,selected_ordinary_appearances=ordinary_count,active_steps=active,existing_layer_phase_terms=terms,max_saved_detail_mean_error=error,As_full_batch_map_capture_layer_calls_derived=terms if arm in ('Q_sat','Q_mix') else 0,mixed_map_where_calls_derived=terms if policy=='mixed' else 0,new_NN_forwards=0,new_loss_terms=0,scope='Exact all indices/counters/detailmean;featuremaps not saved,not repeated feature-SupCon inference')

def ssl_boundary(steps):
 """Recount original E13+E7 organization without exposing or decoding U truth."""
 warmup=active=packets=selected=0
 for n in steps:
  step=n['step'];epoch=(step-1)//222+1;z=n['metrics'];total=z['train/pseudo_total'];take=z['train/pseudo_selected']
  assert n['epoch']==epoch and z['train/pseudo_truth_available']==0 and z['train/pseudo_correct']==0 and 0<=take<=total and z['train/a1_ema_successful_updates']==step
  if epoch<=13:assert total==take==0 and z['train/loss_unlabeled']==0;warmup+=1
  else:assert total in (48,256);active+=1;packets+=total;selected+=take
 assert warmup==2886 and active==1554 and packets==396368
 return dict(status='VERIFIED',labeled_only_updates=warmup,original_U_updates=active,U_appearances=packets,selected_appearances=selected,U_truth_available=False,EMA_successful_updates=steps[-1]['step'],quality_claim=False)

def decision_transitions(reference,candidate,view):
 """Descriptive paired identity errors;fixed source sets,never candidate selection."""
 b,s=reference[view],candidate[view];bc,sc=reference['clean'],candidate['clean']
 assert b['labels']==s['labels']==bc['labels']==sc['labels'] and b['physical_ids']==s['physical_ids']==bc['physical_ids']==sc['physical_ids']
 y=b['labels'];groups={name:[] for name in ('all','both_clean_correct','reference_clean_correct','candidate_clean_correct','both_clean_wrong','clean_correctness_changed')}
 for i,t in enumerate(y):
  rb=bc['predictions'][i]==t;rs=sc['predictions'][i]==t;groups['all'].append(i)
  for name,on in (('both_clean_correct',rb and rs),('reference_clean_correct',rb),('candidate_clean_correct',rs),('both_clean_wrong',not rb and not rs),('clean_correctness_changed',rb!=rs)):
   if on:groups[name].append(i)
 out={}
 for name,take in groups.items():
  repaired=sum(b['predictions'][i]!=y[i] and s['predictions'][i]==y[i] for i in take);broken=sum(b['predictions'][i]==y[i] and s['predictions'][i]!=y[i] for i in take)
  before=sum(b['predictions'][i]==y[i] for i in take);after=sum(s['predictions'][i]==y[i] for i in take);assert after-before==repaired-broken
  out[name]=dict(count=len(take),reference_correct=before,candidate_correct=after,repaired=repaired,broken=broken,net_correct=after-before,delta_pp=100*(after-before)/len(take) if take else None)
 assert out['all']['count']==sum(out[k]['count'] for k in ('both_clean_correct','both_clean_wrong','clean_correctness_changed'))
 return out

def qualification(scores,execution):
 conditions={};comparisons={};eps=1e-10
 for ref in ARMS[:3]:
  delta={v:100*(scores['Q_mix'][v]['accuracy']-scores[ref][v]['accuracy']) for v in VIEWS};mf1=100*(scores['Q_mix']['clean']['macro_F1']-scores[ref]['clean']['macro_F1']);lf1=100*(scores['Q_mix'][VIEWS[-1]]['macro_F1']-scores[ref][VIEWS[-1]]['macro_F1']);dv=100*(execution['Q_mix']['seen_V']['source_val_accuracy']-execution[ref]['seen_V']['source_val_accuracy']);urban=sum(delta[v] for v in VIEWS[4:])/3
  part=dict(clean=delta['clean']>=-eps,clean_F1=mf1>=-eps,low=delta[VIEWS[-1]]>=(5. if ref=='Q0' else 1.)-eps,low_F1=lf1>=-eps,mid=delta[VIEWS[-2]]>=-eps,urban_mean=urban>=(3. if ref=='Q0' else 1.)-eps,each_practical=all(delta[v]>=-.5-eps for v in VIEWS[1:]),seenV=dv>=-.5-eps)
  conditions.update({ref+'_'+k:v for k,v in part.items()});comparisons[ref]=dict(conditions=part,delta_pp=delta,clean_F1_delta_pp=mf1,low_F1_delta_pp=lf1,seenV_delta_pp=dv,urban_mean_delta_pp=urban)
 conditions['healthy']=all(v['final_base_L_objective']<=.5 for v in execution.values())
 return dict(qualified=all(conditions.values()),conditions=conditions,comparisons=comparisons)

def analyze(arms=ARMS):
 OUT.mkdir(parents=True,exist_ok=True)
 scores={};predictions={};execution={};curves=[];seqs=[];args_all=[];result_records={};role_sets=[]
 for arm in arms:
  row=arm+'-RX1-s'+str(SEED);src=PATHS[arm];r=read(src/'satmix_result.json');init=read(src/'initialization.json');done=read(src/'completion.json');frozen=read(src/'source_frozen.json');audit=read(src/'additive_audit.json')
  assert init['scratch_only'] and not init['checkpoint_sources'] and not init['ancestors'] and not init['target_contact']
  assert done['epoch']==frozen['epoch']==20 and done['optimizer_steps']==4440 and not done['target_access'] and done['head_fit_updates']==0 and frozen['selection']=='final_only'
  assert r['actual_steps']==r['additive_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320
  assert r['Ag'] is False and r['As']==(arm!='Q0') and not r['U_truth_read'] and not r['target_access'] and not r['head_fit_updates'] and r['teacher_additive_calls']==0 and not r['checkpoint_sources']
  result_records[arm]=r;contract=read(src/'source_contract.json');assert {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':5040,'U_s':45360,'V':21600};role_sets.append({k:set(v) for k,v in contract['role_ids'].items()})
  expected_As=568320 if arm!='Q0' else 0;weight=.25
  assert r['As_training_observations']==expected_As and r['total_pair_weight']==1+weight and audit['calls']==4440 and audit['As_generated_packets']==expected_As and audit['no_As_U_or_teacher']
  scores[arm]={};predictions[arm]={}
  for view in VIEWS:
   p=read(src/'source_readouts'/(view+'.json'));s=prediction(p)
   assert s['count']==1260 and len({tuple(x) for x in p['physical_ids']})==1260 and {x[1] for x in p['physical_ids']}=={1}
   assert math.isclose(s['accuracy'],r['views'][view]['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],r['views'][view]['macro_F1'],abs_tol=1e-12)
   if view!='clean':assert p['labels']==predictions[arm]['clean']['labels'] and p['physical_ids']==predictions[arm]['clean']['physical_ids']
   scores[arm][view]=s;predictions[arm][view]=p
  native=records(src/'metrics_epoch.jsonl');compact=records(src/'epoch_metrics.jsonl')
  with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
  assert [v['epoch'] for v in native]==list(range(1,21)) and len(compact)==len(table)==20
  for a,b,c in zip(native,compact,table):
   for k in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):assert math.isclose(float(a[k]),float(b[k]),abs_tol=1e-10) and math.isclose(float(a[k]),float(c[k]),abs_tol=1e-10)
  steps=records(src/'step_metrics.jsonl.gz');mix=records(src/'satmix_steps.jsonl.gz');unit=records(src/'unit_steps.jsonl.gz')
  assert all([v['step'] for v in values]==list(range(1,4441)) for values in (steps,mix,unit))
  terms=As=truth=skips=bn=0;formula_error=base_error=0.;aggregate={};mode='none' if arm=='Q0' else 'iid';previous_wall=0.
  for n,v,u in zip(steps,mix,unit):
   finite(n);finite(v);finite(u)
   epoch=(v['step']-1)//222+1;sw=.1*min(max((epoch-5)/10,0.),1.)
   assert v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and v['extra_view_once'] and v['extra_same_IQ_pair']
   assert v['As_packets']==0 and v['extra_As_packets']==(128 if arm!='Q0' else 0) and v['extra_input']==('As(raw)' if arm!='Q0' else 'raw')
   assert v['extra_pair_weight']==weight and v['base_raw_total_weight']==.75 and v['base_ordinary_weight']==.25 and v['total_pair_weight']==1+weight
   assert math.isclose(v['supcon_weight'],sw,abs_tol=1e-12) and len(v['supcon_details'])==(4 if epoch>5 else 0)
   assert sum(v['scene_counts'].values())==sum(v['state_counts'].values())==sum(v['lock_counts'].values())==0
   terms+=len(v['supcon_details']);As+=v['extra_As_packets'];bn+=v['extra_BN_batch_updates']
   if arm=='Q0':assert v['extra_raw_MSE']==0
   assert v['sat_raw_MSE']==0
   assert v['scene_schedule_mode']==mode and sum(v['extra_scene_counts'].values())==v['extra_As_packets']
   assert v['channel_generation_wall_seconds']>=previous_wall;previous_wall=v['channel_generation_wall_seconds']
   base=.5*(v['raw_CE']+v['raw_KL'])+.25*(v['ordinary_CE']+v['ordinary_KL'])+.25*(v['s_slot_CE']+v['s_slot_KL'])+sw*v['supcon']
   extra=v['extra_pair_CE']+v['extra_pair_KL'];total=base+weight*extra
   base_error=max(base_error,abs(base-v['base_total_L']))
   formula_error=max(formula_error,abs(total-v['total_L']),abs(extra-v['extra_pair_loss']),abs(weight*extra-v['extra_pair_weighted']))
   nm=n['metrics']
   for k in ('total_L','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_weight','extra_As_packets'):assert math.isclose(float(nm['satmix/'+k]),v[k],abs_tol=1e-6)
   truth+=int(nm.get('train/pseudo_truth_available',0) or 0);skips+=int(nm.get('train/skipped_nonfinite_loss',0) or nm.get('train/skipped_nonfinite_grad',0) or 0)
   assert u['physical_clip_calls']==v['step']
   sums=aggregate.setdefault(epoch,{})
   for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','ordinary_raw_MSE','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','extra_raw_MSE'):sums[k]=sums.get(k,0.)+v[k]/222
  assert terms==13320 and As==expected_As and truth==skips==0 and formula_error<1e-5 and base_error<1e-5 and bn==audit['extra_BN_updates']
  for k in ('base_total_L','extra_pair_loss'):assert math.isclose(aggregate[20][k],r['final_base_L_objective' if k=='base_total_L' else 'final_extra_pair_objective'],abs_tol=1e-8)
  assert r['final_native_logged_L_CE']==r['final_L_CE'] and math.isclose(compact[-1]['train_loss_tx_labeled'],r['final_L_CE'],abs_tol=1e-10)
  assert math.isclose(previous_wall,r['channel_generation_wall_seconds'],abs_tol=1e-9)
  original_ssl=ssl_boundary(steps)
  route=routing(arm,src,mix)
  sched=read(src/'schedule_recount.json');assert sched['status']=='VERIFIED' and sched['mode']==mode and sched['training_calls']==(4440 if arm!='Q0' else 0) and sched['training_packets']==expected_As
  for v in compact:curves.append(dict(row=row,arm=arm,**v,**aggregate[v['epoch']]))
  coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==As and sum(v['count'] for v in coverage['cells'])==As and not coverage['confidence_gate']
  if arm!='Q0':
   assert {v['scene'] for v in coverage['cells']}==set(VIEWS[1:]) and {v['RX'] for v in coverage['cells']}=={3,4,6,8} and {v['TX'] for v in coverage['cells']}==set(range(6)) and {v['day'] for v in coverage['cells']}=={1,2,3}
  seq=records(src/'training_sequence.jsonl.gz');seqs.append(seq);assert all(p[1] in (3,4,6,8) for v in seq for p in v['ids'])
  observed={role:sum(len(v['ids']) for v in seq if v['role']==role) for role in ('L','U')}
  unique={role:len({tuple(p) for v in seq if v['role']==role for p in v['ids']}) for role in ('L','U')}
  assert observed=={'L':568320,'U':396368} and unique=={'L':5040,'U':45360}
  args=read(src/'actual_native_args.json');args_all.append(args)
  assert args['seed']==SEED and args['use_aug'] and args['source_fold_held_rx']==1 and args['label_epochs']==13 and args['pseudo_epochs']==7 and args['aug_enable_pa_normal'] and args['aug_p_dac']==0 and not args['aug_enable_class_signature']
  ordinary=read(src/'ordinary_augmentation_actual.json');assert ordinary['no_second_pair_redraw'] and ordinary['practical_input_is_raw']
  stdout=(src.parents[1]/'logs'/(src.parent.name+'.stdout.log')).read_text(encoding='utf8',errors='replace')
  assert not any(s in stdout for s in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL')) and 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout
  counters={k:r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')}
  NN=sum(counters.values())-2+r['held_forward_packets']
  execution[arm]=dict(original_SSL=original_ssl,full_steps=4440,full_epochs=20,full_stdout_lines=len(stdout.splitlines()),stdout_warnings=[v for v in stdout.splitlines() if 'warning' in v.lower()],native_nonfinite_placeholder_fields=sorted({k for v in native for k,x in v.items() if isinstance(x,float) and not math.isfinite(x)}),all_step_values_finite_or_explicit_null=True,max_loss_recount_error=formula_error,max_base_loss_recount_error=base_error,supcon_layer_forward_terms=terms,L_forward_packets=r['L_forward_packets'],extra_L_forward_packets=r['extra_L_forward_packets'],As_training_observations=As,As_source_eval_observations=7560,ordinary_generated_packets=568320,physical_observations=observed,unique_physical_training=unique,teacher_additive_calls=0,scene_schedule_mode=mode,schedule_recount=sched,view_routing=route,channel_generation_wall_seconds=previous_wall,teacher_inherited_counter_offset_removed=2,instrumented_counters=counters,actual_network_forward_packets=NN,held_forward_packets=r['held_forward_packets'],elapsed_training_seconds=r['elapsed_seconds'],final_native_logged_first_ordinary_CE=r['final_L_CE'],final_base_L_objective=r['final_base_L_objective'],final_extra_pair_objective=r['final_extra_pair_objective'],total_pair_weight=r['total_pair_weight'],seen_V=r['seen_V'],tracked_BN_module_names=audit['tracked_BN_module_names'],extra_BN_updates=bn,coverage=coverage,ordinary_augmentation=ordinary)
 if len(arms)==1:
  value=dict(status='COMPLETED_SOURCE_ROW_ANALYZED',run_id=RUN,arm=arms[0],scores=scores,execution=execution,source_only=True,target_access=False,comparison_available=False);dump(OUT/('analysis_'+arms[0]+'.json'),value);return value
 assert arms==ARMS and all(s==role_sets[0] for s in role_sets)
 sequence_identical=all(s==seqs[0] for s in seqs)
 differences=[sorted(k for k,v in args_all[0].items() if v!=a.get(k)) for a in args_all[1:]]
 assert all(set(x)<=set(('candidate_id','output_dir','run_id')) and 'candidate_id' in x and 'output_dir' in x for x in differences)
 complete=read(OUT/'completion.json');result=read(OUT/'result.json');assert complete['status']=='ANALYZED' and complete['logical_rows']==4 and complete['new_rows']==2 and complete['reused_rows']==2 and complete['new_epochs']==40 and complete['reused_epochs']==40 and complete['new_optimizer_steps']==8880 and complete['reused_optimizer_steps']==8880
 paired={};strata=[];comparisons={};conditions={}
 for ref in ARMS[:3]:
  paired[ref]={}
  for view in VIEWS:
   b,s=predictions[ref][view],predictions['Q_mix'][view];assert b['labels']==s['labels'] and b['physical_ids']==s['physical_ids']
   repair=sum(p!=y and q==y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']));broken=sum(p==y and q!=y for p,q,y in zip(b['predictions'],s['predictions'],b['labels']))
   delta=100*(scores['Q_mix'][view]['accuracy']-scores[ref][view]['accuracy']);assert math.isclose(delta,100*(repair-broken)/1260,abs_tol=1e-10)
   paired[ref][view]=dict(accuracy_gain_pp=delta,macro_F1_gain_pp=100*(scores['Q_mix'][view]['macro_F1']-scores[ref][view]['macro_F1']),repaired=repair,broken=broken,clean_conditioned=decision_transitions(predictions[ref],predictions['Q_mix'],view))
   for dimension in ('TX','day'):
    for value in (range(6) if dimension=='TX' else (1,2,3)):
     take=[i for i,(y,p) in enumerate(zip(b['labels'],b['physical_ids'])) if (y if dimension=='TX' else p[2])==value]
     acc={a:sum(predictions[a][view]['predictions'][i]==b['labels'][i] for i in take)/len(take) for a in (ref,'Q_mix')}
     strata.append(dict(reference=ref,view=view,dimension=dimension,value=value,count=len(take),reference_accuracy=acc[ref],Q_mix_accuracy=acc['Q_mix'],gain_pp=100*(acc['Q_mix']-acc[ref])))
 q=qualification(scores,execution);conditions=q['conditions'];comparisons=q['comparisons'];qualified=q['qualified']
 assert qualified==complete['qualified'] and conditions==result['qualification']['conditions']
 for ref in ARMS[:3]:
  for view in VIEWS:assert math.isclose(comparisons[ref]['delta_pp'][view],result['qualification']['comparisons'][ref]['delta_pp'][view],abs_tol=1e-12)
 gaps={a:{v:dict(clean_minus_suburban_pp=100*(s['clean']['accuracy']-s[v]['accuracy']),absolute_gap_pp=100*abs(s['clean']['accuracy']-s[v]['accuracy'])) for v in VIEWS[1:4]} for a,s in scores.items()}
 value=dict(status='ANALYZED',actual_execution='VERIFIED',run_id=RUN,source_only=True,model_seed=SEED,qualification=dict(qualified=qualified,conditions=conditions,comparisons=comparisons),scores=scores,paired=paired,suburban_gaps=gaps,execution=execution,all_L_U_sequences_identical=sequence_identical,native_argument_differences=differences,CE_scope='Native final_L_CE is first ordinary smoothed CE;mean readout CE is runtime defaulthead report,hard predictions lack logits for independent CE recount;common base objective includes CE/RDrop/SupCon',new_model_epochs=40,new_optimizer_updates=8880,reused_model_epochs=40,reused_optimizer_updates=8880,new_dispatcher_smoke_NN_packets=2,new_actual_NN_forward_packets=sum(execution[a]['actual_network_forward_packets'] for a in ('Q_sat','Q_mix'))+2,reused_row_NN_forward_packets=sum(execution[a]['actual_network_forward_packets'] for a in ('Q0','Q_ord')),historical_control_origin_note='Original3row sceneCycle cost60modelE/13320updates plussharedsmoke2;2rows40E reusedhere,no oldsmoke reexecution. Thirdbalancedrow notpartofcurrentmatrix.',logical_row_NN_excluding_shared_smokes=sum(v['actual_network_forward_packets'] for v in execution.values()),target_access=False,head_fit_updates=0,Ag_efficacy_proved=False,joint_goal_complete=False,missing=['Independent source confirmation','Validated Ag','Same-contract B00/B10/B01/B11 and interaction','Formal target clean+sixpractical','Cross-seed joint verification'])
 dump(OUT/'analysis_summary.json',value)
 for name,values in (('training_curves.csv',curves),('source_strata.csv',strata)):
  fields=sorted({k for v in values for k in v})
  with (OUT/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
 return value
if __name__=='__main__':
 args=sys.argv[1:];arms=ARMS
 if args:
  assert len(args)==2 and args[0]=='--row' and args[1] in ARMS;arms=(args[1],)
 v=analyze(arms);print(json.dumps({k:v[k] for k in ('status','qualification','new_actual_NN_forward_packets','reused_row_NN_forward_packets') if k in v},ensure_ascii=False))
