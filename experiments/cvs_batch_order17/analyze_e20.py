"""Offline full source/order/loss/cost recount; no model import or new inference."""
import csv,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.cvs_as_pair_ce20.analyze import read,records,dump,finite,prediction,ssl_boundary,decision_transitions,pair_formula
from experiments.cvs_iq_timing20.harvest import score
from .recount_order import audit as order_audit
OUT=ROOT/'local_artifacts/F_L_batch_order_E20_20261017/final'
RUN='20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01'
COMMIT='9059b58502f207331bf9643d9d1399d308a2c2da'
SEED=2026092701;ARMS=('O0','O1')
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def qualification(scores,execution):
 b,s=scores['O0'],scores['O1'];delta={v:100*(s[v]['accuracy']-b[v]['accuracy']) for v in VIEWS};f1=100*(s['clean']['macro_F1']-b['clean']['macro_F1']);sv=100*(execution['O1']['seen_V']['source_val_accuracy']-execution['O0']['seen_V']['source_val_accuracy']);eps=1e-10
 c=dict(clean=delta['clean']>=1.-eps,clean_F1=f1>=-eps,each_practical=all(delta[v]>=-.5-eps for v in VIEWS[1:]),seen_V=sv>=-.5-eps)
 for arm,r in execution.items():
  a=r['actual_L_order'];c[arm+'/execution']=r['full_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['total_pair_weight']==1.25 and r['As_training_observations']==0 and r['head_fit_updates']==0 and not r['target_access'] and r['final_base_L_objective']<=.5 and a['all4440_actual_inputs_checked'] and a['all740_physical_block_multisets_identical'] and (a['moved_positions']==0 if arm=='O0' else a['moved_positions']>0)
 return dict(qualified=all(c.values()),conditions=c,delta_accuracy_pp=delta,delta_clean_F1_pp=f1,delta_seenV_pp=sv,primary='O1',source_only=True,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False)
def analyze(arms=ARMS):
 OUT.mkdir(parents=True,exist_ok=True);scores={};predictions={};execution={};curves=[];orders=[];Ls=[];Us=[];args_all=[];roles=[]
 for arm in arms:
  row=arm+'-RX1-s'+str(SEED);src=OUT/row/'source';r=read(src/'satmix_result.json');init=read(src/'initialization.json');done=read(src/'completion.json');frozen=read(src/'source_frozen.json');audit=read(src/'additive_audit.json')
  assert r['status']=='SOURCE_L_BATCH_ORDER_E20_COMPLETE' and init['scratch_only'] and not init['checkpoint_sources'] and not init['ancestors'] and not init['target_contact']
  assert done['epoch']==frozen['epoch']==20 and done['optimizer_steps']==4440 and not done['target_access'] and done['head_fit_updates']==0 and frozen['selection']=='final_only'
  assert r['actual_steps']==r['additive_steps']==4440 and r['L_forward_packets']==4546560 and r['extra_L_forward_packets']==1136640 and r['supcon_applied']==3330 and r['supcon_layer_forward_terms']==13320
  assert r['Ag'] is False and r['As'] is False and not r['U_truth_read'] and not r['target_access'] and not r['head_fit_updates'] and r['teacher_additive_calls']==0 and not r['checkpoint_sources']
  assert r['As_training_observations']==0 and r['total_pair_weight']==1.25 and audit['calls']==4440 and audit['As_generated_packets']==0 and audit['no_As_U_or_teacher']
  resolved=read(src/'resolved_config.json');assert resolved['commit']==COMMIT and resolved['run_id']==RUN and resolved['row_id']==row
  contract=read(src/'source_contract.json');assert {k:len(v) for k,v in contract['role_ids'].items()}=={'L_s':5040,'U_s':45360,'V':21600};roles.append({k:set(v) for k,v in contract['role_ids'].items()})
  scores[arm]={};predictions[arm]={}
  for view in VIEWS:
   p=read(src/'source_readouts'/(view+'.json'));s=prediction(p)
   assert math.isclose(s['accuracy'],r['views'][view]['accuracy'],abs_tol=1e-12) and math.isclose(s['macro_F1'],r['views'][view]['macro_F1'],abs_tol=1e-12)
   if view!='clean':assert p['labels']==predictions[arm]['clean']['labels'] and p['physical_ids']==predictions[arm]['clean']['physical_ids']
   scores[arm][view]=s;predictions[arm][view]=p
  native=records(src/'metrics_epoch.jsonl');compact=records(src/'epoch_metrics.jsonl')
  with (src/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
  assert [v['epoch'] for v in native]==list(range(1,21)) and [v['epoch'] for v in compact]==list(range(1,21)) and len(table)==20
  for a,b,c in zip(native,compact,table):
   for k in ('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr'):assert math.isclose(float(a[k]),float(b[k]),abs_tol=1e-10) and math.isclose(float(a[k]),float(c[k]),abs_tol=1e-10)
  steps=records(src/'step_metrics.jsonl.gz');mix=records(src/'satmix_steps.jsonl.gz');unit=records(src/'unit_steps.jsonl.gz')
  assert all([v['step'] for v in values]==list(range(1,4441)) for values in (steps,mix,unit))
  terms=truth=skips=bn=0;formula_error=base_error=0.;aggregate={}
  for n,v,u in zip(steps,mix,unit):
   finite(n);finite(v);finite(u);epoch=(v['step']-1)//222+1;sw=.1*min(max((epoch-5)/10,0.),1.)
   assert n['epoch']==v['epoch']==epoch and v['same_IQ_pairs'] and v['all_L_supervised'] and v['extra_view_once'] and v['extra_same_IQ_pair']
   assert v['As_packets']==v['extra_As_packets']==0 and v['extra_input']=='raw' and v['sat_raw_MSE']==v['extra_raw_MSE']==0
   assert v['extra_pair_weight']==.25 and v['base_raw_total_weight']==.75 and v['base_ordinary_weight']==.25 and v['total_pair_weight']==1.25
   assert math.isclose(v['supcon_weight'],sw,abs_tol=1e-12) and len(v['supcon_details'])==(4 if epoch>5 else 0)
   assert all(sum(v[k].values())==0 for k in ('scene_counts','state_counts','lock_counts','extra_scene_counts')) and v['scene_schedule_mode']=='none' and v['channel_generation_wall_seconds']==0
   terms+=len(v['supcon_details']);bn+=v['extra_BN_batch_updates'];base,extra,total=pair_formula(v,'Q0')
   base_error=max(base_error,abs(base-v['base_total_L']));formula_error=max(formula_error,abs(total-v['total_L']),abs(extra-v['extra_pair_loss']),abs(.25*extra-v['extra_pair_weighted']))
   nm=n['metrics']
   for k in ('total_L','base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_weight','extra_As_packets'):assert math.isclose(float(nm['satmix/'+k]),v[k],abs_tol=1e-6)
   truth+=int(nm.get('train/pseudo_truth_available',0) or 0);skips+=int(nm.get('train/skipped_nonfinite_loss',0) or nm.get('train/skipped_nonfinite_grad',0) or 0);assert u['physical_clip_calls']==v['step']
   sums=aggregate.setdefault(epoch,{})
   for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','raw_KL','ordinary_KL','s_slot_KL','supcon','ordinary_raw_MSE','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','extra_raw_MSE'):sums[k]=sums.get(k,0.)+v[k]/222
   sums['native_train_total_loss']=sums.get('native_train_total_loss',0.)+nm['train/loss']/222
  assert terms==13320 and truth==skips==0 and formula_error<1e-5 and base_error<1e-5 and bn==audit['extra_BN_updates']
  for k in ('base_total_L','extra_pair_loss'):assert math.isclose(aggregate[20][k],r['final_base_L_objective' if k=='base_total_L' else 'final_extra_pair_objective'],abs_tol=1e-8)
  assert r['final_native_logged_L_CE']==r['final_L_CE'] and math.isclose(compact[-1]['train_loss_tx_labeled'],r['final_L_CE'],abs_tol=1e-10)
  ssl=ssl_boundary(steps);order,by_epoch,original_L,U=order_audit(src,arm,mix);Ls.append(original_L);Us.append(U)
  for v in compact:curves.append(dict(row=row,arm=arm,**v,**aggregate[v['epoch']]))
  for v in by_epoch:orders.append(dict(arm=arm,epoch=v['epoch'],**{'original_'+k:x for k,x in v['original'].items()},**{'actual_'+k:x for k,x in v['actual'].items()}))
  coverage=read(src/'satellite_supervision_coverage.json');assert coverage['total']==0 and not coverage['cells'] and not coverage['confidence_gate']
  args=read(src/'actual_native_args.json');args_all.append(args);assert args['seed']==SEED and args['use_aug'] and args['source_fold_held_rx']==1 and args['label_epochs']==13 and args['pseudo_epochs']==7 and not args['use_tx_rx_balanced_sampler'] and args['aug_enable_pa_normal'] and args['aug_p_dac']==0 and not args['aug_enable_class_signature']
  ordinary=read(src/'ordinary_augmentation_actual.json');assert ordinary['no_second_pair_redraw'] and ordinary['practical_input_is_raw']
  stdout=(OUT/'logs'/(row+'.stdout.log')).read_text(encoding='utf8',errors='replace');assert not any(s in stdout for s in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL')) and 'SMOKE PASS own scratch checkpoint; query_read=false' in stdout
  counters={k:r[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets')};NN=sum(counters.values())-2+r['held_forward_packets']
  execution[arm]=dict(full_steps=4440,full_epochs=20,full_stdout_lines=len(stdout.splitlines()),stdout_warnings=[v for v in stdout.splitlines() if 'warning' in v.lower()],native_nonfinite_placeholder_fields=sorted({k for v in native for k,x in v.items() if isinstance(x,float) and not math.isfinite(x)}),all_step_values_finite_or_explicit_null=True,max_loss_recount_error=formula_error,max_base_loss_recount_error=base_error,supcon_layer_forward_terms=terms,L_forward_packets=r['L_forward_packets'],extra_L_forward_packets=r['extra_L_forward_packets'],As_training_observations=0,As_source_eval_observations=7560,ordinary_generated_packets=568320,teacher_additive_calls=0,scene_schedule_mode='none',channel_generation_wall_seconds=0,teacher_inherited_counter_offset_removed=2,instrumented_counters=counters,actual_network_forward_packets=NN,held_forward_packets=r['held_forward_packets'],elapsed_training_seconds=r['elapsed_seconds'],final_native_logged_first_ordinary_CE=r['final_L_CE'],final_base_L_objective=r['final_base_L_objective'],final_extra_pair_objective=r['final_extra_pair_objective'],total_pair_weight=r['total_pair_weight'],seen_V=r['seen_V'],tracked_BN_module_names=audit['tracked_BN_module_names'],extra_BN_updates=bn,ordinary_augmentation=ordinary,actual_L_order=order,original_SSL=ssl,head_fit_updates=0,target_access=False)
  dump(OUT/('analysis_'+arm+'.json'),dict(status='COMPLETED_SOURCE_ROW_ANALYZED',run_id=RUN,arm=arm,scores={arm:scores[arm]},execution={arm:execution[arm]},source_only=True,target_access=False,comparison_available=False))
 if len(arms)==1:return read(OUT/('analysis_'+arms[0]+'.json'))
 assert arms==ARMS and roles[0]==roles[1] and Ls[0]==Ls[1]
 differences=sorted(k for k,v in args_all[0].items() if v!=args_all[1].get(k));assert set(differences)<=set(('candidate_id','output_dir','run_id'))
 complete=read(OUT/'completion.json');result=read(OUT/'result.json');assert complete['status']=='ANALYZED' and complete['logical_rows']==complete['new_rows']==2 and complete['new_epochs']==40 and complete['new_optimizer_steps']==8880
 q=qualification(scores,execution);assert q==result['qualification'] and q['qualified']==complete['qualified']
 paired={v:decision_transitions(predictions['O0'],predictions['O1'],v) for v in VIEWS};strata=[]
 for view in VIEWS:
  p=predictions['O0'][view]
  for dimension,values in (('TX',range(6)),('day',(1,2,3)),('TX_day',[(y,d) for y in range(6) for d in (1,2,3)])):
   for v in values:
    take=[i for i,(y,z) in enumerate(zip(p['labels'],p['physical_ids'])) if (y if dimension=='TX' else z[2] if dimension=='day' else (y,z[2]))==v]
    assert len(take)==(210 if dimension=='TX' else 420 if dimension=='day' else 70)
    ss={a:score([p['labels'][i] for i in take],[predictions[a][view]['predictions'][i] for i in take]) for a in ARMS}
    strata.append(dict(view=view,dimension=dimension,value=str(v),count=len(take),O0_accuracy=ss['O0']['accuracy'],O1_accuracy=ss['O1']['accuracy'],delta_pp=100*(ss['O1']['accuracy']-ss['O0']['accuracy']),O0_macro_F1=ss['O0']['macro_F1'],O1_macro_F1=ss['O1']['macro_F1']))
 gaps={a:{v:dict(accuracy=s[v]['accuracy'],clean_minus_suburban_pp=100*(s['clean']['accuracy']-s[v]['accuracy']),absolute_gap_pp=100*abs(s['clean']['accuracy']-s[v]['accuracy'])) for v in VIEWS[1:4]} for a,s in scores.items()}
 value=dict(status='ANALYZED',actual_execution='VERIFIED',execution_commit=COMMIT,run_id=RUN,model_seed=SEED,source_only=True,qualification=q,scores=scores,paired=paired,suburban_gaps=gaps,execution=execution,common_original_L_sequence_identical=True,U_metadata_sequence_identical=Us[0]==Us[1],U_TX_resolutions=0,native_argument_differences=differences,new_model_epochs=40,new_optimizer_updates=8880,new_dispatcher_smoke_NN_packets=2,new_actual_NN_forward_packets=sum(v['actual_network_forward_packets'] for v in execution.values())+2,reused_training_rows=0,parent_training_cost=0,CE_scope='Full original L objective independently recomposed from saved per-step scalar terms;hard source predictions have no logits,so held source mean CE is runtime report only. Native epoch L metric is first ordinary CE,not full L objective.',target_access=False,head_fit_updates=0,new_IQ_Ag=False,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,missing=['Independent source confirmation if fixed criteria pass','Validated IQ Ag','Same-contract B00/B10/B01/B11 and interaction','Formal target clean+sixpractical','Cross-seed joint verification'])
 dump(OUT/'analysis_summary.json',value)
 for name,values in (('training_curves.csv',curves),('batch_order_curves.csv',orders),('source_strata.csv',strata)):
  with (OUT/name).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for v in values for k in v}));w.writeheader();w.writerows(values)
 return value
if __name__=='__main__':
 args=sys.argv[1:];arms=ARMS
 if args:
  assert len(args)==2 and args[0]=='--row' and args[1] in ARMS;arms=(args[1],)
 v=analyze(arms);print(json.dumps({k:v[k] for k in ('status','qualification','new_actual_NN_forward_packets') if k in v},ensure_ascii=False))
