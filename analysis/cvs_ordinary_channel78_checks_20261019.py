"""P78 focused pure-CPU fixtures; no real model, IQ, checkpoint or remote I/O."""
import ast,copy,csv,gzip,json,os,sys,tempfile,unittest
from collections import Counter
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_ordinary_channel78_analysis_20261019 as m

def temporary():
 root=Path(os.environ.get('CVS_ANALYSIS_TEST_TMP',tempfile.gettempdir()));root.mkdir(parents=True,exist_ok=True)
 return tempfile.TemporaryDirectory(dir=root)

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')

def jl(path,values):
 with gzip.open(path,'wt',encoding='utf8') if path.suffix=='.gz' else path.open('w',encoding='utf8') as stream:
  for value in values:stream.write(json.dumps(value,allow_nan=False)+'\n')

def epochs(root):
 values=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)]
 for name in ('metrics_epoch','epoch_metrics'):
  jl(root/(name+'.jsonl'),values)
  with (root/(name+'.csv')).open('w',encoding='utf8',newline='') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)

def metadata(project,count=168000):
 for arm,root in m.rows(project):
  cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=30,cumulative_optimizer_steps=6660,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.))
  put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def source_fixture(project):
 metadata(project);y=np.arange(1800)%6;rx=np.asarray([1,3,4,6,8])[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.asarray(['source'+str(i) for i in range(1800)]);clean=np.zeros((1800,6));clean[np.arange(1800),y]=2.;sat=np.broadcast_to(clean,(2,3,1800,6)).copy()
 for _,root in m.rows(project):np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=clean,satellite=sat)

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)])
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 old=y.copy();old[0]=1;new=y.copy();new[-1]=4;preds={'NOAS30':old,'AS30':y.copy(),'OC_NOAS30':y.copy(),'OC_AS30':new}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for view in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(view+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=view,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS}
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary=summary,decision=m.source_conditions(summary))

def step_fixture(step=1,sat=True,MP=True,AWGN=True,mse=.02):
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64 if sat else 0,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=mse,scene_counts={'practical_low_urban':64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NONE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):r['common_'+suffix]=(.23,.2,.03)[i];r['extra_'+suffix]=(.43,.4,.03)[i];s['shape_'+suffix]=(.45,.4,.05)[i]
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 a=dict(step=step,count=128,enabled=True,input_unchanged=True,primitive_restored=True,full_native_aug_computation=True,post_return_rng_preserved=True,primitive_CPU_rng_advanced=int(MP)+int(AWGN),primitive_CUDA_rng_advanced=0,primitive_Python_rng_advanced=0,returned_ordinary_raw_MSE=mse,extra_NN=0)
 for kind,hit in [('MP',MP),('AWGN',AWGN)]:a.update({kind+'_calls':int(hit),kind+'_packets':128*int(hit),'shadow_'+kind+'_packets':128*int(hit),kind+'_hit':bool(hit)})
 c=dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,ordinary_calls=1,ordinary_packets=128,actual_ORD128_anchor_input='ordinary128 with native MP/AWGN output bypass',same_IQ_each_RDpair=True,**{k:a[k] for k in m.OC_KEYS if k not in ('ordinary_calls','ordinary_packets')})
 native={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};native.update({'ordinary_channel/'+k:c[k] for k in m.OC_KEYS});native.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 ur=dict(step=step,packets=124 if step%222==0 else 256,native_output_passthrough=True,L_primitive_wrappers_inactive=True,post_return_rng_preserved=True,extra_NN=0)
 return r,s,native,c,a,ur

def log_fixture(project,sat=True):
 arm='OC_AS30' if sat else 'OC_NOAS30';root=dict(m.rows(project))[arm];b=root/'source';b.mkdir(parents=True)
 policy=dict(removed=['AWGN','MP'],ordinary_calls_per_step=1,ordinary_packets_per_step=128,first20_full_ordinary_retained=True,U_weak_strong_unchanged=True,As_raw_path_unchanged=True,other_native_operations_unchanged=True,U_augmentor_wrapped=False,whole_training_RNG_matched=False,new_losses=0,new_parameters=0,new_NN=0,formal_test_changed=False)
 cfg=dict(updates=2220,crossview=dict(mode='ORD128'),ordinary_channel=policy);put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=cfg))
 aggregate=dict(mode='L_ordinary_native_AWGN_MP_execute_then_bypass',ordinary_factory_calls=1,ordinary_calls=2220,ordinary_packets=284160,U_calls=2220,U_packets=567000,U256_calls=2210,U124_calls=10,U_native_output_passthrough=True,U_weak_strong_unchanged=True,As_raw_path_unchanged=True,first20_full_ordinary_retained=True,whole_training_RNG_matched=False,full_native_aug_computation=True,new_losses=0,new_parameters=0,extra_NN=0,pair_weights=[.5,.5,.25],total_L_scale=1.25,returned_ordinary_raw_MSE_mean=.02,all_primitive_restored=True,all_post_return_rng_preserved=True)
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080 if sat else 0,total_L_loss_scale=1.25,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,U_steps=2220,U_packets=567000,U_student_NN=567000,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,parent_EMA_restored=True,ordinary_channel=aggregate)
 put(b/'actual_native_args.json',dict(args=dict(use_aug=True,use_unlabeled=True,use_ema_teacher=True)));put(b/'EMA_initialization.json',dict(origin='strict own original P43 stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
 put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,removed_primitives=['AWGN','MP'],no_ordinary_before_practical=True,RDrop_same_IQ=True,use_aug=True,full_native_aug_computation=True,U_strong_guard_only=True,U_augmentor_wrapped=False,whole_training_RNG_matched=False))
 def physical(role,i):return [i%6,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def Lids(step):return [physical('L',((step-1)*128+j)%6300) for j in range(128)]
 refs=[];shapes=[];native=[];units=[];channel=[];draws=[];passthrough=[];coverage=Counter();sums=Counter()
 for step in range(1,2221):
  r,s,n,c,a,ur=step_fixture(step,sat,MP=bool(step%2),AWGN=bool(step%3))
  for j,z in enumerate(Lids(step)):
   if (j+step)%2!=0:coverage[(z[0],z[1],'practical_low_urban' if sat else 'NONE','LOS' if sat else 'NONE','locked' if sat else 'NONE')]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));channel.append(c);draws.append(a);passthrough.append(ur)
  sums.update({k:a[k] for k in ('MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced')});sums.update(MP_hit_steps=int(a['MP_hit']),AWGN_hit_steps=int(a['AWGN_hit']),both_hit_steps=int(a['MP_hit'] and a['AWGN_hit']))
 aggregate.update(sums);result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,records in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('ordinary_channel_steps',channel),('ordinary_draw_steps',draws),('U_passthrough_steps',passthrough)]:jl(b/(name+'.jsonl.gz'),records)
 epochs(b)
 def sequence():
  for step in range(1,2221):yield dict(role='L',ids=Lids(step))
  for epoch in range(10):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All ten epochs complete\n',encoding='utf8');return root

class OrdinaryChannelChecks(unittest.TestCase):
 def test_real_schema_fields_without_importing_training(self):
  path=Path(__file__).resolve().parents[1]/'experiments/cvs_ordinary_channel78/schema.py';tree=ast.parse(path.read_text(encoding='utf8'));fields=next(ast.literal_eval(q.value) for q in tree.body if isinstance(q,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OC_FIELDS' for t in q.targets));self.assertEqual(fields,m.OC_KEYS)
 def test_primitive_zero_and_positive_hits_and_nonzero_other_ordinary(self):
  for MP,AWGN in ((False,False),(True,False),(False,True),(True,True)):
   r,s,n,c,a,u=step_fixture(MP=MP,AWGN=AWGN);self.assertEqual(m.check_ordinary(r,c,a,n),512);self.assertLess(m.check_native_L(r,s,n),3e-5)
  r,s,n,c,a,u=step_fixture(mse=0.);self.assertEqual(m.check_ordinary(r,c,a,n),512)
 def test_wrong_shadow_scope_or_changed_objective_rejected(self):
  r,s,n,c,a,u=step_fixture()
  for key,value in [('count',64),('enabled',False),('primitive_restored',False),('input_unchanged',False),('post_return_rng_preserved',False),('MP_packets',127),('shadow_AWGN_packets',0),('MP_hit',False),('extra_NN',1)]:
   wrong=copy.deepcopy(a);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_ordinary(r,c,wrong,n)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
 def test_U_passthrough_boundary_and_no_double_execution(self):
  for step,count in ((1,256),(222,124)):
   u=step_fixture(step)[-1];m.check_U(u,step,count)
   for key,value in [('packets',128),('native_output_passthrough',False),('L_primitive_wrappers_inactive',False),('post_return_rng_preserved',False),('extra_NN',1)]:
    wrong=dict(u);wrong[key]=value
    with self.subTest(key=key),self.assertRaises(ValueError):m.check_U(wrong,step,count)
 def test_complete_2220_steps_shadow_U_and_actual_NN(self):
  with temporary() as tmp:
   root=log_fixture(Path(tmp),True);q,_=m.parse_new_logs(root,'OC_AS30');self.assertEqual(q['counts']['steps'],2220);self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['counts']['U_forward_packets'],567000);self.assertEqual(len(q['epoch_curves']),10);self.assertFalse(q['U_counterfactual_double_execution_claimed']);self.assertGreater(q['ordinary_channel']['returned_ordinary_raw_MSE_mean'],0.)
   path=root/'source/U_passthrough_steps.jsonl.gz';records=list(m.logs.records(path));jl(path,records[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(root,'OC_AS30')
 def test_aggregate_not_a_primitive_hit_or_MSE_scientific_gate(self):
  counts=Counter(ordinary_calls=2220,ordinary_packets=284160);a=dict(mode='L_ordinary_native_AWGN_MP_execute_then_bypass',ordinary_factory_calls=1,ordinary_calls=2220,ordinary_packets=284160,U_calls=2220,U_packets=567000,U256_calls=2210,U124_calls=10,U_native_output_passthrough=True,U_weak_strong_unchanged=True,As_raw_path_unchanged=True,first20_full_ordinary_retained=True,full_native_aug_computation=True,all_primitive_restored=True,all_post_return_rng_preserved=True,whole_training_RNG_matched=False,new_losses=0,new_parameters=0,extra_NN=0,pair_weights=[.5,.5,.25],total_L_scale=1.25,returned_ordinary_raw_MSE_mean=0.)
  for key in ('MP_calls','MP_packets','AWGN_calls','AWGN_packets','shadow_MP_packets','shadow_AWGN_packets','primitive_CPU_rng_advanced','primitive_CUDA_rng_advanced','primitive_Python_rng_advanced','MP_hit_steps','AWGN_hit_steps','both_hit_steps'):a[key]=0
  m.check_aggregate(a,counts,[0.]*2220)
  a['U124_calls']=0
  with self.assertRaises(ValueError):m.check_aggregate(a,counts,[0.]*2220)
 def test_only_two_source_criteria_and_absolute_pp_guard(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass']);q['OC_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);q['OC_NOAS30']['full_V_clean']=.981;self.assertFalse(m.source_conditions(q)['source_criteria_pass']);self.assertFalse(m.source_conditions(q)['scientific_negative_skips_formal']);self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
 def test_source_never_reads_formal_truth(self):
  with temporary() as tmp:
   project=Path(tmp);source_fixture(project);original=m.read
   def guarded(path):
    if Path(path).name in ('truth.json','scores.json','automatic_summary.json') or 'prediction' in Path(path).parts or 'scoring' in Path(path).parts:raise AssertionError('Formal access during source analysis')
    return original(path)
   with mock.patch.object(m,'read',guarded),mock.patch.object(m,'source_states',return_value=([],[],[],{})):q=m.source_analysis(project)
   self.assertEqual(q['NN'],0);self.assertFalse(q['decision']['source_criteria_pass']);self.assertFalse(q['actual_parent_clean_correct_cohort']['blocking']);self.assertFalse(q['actual_parent_clean_correct_cohort']['SAT18_cache_used'])
 def test_truth_closed_until_all_four_all7(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   project=Path(tmp);source=formal_fixture(project);(dict(m.rows(project))['OC_AS30']/'scoring/complete.json').unlink();original=m.read;touched=[]
   def guarded(path):
    if Path(path).name=='truth.json':touched.append(True);raise AssertionError('Truth opened too early')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(project,project/'truth.json',source)
   self.assertFalse(touched)
 def test_formal_repair_F1_partitions_and_factorial_interaction(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   project=Path(tmp);source=formal_fixture(project);q=m.formal_analysis(project,project/'truth.json',source);p=q['paired']['OC_AS30__NOAS30']['clean'];self.assertEqual((p['repairs'],p['new_errors'],p['net_repairs']),(1,1,0));self.assertEqual(q['partition_reconciliations'],84);self.assertAlmostEqual(q['ordinary_channel_interaction_pp']['clean'],-200/84);self.assertFalse(q['repeatable_clean_positive_from_As_proven']);self.assertFalse(q['new_Ag_success_claimed']);json.dumps(q,allow_nan=False)
 def test_fixed_NoAs_cohort_margin30_empty_strata(self):
  n=6;y=np.arange(n);z=np.eye(6)*30.;data={a:dict(ids=np.arange(n),TX=y,RX=np.asarray([1,3,4,6,8,1]),day=np.ones(n,int),clean=z.copy(),satellite=np.broadcast_to(z,(2,3,n,6)).copy()) for a in m.CONDITIONS};data['OC_AS30']['satellite'][:,:,0]=np.roll(data['OC_AS30']['satellite'][:,:,0],1,axis=-1);shape=(2,3,len(m.state_core.CONDITIONS),n);physics={k:np.zeros(shape) for k in ('state','lock','signal_to_added_noise_db','actual_waveform_quality_db','white_proxy_quality_db')};strata,effects,interaction=m.state_tables(data,physics);r=next(r for r in strata if r['row_id']=='OC_AS30' and r['scene']=='practical_low_urban' and r['draw']==-1 and r['dimension']=='overall');self.assertEqual(r['fixed_NOAS30_clean_correct_count'],12);self.assertAlmostEqual(r['satellite_retention'],5/6);self.assertAlmostEqual(r['mean_margin_over30'],2/3);empty=next(r for r in strata if r['dimension']=='state' and r['stratum']=='BLOCKED');self.assertIsNone(empty['accuracy']);self.assertTrue(empty['empty_cell'])
 def test_reused_audit_yields120_logical80_unique_epochs(self):
  with temporary() as tmp:
   project=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i,ordinary_raw_MSE=.01) for i in range(1,31)];put(project/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in ('NOAS30','AS30')}));lookup=dict(m.rows(project))
   for arm in ('NOAS30','AS30'):put(lookup[arm]/'stage20/result.json',parent);put(lookup[arm]/'source/actual_native_args.json',dict(args={}))
   pc=m.actual_cost(parent);newcost={k:oldcost[k]-pc[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};q=dict(cost=newcost,epoch_curves=curves[20:]);private=dict(args={},ordinary_mse=[.02]*2220)
   with mock.patch.object(m,'parse_new_logs',side_effect=lambda *a:(copy.deepcopy(q),private)),mock.patch.object(m.logs,'compare_streams',return_value=dict(all_equal=True)) as compare:r=m.training_analysis(project)
   self.assertEqual((r['logical_epoch_curve_rows'],r['unique_model_epochs'],r['unique_optimizer_updates']),(120,80,17760));self.assertTrue(all(m.RUN in str(path) for path in compare.call_args.args))
 def test_actual_cost8704668_and_no_reference_recharge(self):
  with temporary() as tmp:
   project=Path(tmp);metadata(project);root=project/'runs'/m.RUN;put(root/'synthetic_smoke.json',dict(status='PASS',target_access=False,NN=1538));put(root/'real_checkpoint_smoke.json',dict(status='PASS',target_query=False,NN=2));cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={a:dict(cost=dict(native_training_and_validation_NN=4797722)) for a in m.CONDITIONS[2:]});q=m.cost_summary(project,training);self.assertEqual(q['new_total_NN'],8704668);self.assertEqual(q['new_updates'],4440);self.assertTrue(q['old_costs_not_recharged'])
 def test_final_stage_needs_saved_source_decision(self):
  with temporary() as tmp:
   project=Path(tmp);out=project/'out';out.mkdir()
   with mock.patch.object(m,'formal_analysis',side_effect=AssertionError('Formal accessed before source')),self.assertRaises(FileNotFoundError):m.run(project,out,'final',project/'truth.json')
 def test_saved_source_final_report_uses_current_interaction_key(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   project=Path(tmp);source=formal_fixture(project);out=project/'analysis';out.mkdir();put(out/'source_summary.json',source);put(out/'source_complete.json',dict(status='SOURCE_ANALYZED',run_id=m.RUN,NN=0,target_scores_read=False));put(out/'source_decision.json',dict(status='SOURCE_DECIDED',run_id=m.RUN,decision=source['decision'],formal_scores_read=False));put(out/'source_state_confusions.json',[])
   cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost,epoch_curves=[]) for a in m.CONDITIONS},parent20={a:dict(cost=dict(native_training_and_validation_NN=4797722)) for a in m.CONDITIONS[2:]});put(out/'training_logs.json',training);root=project/'runs'/m.RUN;put(root/'synthetic_smoke.json',dict(status='PASS',target_access=False,NN=1538));put(root/'real_checkpoint_smoke.json',dict(status='PASS',target_query=False,NN=2))
   with mock.patch.object(m,'p78_plots',return_value=['fixture.png','fixture.pdf']):q=m.run(project,out,'final',project/'truth.json')
   self.assertEqual(q['status'],'ANALYZED');text=(out/'final_analysis.md').read_text(encoding='utf8');self.assertIn('80 唯一模型轮',text);self.assertIn('DAC/class-signature 原本关闭',text);self.assertNotIn('lraw_interaction',text);self.assertFalse(q['repeatable_clean_positive_proven'])

if __name__=='__main__':unittest.main(verbosity=2)
