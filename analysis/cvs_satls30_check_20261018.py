"""Focused synthetic SceneBalance checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_satls30_analysis_20261018 as m

def temporary():
 root=Path(os.environ.get('CVS_ANALYSIS_TEST_TMP',tempfile.gettempdir()));root.mkdir(parents=True,exist_ok=True)
 return tempfile.TemporaryDirectory(dir=root)


def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')
def jl(path,values):
 with gzip.open(path,'wt',encoding='utf8') if path.suffix=='.gz' else path.open('w',encoding='utf8') as f:
  for q in values:f.write(json.dumps(q,allow_nan=False)+'\n')
def csvjson(root,name,values):
 import csv
 jl(root/(name+'.jsonl'),values);fields=list(dict.fromkeys(k for q in values for k in q))
 with (root/(name+'.csv')).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)

def metadata(project,count):
 for arm,root in m.rows(project):
  cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=30,cumulative_optimizer_steps=6660,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.))
  put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def source_fixture(project):
 metadata(project,168000);y=np.arange(1800)%6;rx=np.asarray([1,3,4,6,8])[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.stack([np.zeros(1800,int),rx,day,np.arange(1800),np.arange(1800)],1);clean=np.zeros((1800,6));clean[np.arange(1800),y]=2.;sat=np.broadcast_to(clean,(2,3,1800,6)).copy()
 for _,root in m.rows(project):np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=clean,satellite=sat)

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)])
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;preds={'NOAS30':control,'AS30':y.copy(),'SATLS_AS30':candidate}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,E50_source_eligible=False,fixed_recipe_closed=True,fullV_clean_vs_NoAs30_pp=0.,mid_low_mean_vs_original_As30_pp=0.))

def loss_fixture(step=1,ids=None):
 if ids is None:ids=[[1,(1,3,4,6,8)[j%5],1+j%3,j,j] for j in range(128)]
 positions=[j for j in range(128) if (j+step)%2!=0];details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts={'practical_low_urban':64},state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.445,extra_CE=.415,extra_KL=.03)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05)
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 c=dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,selected_packets=64,As_packets=64,old_epsilon=.01,epsilon=.05,native_label_smoothing=.01,selected_CE_old=.4,selected_CE_new=.43,selected_CE_old_pass0=.35,selected_CE_old_pass1=.45,selected_CE_new_pass0=.37,selected_CE_new_pass1=.49,ordinary_CE_pass0=.3,ordinary_CE_pass1=.5,mixed_CE_old=.4,mixed_CE_new=.415,mixed_CE_delta=.015,loss_delta=.0075,mixed_KL=.03,mixed_RD_old=.43,mixed_RD_new=.445,raw_RD=.45,common_RD=.23,SupCon=.05,SupCon_weight=.1,total_L=r['total_L'],base_total_L=r['total_L']-.0075,total_L_scale=1.25,extra_NN=0,extra_CE_operator_calls=4,extra_CE_logit_packets=256,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],CE_operator_cpu_wall_seconds=.0001,scalarization_cpu_wall_seconds=.0002,explicit_CUDA_synchronize=False,all_six_classes_same_formula=True,confidence_filter=False,same_IQ_each_pair=True,ordinary_redraws=0,selected_L_positions=positions,sample_ids=[ids[j] for j in positions],TX=[j%6 for j in positions],RX=[ids[j][1] for j in positions],day=[ids[j][2] for j in positions],scene_counts=r['scene_counts'])
 n={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};n.update({'satls/'+k:c[k] for k in m.SATLS_KEYS});n.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 return r,s,n,c

def log_fixture(project):
 from collections import Counter
 root=dict(m.rows(project))['SATLS_AS30'];b=root/'source';b.mkdir(parents=True)
 policy=dict(old_epsilon=.01,epsilon=.05,native_label_smoothing=.01,selected_packets_per_step=64,selected_total_CE_weight=.25,extra_CE_operator_calls_per_step=4,extra_CE_logit_packets_per_step=256,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],mixed_KL_unchanged=True,common_raw_U_SupCon_unchanged=True,same_IQ_within_RDpair=True,all_six_classes_same_formula=True,confidence_filter=False,extra_NN=0,explicit_CUDA_synchronize=False,new_losses=0,new_U_views=0,formal_test_changed=False)
 put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=dict(updates=2220,crossview=dict(mode='ORD128'),satellite_label_smoothing=policy)))
 a=dict(mode='selectedAs_CE05',old_epsilon=.01,epsilon=.05,native_label_smoothing=.01,selected_packets=142080,actual_As_packets=142080,channel_calls=2220,pair_weights=[.5,.5,.25],SupCon='ORD128',total_L_scale=1.25,mixed_KL_unchanged=True,common_raw_U_SupCon_unchanged=True,extra_NN=0,extra_CE_operator_calls=8880,extra_CE_logit_packets=568320,baseline_mixed_CE_operator_widths=[128,128],extra_CE_operator_widths=[64,64,64,64],actual_mixed_CE_operator_widths=[128,128,64,64,64,64],CE_operator_cpu_wall_seconds=.222,scalarization_cpu_wall_seconds=.444,timing='perf_counter CPUwall,notCUDAevent;no explicitCUDA synchronization;scalar reads may synchronize',confidence_filter=False,all_six_classes_same_formula=True,new_U_views=0,sampler_changed=False)
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080,total_L_loss_scale=1.25,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,satellite_label_smoothing=a)
 put(b/'actual_native_args.json',dict(args=dict(label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True)));put(b/'EMA_initialization.json',dict(origin='strict own stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [1,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def Lids(step):
  ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)]
  if step==1:ids[1]=list(ids[0])
  return ids
 refs=[];shapes=[];native=[];units=[];satls=[];coverage=Counter()
 for step in range(1,2221):
  ids=Lids(step);r,s,n,c=loss_fixture(step,ids)
  for t,x in zip(c['TX'],c['RX']):coverage[(t,x,'practical_low_urban','LOS','LOCK')]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));satls.append(c)
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('satls_steps',satls)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(1,2221):yield dict(role='L',ids=Lids(i))
  for ep in range(10):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return root

class SatelliteLSChecks(unittest.TestCase):
 def test_actual_CE_only_delta_and_full6L_normalization(self):
  import copy
  r,s,n,c=loss_fixture();e,valid=m.check_satls(r,s,c,n);self.assertLess(e,3e-5);self.assertEqual(valid,512);self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertNotIn('shape/RD',n);self.assertEqual(sum(k.startswith(('concat/','satls/')) for k in n),14)
  for k,v in [('epsilon',.01),('native_label_smoothing',.05),('selected_packets',128),('mixed_CE_delta',.03),('loss_delta',.015),('mixed_KL',.04),('confidence_filter',True),('extra_NN',1),('extra_CE_operator_calls',0),('extra_CE_logit_packets',384),('extra_CE_operator_widths',[128,128]),('selected_L_positions',list(range(64)))]:
   q=copy.deepcopy(c);q[k]=v
   with self.subTest(key=k),self.assertRaises(ValueError):m.check_satls(r,s,q,n)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
  old=(1-.01)+.01/6;new=(1-.05)+.05/6;self.assertAlmostEqual(old,.9916666666666667);self.assertAlmostEqual(new,.9583333333333334);self.assertAlmostEqual(.05/6,.008333333333333333)
 def test_complete_new_tail_loss_cost_scope_and_prefix_rejection(self):
  with temporary() as tmp:
   p=Path(tmp);row=log_fixture(p);q,_=m.parse_new_logs(row,'SATLS_AS30');self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['counts']['U_forward_packets'],567000);self.assertEqual(q['extra_objective_accounting']['extra_CE_operator_calls'],8880);self.assertEqual(q['extra_objective_accounting']['extra_CE_logit_packets'],568320);self.assertTrue(q['extra_objective_accounting']['zero_extra_model_forward_is_not_zero_compute']);self.assertEqual(q['satls']['sampler_repetitions']['batches_with_repeated_physical_ID'],1)
   path=row/'source/satls_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'SATLS_AS30')
 def test_saved_original30_audit_reused_without_old_step_reads(self):
  with temporary() as tmp:
   p=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i,ordinary_raw_MSE=.01) for i in range(1,31)]
   put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in ('NOAS30','AS30')}));lookup=dict(m.rows(p));put(lookup['AS30']/'stage20/result.json',parent);put(lookup['AS30']/'source/actual_native_args.json',dict(args={}))
   newcost={k:oldcost[k]-parent[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};q=dict(cost=newcost,epoch_curves=curves[20:]);private=dict(args={},ordinary_mse=[.01]*2220)
   with mock.patch.object(m,'parse_new_logs',return_value=(q,private)),mock.patch.object(m.logs,'records',side_effect=AssertionError('Old logs reopened')):r=m.training_analysis(p)
   self.assertTrue(r['rows']['NOAS30']['reused']);self.assertEqual(r['new_updates'],2220);self.assertTrue(r['reference_comparability']['four_counters_equal']);self.assertFalse(r['reference_comparability']['old_step_logs_rescanned'])
 def test_two_source_conditions_use_different_frozen_references(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass']);q['SATLS_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);q['AS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(q)['source_criteria_pass']);json.dumps(m.source_conditions(q),allow_nan=False)
  q['SATLS_AS30']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed']);self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
 def test_source_does_not_open_formal_truth(self):
  with temporary() as tmp:
   project=Path(tmp);source_fixture(project);original=m.read
   def guarded(p):
    if Path(p).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal access in source')
    return original(p)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(project)
   self.assertEqual(q['NN'],0);self.assertEqual(q['source_unique_V'],1800);self.assertFalse(q['decision']['source_criteria_pass']);self.assertFalse(q['decision']['scientific_negative_skips_formal'])
 def test_truth_closed_until_all3_sevenview_rows_complete(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['SATLS_AS30']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_full_formal_CM_partitions_errors_and_separate_LS_increment(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],63);self.assertEqual(q['independent_confusion_matrices'],378);self.assertEqual(q['paired']['SATLS_AS30__NOAS30']['clean']['repairs'],1);self.assertEqual(q['paired']['SATLS_AS30__NOAS30']['clean']['new_errors'],1);self.assertTrue(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['incremental_satellite_LS_effect_pp']['clean'],-100/84);self.assertNotIn('contrast_interaction_pp',q);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_main_parameterized_publisher_cost_and_single_archive_collection(self):
  import cvs_satls30_publish_analysis_20261018 as pub
  import cvs_satls30_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_satls30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={'SATLS_AS30':dict(cost=cost)},reference_comparability={'old_step_logs_rescanned':False})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=768,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False));argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertGreater(len((out/'final_report.md').read_text(encoding='utf8').splitlines()),30);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],3162962+12600+590+770)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11);self.assertIn('cvs_satls30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
