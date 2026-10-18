"""Focused synthetic tailNoU checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_tailnou30_analysis_20261018 as m

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
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;preds={'NOAS30':control,'AS30':y.copy(),'TAIL_NO_U_NOAS30':y.copy(),'TAIL_NO_U_AS30':candidate}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,E50_source_eligible=False,fixed_recipe_closed=True,fullV_clean_vs_matched_NoAs_pp=0.,mid_low_mean_vs_original_As30_pp=0.))

def loss_fixture(step=1,sat=True,ids=None):
 As=64 if sat else 0;common=(.23,.2,.03);mixed=(.43,.4,.03);raw=(.45,.4,.05)
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=As,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts={'practical_low_urban':64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NONE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):r['common_'+suffix]=common[i];r['extra_'+suffix]=mixed[i];s['shape_'+suffix]=raw[i]
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 c=dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,U_packets=0,U_student_NN=0,teacher_train_NN=0,teacher_eval_NN=0,EMA_updates=0,L_forwards=6,total_L_scale=1.25)
 native={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};native.update({'no_u/'+k:c[k] for k in m.NO_U_KEYS});native.update({'train/loss':r['total_L'],'train/pseudo_total':0,'train/pseudo_selected':0,'train/pseudo_truth_available':0,'train/a1_ema_successful_updates':step})
 return r,s,native,c

def log_fixture(project,sat=True):
 from collections import Counter
 name='TAIL_NO_U_AS30' if sat else 'TAIL_NO_U_NOAS30';root=dict(m.rows(project))[name];b=root/'source';b.mkdir(parents=True)
 policy=dict(parent_U_epochs=7,parent_EMA_verified=True,parent_EMA_restored_in_child=False,native_label_epochs=0,native_pseudo_epochs=10,use_unlabeled=False,use_ema_teacher=False,U_loader_iteration=False,U_packets=0,U_student_NN=0,teacher_train_NN=0,teacher_eval_NN=0,EMA_updates=0,child_ema_model=None,L_forwards=6,total_L_scale=1.25,RNG_trajectory_matched=False,pure_pseudo_loss_isolated=False,whole_trajectory_NoU=False,new_losses=0,new_parameters=0,formal_test_changed=False)
 cfg=dict(updates=2220,crossview=dict(mode='ORD128'),tail_no_u=policy);put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=cfg,tail_U_packets=0,tail_EMA_updates=0,child_ema_model=None))
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080 if sat else 0,total_L_loss_scale=1.25,student_train_NN=1704960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=0,wall_seconds=10.,teacher_counter_offset=None,teacher_builds=0,U_packets=0,U_student_NN=0,U_iteration_attempts=0,U_steps=0,EMA_local_successful_updates=0,EMA_cumulative_updates=4440,parent_EMA_verified=True,parent_EMA_restored_in_child=False,child_ema_model=None,native_a1_ema_successful_updates_semantics='successful optimizer steps,not actual EMA updates',parent_U_epochs=7,whole_trajectory_NoU=False,tail_no_u=dict(policy,teacher_builds=0,U_iteration_attempts=0),coverage=[dict(TX=0,RX=1,scene='practical_low_urban' if sat else 'NONE',state='LOS' if sat else 'NONE',lock='LOCK' if sat else 'NONE',count=142080)])
 put(b/'result.json',result);put(b/'initialization.json',dict(teacher_origin='disabled in child;own20 EMA provenance verified',parent_EMA_verified=True,parent_EMA_restored_in_child=False,child_ema_model=None));put(b/'EMA_initialization.json',dict(origin='disabled in child;own20 EMA provenance verified',parent_EMA_verified=True,parent_EMA_restored_in_child=False,teacher_constructed=False,teacher_forwards=0,EMA_updates=0,child_ema_model=None));put(b/'actual_native_args.json',dict(args=dict(use_unlabeled=False,use_ema_teacher=False,lambda_u=0.,lambda_ent=0.,epochs=10,label_epochs=0,pseudo_epochs=10)));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def Lids(step):
  values=[((step-1)*128+j)%6300 for j in range(128)];ids=[[1,(1,3,4,6,8)[i%5],1+(i//5)%3,i,i] for i in values]
  if step==1:ids[1]=list(ids[0])
  return ids
 refs=[];shapes=[];native=[];units=[];nou=[]
 for step in range(1,2221):
  r,s,n,c=loss_fixture(step,sat);refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));nou.append(c)
 for filename,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('no_u_steps',nou)]:jl(b/(filename+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs);jl(b/'training_sequence.jsonl.gz',(dict(role='L',ids=Lids(i)) for i in range(1,2221)))
 path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return root

class TailNoUChecks(unittest.TestCase):
 def test_real_producer_relationship_and_full6L_noU_scalars(self):
  import ast,copy
  package=Path(__file__).resolve().parents[1]/'experiments/cvs_tail_no_u30';observer=ast.parse((package/'observer.py').read_text(encoding='utf8'));model=(package/'model.py').read_text(encoding='utf8');self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr=='teacher' for n in ast.walk(observer)));self.assertIn('delta,previous_info=original.hook(self,**kw)',model);self.assertIn("'no_u/'+k",model)
  r,s,n,c=loss_fixture();self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertEqual(m.check_no_u(r,c,n),512);self.assertNotIn('shape/RD',n);self.assertEqual(sum(k.startswith(('concat/','no_u/')) for k in n),14)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
  for key in m.NO_U_KEYS[:5]:
   wrong=copy.deepcopy(c);wrong[key]=1
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_no_u(r,wrong,n)
  wrong=copy.deepcopy(r);wrong['crossview_mode']='MIXED128'
  with self.assertRaises(ValueError):m.check_no_u(wrong,c,n)
 def test_complete_noU_logs_noneEMA_optimizer_counter_and_prefix(self):
  with temporary() as tmp:
   p=Path(tmp);a=log_fixture(p,True);b=log_fixture(p,False);qa,_=m.parse_new_logs(a,'TAIL_NO_U_AS30');qb,_=m.parse_new_logs(b,'TAIL_NO_U_NOAS30')
   self.assertEqual(qa['cost']['native_training_and_validation_NN'],2028962);self.assertEqual(qa['cost']['teacher_eval_NN'],0);self.assertEqual(qa['physical_loader_packets'],dict(L=284160,U=0));self.assertEqual(qa['counts']['As_packets'],142080);self.assertEqual(qb['counts']['As_packets'],0);self.assertEqual(qa['tail_no_u']['child_EMA_updates'],0);self.assertEqual(qa['tail_no_u']['parent_EMA_updates'],4440);self.assertIn('optimizer',qa['tail_no_u']['native_a1_counter_semantics']);self.assertEqual(qa['tail_no_u']['sampler_repetitions']['batches_with_repeated_physical_ID'],1)
   path=a/'source/no_u_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(a,'TAIL_NO_U_AS30')
 def test_noneEMA_and_no_U_consumption_are_actual_metadata_checks(self):
  with temporary() as tmp:
   root=log_fixture(Path(tmp),True);p=root/'source/result.json';q=m.read(p);q['child_ema_model']={};put(p,q)
   with self.assertRaises(ValueError):m.parse_new_logs(root,'TAIL_NO_U_AS30')
   q['child_ema_model']=None;put(p,q);stream=root/'source/training_sequence.jsonl.gz';values=list(m.logs.records(stream));jl(stream,values+[dict(role='U',ids=values[0]['ids'])])
   with self.assertRaises(ValueError):m.parse_new_logs(root,'TAIL_NO_U_AS30')
 def test_saved_original_audit_reused_and_changed_cost_not_equalized(self):
  with temporary() as tmp:
   p=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i,ordinary_raw_MSE=.01) for i in range(1,31)]
   put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in ('NOAS30','AS30')}));lookup=dict(m.rows(p))
   for a in ('NOAS30','AS30'):put(lookup[a]/'stage20/result.json',parent);put(lookup[a]/'source/actual_native_args.json',dict(args=dict(use_unlabeled=True)))
   newcost=dict(student_train_NN=1704960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=0,native_training_and_validation_NN=2028962);q=dict(cost=newcost,epoch_curves=curves[20:]);private=dict(args=dict(use_unlabeled=False),ordinary_mse=[.02]*2220)
   with mock.patch.object(m,'parse_new_logs',side_effect=lambda *a:(dict(q),dict(private))),mock.patch.object(m.logs,'compare_streams',return_value=dict(all_equal=True)) as compare:r=m.training_analysis(p)
   self.assertEqual(r['new_updates'],4440);self.assertTrue(r['rows']['AS30']['reused']);self.assertFalse(r['reference_comparability']['TAIL_NO_U_AS30']['four_counters_equal']);self.assertFalse(r['reference_comparability']['TAIL_NO_U_AS30']['U_rule_unchanged_by_recipe']);self.assertFalse(r['reference_comparability']['TAIL_NO_U_AS30']['RNG_trajectory_matched']);self.assertTrue(all(m.RUN in str(path) for path in compare.call_args.args))
 def test_only_two_source_conditions_and_native_JSON_bool(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass']);q['TAIL_NO_U_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);q['NOAS30']['full_V_clean']=.1;q['AS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(q)['source_criteria_pass']);json.dumps(m.source_conditions(q),allow_nan=False)
  q['TAIL_NO_U_AS30']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed']);self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
 def test_source_does_not_open_formal_truth(self):
  with temporary() as tmp:
   project=Path(tmp);source_fixture(project);original=m.read
   def guarded(p):
    if Path(p).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal access in source')
    return original(p)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(project)
   self.assertEqual(q['NN'],0);self.assertEqual(q['source_unique_V'],1800);self.assertFalse(q['decision']['source_criteria_pass']);self.assertFalse(q['decision']['scientific_negative_skips_formal'])
 def test_truth_closed_until_all4_sevenview_rows_complete(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['TAIL_NO_U_AS30']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_all_CM_partitions_pairs_stage_and_interaction(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],84);self.assertEqual(q['independent_confusion_matrices'],504);self.assertEqual(q['paired']['TAIL_NO_U_AS30__NOAS30']['clean']['repairs'],1);self.assertEqual(q['paired']['TAIL_NO_U_AS30__NOAS30']['clean']['new_errors'],1);self.assertFalse(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertTrue(q['clean_vs_original_NoAs']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['tail_no_u_interaction_pp']['clean'],-200/84);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_main_parameterized_publisher_cost_and_single_archive_collection(self):
  import cvs_tailnou30_publish_analysis_20261018 as pub
  import cvs_tailnou30_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_tailnou30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=1704960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=0,native_training_and_validation_NN=2028962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={a:dict(cost=cost) for a in m.CONDITIONS[2:]},reference_comparability={'different':True})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=1536,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=4,target_query=False));argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertGreater(len((out/'final_report.md').read_text(encoding='utf8').splitlines()),30);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],2*(2028962+12600+590)+1540)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11);self.assertIn('cvs_tailnou30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
