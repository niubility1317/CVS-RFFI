"""Focused synthetic SceneBalance checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_scenebalance30_analysis_20261018 as m

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
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;preds={'NOAS30':control,'AS30':y.copy(),'BAL_AS30':candidate}
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
 common=(.23,.2,.03);mixed=(.43,.4,.03);raw=(.45,.4,.05);cum=4440+step;ep=(step-1)//222+1
 ids=ids if ids is not None else [[1,(1,3,4,6,8)[j%5],1+(j//5)%3,j,j] for j in range(128)]
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 planned=[21,21,21];extra=(cum-1)%3;planned[extra]+=1;seed=cum+10;perm=np.random.default_rng(seed).permutation(64).tolist();indices=np.repeat(np.arange(3),planned)[perm].tolist();scenes={v:planned[j] for j,v in enumerate(m.SOURCE_SCENES)};selected=[j for j in range(128) if (j+step)%2!=0];pids=[ids[j] for j in selected]
 r=dict(step=step,cumulative_step=cum,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts=scenes,state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details)
 s=dict(step=step,cumulative_step=cum,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):r['common_'+suffix]=common[i];r['extra_'+suffix]=mixed[i];s['shape_'+suffix]=raw[i]
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 c=dict(step=step,cumulative_step=cum,epoch=ep,cumulative_epoch=ep+20,namespace=f'FcrossRX/train/e{ep+20}/step{cum}/L_S',train_seed=701,assignment_namespace='scene_balance_assignment_v1',assignment_seed=seed,planned_counts=planned,scene_indices=indices,permutation=perm,extra_scene_index=extra,assignment_permutations=1,input_positions=list(range(64)),selected_L_positions=selected,selected_packets=64,As_packets=64,scene0_packets=planned[0],scene1_packets=planned[1],scene2_packets=planned[2],scene_counts=scenes,sample_id_fields=['eq_i','rx_i','day_i','sig_i','base_index'],sample_ids=pids,TX=[z[4]%6 for z in pids],RX=[z[1] for z in pids],day=[z[2] for z in pids],state=['LOS']*64,lock=['LOCK']*64,session_ids=[f'rx{z[1]}:day{z[2]}:eq1' for z in pids],channel_seeds=list(range(64)),channel_seed_formula='stable_seed(train_seed,namespace,assignedscene,pid)',physical_ID_uniqueness_required=False,sampler_changed=False,global_RNG_used=False,extra_NN=0,new_losses=0,formal_test_changed=False)
 native={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};native.update({'balance/'+k:c[k] for k in m.BALANCE_KEYS});native.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 return r,s,native,c

def log_fixture(project):
 from collections import Counter
 root=dict(m.rows(project))['BAL_AS30'];b=root/'source';b.mkdir(parents=True);scope=dict(assignment_namespace='scene_balance_assignment_v1',selected_packets_per_step=64,permutation_calls_per_step=1,physical_ID_uniqueness_required=False,sampler_changed=False,labels_or_RX_for_assignment=False,extra_NN=0,new_losses=0,new_U_views=0,formal_test_changed=False);cfg=dict(updates=2220,crossview=dict(mode='ORD128'),scene_balance=scope);put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=cfg))
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080,total_L_loss_scale=1.25,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,scene_balance=dict(mode='step21_21_22',selected_packets=142080,actual_As_packets=142080,assignment_permutations=2220,extra_scene_phase='(cumulative_step-1)%3',assignment_namespace='scene_balance_assignment_v1',pair_weights=[.5,.5,.25],SupCon='ORD128',total_L_scale=1.25,channel_calls=2220,extra_NN=0,new_losses=0,global_RNG_used=False,sampler_changed=False,physical_ID_uniqueness_required=False,scene_counts={k:47360 for k in m.SOURCE_SCENES}))
 put(b/'actual_native_args.json',dict(args={}));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [1,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def Lids(step):
  ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)]
  if step==1:ids[1]=list(ids[0])
  return ids
 refs=[];shapes=[];native=[];units=[];balance=[];coverage=Counter()
 for step in range(1,2221):
  r,s,n,c=loss_fixture(step,Lids(step))
  for t,rx,si,st,lk in zip(c['TX'],c['RX'],c['scene_indices'],c['state'],c['lock']):coverage[(t,rx,m.SOURCE_SCENES[si],st,lk)]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));balance.append(c)
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('scene_balance_steps',balance)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(1,2221):yield dict(role='L',ids=Lids(i))
  for ep in range(10):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return root

class SceneBalanceChecks(unittest.TestCase):
 def test_actual_full6L_loss_ORD128_and_quota_negatives(self):
  import copy
  r,s,n,c=loss_fixture();self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertEqual(m.check_scene_balance(r,c,n),512)
  self.assertNotIn('shape/RD',n);self.assertEqual(sum(k.startswith(('concat/','balance/')) for k in n),14)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
  for key,value in [('extra_scene_index',1),('planned_counts',[22,22,20]),('permutation',[0]*64),('scene_indices',[0]*64),('global_RNG_used',True),('physical_ID_uniqueness_required',True),('selected_L_positions',list(range(64)))]:
   wrong=copy.deepcopy(c);wrong[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_scene_balance(r,wrong,n)
  wrong=copy.deepcopy(r);wrong['SupCon_details'][0]['additional_anchors']=64
  with self.assertRaises(ValueError):m.check_scene_balance(wrong,c,n)
 def test_complete_new_tail_logs_allow_sampler_repeats_and_reject_prefix(self):
  with temporary() as tmp:
   p=Path(tmp);row=log_fixture(p);q,_=m.parse_new_logs(row,'BAL_AS30')
   self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['scene_balance']['scene_counts'],{k:47360 for k in m.SOURCE_SCENES});self.assertEqual(q['scene_balance']['sampler_repetitions']['batches_with_repeated_physical_ID'],1);self.assertEqual(sum(q['selected_exposure']['TX'].values()),142080)
   path=row/'source/scene_balance_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'BAL_AS30')
 def test_saved_original30_audit_reused_without_old_step_reads(self):
  with temporary() as tmp:
   p=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i,ordinary_raw_MSE=.01) for i in range(1,31)]
   put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in ('NOAS30','AS30')}));lookup=dict(m.rows(p));put(lookup['AS30']/'stage20/result.json',parent);put(lookup['AS30']/'source/actual_native_args.json',dict(args={}))
   newcost={k:oldcost[k]-parent[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};q=dict(cost=newcost,epoch_curves=curves[20:]);private=dict(args={},ordinary_mse=[.01]*2220)
   with mock.patch.object(m,'parse_new_logs',return_value=(q,private)),mock.patch.object(m.logs,'records',side_effect=AssertionError('Old logs reopened')):r=m.training_analysis(p)
   self.assertTrue(r['rows']['NOAS30']['reused']);self.assertEqual(r['new_updates'],2220);self.assertTrue(r['reference_comparability']['four_counters_equal']);self.assertFalse(r['reference_comparability']['old_step_logs_rescanned'])
 def test_two_source_conditions_use_different_frozen_references(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass']);q['BAL_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);q['AS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(q)['source_criteria_pass']);json.dumps(m.source_conditions(q),allow_nan=False)
  q['BAL_AS30']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed']);self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
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
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['BAL_AS30']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_full_formal_CM_partitions_errors_and_separate_scene_increment(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],63);self.assertEqual(q['independent_confusion_matrices'],378);self.assertEqual(q['paired']['BAL_AS30__NOAS30']['clean']['repairs'],1);self.assertEqual(q['paired']['BAL_AS30__NOAS30']['clean']['new_errors'],1);self.assertTrue(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['incremental_scene_effect_pp']['clean'],-100/84);self.assertNotIn('contrast_interaction_pp',q);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_main_parameterized_publisher_cost_and_single_archive_collection(self):
  import cvs_scenebalance30_publish_analysis_20261018 as pub
  import cvs_scenebalance30_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_scenebalance30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={'BAL_AS30':dict(cost=cost)},reference_comparability={'old_step_logs_rescanned':False})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=768,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False));argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertGreater(len((out/'final_report.md').read_text(encoding='utf8').splitlines()),30);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],3162962+12600+590+770)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11);self.assertIn('cvs_scenebalance30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
