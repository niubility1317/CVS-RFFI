"""Focused synthetic LRaw E50 checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_lraw50_analysis_20261018 as m

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
  budget=int(arm[-2:]);epochs=20 if budget==50 else 10;updates=4440 if budget==50 else 2220;cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=epochs,optimizer_steps=updates,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=budget,cumulative_optimizer_steps=11100 if budget==50 else 6660,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.))
  put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def source_fixture(project):
 metadata(project,168000);y=np.arange(1800)%6;rx=np.asarray([1,3,4,6,8])[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.stack([np.zeros(1800,int),rx,day,np.arange(1800),np.arange(1800)],1);clean=np.zeros((1800,6));clean[np.arange(1800),y]=2.;sat=np.broadcast_to(clean,(2,3,1800,6)).copy()
 for _,root in m.rows(project):np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=clean,satellite=sat)

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)])
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;preds={'NOAS30':control,'AS30':y.copy(),'LRAW_NOAS30':y.copy(),'LRAW_AS30':candidate,'NOAS50':y.copy(),'AS50':y.copy(),'LRAW_NOAS50':control,'LRAW_AS50':y.copy()}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS}
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary=summary,decision=m.source_conditions(summary))

def loss_fixture(step=1,sat=True,ids=None):
 As=64 if sat else 0;common=(.23,.2,.03);mixed=(.43,.4,.03);raw=(.45,.4,.05)
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=6660+step,L_forwards=6,As_packets=As,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=0.,scene_counts={'practical_low_urban':64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NONE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details)
 s=dict(step=step,cumulative_step=6660+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):r['common_'+suffix]=common[i];r['extra_'+suffix]=mixed[i];s['shape_'+suffix]=raw[i]
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD'];discard=.01 if step%2 else 0.
 a=dict(step=step,cumulative_step=6660+step,discardedordinary_MSE=discard,discardedordinary_changed_packets=128 if discard else 0,returned_raw_MSE=0.,full_native_draws=1,count=128,input_clone=True,original_input_unchanged=True,full_native_aug_computation=True)
 c=dict(step=step,cumulative_step=6660+step,epoch=(step-1)//222+1,cumulative_epoch=30+(step-1)//222+1,ordinary_calls=1,ordinary_packets=128,discardedordinary_MSE=discard,discardedordinary_changed_packets=a['discardedordinary_changed_packets'],returned_raw_MSE=0.,full_native_draws=1,extra_NN=0,actual_ORD128_anchor_input='raw128 returned in ordinary slot',same_IQ_each_RDpair=True)
 native={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};native.update({'lraw/'+k:c[k] for k in m.LRAW_KEYS});native.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 return r,s,native,c,a

def log_fixture(project,sat=True):
 from collections import Counter
 name='LRAW_AS50' if sat else 'LRAW_NOAS50';root=dict(m.rows(project))[name];b=root/'source';b.mkdir(parents=True)
 policy=dict(ordinary_calls_per_step=1,ordinary_packets_per_step=128,actual_ordinary_raw_MSE=0.,first20_ordinary_retained=True,full_native_aug_computation=True,RD_same_generated_IQ=True,U_weak_strong_unchanged=True,teacher_unchanged=True,EMA_unchanged=True,whole_trajectory_NoAug=False,whole_training_RNG_matched=False,U_augmentor_wrapped=False,new_losses=0,new_parameters=0,new_NN=0,formal_test_changed=False)
 cfg=dict(updates=4440,crossview=dict(mode='ORD128'),lraw=policy);put(b/'completion.json',dict(epoch=20,optimizer_steps=4440,config=cfg))
 aggregate=dict(mode='ordinary_draw_discard_return_raw',ordinary_factory_calls=1,ordinary_calls=4440,ordinary_packets=568320,full_native_draws=4440,returned_raw_MSE_max=0.,used_ordinary_raw_MSE_max=0.,discardedordinary_MSE_mean=.005,discardedordinary_MSE_max=.01,discardedordinary_changed_packets=2220*128,U_augmentor_wrapped=False,U_weak_strong_unchanged=True,parent_ordinary_epochs=20,whole_trajectory_NoAug=False,full_native_aug_computation=True,whole_training_RNG_matched=False,pair_weights=[.5,.5,.25],SupCon='ORD128;actual raw128 input',total_L_scale=1.25,extra_NN=0,new_losses=0)
 result=dict(optimizer_steps=4440,parent_epochs=30,parent_optimizer_steps=6660,cumulative_epochs=50,cumulative_optimizer_steps=11100,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=3409920,L_As=284160 if sat else 0,total_L_loss_scale=1.25,student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,wall_seconds=10.,U_steps=4440,U_packets=1134000,U_student_NN=1134000,EMA_local_successful_updates=4440,EMA_cumulative_updates=11100,parent_EMA_restored=True,lraw=aggregate)
 put(b/'actual_native_args.json',dict(args=dict(use_aug=True,use_unlabeled=True,use_ema_teacher=True)));put(b/'EMA_initialization.json',dict(origin='strict own LRaw FINAL30 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
 put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True,use_aug=True,full_native_aug_computation=True,returned_input='pre-call raw clone',ordinary_calls=4440,ordinary_packets=568320,U_augmentor_wrapped=False,whole_training_RNG_matched=False))
 def physical(role,i):return [1,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def Lids(step):
  ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)]
  if step==1:ids[1]=list(ids[0])
  return ids
 refs=[];shapes=[];native=[];units=[];lraw=[];draws=[];coverage=Counter()
 for step in range(1,4441):
  ids=Lids(step);r,s,n,c,a=loss_fixture(step,sat,ids)
  for j,z in enumerate(ids):
   if (j+step)%2!=0:coverage[(j%6,z[1],'practical_low_urban' if sat else 'NONE','LOS' if sat else 'NONE','LOCK' if sat else 'NONE')]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=6660+step,physical_clip_calls=step));lraw.append(c);draws.append(a)
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('lraw_steps',lraw),('ordinary_draw_steps',draws)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,21)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(1,4441):yield dict(role='L',ids=Lids(i))
  for ep in range(20):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All20 epochs complete\n',encoding='utf8');return root

class LRaw50Checks(unittest.TestCase):
 def test_same_LRaw_formula_and_raw_return_scope(self):
  r,s,n,c,a=loss_fixture();self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertEqual(m.check_lraw(r,c,a,n),512);self.assertNotIn('shape/RD',n)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
  wrong=dict(c);wrong['returned_raw_MSE']=.1
  with self.assertRaises(ValueError):m.check_lraw(r,wrong,a,n)
 def test_complete_20_epoch_streams_parent_offsets_and_actual_counters(self):
  with temporary() as tmp:
   p=Path(tmp);a=log_fixture(p,True);b=log_fixture(p,False);x,_=m.parse_new_logs(a,'LRAW_AS50');z,_=m.parse_new_logs(b,'LRAW_NOAS50')
   self.assertEqual(x['counts']['steps'],4440);self.assertEqual(x['counts']['As_packets'],284160);self.assertEqual(z['counts']['As_packets'],0);self.assertEqual(x['cost']['native_training_and_validation_NN'],6271922);self.assertEqual(x['physical_loader_packets']['U'],1134000);self.assertEqual(x['EMA']['cumulative_updates'],11100);self.assertEqual([q['epoch'] for q in x['epoch_curves']],list(range(31,51)))
   path=a/'source/ordinary_draw_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(a,'LRAW_AS50')
 def test_source_is_fixed_budget_readout_with_no_new_gate(self):
  q={a:dict(full_V_clean=np.float64(.98),mid_low_mean=np.float64(.6)) for a in m.CONDITIONS};d=m.source_conditions(q);self.assertFalse(d['new_scientific_gate']);self.assertNotIn('source_criteria_pass',d);self.assertFalse(d['automatic_extension']);self.assertTrue(d['E50_already_authorized_from_saved_E30_source']);json.dumps(d,allow_nan=False)
  q['LRAW_AS50']['full_V_clean']=.1;q['LRAW_AS50']['mid_low_mean']=.1;d=m.source_conditions(q);self.assertNotIn('fixed_recipe_closed',d);self.assertFalse(d['scientific_negative_skips_formal'])
 def test_eight_condition_budget_and_method_interactions(self):
  q=dict(zip(m.CONDITIONS,[.1,.2,.3,.5,.15,.35,.5,.9]));f=m.factorial_effects(q);self.assertAlmostEqual(f['two_factor_interactions_pp']['LRaw_x_As_at_E30'],10.);self.assertAlmostEqual(f['two_factor_interactions_pp']['LRaw_x_As_at_E50'],20.);self.assertAlmostEqual(f['three_factor_interaction_pp'],10.);self.assertAlmostEqual(f['paired_accuracy_pp']['LRAW_AS50__LRAW_AS30'],40.)
 def test_source_never_opens_formal_truth(self):
  with temporary() as tmp:
   p=Path(tmp);source_fixture(p);read=m.read
   def guarded(path):
    if Path(path).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal access in source')
    return read(path)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(p)
   self.assertEqual(q['NN'],0);self.assertFalse(q['decision']['new_scientific_gate'])
 def test_all8x7_before_truth_and_full_formal_pairs(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);source=formal_fixture(p);row=dict(m.rows(p))['LRAW_AS50'];missing=row/'scoring/complete.json';value=m.read(missing);missing.unlink();read=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return read(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertFalse(touch);put(missing,value);q=m.formal_analysis(p,p/'truth.json',source);self.assertEqual(q['independent_confusion_matrices'],1008);self.assertEqual(q['partition_reconciliations'],168);self.assertEqual(q['paired']['LRAW_AS50__LRAW_NOAS50']['clean']['repairs'],1);self.assertTrue(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['factorial']['clean']['three_factor_interaction_pp'],300/84)
 def test_terminal_process_handles_no_infinite_wait(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir()
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on terminal handle'))
 def test_reuse_old_audits_and_complete_trajectory_without_old_step_scan(self):
  with temporary() as tmp:
   p=Path(tmp);base=p/'runs';roots=dict(m.rows(p));c20=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);c10=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,local_updates=2220,wall_seconds=1.);c30=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_updates=6660,wall_seconds=2.);c50=dict(student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_training_and_validation_NN=6271922,local_updates=4440,wall_seconds=1.)
   curve=lambda a,b:[dict(epoch=e) for e in range(a,b+1)];prior=dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=c30,epoch_curves=curve(1,30)) for a in ('NOAS30','AS30')},parent20={a:dict(cost=c20,epochs=20,updates=4440,wall_seconds=1.) for a in ('LRAW_NOAS30','LRAW_AS30')});prior['rows'].update({a:dict(cost=c10,epoch_curves=curve(21,30)) for a in ('LRAW_NOAS30','LRAW_AS30')})
   put(base/m.LRAW30_RUN/'analysis_auto_r01/training_logs.json',prior);put(base/m.ORIGINAL50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=c50,epoch_curves=curve(31,50)) for a in ('NOAS50','AS50')}))
   for name in ('NOAS50','AS50'):put(roots[name]/'source/actual_native_args.json',dict(args={}))
   def new(*args):return dict(cost=dict(c50),epoch_curves=curve(31,50)),dict(args={})
   with mock.patch.object(m,'parse_new_logs',side_effect=new) as parser,mock.patch.object(m.logs,'records',side_effect=AssertionError('Old step rescan')):value=m.training_analysis(p)
   self.assertEqual(parser.call_count,2);self.assertEqual(value['new_updates'],8880);self.assertEqual(value['rows']['LRAW_AS50']['cumulative_native_cost']['native_training_and_validation_NN'],14232606);self.assertEqual([q['epoch'] for q in value['rows']['LRAW_AS50']['complete_trajectory_curves']],list(range(1,51)));self.assertTrue(value['rows']['AS50']['reused'])
 def test_main_publisher_output_and_saved_collection(self):
  import cvs_lraw50_publish_analysis_20261018 as pub
  import cvs_lraw50_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_lraw50_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[]
   c30=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,wall_seconds=1.)
   c10=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.)
   c50=dict(student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_training_and_validation_NN=6271922,wall_seconds=1.)
   training=dict(rows={a:dict(cost=c30 if a in ('NOAS30','AS30') else c10 if a.endswith('30') else c50,cumulative_native_cost=c30 if a.endswith('30') else m.cumulative_cost(c30,c50)) for a in m.CONDITIONS},parent30={a:dict(cumulative_native_cost=c30) for a in m.CONDITIONS[-2:]})

   for a in m.CONDITIONS[-2:]:training['rows'][a].update(lraw=dict(ordinary_calls=4440,ordinary_packets=568320,discardedordinary_MSE_mean=.005,discardedordinary_MSE_max=.01,used_ordinary_raw_MSE_max=0.),EMA=dict(local_updates=4440,cumulative_updates=11100),physical_loader_packets=dict(U=1134000))
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=1536,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=4,target_query=False))
   argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertEqual(m.read(out/'costs.json')['new_total_NN'],2*(6271922+12600+590)+1540)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11)
   self.assertIn('cvs_lraw50_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
