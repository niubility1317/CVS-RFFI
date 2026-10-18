"""Focused synthetic AsWeight checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_asweight30_analysis_20261018 as m

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
 n=6;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n);rx=np.asarray(['r0']*3+['r1']*3);day=np.asarray(['d0','d1']*3)
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 preds={'NOAS30':np.asarray([1,1,2,3,4,5]),'AS30':y.copy(),'WEIGHT_NOAS30':y.copy(),'WEIGHT_AS30':np.asarray([0,1,2,3,4,4])}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,E50_source_eligible=False,fixed_recipe_closed=True,fullV_clean_vs_original_As30_pp=0.,mid_low_mean_vs_original_As30_pp=0.,fullV_clean_interaction_pp=0.,mid_low_mean_interaction_pp=0.))


def loss_fixture(step=1,sat=True):
 As=64 if sat else 0;common=(.23,.2,.03);ordinary=(.34,.3,.04);selected=(.52,.5,.02);raw=(.45,.4,.05)
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=As,selected_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,selected_actual_view='As_raw' if sat else 'raw',selected_ordinary_MSE=.02 if sat else .01,selected_raw_MSE=.1 if sat else 0.,scene_counts={'practical_low_urban':64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NONE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,ordinary_group_weight=.125,selected_group_weight=.375,SupCon_details=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')])
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):
  r['common_'+suffix]=common[i];r['ordinary64_'+suffix]=ordinary[i];r['selected64_'+suffix]=selected[i];r['extra_'+suffix]=.5*(ordinary[i]+selected[i]);r['mixed_weighted_'+suffix]=.125*ordinary[i]+.375*selected[i];s['shape_'+suffix]=raw[i]
 r['base_L_before_shape']=.5*r['common_RD']+r['mixed_weighted_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 m={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
 # Keep the emitted names identical to model.hook; no shape/RD is emitted.
 m.update({'weight/'+k:r[k] for k in ('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')});m.update({'weight/raw_'+suffix:s['shape_'+suffix] for suffix in ('RD','CE','KL')});m.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 w={k:r[k] for k in ('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')};w.update(selected_actual_view=r['selected_actual_view'],selected_ordinary_MSE=r['selected_ordinary_MSE'],selected_raw_MSE=r['selected_raw_MSE'],step=step,cumulative_step=4440+step,As_packets=As,selected_packets=64,selected_positions=[j for j in range(128) if (j+step)%2!=0]);w.update({'unweighted_mixed_'+suffix:r['extra_'+suffix] for suffix in ('RD','CE','KL')})
 return r,s,m,w

def log_fixture(project,sat=True):
 from collections import Counter
 name='WEIGHT_AS30' if sat else 'WEIGHT_NOAS30';root=dict(m.rows(project))[name];b=root/'source';b.mkdir(parents=True);put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=dict(updates=2220)))
 result=dict(optimizer_steps=2220,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080 if sat else 0,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,mixed_weight=dict(ordinary=.125,selected=.375,primary_ordinary=.5,raw=.25,total_L_scale=1.25,selected_packets=142080,actual_As_packets=142080 if sat else 0,channel_calls=2220 if sat else 0,identity_control_calls=0 if sat else 2220,unchanged_IQ_and_forward_order=True))
 put(b/'actual_native_args.json',dict(args={}));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [0,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 refs=[];shapes=[];native=[];units=[];weights=[];coverage=Counter()
 for step in range(1,2221):
  r,s,n,w=loss_fixture(step,sat);ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)];positions=w['selected_positions'];w['selected_TX']=[0]*64;w['selected_RX']=[ids[j][1] for j in positions];w['selected_day']=[ids[j][2] for j in positions]
  for t,x in zip(w['selected_TX'],w['selected_RX']):coverage[(t,x,'practical_low_urban' if sat else 'NONE','LOS' if sat else 'NONE','LOCK' if sat else 'NONE')]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));weights.append(w)
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('weight_steps',weights)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(2220):yield dict(role='L',ids=[physical('L',(i*128+j)%6300) for j in range(128)])
  for ep in range(10):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return root

class AsWeightChecks(unittest.TestCase):
 def test_actual_full6L_weighted_means_and_negative_wrong_semantics(self):
  r,s,n,w=loss_fixture();self.assertLess(m.check_weighted_L(r,s,n,w),3e-5)
  self.assertNotIn('shape/RD',n);self.assertEqual(n['concat/L_forwards'],6)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_weighted_L(r,s,wrong,w)
  wrong=dict(r);wrong['mixed_weighted_RD']=.5*r['extra_RD']
  with self.assertRaises(ValueError):m.check_weighted_L(wrong,s,n,w)
  wrong=dict(w);wrong['mixed_weighted_KL']+=.01
  with self.assertRaises(ValueError):m.check_weighted_L(r,s,n,wrong)
 def test_complete_actual_As_and_NoAs_logs_and_missing_last(self):
  with tempfile.TemporaryDirectory() as tmp:
   project=Path(tmp);a=log_fixture(project,True);b=log_fixture(project,False);asrow,_=m.parse_new_logs(a,'WEIGHT_AS30');norow,_=m.parse_new_logs(b,'WEIGHT_NOAS30')
   self.assertEqual(asrow['counts']['As_packets'],142080);self.assertEqual(norow['counts']['As_packets'],0);self.assertEqual(norow['counts']['selected_packets'],142080);self.assertEqual(asrow['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(norow['cost']['native_training_and_validation_NN'],3162962)
   self.assertTrue(norow['selected_exposure']['scene_counts_are_identity_control_not_As']);self.assertEqual(sum(asrow['selected_exposure']['TX'].values()),142080)
   q=m.compare_old_tail(a,a);self.assertTrue(q['physical_sequence']['all_equal']);self.assertEqual(q['ordinary_MSE_records_different'],0)
   path=a/'source/weight_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(a,'WEIGHT_AS30')
 def test_source_fixed_scientific_conditions(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass'])
  q['WEIGHT_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass'])
  q['WEIGHT_NOAS30']=dict(full_V_clean=.1,mid_low_mean=.01);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);json.dumps(m.source_conditions(q),allow_nan=False)
  q['WEIGHT_AS30']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed'])
  self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
 def test_source_does_not_open_formal_truth(self):
  with tempfile.TemporaryDirectory() as tmp:
   project=Path(tmp);source_fixture(project);original=m.read
   def guarded(p):
    if Path(p).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal access in source')
    return original(p)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(project)
   self.assertEqual(q['NN'],0);self.assertEqual(q['source_unique_V'],1800);self.assertFalse(q['decision']['source_criteria_pass']);self.assertFalse(q['decision']['scientific_negative_skips_formal'])
 def test_truth_closed_until_all4_sevenview_rows_complete(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['WEIGHT_AS30']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_full_formal_CM_F1_partition_and_pairs(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],84);self.assertEqual(q['independent_confusion_matrices'],308)
   self.assertEqual(q['paired']['WEIGHT_AS30__NOAS30']['clean']['repairs'],1);self.assertEqual(q['paired']['WEIGHT_AS30__NOAS30']['clean']['new_errors'],1);self.assertFalse(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertTrue(q['clean_vs_original_NoAs']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['weight_interaction_pp']['clean'],-100/3);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[]
   m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_main_publisher_output_and_saved_collection(self):
  import cvs_asweight30_publish_analysis_20261018 as pub
  import cvs_asweight30_collect_20261018 as collect
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_asweight30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.)
   training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={a:dict(cost=cost) for a in m.CONDITIONS[2:]},physical_and_ordinary_comparison_to_original_tails={'different':True})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=1536,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=4,target_query=False))
   argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertEqual(m.read(out/'costs.json')['new_total_NN'],2*(3162962+12600+44)+1540)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11)
   self.assertIn('cvs_asweight30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
