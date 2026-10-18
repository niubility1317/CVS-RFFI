"""Focused synthetic stepALT checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_stepalt30_analysis_20261018 as m

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
 preds={'NOAS30_ORIGINAL':np.asarray([1,1,2,3,4,5]),'AS30_ORIGINAL':y.copy(),'AS30_STEPALT':np.asarray([0,1,2,3,4,4])}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,decision=dict(source_criteria_pass=False,fixed_recipe_closed=True,fullV_clean_vs_NoAs30_pp=0.,mid_low_mean_vs_original_As30_pp=0.))

def log_fixture(project):
 root=dict(m.rows(project))['AS30_STEPALT'];b=root/'source';b.mkdir(parents=True);put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=dict(updates=2220)))
 put(b/'result.json',dict(optimizer_steps=2220,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.))
 put(b/'actual_native_args.json',dict(args={}));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [0,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 refs=[];shapes=[];steps=[];units=[];exposures=[]
 for i in range(1,2221):
  ep=(i-1)//222+1;on=i%2==1;As=128 if on else 0;before=.5*.23+.5*.34+.1*.05;total=before+.25*.45;ids=[physical('L',((i-1)*128+j)%6300) for j in range(128)]
  refs.append(dict(step=i,cumulative_step=4440+i,L_forwards=6,As_packets=As,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.34,extra_CE=.3,extra_KL=.04,SupCon_weight=.1,SupCon=.05,total_L=total,base_L_before_shape=before,ordinary_raw_MSE=.01,mixed_ordinary_MSE=.1 if on else 0.,scene_counts={'practical_low_urban':128} if on else {},state_counts={'LOS':128} if on else {},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,stepALT=dict(cumulative_epoch=20+ep,cumulative_step=4440+i,on=on,As_packets=As,L_NN=768,L_loss_scale=1.25),SupCon_details=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]))
  shapes.append(dict(step=i,cumulative_step=4440+i,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05))
  steps.append(dict(step=i,epoch=ep,metrics={'concat/L_forwards':6,'concat/As_packets':As,'concat/total_L':before,'shape/RD':.45,'train/loss':total+.1,'train/pseudo_total':124 if i%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0}));units.append(dict(step=i,cumulative_step=4440+i,physical_clip_calls=i));exposures.append(dict(step=i,physical_ids=ids,TX=[0]*128,RX=[z[1] for z in ids],day=[z[2] for z in ids],on=on,As_packets=As,scenes=['practical_low_urban']*As,sample_ids=list(range(As))))
 from collections import Counter
 cells=Counter()
 for e in exposures:
  if e['on']:
   for t,r,d,s in zip(e['TX'],e['RX'],e['day'],e['scenes']):cells[(t,r,d,s)]+=1
 result=m.read(b/'result.json');result['stepALT']=dict(As_packets=142080,on_steps=1110,off_steps=1110,channel_calls=1110,strata=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state='LOS',lock='LOCK',count=v) for k,v in cells.items()]);put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',steps),('unit_steps',units),('stepalt_exposure',exposures)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)]
 csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(2220):yield dict(role='L',ids=[physical('L',(i*128+j)%6300) for j in range(128)])
  for ep in range(10):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=root.parent/'logs'/(root.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return root

class StepALTChecks(unittest.TestCase):
 def test_source_fixed_scientific_conditions(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass'])
  q['AS30_STEPALT']=dict(full_V_clean=.979,mid_low_mean=.6001);self.assertTrue(m.source_conditions(q)['source_criteria_pass'])
  q['AS30_STEPALT']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed'])
  self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
 def test_source_does_not_open_formal_truth(self):
  with tempfile.TemporaryDirectory() as tmp:
   project=Path(tmp);source_fixture(project);original=m.read
   def guarded(p):
    if Path(p).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal access in source')
    return original(p)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(project)
   self.assertEqual(q['NN'],0);self.assertEqual(q['source_unique_V'],1800);self.assertFalse(q['decision']['source_criteria_pass']);self.assertFalse(q['decision']['scientific_negative_skips_formal'])
 def test_truth_closed_until_all3_sevenview_rows_complete(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['AS30_STEPALT']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_full_formal_CM_F1_partition_and_pairs(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],63);self.assertEqual(q['independent_confusion_matrices'],231)
   self.assertEqual(q['paired']['AS30_STEPALT__NOAS30_ORIGINAL']['clean']['repairs'],1);self.assertEqual(q['paired']['AS30_STEPALT__NOAS30_ORIGINAL']['clean']['new_errors'],1);self.assertTrue(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[]
   m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_complete_new10_logs_and_actual_exposure_missing_last(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=log_fixture(Path(tmp));q,_=m.parse_new_logs(root);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['counts']['L_NN'],1704960);self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['step_alternation']['off_steps'],1110)
   self.assertEqual(sum(q['satellite_exposure']['scenes'].values()),142080);self.assertEqual([x['epoch'] for x in q['epoch_curves']],list(range(21,31)))
   p=root/'source/stepalt_exposure.jsonl.gz';r=list(m.logs.records(p));jl(p,r[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(root)
 def test_main_publisher_output_and_saved_collection(self):
  import cvs_stepalt30_publish_analysis_20261018 as pub
  import cvs_stepalt30_collect_20261018 as collect
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_stepalt30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.)
   training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20=dict(cost=cost),physical_and_ordinary_comparison_to_original_As30_tail={'different':True})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=1536,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False))
   argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertEqual(m.read(out/'costs.json')['new_total_NN'],3162962+12600+44+1538)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11)
   self.assertIn('cvs_stepalt30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
