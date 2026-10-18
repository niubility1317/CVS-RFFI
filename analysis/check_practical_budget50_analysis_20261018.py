"""Targeted synthetic checks; no real source/target/IQ/checkpoint access."""
import csv,gzip,json,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_analysis_20261018 as m

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')
def jl(path,values):
 with gzip.open(path,'wt',encoding='utf8') if path.suffix=='.gz' else path.open('w',encoding='utf8') as f:
  for q in values:f.write(json.dumps(q,allow_nan=False)+'\n')
def csvjson(root,name,values):
 jl(root/(name+'.jsonl'),values);fields=list(dict.fromkeys(k for r in values for k in r))
 with (root/(name+'.csv')).open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)

def metadata(project,count):
 for arm,root in m.rows(project):
  new=arm.endswith('50');cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=m.RUN if new else m.REFERENCE_RUN)
  done=dict(epoch=20 if new else 10,optimizer_steps=4440 if new else 2220,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=50,cumulative_optimizer_steps=11100,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)}))
  put(root/'source/completion.json',done);put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg))
  put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.))
  put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)))
  put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def formal_fixture(project):
 n=6;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n);rx=np.asarray(['r0']*3+['r1']*3);day=np.asarray(['d0','d1']*3)
 truth={s:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,s in enumerate(ids)};put(project/'truth.json',truth)
 preds={'NOAS30':np.asarray([1,1,2,3,4,5]),'AS30':y.copy(),'NOAS50':y.copy(),'AS50':np.asarray([0,1,2,3,4,4])};scores={}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   logits=np.zeros((n,6));logits[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=logits)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records));m0=m.metric(y,p);stats=dict(accuracy=m0['accuracy'],macro_F1=m0['macro_f1'],worst_RX=min((p[rx==r]==y[rx==r]).mean() for r in set(rx)));scores[arm]={v:stats for v in m.VIEWS};scores[arm]['urban_mean']=dict(accuracy=stats['accuracy'])
 pairs={a+'_minus_'+b:{v:100*(scores[a][v]['accuracy']-scores[b][v]['accuracy']) for v in (*m.VIEWS,'urban_mean')} for a,b in m.PAIRS}
 inter={v:pairs['AS50_minus_NOAS50'][v]-pairs['AS30_minus_NOAS30'][v] for v in (*m.VIEWS,'urban_mean')}
 put(project/'runs'/m.RUN/'automatic_summary.json',dict(summary=scores,paired_accuracy_pp=pairs,budget_interaction_pp=inter,stage_clean_0p1pp=False))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,decision={'source_reference':True})

def new_log_fixture(project):
 root=project/'runs'/m.RUN/'B01-s2026092701';b=root/'source';b.mkdir(parents=True);cfg=dict(updates=4440)
 put(b/'completion.json',dict(epoch=20,optimizer_steps=4440,config=cfg));result=dict(optimizer_steps=4440,cumulative_epochs=50,cumulative_optimizer_steps=11100,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=3409920,L_As=284160,student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_NN=6271922,wall_seconds=10.,teacher_counter_offset={'train':0,'eval':2})
 put(b/'result.json',result);put(b/'actual_native_args.json',dict(args={'run_id':m.RUN}));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 refs=[];shapes=[];steps=[];units=[]
 for i in range(1,4441):
  ep=(i-1)//222+1;before=.5*(.2+.03)+.5*(.3+.04)+.1*.05;total=before+.25*(.4+.05)
  refs.append(dict(step=i,cumulative_step=6660+i,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.34,extra_CE=.3,extra_KL=.04,SupCon_weight=.1,SupCon=.05,total_L=total,base_L_before_shape=before,ordinary_raw_MSE=.01,scene_counts={'practical_low_urban':64},state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]))
  shapes.append(dict(step=i,cumulative_step=6660+i,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05))
  steps.append(dict(step=i,epoch=ep,metrics={'concat/L_forwards':4,'concat/As_packets':64,'concat/total_L':before,'shape/RD':.45,'train/loss':total+.1,'train/pseudo_total':124 if i%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0}));units.append(dict(step=i,cumulative_step=6660+i,physical_clip_calls=i))
 for n,v in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',steps),('unit_steps',units)]:jl(b/(n+'.jsonl.gz'),v)
 epoch=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2,**({'early':0} if i<10 else {'late':0})) for i in range(1,21)]
 csvjson(b,'metrics_epoch',epoch);csvjson(b,'epoch_metrics',epoch)
 def ids(role,i):return [0,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def sequence():
  for i in range(4440):yield dict(role='L',ids=[ids('L',(i*128+j)%6300) for j in range(128)])
  for ep in range(20):
   for i in range(222):yield dict(role='U',ids=[ids('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());log=root.parent/'logs'/(root.name+'.log');log.parent.mkdir(exist_ok=True);log.write_text('SMOKE PASS\nAll20epochs\n',encoding='utf8')
 return root

class BudgetChecks(unittest.TestCase):
 def test_absolute_stage_boundary(self):
  self.assertTrue(m.stable.stage_clean_pass(-168,168000));self.assertFalse(m.stable.stage_clean_pass(-169,168000));self.assertFalse(m.stable.stage_clean_pass(-212,168000))
 def test_source_arrays_and_native_fullV_without_truth(self):
  with tempfile.TemporaryDirectory() as tmp:
   project=Path(tmp);metadata(project,168000);y=np.arange(1800)%6;rx=np.asarray((1,3,4,6,8))[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.stack([np.zeros(1800,int),rx,day,np.arange(1800),np.arange(1800)],1)
   clean=np.zeros((1800,6));clean[np.arange(1800),y]=2.;sat=np.broadcast_to(clean,(2,3,1800,6)).copy();rows={};paired={}
   for arm,root in m.rows(project):
    np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=clean,satellite=sat)
    panel={v:dict(accuracy=1.,macro_F1=1.,worst_RX=1.) for v in ('clean',*m.SOURCE_SCENES)};rows[arm]=dict(full_V_clean=.98,full_V_RX_accuracy={str(r):.98 for r in (1,3,4,6,8)},panel=panel,mid_low_mean=1.)
   for a,b in m.PAIRS:paired[a+'_minus_'+b]={v:dict(repaired=0,new_errors=0,net=0,accuracy_pp=0.,count=1800 if v=='clean' else 3600,macro_F1_pp=0.,worst_RX_pp=0.) for v in ('clean',*m.SOURCE_SCENES)}
   put(project/'runs'/m.RUN/'source_matrix_decision.json',dict(status='SOURCE_BUDGET_READOUT_COMPLETE',target_scores_read=False,rows=rows,paired=paired,fullV_As50_minus_NoAs50_clean_pp=0.,source_clean_reference_within_0p1pp=True,fullV_clean_budget_interaction_pp=0.,panel_accuracy_budget_interaction_pp={v:0. for v in ('clean',*m.SOURCE_SCENES)},mid_low_mean_budget_interaction_pp=0.))
   original=m.read
   def source_only(p):
    if Path(p).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Source opened formal material')
    return original(p)
   with mock.patch.object(m,'read',source_only):value=m.source_analysis(project)
   self.assertEqual(value['source_unique_V'],1800);self.assertFalse(value['source_high_urban_available']);self.assertFalse(value['target_scores_read']);self.assertEqual(value['NN'],0)
 def test_actual_counter_schema(self):
  x=dict(student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_NN=6271922)
  self.assertEqual(m.actual_counter_cost(x)['native_training_and_validation_NN'],6271922)
  x['student_eval_NN']+=1
  with self.assertRaises(ValueError):m.actual_counter_cost(x)
 def test_metric_and_paired_errors(self):
  y=np.arange(6);p=y.copy();p[0]=1;m0=m.metric(y,p);self.assertEqual(m0['correct'],5);self.assertEqual(sum(map(sum,m0['confusion'])),6)
  q=m.paired(y,p,y,np.ones(6,bool));self.assertEqual((q['repairs'],q['new_errors'],q['net_repairs']),(1,0,1))
 def test_final_truth_closed_until_last_row(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   project=Path(tmp);s=formal_fixture(project);last=dict(m.rows(project))['AS50']/'scoring/complete.json';last.unlink();original=m.read;touch=[]
   def guarded(p):
    if Path(p).name=='truth.json':touch.append(True);raise AssertionError('Truth touched early')
    return original(p)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(project,project/'truth.json',s)
   self.assertEqual(touch,[])
 def test_full_formal_CM_F1_partitions_pairs_interaction(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   project=Path(tmp);s=formal_fixture(project);q,_=m.formal_analysis(project,project/'truth.json',s)
   self.assertEqual(q['independent_recount'],'VERIFIED');self.assertEqual(q['partition_reconciliations'],84)
   self.assertEqual(q['paired']['AS50__NOAS50']['clean']['new_errors'],1);self.assertAlmostEqual(q['budget_interaction_pp']['clean'],-100/3)
   self.assertFalse(q['stage']['50']['clean_absolute_0p1pp_guard_met']);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_real_handles_or_terminal(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();calls=[]
   def ready(_):return bool(calls)
   def sleep(_):calls.append(True)
   m.wait_until(p,p/'release','SOURCE',out,readiness=ready,probe=lambda p,r:([3],False),sleep=sleep)
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('dead handle waited'))
 def test_dynamic_output_and_publisher_integration(self):
  import cvs_practical_budget50_publish_analysis_20261018 as pub
  import cvs_practical_budget50_collect_20261018 as collect
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'runs'/m.RUN/'analysis_auto_r07';self.assertEqual(m.validate_output(p,out),out)
   rt=dict(project=str(p),python='python',user='ordinary',host='host');c=pub.analysis_config(rt,'a'*40,release='cvs_F_practical_budget50_analysis_20261018_r07',output='analysis_auto_r07')
   self.assertEqual(Path(c['output']),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   self.assertIn('cvs_practical_budget50_analysis_20261018.py',pub.FILES)
   # Drive main with exact publisher-derived output/release and synthetic
   # completed artifacts; actual formal recount, cost, render and exports run.
   with mock.patch.object(m,'COUNT',6):
    source=formal_fixture(p);source.update(metrics=[],decision=dict(fullV_As50_minus_NoAs50_clean_pp=0.,mid_low_mean_budget_interaction_pp=0.))
    cost30=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,wall_seconds=1.)
    cost50=dict(student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_training_and_validation_NN=6271922,wall_seconds=1.)
    training=dict(rows={a:dict(cost=cost50 if a.endswith('50') else cost30) for a,_ in m.rows(p)})
    put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=1536,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False))
    argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
    with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
    complete=m.read(out/'complete.json');self.assertEqual(complete['status'],'ANALYZED');self.assertTrue(complete['formal_complete']);self.assertEqual(complete['NN'],0)
    total=m.read(out/'costs.json');self.assertEqual(total['new_segment_and_readouts_NN'],2*(6271922+12600+44));self.assertEqual(total['shared_smoke_NN'],1538)
    self.assertTrue((out/'final_report.md').is_file())
    for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
 def test_complete_new_local20_log_and_missing_last(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=new_log_fixture(Path(tmp));q,_=m.parse_new_logs(root,'AS50');self.assertEqual(q['counts']['steps'],4440);self.assertEqual(q['cost']['native_training_and_validation_NN'],6271922);self.assertEqual([r['epoch'] for r in q['epoch_curves']],list(range(31,51)))
   path=root/'source/unit_steps.jsonl.gz';values=list(m.logs.records(path));jl(path,values[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(root,'AS50')

if __name__=='__main__':unittest.main(verbosity=2)
