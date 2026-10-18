"""Synthetic tests for added complete-three-seed aggregation and interfaces only."""
import csv,json,tempfile,sys
from pathlib import Path
from unittest import TestCase,main,mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_practical_budget50_replication_analysis_20261018 as m

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')
def metadata(project,n):
 for arm,root in m.rows(project):
  condition,seed=m.split_label(arm);fifty=condition.endswith('50');cfg=dict(model_seed=seed,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=20 if fifty else 10,optimizer_steps=4440 if fifty else 2220,target_access=False,config=cfg,cumulative_epochs=50,cumulative_optimizer_steps=11100,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,clean_accuracy=1.,source_accuracies=[[1.]*3]*2))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=n,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=n*7+2,wall_seconds=1.));put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def source_fixture(project):
 metadata(project,6);y=np.arange(1800)%6;rx=np.asarray((1,3,4,6,8))[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.stack([np.zeros(1800,int),rx,day,np.arange(1800),np.arange(1800)],1);clean=np.zeros((1800,6));clean[np.arange(1800),y]=2.;sat=np.broadcast_to(clean,(2,3,1800,6)).copy();summary={};views=('clean',*m.SOURCE_SCENES)
 for arm,root in m.rows(project):
  np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=clean,satellite=sat);summary[arm]=dict(full_V_clean=.98,full_V_RX_accuracy={str(r):.98 for r in (1,3,4,6,8)},panel={v:dict(accuracy=1.,macro_F1=1.,worst_RX=1.) for v in views},mid_low_mean=1.)
 ps={}
 for seed in m.SEEDS:
  pairs={name:{v:dict(repaired=0,new_errors=0,net=0,accuracy_pp=0.,count=1800 if v=='clean' else 3600,macro_F1_pp=0.,worst_RX_pp=0.) for v in views} for name in m.PAIR_NAMES}
  for p in pairs.values():p.update(fullV_clean_pp=0.,mid_low_mean_pp=0.)
  ps[str(seed)]=dict(paired=pairs,source_clean_reference_within_0p1pp=True,panel_accuracy_budget_interaction_pp={v:0. for v in views},fullV_clean_budget_interaction_pp=0.,mid_low_mean_budget_interaction_pp=0.)
 aggregate={name:{v:m.stats3([0.,0.,0.]) for v in views} for name in m.PAIR_NAMES};put(project/'runs'/m.RUN/'source_matrix_decision.json',dict(status='SOURCE_BUDGET_REPLICATION_READOUT_COMPLETE',target_scores_read=False,rows=summary,per_seed=ps,paired_accuracy_pp_three_seed=aggregate))
 return m.source_analysis(project)

def formal_fixture(project):
 metadata(project,6);ids=np.asarray(['opaque'+str(i) for i in range(6)]);y=np.arange(6);rx=np.asarray(['r0']*3+['r1']*3);day=np.asarray(['d0','d1']*3);put(project/'truth.json',{s:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,s in enumerate(ids)});summary={};pred={}
 for label,root in m.rows(project):
  condition,seed=m.split_label(label);p=y.copy()
  if condition=='NOAS30' or condition=='NOAS50' and seed!=m.SEEDS[0]:p[0]=1
  if condition=='AS50' and seed==m.SEEDS[0]:p[-1]=4
  if condition=='AS30' and seed==m.SEEDS[1]:p[1]=2
  pred[label]=p;np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   logits=np.zeros((6,6));logits[np.arange(6),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=logits)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(6,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records));q=m.metric(y,p);s=dict(accuracy=q['accuracy'],macro_F1=q['macro_f1'],worst_RX=min(float((p[rx==r]==y[rx==r]).mean()) for r in set(rx)));summary[label]={v:s for v in m.VIEWS};summary[label]['urban_mean']=dict(accuracy=s['accuracy'])
 ps={}
 for seed in m.SEEDS:
  pp={name:{v:100*(summary[a][v]['accuracy']-summary[b][v]['accuracy']) for v in (*m.VIEWS,'urban_mean')} for name,a,b in m.pairs_for(seed)};ps[str(seed)]=dict(paired_accuracy_pp=pp,budget_interaction_pp={v:pp['AS50_minus_NOAS50'][v]-pp['AS30_minus_NOAS30'][v] for v in (*m.VIEWS,'urban_mean')},stage_clean_0p1pp=pp['AS50_minus_NOAS50']['clean']>=-.1-1e-12)
 agg={name:{v:m.stats3([ps[str(s)]['paired_accuracy_pp'][name][v] for s in m.SEEDS]) for v in (*m.VIEWS,'urban_mean')} for name in m.PAIR_NAMES};inter={v:m.stats3([ps[str(s)]['budget_interaction_pp'][v] for s in m.SEEDS]) for v in (*m.VIEWS,'urban_mean')};put(project/'runs'/m.RUN/'automatic_summary.json',dict(summary=summary,per_seed=ps,paired_accuracy_pp_three_seed=agg,budget_interaction_pp_three_seed=inter))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False)

class ReplicationChecks(TestCase):
 def test_three_seed_complete_mean_and_ddof1(self):
  q=m.stats3([1.,2.,3.]);self.assertEqual(q['mean'],2.);self.assertEqual(q['sample_SD'],1.);self.assertEqual(q['values'],[1.,2.,3.])
  with self.assertRaises(ValueError):m.stats3([1.,2.])
  self.assertEqual(len(m.rows(Path('project'))),12)
 def test_last_logical_row_missing_truth_stays_closed(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);s=formal_fixture(p);last=dict(m.rows(p))[m.label('AS50',m.SEEDS[-1])]/'scoring/complete.json';last.unlink();original=m.read;touched=[]
   def read(path):
    if Path(path).name=='truth.json':touched.append(True);raise AssertionError('Truth opened early')
    return original(path)
   with mock.patch.object(m,'read',read),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',s)
   self.assertEqual(touched,[])
 def test_twelve_rows_CM_pairs_SD_and_actual_partitions(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);s=formal_fixture(p);q,_=m.formal_analysis(p,p/'truth.json',s);self.assertEqual(len(q['summary']),12);self.assertEqual(q['partition_reconciliations'],252);self.assertEqual(q['independent_confusion_matrices'],924)
   a=q['paired_accuracy_pp_three_seed']['AS50_minus_NOAS50']['clean'];self.assertAlmostEqual(a['mean'],100/18);self.assertAlmostEqual(a['sample_SD'],float(np.std([-100/6,100/6,100/6],ddof=1)));self.assertFalse(q['declaration']['fixed_three_seed_consistent_double_improvement'])
 def test_source_all_twelve_and_native_fullV_panel_separate(self):
  with tempfile.TemporaryDirectory() as tmp:
   q=source_fixture(Path(tmp));self.assertEqual(len(q['summary']),12);self.assertEqual(q['summary'][m.label('AS50',m.SEEDS[0])]['full_V_clean'],.98);self.assertEqual(q['summary'][m.label('AS50',m.SEEDS[0])]['panel']['clean']['accuracy'],1.);self.assertFalse(q['target_scores_read']);self.assertFalse(q['source_high_urban_available'])
 def test_absolute_integer_boundaries_and_fixed_claim(self):
  def pairs(nets):return {str(s)+'__AS50_minus_NOAS50':{v:dict(net_repairs=net if v=='clean' else 1) for v in ('clean','practical_mid_urban','practical_low_urban')} for s,net in zip(m.SEEDS,nets)}
  q=m.declaration_from_pairs(pairs([-168,-168,-168]),168000);self.assertTrue(q['mean_clean_within_0p1pp']);self.assertTrue(q['all_seeds_clean_within_0p1pp'])
  self.assertFalse(m.declaration_from_pairs(pairs([-169,-168,-168]),168000)['mean_clean_within_0p1pp'])
  self.assertTrue(m.declaration_from_pairs(pairs([168,168,168]),168000)['fixed_three_seed_consistent_double_improvement'])
  self.assertFalse(m.declaration_from_pairs(pairs([167,168,168]),168000)['fixed_three_seed_consistent_double_improvement'])
 def test_actual_new_parent_audit_schema_does_not_double_count_readout(self):
  counters=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900)
  a=dict(cost=dict(source_counters=counters,native_training_and_validation_identity_NN=7960684,new_updates=6660,source_wall_seconds=2.,L_identity_NN=5114880,frozen_source_readout_NN=12600,source_identity_NN=7973284),curve=[{}]*30,full_records=dict(actual_steps=6660));q=m.parent_audit_cost(a);self.assertEqual(q['native_training_and_validation_NN'],7960684);self.assertEqual(q['frozen_source_readout_NN_already_in_old_source_identity_NN'],12600)
  a['cost']['native_training_and_validation_identity_NN']+=12600
  with self.assertRaises(ValueError):m.parent_audit_cost(a)
 def test_publisher_dynamic_output_runs_actual_pipeline(self):
  import cvs_practical_budget50_replication_publish_analysis_20261018 as pub
  import cvs_practical_budget50_replication_collect_20261018 as collect
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);formal_source=formal_fixture(p);source=dict(formal_source,metrics=[],per_seed={str(s):dict(paired={'AS50_minus_NOAS50':dict(fullV_clean_pp=0.,clean=dict(accuracy_pp=0.),mid_low_mean_pp=0.)}) for s in m.SEEDS});native=dict(student_train_NN=1,student_eval_NN=2,teacher_train_NN=0,teacher_eval_NN=3,native_training_and_validation_NN=6,wall_seconds=1.);training=dict(rows={a:dict(cost=native) for a,_ in m.rows(p)})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=3072,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False));c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_practical_budget50_replication_analysis_20261018_r07',output='analysis_auto_r07');self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   out=Path(c['output']);q=m.read(out/'complete.json');self.assertEqual(q['status'],'ANALYZED');self.assertEqual(q['model_seed_count'],3);self.assertEqual(q['new_model_rows'],4);cost=m.read(out/'costs.json');self.assertEqual(cost['new_shared_smoke_NN'],3074)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)

if __name__=='__main__':main(verbosity=2)
