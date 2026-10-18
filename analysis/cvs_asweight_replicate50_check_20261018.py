"""Focused new-seed/stage/cost/24logical checks; no real data or model access."""
import gzip,json,tempfile,unittest,sys
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_asweight_replicate50_analysis_20261018 as m
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


def eligibility_fixture(project,passed=True):
 decision=dict(source_criteria_pass=passed,fixed_02_03_source_eligible=passed,clean_nonnegative=passed,mid_low_gain_at_least_0p5pp=passed,newAs50_minus_ownAs30_fullV_clean_pp=0.,newAs50_minus_ownAs30_mid_low_mean_pp=.6)
 put(project/'runs'/m.base.RUN/'analysis_auto_r01/source_summary.json',dict(status='SOURCE_ANALYZED',run_id=m.base.RUN,target_scores_read=False,NN=0,IQ_reads=0,checkpoint_reads=0,updates=0,decision=decision))

def metadata(project,count):
 eligibility_fixture(project)
 for arm,root in m.rows(project):
  condition,seed=m.split_label(arm);budget=int(condition[-2:]);E=10 if budget==30 else 20;new=seed in m.NEW_SEEDS and condition.startswith('WEIGHT_');cfg=dict(model_seed=seed,row_id=m.physical_row(condition,seed),run_id=m.RUN if new else root.parent.name)
  put(root/'source/completion.json',dict(epoch=E,optimizer_steps=E*222,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=budget,cumulative_optimizer_steps=budget*222,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.));put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def source_fixture(project):
 metadata(project,168000);y=np.arange(1800)%6;rx=np.asarray([1,3,4,6,8])[np.arange(1800)%5];day=np.arange(1800)%3+1;ids=np.stack([np.zeros(1800,int),rx,day,np.arange(1800),np.arange(1800)],1);z=np.zeros((1800,6));z[np.arange(1800),y]=2.;sat=np.broadcast_to(z,(2,3,1800,6)).copy()
 for _,root in m.rows(project):np.savez(root/'source_qualification/predictions.npz',ids=ids,TX=y,RX=rx,day=day,clean=z,satellite=sat)

def formal_fixture(project):
 metadata(project,6);n=6;ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n);rx=np.asarray(['r0']*3+['r1']*3);day=np.asarray(['d0','d1']*3);put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 for arm,root in m.rows(project):
  condition,seed=m.split_label(arm);p=y.copy()
  if condition=='NOAS30' or (condition=='WEIGHT_NOAS50' and seed!=m.SEEDS[-1]) or (condition=='WEIGHT_AS50' and seed==m.SEEDS[-1]):p[0]=1
  np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for view in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(view+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=view,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,eligibility=m.seed01_source_eligibility(project))

def loss_fixture(step=1,sat=True):
 As=64 if sat else 0;common=(.23,.2,.03);ordinary=(.34,.3,.04);selected=(.52,.5,.02);raw=(.45,.4,.05)
 r=dict(step=step,cumulative_step=6660+step,L_forwards=6,As_packets=As,selected_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,selected_actual_view='As_raw' if sat else 'raw',selected_ordinary_MSE=.02 if sat else .01,selected_raw_MSE=.1 if sat else 0.,scene_counts={'practical_low_urban':64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NONE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,ordinary_group_weight=.125,selected_group_weight=.375,SupCon_details=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')])
 s=dict(step=step,cumulative_step=6660+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None)
 for i,suffix in enumerate(('RD','CE','KL')):
  r['common_'+suffix]=common[i];r['ordinary64_'+suffix]=ordinary[i];r['selected64_'+suffix]=selected[i];r['extra_'+suffix]=.5*(ordinary[i]+selected[i]);r['mixed_weighted_'+suffix]=.125*ordinary[i]+.375*selected[i];s['shape_'+suffix]=raw[i]
 r['base_L_before_shape']=.5*r['common_RD']+r['mixed_weighted_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 m={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
 # Keep the emitted names identical to model.hook; no shape/RD is emitted.
 m.update({'weight/'+k:r[k] for k in ('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')});m.update({'weight/raw_'+suffix:s['shape_'+suffix] for suffix in ('RD','CE','KL')});m.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0})
 w={k:r[k] for k in ('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')};w.update(selected_actual_view=r['selected_actual_view'],selected_ordinary_MSE=r['selected_ordinary_MSE'],selected_raw_MSE=r['selected_raw_MSE'],step=step,cumulative_step=6660+step,As_packets=As,selected_packets=64,selected_positions=[j for j in range(128) if (j+step)%2!=0]);w.update({'unweighted_mixed_'+suffix:r['extra_'+suffix] for suffix in ('RD','CE','KL')})
 return r,s,m,w

def log_fixture(project,seed,budget,sat=True):
 from collections import Counter
 name=('WEIGHT_AS' if sat else 'WEIGHT_NOAS')+str(budget);root=dict(m.rows(project))[m.label(name,seed)];E=10 if budget==30 else 20;N=E*222;offset=4440 if budget==30 else 6660;b=root/'source';b.mkdir(parents=True);put(b/'completion.json',dict(epoch=E,optimizer_steps=N,config=dict(updates=N,model_seed=seed,target_budget=budget,seed_binding=dict(model_and_native_training=seed,F_factory_model_seed=seed))))
 result=dict(actual_F_factory_seed=seed,optimizer_steps=N,cumulative_epochs=budget,cumulative_optimizer_steps=offset+N,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=N*768,L_As=N*64 if sat else 0,student_train_NN=N*768+E*56700,student_eval_NN=E*27000+54002,teacher_train_NN=0,teacher_eval_NN=E*56700,wall_seconds=10.,mixed_weight=dict(ordinary=.125,selected=.375,primary_ordinary=.5,raw=.25,total_L_scale=1.25,selected_packets=N*64,actual_As_packets=N*64 if sat else 0,channel_calls=N if sat else 0,identity_control_calls=0 if sat else N,unchanged_IQ_and_forward_order=True))
 put(b/'actual_native_args.json',dict(args={}));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [0,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 refs=[];shapes=[];native=[];units=[];weights=[];coverage=Counter()
 for step in range(1,N+1):
  r,s,n,w=loss_fixture(step,sat)
  for q in (r,s,w):q['cumulative_step']=offset+step
  ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)];positions=w['selected_positions'];w['selected_TX']=[0]*64;w['selected_RX']=[ids[j][1] for j in positions];w['selected_day']=[ids[j][2] for j in positions]
  for t,x in zip(w['selected_TX'],w['selected_RX']):coverage[(t,x,'practical_low_urban' if sat else 'NONE','LOS' if sat else 'NONE','LOCK' if sat else 'NONE')]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=offset+step,physical_clip_calls=step));weights.append(w)
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('weight_steps',weights)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,E+1)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(N):yield dict(role='L',ids=[physical('L',(i*128+j)%6300) for j in range(128)])
  for ep in range(E):
   for i in range(222):yield dict(role='U',ids=[physical('U',i*256+j) for j in range(124 if i==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=project/'runs'/m.RUN/'logs'/(root.parent.name+'.log');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('All20 epochs complete\n',encoding='utf8');return root

def cost_fixture():
 c20=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.)
 c10=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.,local_updates=2220)
 c30=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,wall_seconds=2.,new_updates=6660)
 c50=dict(student_train_NN=4543920,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=1134000,native_training_and_validation_NN=6271922,wall_seconds=1.,local_updates=4440)
 return c20,c10,c30,c50

def training_fixture(project):
 c20,c10,c30,c50=cost_fixture();curve=lambda a,b:[dict(epoch=i) for i in range(a,b+1)];original={};single={};roots=dict(m.rows(project))
 for seed in m.SEEDS:
  for condition in ('NOAS30','AS30','NOAS50','AS50'):
   budget=int(condition[-2:]);original[m.label(condition,seed)]=dict(cost=c30 if budget==30 else c50,epoch_curves=curve(1,30) if budget==30 else curve(31,50))
  if seed in m.NEW_SEEDS:
   for condition in ('NOAS30','AS30'):put(roots[m.label(condition,seed)]/'stage20/result.json',c20)
 for condition in ('WEIGHT_NOAS30','WEIGHT_AS30','WEIGHT_NOAS50','WEIGHT_AS50'):
  budget=int(condition[-2:]);single[condition]=dict(cost=c10 if budget==30 else c50,cumulative_native_cost=c30 if budget==30 else m.base.cumulative_cost(c30,c50),epoch_curves=curve(21,30) if budget==30 else curve(31,50),complete_trajectory_curves=curve(1,budget))
 put(project/'runs'/m.ORIGINAL50_REPLICATION/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows=original));put(project/'runs'/m.base.RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows=single))
 def new(root,condition,stdout):
  budget=int(condition[-2:]);E=10 if budget==30 else 20;return dict(cost=c10 if budget==30 else c50,counts=dict(steps=E*222),epoch_curves=curve(21,30) if budget==30 else curve(31,50)),dict(args={})
 with mock.patch.object(m,'parse_segment',side_effect=new) as parser,mock.patch.object(m.logs,'records',side_effect=AssertionError('Old steps reopened')):value=m.training_analysis(project)
 return value,parser.call_count

class ReplicateChecks(unittest.TestCase):
 def test_exact_24_nested_stages_seed_binding(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);metadata(p,168000);items=dict(m.rows(p));self.assertEqual(len(items),24);q=items['WEIGHT_AS50-s2026092703'];self.assertEqual(q.parts[-2:],('B01-s2026092703','weighted50'));self.assertIn(m.ORIGINAL30_REPLICATION,str(items['AS30-s2026092702']));self.assertEqual(len(m.ensure_metadata(p,'source')),24)
   path=q/'source/completion.json';saved=m.read(path);saved['config']['model_seed']=2026092702;put(path,saved)
   with self.assertRaises(ValueError):m.ensure_metadata(p,'source')
 def test_conditional_source_pass_only_no_registration(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);eligibility_fixture(p,False)
   with self.assertRaises(ValueError):m.seed01_source_eligibility(p)
   eligibility_fixture(p,True);d=m.seed01_source_eligibility(p);self.assertFalse(d['target_scores_read']);json.dumps(d,allow_nan=False)
 def test_three_seed_source_never_reads_formal(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);source_fixture(p);original=m.read
   def guarded(path):
    if Path(path).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Source opened formal')
    return original(path)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(p)
   self.assertEqual(len(q['summary']),24);self.assertEqual(q['model_seed_count'],3);self.assertEqual(q['channel_draws'],2);self.assertTrue(q['channel_draws_are_not_model_seeds']);self.assertEqual(m.stats3([0,1,2])['sample_SD'],1.)
 def test_truth_closed_until_last_of24_all7_scores(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['WEIGHT_AS50-s2026092703']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_24_recount_paired_mean_SD_all_vs_mean_stage(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['independent_confusion_matrices'],1848);self.assertEqual(q['partition_reconciliations'],504);s=q['stage_summary']['50'];self.assertTrue(s['mean_clean_stage_within_0p1pp']);self.assertFalse(s['all_seeds_clean_stage_within_0p1pp']);self.assertFalse(s['fixed_three_seed_consistent_double_improvement_observation']);self.assertEqual(len(s['clean_paired_pp']['values']),3);self.assertTrue(q['summary']['WEIGHT_AS50-s2026092701']['reused']);self.assertFalse(q['summary']['WEIGHT_AS50-s2026092703']['reused'])
 def test_new_seed10_and20_complete_streams_offsets_and_truncation(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);a=log_fixture(p,2026092702,30,False);b=log_fixture(p,2026092703,50,True);stdout=lambda seed,condition:p/'runs'/m.RUN/'logs'/(m.physical_row(condition,seed)+'.log');q,_=m.parse_segment(a,'WEIGHT_NOAS30',stdout(2026092702,'WEIGHT_NOAS30'));r,_=m.parse_segment(b,'WEIGHT_AS50',stdout(2026092703,'WEIGHT_AS50'));self.assertEqual(q['cost']['cumulative_updates'],6660);self.assertEqual(r['cost']['cumulative_updates'],11100);self.assertEqual(q['counts']['As_packets'],0);self.assertEqual(r['counts']['As_packets'],284160);self.assertEqual(q['counts']['steps']+r['counts']['steps'],6660)
   factory=b/'source/result.json';saved=m.read(factory);saved['actual_F_factory_seed']=2026092702;put(factory,saved)
   with self.assertRaises(ValueError):m.parse_segment(b,'WEIGHT_AS50',stdout(2026092703,'WEIGHT_AS50'))
   saved['actual_F_factory_seed']=2026092703;put(factory,saved)
   path=b/'source/weight_steps.jsonl.gz';values=list(m.logs.records(path));jl(path,values[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_segment(b,'WEIGHT_AS50',stdout(2026092703,'WEIGHT_AS50'))
 def test_old_audits_reused_26640_new_steps_and_cost_not_double_parent(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);metadata(p,6);t,calls=training_fixture(p);self.assertEqual(calls,8);self.assertEqual(t['new_local_steps'],26640);self.assertEqual(t['rows']['WEIGHT_AS50-s2026092703']['cumulative_native_cost']['native_training_and_validation_NN'],14232606);self.assertEqual(len(t['rows']['WEIGHT_AS50-s2026092703']['complete_trajectory_curves']),50)
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=3072,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=8,target_query=False));c=m.cost_summary(p,t);self.assertEqual(c['new_total_NN'],4*(3162962+6271922+2*(12600+44))+3080);self.assertTrue(c['intermediate30_not_double_counted_as_parent']);self.assertEqual(len(c['groups']['new_weighted30_segments']['rows']),4)
 def test_pipeline_publisher_dynamic_output_and_collection(self):
  import cvs_asweight_replicate50_publish_analysis_20261018 as pub
  import cvs_asweight_replicate50_collect_20261018 as collector
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(m,'COUNT',6):
   p=Path(tmp);source=formal_fixture(p);source['metrics']=[];training,_=training_fixture(p);runtime=dict(project=str(p),python='python',user='ordinary',host='host');config=pub.analysis_config(runtime,'a'*40,release='cvs_F_asweight_replicate50_analysis_20261018_r07',output='analysis_auto_r07');self.assertEqual(Path(config['training_release']).name,m.TRAINING_RELEASE);out=Path(config['output'])
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=3072,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=8,target_query=False));argv=['analysis','--project',config['project'],'--release',config['training_release'],'--output',config['output'],'--truth',str(p/'truth.json'),'--code-commit',config['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['reused_model_rows'],16);archive=p/'aggregate.tar.gz';saved=collector.pack_public(out,archive,'final');r=collector.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(r['status'],'VERIFIED');self.assertEqual(r['file_count'],11)
 def test_waits_only_observable_live_handles_and_terminal_missing(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();calls=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(calls),probe=lambda p,r:([321],False),sleep=lambda _:calls.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Dead handle waited'))
if __name__=='__main__':unittest.main(verbosity=2)
