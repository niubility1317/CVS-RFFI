"""Focused saved-metadata fixtures; no real model/CP/IQ or current results."""
import ast,csv,gzip,json,os,sys,tempfile,unittest,copy
from collections import Counter,defaultdict
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_u_qualified30_analysis_20261019 as m

def temporary():return tempfile.TemporaryDirectory(dir=os.environ.get('CVS_ANALYSIS_TEST_TMP'))
def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf-8')
def jl(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'wt',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
def csvjson(root,name,rows):
 (root/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
 with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def u_fixture(step=1,arm='UQUALIFIED30',k=None):
 n=124 if step%222==0 else 256;j=(step-1)%222;domain=j%15;rx=[1,3,4,6,8][domain//3];day=domain%3+1
 if k is None:k=(0,2,6)[step%3]
 qualified=list(range(k));positions=qualified if arm=='UQUALIFIED30' else list(range(n-k,n));selected=set(positions);elig=[i<k for i in range(n)];gaussian=[i for i in range(n) if i not in selected];ids=[[1,rx,day,100000+j*256+i,100000+j*256+i] for i in range(n)]
 z=np.zeros((n,6));z[:,1]=1.;lp=z-np.log(np.exp(z).sum(1,keepdims=True));ce=float(-lp[:k,0].mean()) if k else 0.;ent=float(-(np.exp(lp)*np.log(np.maximum(np.exp(lp),1e-8))).sum(1).mean());mode='QUALIFIED_PRACTICAL' if arm=='UQUALIFIED30' else 'COUNT_RANDOM'
 physics=[dict(selected_position=i,scene=m.SOURCE_SCENES[i%3],sample_id=ids[i],session_id='synthetic',namespace='synthetic',state=('LOS','PARTIAL','BLOCKED')[i%3],lock=('locked','degraded','unlocked')[i%3],signal_to_added_noise_db=(1.,5.,10.,20.)[i%4],quality_snr_db=10.) for i in positions]
 q=dict(mode=mode,routing_rule=mode,step=step,epoch=j//222+(step-1)//222+1,cumulative_step=4440+step,cumulative_epoch=21+(step-1)//222,U_packets=n,selected_positions=positions,Gaussian_positions=gaussian,selected_slots=k,As_packets=k,qualification_count=k,qualification_positions=qualified,same_future_Ascount_claim=False,selection_uses_current_teacher_qualification=True,selection_uses_TX_or_pseudo_class=False,sample_id_fields=['eq_i','rx_i','day_i','sig_i','base_index'],sample_ids=ids,RX=[rx]*n,day=[day]*n,eq=[1]*n,U_truth_available=False,ordinary_before_practical=False,original_strong_draw_packets=n,original_strong_std=.015,original_strong_before_overwrite=True,C_returns_original_strong_object=not k,teacher_qualification_before_student_view=True,student_passes=1,extra_NN=0,extra_CE_operator_calls=0,physics=physics,scene_counts=dict(Counter(v['scene'] for v in physics)),global_RNG_used_for_routing=False,selection_uses_TX_or_pseudo=False,selection_without_replacement=True,teacher_conf=[.98 if i<k else .5 for i in range(n)],teacher_pseudo=[0]*n,domains=[domain]*n,conf_mask=elig.copy(),temporal_mask=elig.copy(),eligibility=elig.copy(),threshold_policy=dict(mode='rx_day_quantile',quantile=.86,tau_min=.92,tau_max=.97),temporal_policy=dict(mode='batch_neighbor',window=2,minimum_confidence=.8),domain_gate=False,strong_agreement=False,teacher_raw_same_input=True,teacher_eval_no_grad=True,teacher_qualification_completed_before_strong=True,teacher_NN=n,student_NN=n,student_logits=z.tolist(),native_loss_u=ce,native_loss_entropy=ent,lambda_u=.16,lambda_entropy=.01,CE_denominator=k,entropy_denominator=n,eligible_selected_slots=sum(elig[i] for i in positions),eligible_Gaussian_slots=sum(elig[i] for i in gaussian),eligible_As_slots=sum(elig[i] for i in positions),CE_label_smoothing=0.,native_global_L_label_smoothing=.01,entropy_all_U=True,entropy_includes_unqualified=True,entropy_includes_As=bool(k),U_truth_used=False,extra_teacher=False,native_losses_preserved=True,native_feature_adjustment=None,native_optimizer_step=step,native_hook='model._source_unlabeled_feature_hook')
 q.update({key:.0001 for key in m.U_TIMERS})
 if not k:q['simulation_adapter_cpu_wall_seconds']=q['replacement_copy_cpu_wall_seconds']=0.
 return q

def L_fixture(step):
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts={'synthetic':64},state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.27,extra_CE=.25,extra_KL=.02)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05);r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD'];return r,s

def full_fixture(project,arm):
 row=dict(m.rows(project))[arm];b=row/'source';b.mkdir(parents=True);mode='QUALIFIED_PRACTICAL' if arm=='UQUALIFIED30' else 'COUNT_RANDOM'
 p=dict(mode=mode,slot_fraction=None,U_packets=567000,U_As_packets=None,selected_slots=None,Gaussian_std=.015,lambda_unlabeled=.16,lambda_unlabeled_entropy=.01,pseudo_strong_agreement=False,pseudo_domain_gate=False,U_truth_used=False,extra_NN=0,new_losses=0,L_changed=False,formal_test_changed=False,ordinary_before_practical=False,teacher_trajectory_equality_claim=False,original_strong_draw_all_packets=True,teacher_only_eligibility=True,pseudo_temporal_gate=True,teacher_quantile_unchanged=True,batch_neighbor_unchanged=True,teacher_qualification_before_student_view=True,pseudo_temporal_mode='batch_neighbor')
 cfg=dict(updates=2220,reference_row_id='B01-s2026092701',crossview=dict(mode='ORD128'),As=True,u_practical=p)
 put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=cfg));put(b/'EMA_initialization.json',dict(origin='strict own stage20 EMA',reference_row_id='B01-s2026092701',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True,U_DataAugmentation_used=False));put(b/'actual_native_args.json',dict(args=dict(label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True,pseudo_strong_agreement=False,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor')))
 rs=[];ss=[];ns=[];us=[];units=[];coverage=defaultdict(Counter);scenes=Counter()
 for i in range(1,2221):
  r,s=L_fixture(i);q=u_fixture(i,arm);rs.append(r);ss.append(s);us.append(q);metrics={'concat/'+key:r[key] for key in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};metrics.update({'train/loss':r['total_L']+.16*q['native_loss_u']+.01*q['native_loss_entropy'],'train/pseudo_total':q['U_packets'],'train/pseudo_selected':q['CE_denominator'],'train/pseudo_truth_available':0});ns.append(dict(step=i,epoch=(i-1)//222+1,metrics=metrics));units.append(dict(step=i,cumulative_step=4440+i,physical_clip_calls=i));scenes.update(q['scene_counts']);selected=set(q['selected_positions'])
  for j in range(q['U_packets']):coverage[(q['RX'][j],q['day'][j],'As' if j in selected else 'Gaussian')].update(count=1,qualified=int(q['eligibility'][j]))
 total=sum(q['As_packets'] for q in us);qual=sum(q['CE_denominator'] for q in us);asqual=sum(q['eligible_As_slots'] for q in us);calls=sum(bool(q['As_packets']) for q in us)
 actual=dict(mode=mode,calls=2220,U_packets=567000,selected_slots=total,As_packets=total,qualification_count=qual,same_future_Ascount_claim=False,channel=dict(calls=calls,packets=total),qualified_packets=qual,qualified_As_packets=asqual,qualified_Gaussian_packets=qual-asqual,original_strong_draw_packets=567000,teacher_only_eligibility=True,pseudo_strong_agreement=False,extra_NN=0,extra_CE_operator_calls=0,U_truth_used=False,teacher_trajectory_equality_claim=False,lambda_u=.16,lambda_entropy=.01,RX_day_route_exposure=[dict(RX=k[0],day=k[1],route=k[2],count=v['count'],qualified=v['qualified']) for k,v in sorted(coverage.items())],scene_counts=dict(scenes),timing={key:sum(q[key] for q in us) for key in m.U_TIMERS},timing_note='synthetic hostwall;inclusive route')
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,Ag_packets=0,L_NN=1704960,L_As=142080,U_As=total,total_L_loss_scale=1.25,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,U_steps=2220,U_packets=567000,U_student_NN=567000,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,u_practical=actual);put(b/'result.json',result)
 for name,records in [('reference_steps',rs),('shape_steps',ss),('step_metrics',ns),('unit_steps',units),('u_steps',us)]:jl(b/(name+'.jsonl.gz'),records)
 erows=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',erows);csvjson(b,'epoch_metrics',erows)
 def seq():
  for i,q in enumerate(us,1):
   indices=((i-1)*128+np.arange(128))%6300;yield dict(role='L',ids=[[1,[1,3,4,6,8][int(k)%5],int(k//5)%3+1,int(k),int(k)] for k in indices]);yield dict(role='U',ids=q['sample_ids'])
 jl(b/'training_sequence.jsonl.gz',seq());log=row.parent/'logs'/(row.name+'.log');log.parent.mkdir(exist_ok=True);log.write_text('All10 complete\n',encoding='utf-8');return row

def metadata(project,count):
 for arm,root in m.rows(project):
  cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=30,cumulative_optimizer_steps=6660,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={'1':.98})));put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.]*3]*2));put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.));put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)]);put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 for j,(arm,root) in enumerate(m.rows(project)):
  p=y.copy();p[j]=(p[j]+1)%6;np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{key:q[key] for key in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 summary={arm:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for arm in m.CONDITIONS}
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary=summary,decision=m.source_conditions(summary))

def training_fixture():
 c=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,local_updates=2220,cumulative_updates=6660,wall_seconds=1.)
 u=dict(qualified_packets=0,qualified_As_packets=0,qualified_Gaussian_packets=0,U_As=0,As_channel_calls=0,zero_As_steps=2220,timing={key:0. for key in m.U_TIMERS})
 return dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=m.RUN,target_scores_read=False,rows={arm:dict(status='VERIFIED_COMPLETE_SOURCE_LOGS',cost=c,counts=dict(steps=2220,U_As=0),u_practical=u,epoch_curves=[dict(epoch=i) for i in range(21,31)]) for arm in m.CONDITIONS},parent20={arm:dict(cost=c) for arm in m.CONDITIONS},comparator_parent20={arm:dict(cost=c) for arm in m.CONDITIONS[:2]},unique_parent20_costs={'original_AS20':c})

def smoke_fixture(root,project):
 records=[dict(row_id=p.name,reference_row_id='B01-s2026092701',L_NN=768,U_teacher_NN=8,U_student_NN=8,NN=784,optimizer_updates=0) for arm,p in m.rows(project) if arm in m.CONDITIONS[2:]]
 put(root/'synthetic_smoke.json',dict(status='PASS',records=records,NN=1568,real_checkpoint_NN=2,total_NN=1570,optimizer_updates=0,real_IQ=0,target_access=False));put(root/'real_checkpoint_smoke.json',dict(status='PASS',records=[dict(reference_row_id='B01-s2026092701',NN=2)],NN=2,optimizer_updates=0,real_IQ=0,target_query=False))

class QualifiedChecks(unittest.TestCase):
 def test_dynamic_route_qualification_and_native_loss(self):
  for arm in m.CONDITIONS[2:]:
   for step,k in ((1,0),(1,2),(222,6),(222,124)):
    q=u_fixture(step,arm,k);v=m.check_u_step(q,arm);self.assertEqual(q['As_packets'],k);self.assertEqual(v['eligibility'].sum(),k);self.assertEqual(q['eligible_As_slots'],k if arm=='UQUALIFIED30' else max(0,2*k-q['U_packets']))
  base=u_fixture(1,'UQUALIFIED30',2)
  for key,value in (('qualification_count',3),('qualification_positions',[2,3]),('selected_positions',[2,3]),('native_losses_preserved',False),('same_future_Ascount_claim',True),('extra_NN',1),('native_loss_entropy',0.)):
   q=copy.deepcopy(base);q[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_u_step(q,'UQUALIFIED30')
  q=u_fixture(1,'URANDOM30',0);q['simulation_adapter_cpu_wall_seconds']=1.
  with self.assertRaises(ValueError):m.check_u_step(q,'URANDOM30')
 def test_two_complete_dynamic2220step_streams(self):
  with temporary() as tmp:
   p=Path(tmp)
   for arm in m.CONDITIONS[2:]:
    row=full_fixture(p,arm);r,_=m.parse_new_logs(row,arm);self.assertEqual(r['counts']['steps'],2220);self.assertEqual(r['counts']['U_As'],5920);self.assertEqual(r['u_practical']['As_channel_calls'],1480);self.assertEqual(r['u_practical']['zero_As_steps'],740);self.assertEqual(r['cost']['native_training_and_validation_NN'],3162962)
   jl(row/'source/u_steps.jsonl.gz',[u_fixture(1)])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'UQUALIFIED30')
 def test_reuse_N_C_audits_and_one_As20_parent(self):
  with temporary() as tmp:
   p=Path(tmp);t=training_fixture();t['run_id']=m.COMPARATOR_RUN;put(p/'runs'/m.COMPARATOR_RUN/'analysis_auto_r01/training_logs.json',t);put(p/'runs'/m.REFERENCE_RUN/'B01-s2026092701/stage20/result.json',dict(t['rows']['UTEACHER30']['cost']))
   with mock.patch.object(m,'parse_new_logs',return_value=(copy.deepcopy(t['rows']['UTEACHER30']),{})) as parser:r=m.training_analysis(p)
   self.assertEqual(parser.call_count,2);self.assertEqual(r['new_updates'],4440);self.assertEqual(r['shared_parent_count'],1);self.assertTrue(r['rows']['MATCHED_NOAS30']['reused']);self.assertEqual(len(r['unique_parent20_costs']),1)
 def test_only_two_source_conditions(self):
  s={a:dict(full_V_clean=.98,mid_low_mean=.5) for a in m.CONDITIONS};s['UQUALIFIED30']=dict(full_V_clean=.979,mid_low_mean=.505);d=m.source_conditions(s);self.assertTrue(d['source_criteria_pass']);self.assertEqual(len(d['checks']),2);json.dumps(d,allow_nan=False);s['URANDOM30']=dict(full_V_clean=.999,mid_low_mean=.999);self.assertTrue(m.source_conditions(s)['source_criteria_pass']);s['MATCHED_NOAS30']['full_V_clean']=.981;self.assertFalse(m.source_conditions(s)['source_criteria_pass'])
 def test_all4_all7_truthlast_504CM84partitions(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);s=formal_fixture(p);path=dict(m.rows(p))['URANDOM30']/'prediction/complete.json';saved=path.read_bytes();path.unlink()
   with self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',s)
   path.write_bytes(saved);f=m.formal_analysis(p,p/'truth.json',s);self.assertEqual(f['independent_confusion_matrices'],504);self.assertEqual(f['partition_reconciliations'],84);self.assertEqual(len(f['paired']),6);self.assertEqual(f['stage']['clean_delta_pp'],f['paired']['UQUALIFIED30__MATCHED_NOAS30']['clean']['accuracy_delta_pp']);self.assertNotIn('interaction',f)
 def test_two_phases_and_actual1570_smoke_no_old_failed770(self):
  import cvs_u_qualified30_publish_analysis_20261019 as pub
  import cvs_u_qualified30_collect_20261019 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);source=formal_fixture(p);source['metrics']=[];training=training_fixture();root=p/'runs'/m.RUN;smoke_fixture(root,p);out=root/'analysis_auto_r01';argv=['analysis','--project',str(p),'--release',str(p/m.TRAINING_RELEASE),'--output',str(out),'--code-commit','a'*40]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source),mock.patch.object(m,'formal_analysis') as never:m.main();never.assert_not_called()
   before={name:(out/name).read_bytes() for name in collect.SOURCE_FILES};self.assertFalse((out/'formal_summary.json').exists());self.assertNotIn('truth',pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40))
   with mock.patch.object(sys,'argv',argv+['--phase','final','--truth',str(p/'truth.json')]),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',side_effect=AssertionError('Old source audit must be reused')):m.main()
   costs=m.read(out/'costs.json');self.assertEqual(costs['new_total_NN'],2*(3162962+12600+590)+1570);self.assertEqual(costs['comparator_technical_failure_NN_recharged'],0);self.assertNotIn('technical_failed_NN',costs);self.assertEqual(before,{name:(out/name).read_bytes() for name in collect.SOURCE_FILES});archive=p/'public.tar.gz';packed=collect.pack_public(out,archive,'final');self.assertEqual(collect.extract_public(archive,p/'collected','final',packed['archive_SHA256'])['file_count'],11)
 def test_terminal_missing_and_AST(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir()
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda x:False,probe=lambda *a:([],False),sleep=lambda _:None)
  for suffix in ('analysis','check','publish_analysis','collect'):
   path=Path(__file__).parent/('cvs_u_qualified30_'+suffix+'_20261019.py');ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))

if __name__=='__main__':unittest.main(verbosity=2)
