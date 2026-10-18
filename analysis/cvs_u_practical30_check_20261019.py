"""Focused metadata-only checks for staged U practical analysis;0NN/CP/IQ."""
import ast,csv,gzip,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_u_practical30_analysis_20261019 as m


def temporary():
 root=Path(os.environ.get('CVS_ANALYSIS_TEST_TMP',tempfile.gettempdir()));root.mkdir(parents=True,exist_ok=True)
 return tempfile.TemporaryDirectory(dir=root)

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')


def technical_history_fixture(root):
 value=dict(status='FAILED_PREFLIGHT',run_id=m.FAILED_RUN,code_commit='17b5939f440e370dacb6e92b6b523940d726115b',NN=770,optimizer_updates=0,real_IQ=0,propagation_simulations=0,worker_count=0,preserved=True,error='missing module trusted hook;actual model featurehook used inR02')
 put(root/'technical_history/r01/cost.json',value);return value

def metadata(project,count):
 for arm,root in m.rows(project):
  cfg=dict(model_seed=m.SEED,row_id=root.name,run_id=root.parent.name)
  put(root/'source/completion.json',dict(epoch=10,optimizer_steps=2220,target_access=False,head_fit_updates=0,config=cfg,cumulative_epochs=30,cumulative_optimizer_steps=6660,final_source_metrics=dict(source_val_count=27000,source_val_accuracy=.98,source_val_rx_accuracy={str(r):.98 for r in (1,3,4,6,8)})))
  put(root/'source/source_frozen.json',dict(selection='final_only',target_access=False,config=cfg));put(root/'source_qualification/complete.json',dict(target_access=False,NN=12600,training_updates=0,head_fit_updates=0,clean_accuracy=1.,source_accuracies=[[1.,1.,1.],[1.,1.,1.]]))
  put(root/'prediction/complete.json',dict(status='PREDICTIONS_COMPLETE',views=list(m.VIEWS),count=count,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=count*7+2,wall_seconds=1.))
  put(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',views=list(m.VIEWS)));put(root/'complete.json',dict(status='TRAINED_AND_SCORED'))

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)])
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;iid=y.copy();iid[1]=0;preds={'NOAS30':control,'AS30':y.copy(),'MATCHED_NOAS30':control.copy(),'UTEACHER30':iid,'UPRACTICAL30':candidate}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},quality_groups={a:{v:{g:dict(count=1800,accuracy=.6,macro_F1=.58,mean_trueclass_margin_over30=.01) for g in ('hard_lt8','easy_ge8')} for v in m.SOURCE_SCENES} for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,M3_source_eligible=False,fixed_stage1_closed=True,hard_lt8_S_minus_I_equal_scene_accuracy_pp=0.,easy_ge8_S_minus_I_equal_scene_accuracy_pp=0.,fullV_clean_S_minus_originalAs_pp=0.))


def jl(path,values):
 path.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(path,'wt',encoding='utf-8') as f:
  for value in values:f.write(json.dumps(value,allow_nan=False)+'\n')
def csvjson(root,name,rows):
 jlpath=root/(name+'.jsonl');jlpath.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
 with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def u_step_fixture(step=1,arm='UPRACTICAL30'):
 n=124 if step%222==0 else 256;j=(step-1)%222;domain=j%15;rx=[1,3,4,6,8][domain//3];day=domain%3+1;ids=[[1,rx,day,100000+j*256+i,100000+j*256+i] for i in range(n)];sat=arm=='UPRACTICAL30';pos=list(range(n//4));gaussian=list(range(n//4,n)) if sat else list(range(n));z=np.zeros((n,6));z[:,1]=1.;shift=z-z.max(1,keepdims=True);lp=shift-np.log(np.exp(shift).sum(1,keepdims=True));ce=float(-lp[:,0].mean());ent=float(-(np.exp(lp)*np.log(np.maximum(np.exp(lp),1e-8))).sum(1).mean());physics=[dict(selected_position=i,scene=m.SOURCE_SCENES[i%3],sample_id=ids[i],session_id='synthetic',namespace='synthetic',state=['LOS','PARTIAL','BLOCKED'][i%3],lock=['locked','degraded','unlocked'][i%3],signal_to_added_noise_db=[1.,5.,10.,20.][i%4],quality_snr_db=10.) for i in pos] if sat else []
 q=dict(mode='QUARTER_PRACTICAL' if sat else 'GAUSSIAN_CONTROL',step=step,epoch=(step-1)//222+1,cumulative_step=4440+step,cumulative_epoch=21+(step-1)//222,U_packets=n,selected_positions=pos,Gaussian_positions=gaussian,selected_slots=n//4,As_packets=n//4 if sat else 0,sample_id_fields=['eq_i','rx_i','day_i','sig_i','base_index'],sample_ids=ids,RX=[rx]*n,day=[day]*n,eq=[1]*n,duplicate_PID_occurrences=0,U_truth_available=False,raw_to_practical=True,ordinary_before_practical=False,original_strong_draw_packets=n,original_strong_std=.015,original_strong_before_overwrite=True,C_returns_original_strong_object=not sat,teacher_qualification_before_student_view=True,student_passes=1,extra_NN=0,extra_CE_operator_calls=0,physics=physics,scene_counts=dict(__import__('collections').Counter(p['scene'] for p in physics)),slot_namespace='FcrossRX/UPractical/slots/v1',slot_private_seed=step,slot_augmentation_seed=20261016421,global_RNG_used_for_routing=False,selection_uses_TX_or_pseudo=False,selection_without_replacement=True,native_epoch=(step-1)//222+1,native_batch_idx=j+1,teacher_conf=[.98]*n,teacher_pseudo=[0]*n,domains=[domain]*n,conf_mask=[True]*n,temporal_mask=[True]*n,eligibility=[True]*n,threshold_policy=dict(mode='rx_day_quantile',quantile=.86,tau_min=.92,tau_max=.97),temporal_policy=dict(mode='batch_neighbor',window=2,minimum_confidence=.8),domain_gate=False,strong_agreement=False,teacher_raw_same_input=True,teacher_eval_no_grad=True,teacher_qualification_completed_before_strong=True,teacher_NN=n,student_NN=n,student_logits=z.tolist(),native_loss_u=ce,native_loss_entropy=ent,lambda_u=.16,lambda_entropy=.01,CE_denominator=n,entropy_denominator=n,eligible_selected_slots=n//4,eligible_Gaussian_slots=len(gaussian),eligible_As_slots=n//4 if sat else 0,CE_label_smoothing=0.,native_global_L_label_smoothing=.01,entropy_all_U=True,entropy_includes_unqualified=True,entropy_includes_As=sat,U_truth_used=False,extra_teacher=False,native_losses_preserved=True,native_feature_adjustment=None,native_optimizer_step=step,native_hook='model._source_unlabeled_feature_hook',eligibility_source='actual frozen native base_mask_and_ones with no selectionhook',CPU_timing='synthetic CPU host only')
 q.update({k:.0001 for k in m.U_TIMERS});return q

def L_step_fixture(step,as_l=True):
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64 if as_l else 0,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts={'synthetic':64},state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.27,extra_CE=.25,extra_KL=.02)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05);r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD'];return r,s

def full_log_fixture(project,arm):
 from collections import Counter,defaultdict
 row=dict(m.rows(project))[arm];b=row/'source';b.mkdir(parents=True);as_l=arm!='MATCHED_NOAS30';sat=arm=='UPRACTICAL30';ref='B01-s2026092701' if as_l else 'B00-s2026092701';mode='QUARTER_PRACTICAL' if sat else 'GAUSSIAN_CONTROL';p=dict(mode=mode,slot_fraction=.25,U_packets=567000,U_As_packets=141750 if sat else 0,selected_slots=141750,Gaussian_std=.015,lambda_unlabeled=.16,lambda_unlabeled_entropy=.01,pseudo_strong_agreement=False,pseudo_domain_gate=False,U_truth_used=False,extra_NN=0,new_losses=0,L_changed=False,formal_test_changed=False,ordinary_before_practical=False,teacher_trajectory_equality_claim=False,original_strong_draw_all_packets=True,teacher_only_eligibility=True,pseudo_temporal_gate=True,teacher_quantile_unchanged=True,batch_neighbor_unchanged=True,teacher_qualification_before_student_view=True,pseudo_temporal_mode='batch_neighbor');cfg=dict(updates=2220,reference_row_id=ref,crossview=dict(mode='ORD128'),As=as_l,u_practical=p)
 put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=cfg));put(b/'EMA_initialization.json',dict(origin='strict own stage20 EMA',reference_row_id=ref,extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True,U_DataAugmentation_used=False));put(b/'actual_native_args.json',dict(args=dict(label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True,pseudo_strong_agreement=False,pseudo_domain_gate=False,pseudo_temporal_gate=True,pseudo_temporal_mode='batch_neighbor')))
 rs=[];ss=[];ns=[];us=[];units=[];coverage=defaultdict(Counter);scenes=Counter()
 for i in range(1,2221):
  r,s=L_step_fixture(i,as_l);q=u_step_fixture(i,arm);rs.append(r);ss.append(s);us.append(q);metrics={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};metrics.update({'train/loss':r['total_L']+.16*q['native_loss_u']+.01*q['native_loss_entropy'],'train/pseudo_total':q['U_packets'],'train/pseudo_selected':q['CE_denominator'],'train/pseudo_truth_available':0});ns.append(dict(step=i,epoch=(i-1)//222+1,metrics=metrics));units.append(dict(step=i,cumulative_step=4440+i,physical_clip_calls=i));scenes.update(q['scene_counts'])
  selected=set(q['selected_positions'])
  for j in range(q['U_packets']):coverage[(q['RX'][j],q['day'][j],'As' if sat and j in selected else 'Gaussian')].update(count=1,qualified=1)
 actual=dict(mode=mode,calls=2220,U_packets=567000,selected_slots=141750,As_packets=141750 if sat else 0,qualified_packets=567000,qualified_As_packets=141750 if sat else 0,qualified_Gaussian_packets=425250 if sat else 567000,original_strong_draw_packets=567000,teacher_only_eligibility=True,pseudo_strong_agreement=False,extra_NN=0,extra_CE_operator_calls=0,U_truth_used=False,teacher_trajectory_equality_claim=False,lambda_u=.16,lambda_entropy=.01,RX_day_route_exposure=[dict(RX=k[0],day=k[1],route=k[2],count=v['count'],qualified=v['qualified']) for k,v in sorted(coverage.items())],scene_counts=dict(scenes),timing={k:sum(q[k] for q in us) for k in m.U_TIMERS},timing_note='synthetic route inclusive;not CUDAevent')
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,Ag_packets=0,L_NN=1704960,L_As=142080 if as_l else 0,U_As=141750 if sat else 0,total_L_loss_scale=1.25,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,U_steps=2220,U_packets=567000,U_student_NN=567000,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.,u_practical=actual);put(b/'result.json',result)
 for name,records in [('reference_steps',rs),('shape_steps',ss),('step_metrics',ns),('unit_steps',units),('u_steps',us)]:jl(b/(name+'.jsonl.gz'),records)
 erows=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',erows);csvjson(b,'epoch_metrics',erows)
 def seq():
  for i,q in enumerate(us,1):
   indices=((i-1)*128+np.arange(128))%6300;yield dict(role='L',ids=[[1,[1,3,4,6,8][int(k)%5],int(k//5)%3+1,int(k),int(k)] for k in indices]);yield dict(role='U',ids=q['sample_ids'])
 jl(b/'training_sequence.jsonl.gz',seq());log=row.parent/'logs'/(row.name+'.log');log.parent.mkdir(exist_ok=True);log.write_text('All10 complete\n',encoding='utf-8');return row


class UPracticalChecks(unittest.TestCase):
 def test_r02_actual_feature_hook_logging_schema(self):
  import copy
  for arm in m.CONDITIONS[2:]:
   for step in (1,222):
    q=u_step_fixture(step,arm);self.assertNotIn('native_loss_returned_unchanged',q);self.assertEqual(q['native_batch_idx'],(step-1)%222+1);m.check_u_step(q,arm)
  base=u_step_fixture()
  for key,value in (('native_losses_preserved',False),('native_feature_adjustment',0.),('native_hook','_trusted_pseudo_loss_hook'),('native_optimizer_step',0)):
   q=copy.deepcopy(base);q[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_u_step(q,'UPRACTICAL30')
  for suffix in ('analysis','check'):
   path=Path(__file__).parent/('cvs_u_practical30_'+suffix+'_20261019.py');ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))

 def test_r02_ID_AST_and_failed770_cost(self):
  import cvs_u_practical30_publish_analysis_20261019 as pub
  import cvs_u_practical30_collect_20261019 as collect
  self.assertEqual(m.RUN,'20261019-phase1-FcrossRX-UPractical-e30-s2026092701-m3-r02');self.assertEqual(pub.RUN,m.RUN);self.assertEqual(collect.RUN,m.RUN);self.assertEqual(m.TRAINING_RELEASE,'cvs_FcrossRX_UPractical_E30_20261019_r02');self.assertEqual(pub.TRAINING_RELEASE,m.TRAINING_RELEASE);self.assertEqual(pub.RELEASE,'cvs_F_u_practical30_analysis_20261019_r02')
  config=pub.analysis_config(dict(project='/example/project',python='python',user='ordinary',host='host'),'a'*40);self.assertTrue(config['training_release'].endswith(m.TRAINING_RELEASE));self.assertIn('/runs/'+m.RUN+'/',config['output']);self.assertNotIn('truth',config)
  for suffix in ('analysis','check','publish_analysis','collect'):
   path=Path(__file__).parent/('cvs_u_practical30_'+suffix+'_20261019.py');ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
  for name in ('PREFLIGHT','TRANSFER','LAUNCH'):ast.parse(getattr(pub,name).replace('CONFIG',repr(config)).replace('RUN_VALUE',m.RUN))
  with temporary() as tmp:
   p=Path(tmp);root=p/'runs'/m.RUN;value=technical_history_fixture(root)
   with mock.patch.object(m,'_current_cost_summary',return_value=dict(new_total_NN=13057048)):
    c=m.cost_summary(p,{});self.assertEqual(c['new_total_NN'],13057048);self.assertEqual(c['technical_failed_NN'],770);self.assertEqual(c['candidate_whole_attempts_NN'],13057818);self.assertEqual(c['technical_failed_attempts'][0]['run_id'],m.FAILED_RUN)
    value['NN']=769;put(root/'technical_history/r01/cost.json',value)
    with self.assertRaises(ValueError):m.cost_summary(p,{})

 def test_source_exact_two_conditions_and_native_JSON(self):
  summary={a:dict(full_V_clean=.98,mid_low_mean=.5) for a in m.CONDITIONS};summary['UPRACTICAL30']=dict(full_V_clean=.979,mid_low_mean=.505);d=m.source_conditions(summary);self.assertTrue(d['source_criteria_pass']);self.assertEqual(len(d['checks']),2);json.dumps(d,allow_nan=False);summary['NOAS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(summary)['source_criteria_pass']);summary['MATCHED_NOAS30']['full_V_clean']=.981;self.assertFalse(m.source_conditions(summary)['source_criteria_pass']);summary['MATCHED_NOAS30']['full_V_clean']=.98;summary['UPRACTICAL30']['mid_low_mean']=.50499;self.assertFalse(m.source_conditions(summary)['source_criteria_pass'])
 def test_actual_U_teacher_mask_CE_entropy_route_scope(self):
  import copy
  for arm in m.CONDITIONS[2:]:
   for step in (1,222):
    q=u_step_fixture(step,arm);v=m.check_u_step(q,arm);self.assertGreater(v['accepted_student_disagreement'],0);self.assertEqual(q['CE_denominator'],q['U_packets']);self.assertEqual(q['As_packets'],q['U_packets']//4 if arm=='UPRACTICAL30' else 0)
  q=u_step_fixture();q.update(teacher_conf=[.5]*256,conf_mask=[False]*256,temporal_mask=[False]*256,eligibility=[False]*256,CE_denominator=0,eligible_selected_slots=0,eligible_Gaussian_slots=0,eligible_As_slots=0,native_loss_u=0.);v=m.check_u_step(q,'UPRACTICAL30');self.assertEqual(v['U_CE'],0.);self.assertGreater(v['U_entropy'],0.)
  base=u_step_fixture()
  for key,mutate in [('student_gate',lambda q:q.__setitem__('strong_agreement',True)),('Utruth',lambda q:q.__setitem__('U_truth_used',True)),('CEmean',lambda q:q.__setitem__('CE_denominator',64)),('entropy',lambda q:q.__setitem__('native_loss_entropy',0.)),('lambda',lambda q:q.__setitem__('lambda_u',.3)),('temporal',lambda q:q['temporal_mask'].__setitem__(0,False)),('routes',lambda q:q['selected_positions'].__setitem__(0,1)),('PID',lambda q:q['physics'][0]['sample_id'].__setitem__(3,0)),('NN',lambda q:q.__setitem__('extra_NN',1))]:
   q=copy.deepcopy(base);mutate(q)
   with self.subTest(key=key),self.assertRaises(ValueError):m.check_u_step(q,'UPRACTICAL30')
 def test_three_full2220step_logs_and_native_cost(self):
  with temporary() as tmp:
   p=Path(tmp)
   for arm in m.CONDITIONS[2:]:
    row=full_log_fixture(p,arm);public,_=m.parse_new_logs(row,arm);self.assertEqual(public['counts']['steps'],2220);self.assertEqual(public['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(public['counts']['U_packets'],567000);self.assertEqual(public['counts']['U_As'],141750 if arm=='UPRACTICAL30' else 0);self.assertEqual(public['physical_loader_packets'],dict(L=284160,U=567000));self.assertNotIn('bank',public)
   jl(row/'source/u_steps.jsonl.gz',[u_step_fixture(1,'UPRACTICAL30')])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'UPRACTICAL30')
 def test_old_audits_reused_three_tails_two_parents(self):
  with temporary() as tmp:
   p=Path(tmp);lookup=dict(m.rows(p));pc=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);tc=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962);oc={k:pc[k]+tc[k] for k in tc};oc.update(new_updates=6660,new_epochs=30,wall_seconds=2.);put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oc,epoch_curves=[dict(epoch=i) for i in range(1,31)]) for a in m.CONDITIONS[:2]}))
   for a in m.CONDITIONS[:2]:put(lookup[a]/'stage20/result.json',pc);put(lookup[a]/'source/actual_native_args.json',dict(args={}))
   with mock.patch.object(m,'parse_new_logs',return_value=(dict(cost=tc),dict(args={}))) as parser:r=m.training_analysis(p)
   self.assertEqual(parser.call_count,3);self.assertEqual(r['new_updates'],6660);self.assertEqual(r['shared_parent_count'],2);self.assertEqual(set(r['unique_parent20_costs']),{'NOAS30','AS30'});self.assertEqual(r['parent20']['MATCHED_NOAS30']['reference_condition'],'NOAS30');self.assertEqual(r['parent20']['UPRACTICAL30']['reference_condition'],'AS30')
 def test_all5_all7_before_truth_630CM105partitions(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);s=formal_fixture(p);lookup=dict(m.rows(p));path=lookup['MATCHED_NOAS30']/'prediction/complete.json';saved=path.read_bytes();path.unlink()
   with self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',s)
   path.write_bytes(saved);f=m.formal_analysis(p,p/'truth.json',s);self.assertEqual(f['independent_confusion_matrices'],630);self.assertEqual(f['partition_reconciliations'],105);self.assertEqual(len(f['comparisons']),10);self.assertEqual(f['stage']['clean_delta_pp'],f['paired']['UPRACTICAL30__MATCHED_NOAS30']['clean']['accuracy_delta_pp']);self.assertNotIn('interaction',f)
 def test_wait_terminal_missing_stops(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir()
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda x:False,probe=lambda *a:([],False),sleep=lambda _:None)
 def test_two_phase_source_truth_isolation_cost_and_single_archive(self):
  import cvs_u_practical30_publish_analysis_20261019 as pub
  import cvs_u_practical30_collect_20261019 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);s=formal_fixture(p);s['decision']=m.source_conditions(s['summary']);s['metrics']=[];tc=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);uq=dict(qualified_packets=0,qualified_As_packets=0,qualified_Gaussian_packets=0,U_As=0,timing={k:0. for k in m.U_TIMERS});training=dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',run_id=m.RUN,target_scores_read=False,rows={a:dict(cost=tc,u_practical=uq,counts=dict(U_As=0)) for a in m.CONDITIONS},parent20={a:dict(cost=tc) for a in m.CONDITIONS[2:]},unique_parent20_costs={a:tc for a in m.CONDITIONS[:2]});r=p/'runs'/m.RUN;records=[dict(row_id=root.name,reference_row_id='B00-s2026092701' if a=='MATCHED_NOAS30' else 'B01-s2026092701',L_NN=768,U_teacher_NN=8,U_student_NN=8,NN=784,optimizer_updates=0) for a,root in m.rows(p) if a in m.CONDITIONS[2:]];real=[dict(reference_row_id='B00-s2026092701',NN=2),dict(reference_row_id='B01-s2026092701',NN=2)];put(r/'synthetic_smoke.json',dict(status='PASS',records=records,NN=2352,real_checkpoint_NN=4,total_NN=2356,optimizer_updates=0,real_IQ=0,target_access=False));put(r/'real_checkpoint_smoke.json',dict(status='PASS',records=real,NN=4,optimizer_updates=0,real_IQ=0,target_query=False));out=r/'analysis_auto_r01';argv=['analysis','--project',str(p),'--release',str(p/m.TRAINING_RELEASE),'--output',str(out),'--code-commit','a'*40]
   technical_history_fixture(r)
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=s),mock.patch.object(m,'formal_analysis') as never:m.main();never.assert_not_called()
   self.assertFalse((out/'formal_summary.json').exists());before={n:(out/n).read_bytes() for n in collect.SOURCE_FILES};config=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40);self.assertNotIn('truth',config);self.assertIn('truth',pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,phase='final'))
   with mock.patch.object(sys,'argv',argv+['--phase','final','--truth',str(p/'truth.json')]),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',side_effect=AssertionError('Do not rescan frozen source')):m.main()
   self.assertEqual(m.read(out/'complete.json')['new_model_rows'],3);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],3*(3162962+12600+590)+2356);self.assertEqual(before,{n:(out/n).read_bytes() for n in collect.SOURCE_FILES});archive=p/'aggregate.tar.gz';packed=collect.pack_public(out,archive,'final');self.assertEqual(collect.extract_public(archive,p/'collected','final',packed['archive_SHA256'])['file_count'],11)

if __name__=='__main__':unittest.main(verbosity=2)
