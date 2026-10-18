"""Focused synthetic As64 allocation checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_satalloc30_analysis_20261018 as m

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
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;preds={'NOAS30':control,'AS30':y.copy(),'ALLOC_AS30':candidate}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,E50_source_eligible=False,fixed_recipe_closed=True,fullV_clean_vs_NoAs30_pp=0.,mid_low_mean_vs_original_As30_pp=0.))

def allocation_fixture(step=1,ids=None):
 if ids is None:ids=[[1,(1,3,4,6,8)[j%5],1+(j//5)%3,j,j] for j in range(128)]
 TX=[z[3]%6 for z in ids];RX=[z[1] for z in ids];priorities=(np.arange(128)+.5)/128;grouped={}
 for j,key in enumerate(zip(TX,RX)):grouped.setdefault(key,[]).append(j)
 members=sorted(grouped.values(),key=lambda ix:ix[0]);quotas=[0]*len(members);chosen=[];rounds=[]
 while len(chosen)<64:
  active=[g for g,ix in enumerate(members) if quotas[g]<len(ix)];order=list(active);used=order[:min(len(order),64-len(chosen))]
  for g in used:chosen.append(members[g][quotas[g]]);quotas[g]+=1
  rounds.append(dict(active_groups=active,visit_order=order,selected_groups=used))
 positions=sorted(chosen);old=[j for j in range(128) if (j+step)%2!=0];cells=[]
 for g,ix in enumerate(members):cells.append(dict(canonical_index=g,TX=TX[ix[0]],RX=RX[ix[0]],available=len(ix),selected=quotas[g],input_L_positions=ix,priority_ordered_L_positions=ix,selected_L_positions=sorted(set(ix)&set(positions)),old_selected_L_positions=sorted(set(ix)&set(old)),old_selected=len(set(ix)&set(old))))
 return dict(step=step,cumulative_step=4440+step,epoch=(step-1)//222+1,cumulative_epoch=20+(step-1)//222+1,selected_packets=64,ordinary_packets=64,As_packets=64,cells=cells,rounds=rounds,priorities=priorities.tolist(),selected_L_positions=positions,ordinary_L_positions=sorted(set(range(128))-set(positions)),old_selected_L_positions=old,active_cells=len(cells),quota_min=min(quotas),quota_max=max(quotas),priority_ties=0,changed_positions=len(set(positions)^set(old)),augmentation_seed=20261016421,allocation_namespace='FcrossRX/AsAllocation/waterfill64/v1',allocation_seed=123,private_CPU_rng='numpy.default_rng(PCG64)',global_RNG_used=False,sampler_changed=False,physical_ID_uniqueness_required=False,labels_used='trueL TX only',RX_used='existing originalRX',confidence_or_quality_used=False,U_truth_used=False,extra_NN=0,new_losses=0,formal_test_changed=False,allocation_cpu_wall_seconds=.0001,allocation_hook_cpu_wall_seconds=.0002,TX=TX,RX=RX,day=[z[2] for z in ids],sample_id_fields=['eq_i','rx_i','day_i','sig_i','base_index'],sample_ids=ids,selected_sample_ids=[ids[j] for j in positions],duplicate_PID_occurrences=128-len({tuple(z) for z in ids}),scene=['practical_low_urban']*64,state=['LOS']*64,lock=['LOCK']*64,scene_counts={'practical_low_urban':64},same_IQ_each_pair=True,ordinary_redraws=0,global_Torch_RNG_draws=0,global_CUDA_RNG_draws=0,CPU_timing='perf_counter CPUwall;not CUDAevent',total_L_scale=1.25,L_forwards=6,L_NN=768,extra_CE_operator_calls=0)


def loss_fixture(step=1,ids=None):
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts={'practical_low_urban':64},state_counts={'LOS':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=.43,extra_CE=.4,extra_KL=.03)
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05)
 r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD'];c=allocation_fixture(step,ids);c['total_L']=r['total_L'];n={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};n.update({'allocation/'+k:c[k] for k in m.ALLOCATION_KEYS});n.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0});return r,s,n,c


def log_fixture(project):
 from collections import Counter
 row=dict(m.rows(project))['ALLOC_AS30'];b=row/'source';b.mkdir(parents=True)
 policy=dict(mode='TX_RX_waterfill64',selected_packets_per_step=64,ordinary_complement_per_step=64,augmentation_seed=20261016421,allocation_namespace='FcrossRX/AsAllocation/waterfill64/v1',original128positions_unchanged=True,each_physical_position_one_mixed_view=True,primary_ordinary128_unchanged=True,ORD128_anchors_unchanged=True,loader_changed=False,physical_ID_uniqueness_required=False,physical_sample_deduplication=False,reorder_As_to_fixed_mask=False,global_RNG_used=False,extra_NN=0,extra_CE_operator_calls=0,new_losses=0,new_U_views=0,formal_test_changed=False)
 put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=dict(updates=2220,crossview=dict(mode='ORD128'),satellite_allocation=policy)))
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080,total_L_loss_scale=1.25,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,U_steps=2220,U_packets=567000,U_student_NN=567000,wall_seconds=10.,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,satellite_allocation={})
 put(b/'actual_native_args.json',dict(args=dict(label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True)));put(b/'EMA_initialization.json',dict(origin='strict own stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True))
 def physical(role,i):return [1,(1,3,4,6,8)[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]
 def Lids(step):
  ids=[physical('L',((step-1)*128+j)%6300) for j in range(128)]
  if step==1:ids[1]=list(ids[0])
  return ids
 refs=[];shapes=[];native=[];units=[];allocations=[];coverage=Counter();available=Counter();selected=Counter();oldmask=Counter()
 for step in range(1,2221):
  r,s,n,c=loss_fixture(step,Lids(step))
  for j in range(128):available[(c['TX'][j],c['RX'][j],c['day'][j])]+=1
  for j in c['selected_L_positions']:
   coverage[(c['TX'][j],c['RX'][j],'practical_low_urban','LOS','LOCK')]+=1;selected[(c['TX'][j],c['RX'][j],c['day'][j])]+=1
  for j in c['old_selected_L_positions']:oldmask[(c['TX'][j],c['RX'][j],c['day'][j])]+=1
  refs.append(r);shapes.append(s);native.append(dict(step=step,epoch=(step-1)//222+1,metrics=n));units.append(dict(step=step,cumulative_step=4440+step,physical_clip_calls=step));allocations.append(c)
 def strata(counter):return [dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(counter.items())]
 result['satellite_allocation']=dict(mode='TX_RX_waterfill64',allocation_calls=2220,selected_packets=142080,ordinary_complement_packets=142080,actual_As_packets=142080,channel_calls=2220,augmentation_seed=20261016421,allocation_namespace='FcrossRX/AsAllocation/waterfill64/v1',pair_weights=[.5,.5,.25],SupCon='ORD128',total_L_scale=1.25,per_step_L_NN=768,extra_NN=0,extra_CE_operator_calls=0,pure_gradient_variance_claim=False,global_RNG_used=False,sampler_changed=False,physical_ID_uniqueness_required=False,common_U_trajectory_equality_claim=False,oldmask_is_counterfactual=True,each_physical_position_one_mixed_view=True,availability_TX_RX_day_exposure=strata(available),selected_TX_RX_day_exposure=strata(selected),within_current_trajectory_oldmask_TX_RX_day_exposure=strata(oldmask),changed_positions=sum(z['changed_positions'] for z in allocations),allocation_cpu_wall_seconds=sum(z['allocation_cpu_wall_seconds'] for z in allocations),allocation_hook_cpu_wall_seconds=sum(z['allocation_hook_cpu_wall_seconds'] for z in allocations),timing='perf_counter CPUwall;selector included in hook')
 result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('allocation_steps',allocations)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(1,2221):
   yield dict(role='L',ids=Lids(i));u=(i-1)%222;yield dict(role='U',ids=[physical('U',u*256+j) for j in range(124 if u==221 else 256)])
 jl(b/'training_sequence.jsonl.gz',sequence());path=row.parent/'logs'/(row.name+'.log');path.parent.mkdir(exist_ok=True);path.write_text('All10 epochs complete\n',encoding='utf8');return row

class SatelliteAllocationChecks(unittest.TestCase):
 def test_allocation_actual_schema_feasibility_and_full6L(self):
  import copy
  r,s,n,c=loss_fixture();e,valid=m.check_allocation(r,s,c,n);self.assertLess(e,3e-5);self.assertEqual(valid,512);self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertNotIn('shape/RD',n);self.assertEqual(sum(k.startswith(('concat/','allocation/')) for k in n),14)
  for name,mutate in [('duplicate',lambda q:q['selected_L_positions'].__setitem__(0,q['selected_L_positions'][1])),('complement',lambda q:q['ordinary_L_positions'].__setitem__(0,q['selected_L_positions'][0])),('quota',lambda q:q['cells'][0].__setitem__('selected',q['cells'][0]['available']+1)),('missingcell',lambda q:q['cells'].pop()),('oldmask',lambda q:q['old_selected_L_positions'].reverse()),('round',lambda q:q['rounds'][0]['selected_groups'].pop()),('priority',lambda q:q['priorities'].__setitem__(0,1.1)),('filter',lambda q:q.__setitem__('confidence_or_quality_used',True)),('sampler',lambda q:q.__setitem__('sampler_changed',True)),('Utruth',lambda q:q.__setitem__('U_truth_used',True)),('NN',lambda q:q.__setitem__('extra_NN',1)),('maskchange',lambda q:q.__setitem__('changed_positions',0))]:
   q=copy.deepcopy(c);mutate(q)
   with self.subTest(name=name),self.assertRaises(ValueError):m.check_allocation(r,s,q,n)
  wrong=dict(n);wrong['concat/total_L']=r['base_L_before_shape']
  with self.assertRaises(ValueError):m.check_native_L(r,s,wrong)
 def test_complete_tail_actual_U_EMA_cost_exposure_and_prefix(self):
  with temporary() as tmp:
   p=Path(tmp);row=log_fixture(p);q,_=m.parse_new_logs(row,'ALLOC_AS30');self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['counts']['U_forward_packets'],567000);self.assertEqual(q['selector_accounting']['extra_CE_operator_calls'],0);self.assertTrue(q['selector_accounting']['zero_extra_model_forward_is_not_zero_compute']);self.assertEqual(q['allocation']['sampler_repetitions']['batches_with_repeated_physical_ID'],1);self.assertTrue(q['selected_exposure']['old_parity_is_counterfactual_on_candidate_sequence_not_original_run_replay'])
   path=row/'source/allocation_steps.jsonl.gz';value=list(m.logs.records(path));jl(path,value[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'ALLOC_AS30')

 def test_saved_original30_audit_reused_without_old_step_reads(self):
  with temporary() as tmp:
   p=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i,ordinary_raw_MSE=.01) for i in range(1,31)]
   put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in ('NOAS30','AS30')}));lookup=dict(m.rows(p));put(lookup['AS30']/'stage20/result.json',parent);put(lookup['AS30']/'source/actual_native_args.json',dict(args={}))
   newcost={k:oldcost[k]-parent[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')};q=dict(cost=newcost,epoch_curves=curves[20:]);private=dict(args={},ordinary_mse=[.01]*2220)
   with mock.patch.object(m,'parse_new_logs',return_value=(q,private)),mock.patch.object(m.logs,'records',side_effect=AssertionError('Old logs reopened')):r=m.training_analysis(p)
   self.assertTrue(r['rows']['NOAS30']['reused']);self.assertEqual(r['new_updates'],2220);self.assertTrue(r['reference_comparability']['four_counters_equal']);self.assertFalse(r['reference_comparability']['old_step_logs_rescanned'])
 def test_two_source_conditions_use_different_frozen_references(self):
  q={a:dict(full_V_clean=.98,mid_low_mean=.6) for a in m.CONDITIONS};self.assertFalse(m.source_conditions(q)['source_criteria_pass']);q['ALLOC_AS30']=dict(full_V_clean=.979,mid_low_mean=.605);self.assertTrue(m.source_conditions(q)['source_criteria_pass']);q['AS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(q)['source_criteria_pass']);json.dumps(m.source_conditions(q),allow_nan=False)
  q['ALLOC_AS30']['full_V_clean']=.97899;self.assertTrue(m.source_conditions(q)['fixed_recipe_closed']);self.assertTrue(m.old.stable.stage_clean_pass(-168,168000));self.assertFalse(m.old.stable.stage_clean_pass(-169,168000))
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
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['ALLOC_AS30']/'scoring/complete.json').unlink();original=m.read;touch=[]
   def guarded(path):
    if Path(path).name=='truth.json':touch.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(touch,[])
 def test_full_formal_CM_partitions_errors_and_separate_allocation_increment(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['partition_reconciliations'],63);self.assertEqual(q['independent_confusion_matrices'],378);self.assertEqual(q['paired']['ALLOC_AS30__NOAS30']['clean']['repairs'],1);self.assertEqual(q['paired']['ALLOC_AS30__NOAS30']['clean']['new_errors'],1);self.assertTrue(q['stage']['clean_absolute_0p1pp_guard_met']);self.assertAlmostEqual(q['incremental_satellite_allocation_effect_pp']['clean'],-100/84);self.assertNotIn('contrast_interaction_pp',q);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_observes_real_handles_and_terminal_missing(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Waited on dead handle'))
 def test_main_parameterized_publisher_cost_and_single_archive_collection(self):
  import cvs_satalloc30_publish_analysis_20261018 as pub
  import cvs_satalloc30_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);c=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_satalloc30_analysis_20261018_r07',output='analysis_auto_r07');out=Path(c['output']);self.assertEqual(m.validate_output(p,out),out);self.assertEqual(Path(c['training_release']).name,m.TRAINING_RELEASE)
   source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost,allocation={}) for a in m.CONDITIONS},parent20={'ALLOC_AS30':dict(cost=cost)},reference_comparability={'old_step_logs_rescanned':False})
   put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',NN=768,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False));argv=['analysis','--project',c['project'],'--release',c['training_release'],'--output',c['output'],'--truth',str(p/'truth.json'),'--code-commit',c['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   self.assertEqual(m.read(out/'complete.json')['status'],'ANALYZED');self.assertGreater(len((out/'final_report.md').read_text(encoding='utf8').splitlines()),30);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],3162962+12600+590+770)
   for name in collect.required_names('final'):self.assertTrue((out/name).is_file(),name)
   archive=p/'aggregate.tar.gz';saved=collect.pack_public(out,archive,'final');result=collect.extract_public(archive,p/'collected','final',saved['archive_SHA256']);self.assertEqual(result['status'],'VERIFIED');self.assertEqual(result['file_count'],11);self.assertIn('cvs_satalloc30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
