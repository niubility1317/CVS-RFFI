"""Focused synthetic finite-bank quality risk checks; no real data/model/remote operations."""
import gzip,json,tempfile,unittest,sys,os
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_qualityrisk30_analysis_20261018 as m

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
 panel=project/m.old.stable.PANEL;put(panel/'data_roles.json',dict(physical_ids=ids.tolist()));obs=panel/'observations';obs.mkdir();ratio=np.broadcast_to(np.resize(np.asarray([0.,5.,10.,20.]),1800),(2,3,11,1800)).copy();np.savez(obs/'physics.npz',state=np.zeros_like(ratio,int),lock=np.zeros_like(ratio,int),signal_to_added_noise_db=ratio)

def formal_fixture(project):
 n=84;metadata(project,n);ids=np.asarray(['opaque'+str(i) for i in range(n)]);y=np.arange(n)%6;rx=np.asarray(['r'+str(i%7) for i in range(n)]);day=np.asarray(['d'+str(i%4) for i in range(n)])
 put(project/'truth.json',{z:dict(label=int(y[i]),receiver=str(rx[i]),day=str(day[i])) for i,z in enumerate(ids)})
 control=y.copy();control[0]=1;candidate=y.copy();candidate[-1]=4;iid=y.copy();iid[1]=0;preds={'NOAS30':control,'AS30':y.copy(),'BANK_IID30':iid,'BANK_STRAT30':candidate}
 for arm,root in m.rows(project):
  p=preds[arm];np.savez(root/'prediction/predictions.npz',ids=ids,**{v:p for v in m.VIEWS});records=[]
  for v in m.VIEWS:
   z=np.zeros((n,6));z[np.arange(n),p]=2.;np.savez(root/'prediction'/(v+'_scores.npz'),ids=ids,logits=z)
   for dim,values in [('overall',['ALL']),('receiver',sorted(set(rx))),('day',sorted(set(day))),('transmitter',range(6))]:
    for value in values:
     mask=np.ones(n,bool) if dim=='overall' else rx==value if dim=='receiver' else day==value if dim=='day' else y==value;q=m.metric(y[mask],p[mask]);records.append(dict(view=v,dimension=dim,stratum=str(value),query_count=q['count'],**{k:q[k] for k in ('accuracy','macro_f1','confusion')}))
  put(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=records))
 return dict(status='SOURCE_ANALYZED',run_id=m.RUN,target_scores_read=False,summary={a:dict(full_V_clean=.98,panel=dict(clean=dict(accuracy=1.)),mid_low_mean=.6) for a in m.CONDITIONS},quality_groups={a:{v:{g:dict(count=1800,accuracy=.6,macro_F1=.58,mean_trueclass_margin_over30=.01) for g in ('hard_lt8','easy_ge8')} for v in m.SOURCE_SCENES} for a in m.CONDITIONS},decision=dict(source_criteria_pass=False,M3_source_eligible=False,fixed_stage1_closed=True,hard_lt8_S_minus_I_equal_scene_accuracy_pp=0.,easy_ge8_S_minus_I_equal_scene_accuracy_pp=0.,fullV_clean_S_minus_originalAs_pp=0.))

def bank_fixture(project):
 from collections import Counter
 b=project/'runs'/m.RUN/'proposal_bank';b.mkdir(parents=True,exist_ok=True);n=m.BANK_SIZE;L=m.BANK_L;si=np.tile(np.arange(L),40);tx=si%6;rxindex=(si//6)%5;rx=np.asarray([1,3,4,6,8])[rxindex];day=(si//30)%3+1;domain=rxindex*3+day-1;ids=np.stack([np.ones(n,int),rx,day,si,si],axis=1);context=np.repeat(np.arange(40),L);bb=np.resize(np.asarray([0,1,1,2,2,2,3,3,3,3],np.int8),n);ratio=np.asarray([1.,5.,10.,20.])[bb]
 meta=dict(TX=tx,RX=rx,day=day,domain=domain,bin=bb,scene=np.asarray(m.SOURCE_SCENES)[np.arange(n)%3],state=np.asarray(['LOS','PARTIAL','BLOCKED'])[np.arange(n)%3],lock=np.asarray(['locked','degraded','unlocked'])[np.arange(n)%3],signal_to_added_noise_db=ratio,quality_snr_db=ratio+.1,cumulative_epoch=21+context//4,replica=context%4,context_step=(20+context//4)*222+1+context%4,source_L_index=si,sample_ids=ids)
 np.savez(b/'metadata.npz',**meta);(b/'iq.npy').write_bytes(b'SYNTHETIC_METADATA_ONLY');(b/'physics.jsonl.gz').write_bytes(b'SYNTHETIC_METADATA_ONLY');counts=np.bincount(bb,minlength=4);contexts=[dict(cumulative_epoch=e,replica=r,cumulative_step=(e-1)*222+1+r) for e in range(21,31) for r in range(4)];strata=Counter(zip(*(meta[k].tolist() for k in ('TX','RX','day','bin','scene','state','lock'))))
 done=dict(status='SOURCE_L_BANK_COMPLETE',run_id=m.RUN,role='L',source_role_key='L_s',source_role_counts=dict(L_s=L,U_s=56700,V=27000),role_ids=['opaqueL'+str(i) for i in range(L)],count=n,source_L=L,each_L_views=40,bin_field='signal_to_added_noise_db',bin_edges=[3.,8.,15.],NN=0,checkpoint_reads=0,U_truth_used=False,U_iterations=0,target_access=False,quality_rejection=False,retries=0,bank_online_risk_equivalence_claim=False,bin_counts=counts.tolist(),nominal_bin_mass=(counts/n).tolist(),source_L_index_counts=[40]*L,contexts=contexts,propagation_simulations=n,shape=[n,2,256],dtype='float32',IQ_bytes=n*2*256*4,strata=[dict(TX=k[0],RX=k[1],day=k[2],bin=k[3],scene=k[4],state=k[5],lock=k[6],count=v) for k,v in strata.items()],wall_seconds=1.,timing='synthetic CPUwall',raw_dataset_cache_cpu_wall_seconds=.1,simulation_adapter_cpu_wall_seconds=.5,IQ_cache_write_cpu_wall_seconds=.1,metadata_write_cpu_wall_seconds=.1);put(b/'complete.json',done);return m.audit_bank(project)


def bank_step_fixture(bank,step=1,mode='STRATIFIED',ids=None):
 meta=bank['meta'];positions=[j for j in range(128) if (j+step)%2!=0];ordinary=[j for j in range(128) if j not in positions]
 if ids is None:ids=meta['sample_ids'][np.arange(128)%m.BANK_L].tolist()
 lookup={tuple(z):int(t) for z,t in zip(bank['source_ids'],meta['TX'][:m.BANK_L])};TX=[lookup[tuple(z)] for z in ids];RX=[z[1] for z in ids];day=[z[2] for z in ids];dom=[bank['domain_map'][(r,d)] for r,d in zip(RX,day)]
 if mode=='STRATIFIED':indices=np.concatenate([np.flatnonzero(meta['bin']==g)[(step*16+np.arange(16))%int(bank['bin_counts'][g])] for g in range(4)])
 else:indices=(step*64+np.arange(64))%m.BANK_SIZE
 bb=meta['bin'][indices];weights=np.ones(64) if mode=='IID' else 4*bank['nominal_mass'][bb];mix_y=np.asarray(TX).copy();mix_y[positions]=meta['TX'][indices];mixdom=np.asarray(dom).copy();mixdom[positions]=meta['domain'][indices];ca=.2+np.arange(128)*.001;cb=ca+.05;ce=.5*(ca+cb);kl=.02+np.arange(128)*.00001;w=np.ones(128);w[positions]=weights.astype(np.float32)
 details=[dict(phase=p,level=l,mode='ORD128',loss=.05,total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')]
 r=dict(step=step,cumulative_step=4440+step,L_forwards=6,As_packets=64,Ag_packets=0,extra_control_packets=128,Ag_mode='NONE',ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,SupCon_weight=.1,SupCon=.05,ordinary_raw_MSE=.01,scene_counts=dict(__import__('collections').Counter(meta['scene'][indices].tolist())),state_counts=dict(__import__('collections').Counter(meta['state'][indices].tolist())),crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=details,common_RD=.23,common_CE=.2,common_KL=.03,extra_RD=float(((ce+kl)*w).mean()),extra_CE=float((ce*w).mean()),extra_KL=float((kl*w).mean()))
 s=dict(step=step,cumulative_step=4440+step,mode='NONE',NN=256,packets=128,applied=0,weight=.25,new_boundary_samples=0,same_IQ_each_pair=True,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_RD=.45,shape_CE=.4,shape_KL=.05);r['base_L_before_shape']=.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'];r['total_L']=r['base_L_before_shape']+.25*s['shape_RD']
 c=dict(step=step,epoch=(step-1)//222+1,cumulative_epoch=21+(step-1)//222,cumulative_step=4440+step,mode=mode,selected_L_positions=positions,ordinary_L_positions=ordinary,current_L_TX=TX,current_L_RX=RX,current_L_day=day,current_L_sample_ids=ids,bank_indices=indices.tolist(),bin_counts=np.bincount(bb,minlength=4).tolist(),bank_bin_counts=bank['bin_counts'].tolist(),nominal_bin_mass=bank['nominal_mass'].tolist(),weights=weights.tolist(),bank_weight_sum=float(weights.sum()),Torch_bank_weight_sum=float(weights.sum()),ordinary_weight_sum=64.,applied_bank_weights=weights.astype(np.float32).astype(float).tolist(),mixed_labels=mix_y.tolist(),mixed_domains=mixdom.tolist(),with_replacement=True,global_RNG_used=False,extra_NN=0,As_packets=64,bank_cache_reads=64,propagation_simulations=0,L_forwards=6,L_NN=768,total_L_scale=1.25,CE_logit_packets=256,new_weight_multiplication_packets=128,CE_operator_widths=[128,128],baseline_CE_operator_widths=[128,128],extra_CE_operator_calls=0,ordinary_redraws=0,bank_redraws_in_second_forward=0,same_IQ_each_pair=True,per_packet_CE_a=ca.tolist(),per_packet_CE_b=cb.tolist(),per_packet_KL=kl.tolist(),ordinary_CE=float(ce[ordinary].mean()),ordinary_KL=float(kl[ordinary].mean()),bank_unweighted_CE=float(ce[positions].mean()),bank_unweighted_KL=float(kl[positions].mean()),bank_weighted_CE=float((ce[positions]*weights.astype(np.float32)).mean()),bank_weighted_KL=float((kl[positions]*weights.astype(np.float32)).mean()),mixed_weighted_CE=r['extra_CE'],mixed_weighted_KL=r['extra_KL'],mixed_weighted_RD=r['extra_RD'],total_L=r['total_L'],scene_counts=r['scene_counts'],state_counts=r['state_counts'],sampling_seed=20261016421,sampling_namespace='FcrossRX/qualityRisk/bank_sampling/v1',sampling_cpu_wall_seconds=.0001,cache_gather_cpu_wall_seconds=.0002,cache_transfer_cpu_wall_seconds=.0003,weighted_loss_cpu_wall_seconds=.0004)
 mapping={'bank_TX':'TX','bank_RX':'RX','bank_day':'day','bank_domain':'domain','bank_sample_ids':'sample_ids','bank_source_L_index':'source_L_index','bin':'bin','scene':'scene','state':'state','lock':'lock','signal_to_added_noise_db':'signal_to_added_noise_db','quality_snr_db':'quality_snr_db','bank_context_epoch':'cumulative_epoch','bank_replica':'replica','bank_context_step':'context_step'}
 c.update({k:meta[v][indices].tolist() for k,v in mapping.items()});c['mixed_label_replacements']=sum(TX[p]!=t for p,t in zip(positions,c['bank_TX']));c['currentL_bank_PID_matches']=sum(tuple(ids[p])==tuple(z) for p,z in zip(positions,c['bank_sample_ids']))
 for i,n in enumerate(c['bin_counts']):c['bin'+str(i)]=n
 n={'concat/'+k:r[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')};n.update({'bank/'+k:c[k] for k in m.BANK_KEYS});n.update({'train/loss':r['total_L']+.1,'train/pseudo_total':124 if step%222==0 else 256,'train/pseudo_selected':100,'train/pseudo_truth_available':0});return r,s,c,n


def log_fixture(project,bank,mode='STRATIFIED'):
 from collections import Counter
 arm='BANK_STRAT30' if mode=='STRATIFIED' else 'BANK_IID30';row=dict(m.rows(project))[arm];b=row/'source';b.mkdir(parents=True)
 p=dict(mode=mode,bank_size=m.BANK_SIZE,source_L=m.BANK_L,each_L_bank_views=40,bank_packets_per_step=64,mixed_denominator=128,mixed_outer_weight=.5,As_expected_outer_weight=.25,all_loss_scale=1.25,CE_label_smoothing=.01,bin_field='signal_to_added_noise_db',bin_edges=[3.,8.,15.],per_bin_packets=16 if mode=='STRATIFIED' else None,bank_weight='4*N_bin/N_bank' if mode=='STRATIFIED' else '1',sample_with_replacement=True,KL_same_weights_as_CE=True,private_RNG=True,original_alternating_mask=True,mixed_labels_replaced=True,mixed_domain_metadata_replaced=True,ORD128_currentL_only=True,global_RNG_used=False,online_bank_risk_equivalence_claim=False,common_U_trajectory_equality_claim=False,extra_NN=0,extra_CE_operator_calls=0,formal_test_changed=False)
 put(b/'completion.json',dict(epoch=10,optimizer_steps=2220,config=dict(updates=2220,crossview=dict(mode='ORD128'),quality_risk=p)))
 result=dict(optimizer_steps=2220,parent_epochs=20,parent_optimizer_steps=4440,parent_reference_row_id='B01-s2026092701',cumulative_epochs=30,cumulative_optimizer_steps=6660,U_truth_read=False,target_access=False,head_fit_updates=0,extra_U_NN=0,U_As=0,Ag_packets=0,L_NN=1704960,L_As=142080,total_L_loss_scale=1.25,EMA_local_successful_updates=2220,EMA_cumulative_updates=6660,U_steps=2220,U_packets=567000,U_student_NN=567000,student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,wall_seconds=10.)
 put(b/'EMA_initialization.json',dict(origin='strict own stage20 EMA',extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True));put(b/'ordinary_augmentation_actual.json',dict(schedule_frozen_at=13,schedule_args_epochs=20,no_ordinary_before_practical=True,RDrop_same_IQ=True));put(b/'actual_native_args.json',dict(args=dict(label_smoothing=.01,use_aug=True,use_unlabeled=True,use_ema_teacher=True)))
 refs=[];shapes=[];native=[];units=[];steps=[];index_count=np.zeros(m.BANK_SIZE,int);physical_count=np.zeros(m.BANK_L,int);physical_mass=np.zeros(m.BANK_L,float);bins_count=np.zeros(4,int);bins_mass=np.zeros(4,float);strata=Counter();weighted=Counter();coverage=Counter()
 def Lids(step):return bank['source_ids'][((step-1)*128+np.arange(128))%m.BANK_L].tolist()
 for i in range(1,2221):
  r,s,c,n=bank_step_fixture(bank,i,mode,Lids(i));refs.append(r);shapes.append(s);native.append(dict(step=i,epoch=(i-1)//222+1,metrics=n));units.append(dict(step=i,cumulative_step=4440+i,physical_clip_calls=i));steps.append(c);ix=c['bank_indices'];si=c['bank_source_L_index'];w=np.asarray(c['weights']);bb=np.asarray(c['bin']);np.add.at(index_count,ix,1);np.add.at(physical_count,si,1);np.add.at(physical_mass,si,w);bins_count+=np.bincount(bb,minlength=4);bins_mass+=np.bincount(bb,weights=w,minlength=4)
  for tx,rx,day,g,scene,state,lock,weight in zip(c['bank_TX'],c['bank_RX'],c['bank_day'],bb,c['scene'],c['state'],c['lock'],w):key=(tx,rx,day,int(g),scene,state,lock);strata[key]+=1;weighted[key]+=float(weight);coverage[(tx,rx,scene,state,lock)]+=1
 def sr(counter):return [dict(TX=k[0],RX=k[1],day=k[2],bin=k[3],scene=k[4],state=k[5],lock=k[6],mass=float(v) if isinstance(v,(float,np.floating)) else int(v)) for k,v in counter.items()]
 a=dict(mode=mode,bank_size=m.BANK_SIZE,bank_sampling_calls=2220,bank_cache_reads=142080,selected_packets=142080,ordinary_complement_packets=142080,training_propagation_simulations=0,shared_propagation_simulations=m.BANK_SIZE,per_step_L_NN=768,pair_weights=[.5,.5,.25],total_L_scale=1.25,As_expected_weight=.25,bank_bin_counts=bank['bin_counts'].tolist(),nominal_bin_mass=bank['nominal_mass'].tolist(),shared_bank_cost_counted_once_at_run_level=True,selected_true_label_metadata=True,old_currentL_labels_not_reused_for_bank=True,finite_bank_risk_equivalence_only=True,extra_NN=0,extra_CE_operator_calls=0,online_risk_equivalence_claim=False,common_U_trajectory_equality_claim=False,global_RNG_used=False,sampler_current_L_changed=False,CE_operator_calls=4440,baseline_CE_operator_calls=4440,CE_logit_packets=568320,baseline_CE_logit_packets=568320,new_weight_multiplication_packets=284160,bin_counts=bins_count.tolist(),bin_weighted_mass=bins_mass.tolist(),selected_TX_RX_day_bin_scene_state_lock_exposure=sr(strata),weighted_TX_RX_day_bin_scene_state_lock_mass=sr(weighted),timing='disjoint CPUwall;not CUDAevent');a.update({k:sum(z[k] for z in steps) for k in m.TIMERS});result['quality_risk']=a;result['coverage']=[dict(TX=k[0],RX=k[1],scene=k[2],state=k[3],lock=k[4],count=v) for k,v in coverage.items()];put(b/'result.json',result)
 np.savez_compressed(b/'bank_exposure.npz',bank_index_counts=index_count,source_L_counts=physical_count,source_L_weighted_mass=physical_mass,bin_counts=bins_count,bin_weighted_mass=bins_mass)
 for name,value in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',native),('unit_steps',units),('bank_steps',steps)]:jl(b/(name+'.jsonl.gz'),value)
 epochs=[dict(epoch=i,lr=.0002,val_tx_acc=98.,epoch_time_s=.5,train_grad_backbone=.1,train_grad_total=.2) for i in range(1,11)];csvjson(b,'metrics_epoch',epochs);csvjson(b,'epoch_metrics',epochs)
 def sequence():
  for i in range(1,2221):
   yield dict(role='L',ids=Lids(i));j=(i-1)%222;uu=[ [1,[1,3,4,6,8][k%5],1+(k//5)%3,100000+k,100000+k] for k in range(j*256,j*256+(124 if j==221 else 256))];yield dict(role='U',ids=uu)
 jl(b/'training_sequence.jsonl.gz',sequence());log=row.parent/'logs'/(row.name+'.log');log.parent.mkdir(exist_ok=True);log.write_text('All10 complete\n',encoding='utf8');return row

class QualityRiskChecks(unittest.TestCase):
 def test_actual_bank_scope_weighted_CE_KL_and_no_renormalization(self):
  import copy
  with temporary() as tmp,mock.patch.object(m,'BANK_L',90),mock.patch.object(m,'BANK_SIZE',3600):
   p=Path(tmp);public,bank=bank_fixture(p);self.assertEqual(public['observations'],3600);self.assertEqual(public['bin_counts'],[360,720,1080,1440])
   for mode in ('IID','STRATIFIED'):
    r,s,c,n=bank_step_fixture(bank,mode=mode);e,valid=m.check_bank_step(r,s,c,n,bank,mode);self.assertLess(e,3e-5);self.assertEqual(valid,512);self.assertLess(m.check_native_L(r,s,n),3e-5);self.assertEqual(sum(k.startswith(('concat/','bank/')) for k in n),15);self.assertNotIn('shape/RD',n)
   r,s,c,n=bank_step_fixture(bank)
   for name,mutate in [('weights',lambda z:z['weights'].__setitem__(0,1.)),('applied',lambda z:z['applied_bank_weights'].__setitem__(0,1.)),('mixedlabel',lambda z:z['mixed_labels'].__setitem__(z['selected_L_positions'][0],(z['mixed_labels'][z['selected_L_positions'][0]]+1)%6)),('CE',lambda z:z.__setitem__('mixed_weighted_CE',2*z['mixed_weighted_CE'])),('KL',lambda z:z.__setitem__('mixed_weighted_KL',z['bank_unweighted_KL'])),('counts',lambda z:z['bin_counts'].__setitem__(0,15)),('sameIQ',lambda z:z.__setitem__('bank_redraws_in_second_forward',1)),('NN',lambda z:z.__setitem__('extra_NN',1))]:
    z=copy.deepcopy(c);mutate(z)
    with self.subTest(name=name),self.assertRaises(ValueError):m.check_bank_step(r,s,z,n,bank,'STRATIFIED')
 def test_complete_stratified_tail_actual_cost_exposure_and_prefix(self):
  with temporary() as tmp:
   p=Path(tmp);_,bank=bank_fixture(p);row=log_fixture(p,bank);q,_=m.parse_new_logs(row,'BANK_STRAT30',bank);self.assertEqual(q['cost']['native_training_and_validation_NN'],3162962);self.assertEqual(q['counts']['As_packets'],142080);self.assertEqual(q['physical_loader_packets']['U'],567000);self.assertEqual(q['exposure_summary']['bin_counts'],[35520]*4);self.assertEqual(q['risk_accounting']['extra_CE_operator_calls'],0);self.assertEqual(q['risk_accounting']['CE_operator_calls'],4440)
   path=row/'source/bank_steps.jsonl.gz';values=list(m.logs.records(path));jl(path,values[:-1])
   with self.assertRaises(m.logs.IncompleteTrainingLogs):m.parse_new_logs(row,'BANK_STRAT30',bank)
 def test_fixed_three_source_criteria_equal_scene_not_count_weighted(self):
  summary={a:dict(full_V_clean=.98) for a in m.CONDITIONS};quality={a:{v:{g:dict(accuracy=.6,count=10 if v==m.SOURCE_SCENES[1] else 1000) for g in ('hard_lt8','easy_ge8')} for v in m.SOURCE_SCENES[1:]} for a in m.CONDITIONS};S=quality['BANK_STRAT30'];S[m.SOURCE_SCENES[1]]['hard_lt8']['accuracy']=.61;S[m.SOURCE_SCENES[2]]['hard_lt8']['accuracy']=.6
  for v in m.SOURCE_SCENES[1:]:S[v]['easy_ge8']['accuracy']=.598
  summary['BANK_STRAT30']['full_V_clean']=.979;q=m.source_conditions(summary,quality);self.assertTrue(q['source_criteria_pass']);self.assertAlmostEqual(q['hard_lt8_S_minus_I_equal_scene_accuracy_pp'],.5);self.assertFalse(q['automatic_extension']);json.dumps(q,allow_nan=False);summary['NOAS30']['full_V_clean']=.999;self.assertTrue(m.source_conditions(summary,quality)['source_criteria_pass']);S[m.SOURCE_SCENES[1]]['hard_lt8']['accuracy']=.6099;self.assertTrue(m.source_conditions(summary,quality)['fixed_stage1_closed'])
 def test_source_physics_bins_counts_and_truth_closed(self):
  with temporary() as tmp:
   p=Path(tmp);source_fixture(p);original=m.read
   def guarded(path):
    if Path(path).name in ('truth.json','scores.json','automatic_summary.json'):raise AssertionError('Formal read in source')
    return original(path)
   with mock.patch.object(m,'read',guarded):q=m.source_analysis(p)
   self.assertEqual(q['NN'],0);self.assertFalse(q['decision']['source_criteria_pass']);self.assertEqual(q['quality_groups']['BANK_STRAT30']['practical_mid_urban']['hard_lt8']['count'],1800);self.assertEqual(q['quality_groups']['BANK_STRAT30']['practical_mid_urban']['easy_ge8']['count'],1800);self.assertTrue(any(z['dimension']=='state_lock' for z in q['metrics']));self.assertTrue(any(z['dimension']=='added_noise_ratio' and z['stratum']=='8-15' for z in q['metrics']))
 def test_old_audit_reused_two_new_tails_shared_parent_once(self):
  with temporary() as tmp:
   p=Path(tmp);parent=dict(student_train_NN=3806820,student_eval_NN=594002,teacher_train_NN=0,teacher_eval_NN=396900,native_training_and_validation_NN=4797722,wall_seconds=1.);oldcost=dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900,native_training_and_validation_NN=7960684,new_epochs=30,new_updates=6660,wall_seconds=2.);curves=[dict(epoch=i) for i in range(1,31)];put(p/'runs'/m.PRIOR_E50_RUN/'analysis_auto_r01/training_logs.json',dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',rows={a:dict(cost=oldcost,epoch_curves=curves) for a in m.CONDITIONS[:2]}));lookup=dict(m.rows(p));put(lookup['AS30']/'stage20/result.json',parent);put(lookup['AS30']/'source/actual_native_args.json',dict(args={}));cost={k:oldcost[k]-parent[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','native_training_and_validation_NN')}
   with mock.patch.object(m,'audit_bank',return_value=(dict(observations=252000),{})),mock.patch.object(m,'parse_new_logs',side_effect=lambda *args:(dict(cost=cost,epoch_curves=curves[20:]),dict(args={}))),mock.patch.object(m.logs,'records',side_effect=AssertionError('Old stream reopened')):q=m.training_analysis(p)
   self.assertEqual(q['new_updates'],4440);self.assertEqual(q['shared_parent_count'],1);self.assertTrue(q['reference_comparability']['BANK_IID30']['four_counters_equal'])
 def test_all4_all7_before_formal_truth(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);source=formal_fixture(p);(dict(m.rows(p))['BANK_IID30']/'scoring/complete.json').unlink();original=m.read;opened=[]
   def guarded(path):
    if Path(path).name=='truth.json':opened.append(True);raise AssertionError('Early truth')
    return original(path)
   with mock.patch.object(m,'read',guarded),self.assertRaises(m.AwaitingArtifacts):m.formal_analysis(p,p/'truth.json',source)
   self.assertEqual(opened,[])
 def test_formal504CM84partition_sixpairs_no_factorial(self):
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);q=m.formal_analysis(p,p/'truth.json',formal_fixture(p));self.assertEqual(q['independent_confusion_matrices'],504);self.assertEqual(q['partition_reconciliations'],84);self.assertEqual(len(q['paired']),6);self.assertEqual(q['paired']['BANK_STRAT30__BANK_IID30']['clean']['repairs'],1);self.assertEqual(q['paired']['BANK_STRAT30__BANK_IID30']['clean']['new_errors'],1);self.assertNotIn('interaction',q);self.assertFalse(q['repeatable_clean_positive_from_As_proven'])
 def test_wait_real_handles_terminal_missing(self):
  with temporary() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();slept=[];m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:bool(slept),probe=lambda p,r:([123],False),sleep=lambda _:slept.append(True))
   with self.assertRaises(m.AwaitingArtifacts):m.wait_until(p,p/'release','SOURCE',out,readiness=lambda _:False,probe=lambda p,r:([],False),sleep=lambda _:self.fail('Wait on dead handle'))
 def test_r03_existing_bank_Ls_scope_null_timers_and_new_output(self):
  import cvs_qualityrisk30_publish_analysis_20261018 as pub
  import cvs_qualityrisk30_collect_20261018 as collect
  with temporary() as tmp:
   p=Path(tmp);_,bank=bank_fixture(p);path=p/'runs'/m.RUN/'proposal_bank/complete.json';d=m.read(path);self.assertEqual(d['source_role_key'],'L_s');self.assertEqual(d['source_role_counts'],dict(L_s=6300,U_s=56700,V=27000));self.assertEqual(len(d['role_ids']),6300)
   for k in ('wall_seconds','raw_dataset_cache_cpu_wall_seconds','simulation_adapter_cpu_wall_seconds','IQ_cache_write_cpu_wall_seconds','metadata_write_cpu_wall_seconds'):d[k]=None
   d.update(timing_status='UNAVAILABLE_R02_FINALIZER_EXCEPTION',recovered_existing_bank=True,recovery_release='r03',generation_release='r02',no_additional_propagation=True,finalization_cpu_wall_seconds=.1,IQ_payload_reads=0,timing='original generation timings were not persisted;not imputed from file time bounds')
   put(path,d);public,_=m.audit_bank(p);self.assertEqual(public['propagation_simulations'],252000);self.assertTrue(public['bank_cost_counted_once']);self.assertEqual(public['source_role_key'],'L_s');self.assertIsNone(public['wall_seconds']);self.assertTrue(all(v is None for v in public['CPUwall'].values()));self.assertEqual(len(public['missing_timing_fields']),5);self.assertTrue(public['timing_missing_not_imputed']);self.assertEqual(public['generation_release'],'r02');self.assertNotIn('role_ids',public)
   cfg=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40);self.assertEqual(Path(cfg['release']).name,'cvs_F_qualityrisk30_analysis_20261018_r02');self.assertEqual(Path(cfg['output']).name,'analysis_auto_r02');self.assertEqual(Path(cfg['training_release']).name,m.TRAINING_RELEASE);self.assertEqual(collect.OUTPUT,'analysis_auto_r02')

 def test_r02_preserved_r01_technical_cost_once(self):
  import cvs_qualityrisk30_publish_analysis_20261018 as pub
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);metadata(p,84);cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={a:dict(cost=cost) for a in m.CONDITIONS[2:]},bank=dict(observations=252000));r=p/'runs'/m.RUN
   put(r/'synthetic_smoke.json',dict(status='PASS',actual_L_NN=768,real_checkpoint_NN=2,total_NN=770,extra_CE_operator_calls=4,extra_CE_logit_packets=512,target_access=False));put(r/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False))
   baseline=m.cost_summary(p,training);h=dict(total_NN=258,real_parent_forward_packets=2,synthetic_ordinary_forward_packets=256,optimizer_updates=0,real_IQ=0,propagation_simulations=0);put(r/'technical_history/r01/cost.json',h);actual=m.cost_summary(p,training)
   self.assertEqual(actual['new_total_NN'],baseline['new_total_NN']+258);self.assertEqual(actual['scientific_new_training_and_readouts_plus_smoke_NN'],baseline['new_total_NN']);self.assertEqual(actual['new_shared_smoke_NN'],770);self.assertEqual(actual['technical_failure_NN'],258);self.assertEqual(len(actual['technical_failure_history']),1);self.assertTrue(actual['technical_history_counted_once']);self.assertTrue(m.TRAINING_RELEASE.endswith('_r03'));self.assertEqual(pub.TRAINING_RELEASE,m.TRAINING_RELEASE)
   h['optimizer_updates']=1;put(r/'technical_history/r01/cost.json',h)
   with self.assertRaises(ValueError):m.cost_summary(p,training)

 def test_pipeline_publish_cost_collect_single_archive(self):
  import cvs_qualityrisk30_publish_analysis_20261018 as pub
  import cvs_qualityrisk30_collect_20261018 as collect
  with temporary() as tmp,mock.patch.object(m,'COUNT',84):
   p=Path(tmp);config=pub.analysis_config(dict(project=str(p),python='python',user='ordinary',host='host'),'a'*40,release='cvs_F_qualityrisk30_analysis_20261018_r07',output='analysis_auto_r07');self.assertEqual(Path(config['training_release']).name,m.TRAINING_RELEASE);source=formal_fixture(p);source['metrics']=[];cost=dict(student_train_NN=2271960,student_eval_NN=324002,teacher_train_NN=0,teacher_eval_NN=567000,native_training_and_validation_NN=3162962,wall_seconds=1.);training=dict(rows={a:dict(cost=cost) for a in m.CONDITIONS},parent20={a:dict(cost=cost) for a in m.CONDITIONS[2:]},bank=dict(observations=252000));put(p/'runs'/m.RUN/'synthetic_smoke.json',dict(status='PASS',actual_L_NN=768,real_checkpoint_NN=2,total_NN=770,extra_CE_operator_calls=4,extra_CE_logit_packets=512,target_access=False));put(p/'runs'/m.RUN/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,target_query=False));argv=['analysis','--project',config['project'],'--release',config['training_release'],'--output',config['output'],'--truth',str(p/'truth.json'),'--code-commit',config['commit']]
   with mock.patch.object(sys,'argv',argv),mock.patch.object(m,'wait_until'),mock.patch.object(m,'training_analysis',return_value=training),mock.patch.object(m,'source_analysis',return_value=source):m.main()
   out=Path(config['output']);self.assertEqual(m.read(out/'complete.json')['new_model_rows'],2);self.assertEqual(m.read(out/'costs.json')['new_total_NN'],2*(3162962+12600+590)+770)
   archive=p/'aggregates.tar.gz';packed=collect.pack_public(out,archive,'final');self.assertEqual(collect.extract_public(archive,p/'collected','final',packed['archive_SHA256'])['file_count'],11);self.assertIn('cvs_qualityrisk30_analysis_20261018.py',pub.FILES)
if __name__=='__main__':unittest.main(verbosity=2)
