import json,tempfile,unittest,math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_continuous52_results_20261018 as a
import cvs_continuous52_pipeline_20261018 as p

class Checks(unittest.TestCase):
 def rates(self):
  r={k:dict(V=dict(source_val_accuracy=.9,source_val_rx_accuracy={str(rx):.9 for rx in [1,3,4,6,8]}),panel=[[.8,.5,.4],[.8,.5,.4]]) for k in [x['key'] for x in a.rows()]};r['CONT_B01']['panel']=[[.8,.52,.42],[.8,.52,.42]];return r
 def test_source_fixed_conditions(self):
  r=self.rates();self.assertTrue(a.qualify(r)['source_qualified']);r['CONT_B01']['V']['source_val_accuracy']=.899;self.assertFalse(a.qualify(r)['source_qualified']);r=self.rates();r['CONT_B01']['panel'][1][2]=float('nan')
  with self.assertRaises(ValueError):a.qualify(r)
 def test_interaction_is_trajectory_not_Ag_or_seed(self):
  self.assertEqual(len(a.rows()),4);self.assertEqual({x['seed'] for x in a.rows()},{2026092701});self.assertAlmostEqual(a.interaction(dict(STAGED_B00=.8,STAGED_B01=.85,CONT_B00=.82,CONT_B01=.90)),.03)
  with self.assertRaises(ValueError):a.interaction(dict(STAGED_B00=.8))
 def test_fixed_reference_cohort(self):
  q=a.clean_cohort_effect(np.array([0,1,2,3]),np.array([0,0,2,3]),np.array([0,1,0,3]),np.array([1,1,2,0]));self.assertEqual((q['repairs'],q['new_errors'],q['net']),(2,1,1));self.assertEqual(q['fixed_B00_clean_correct_count'],3)
  q=a.clean_cohort_effect(np.array([0]),np.array([1]),np.array([0]),np.array([1]));self.assertIsNone(q['B01_retention'])
 def test_truth_closed_if_last_row_incomplete(self):
  seen=[]
  def guard(row):
   seen.append(row['key'])
   if row==a.rows()[-1]:raise ValueError('unfinished')
  with patch.object(a,'read',return_value=dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=1)),patch.object(a,'formal_complete',side_effect=guard),patch.object(a.base,'collect_row') as collect:
   with self.assertRaises(ValueError):a.final_analysis(Path('/unused'),Path('/source'),Path('/closedtruth'))
   collect.assert_not_called();self.assertEqual(len(seen),4)
 def test_full_synthetic_fourrow_aggregate_and_cost(self):
  seen=[];ys=np.array([0,1,2,3]);ids=np.array(['p0','p1','p2','p3'])
  def guard(row):seen.append(row['key']);return {},{},{}
  def read(path):
   path=Path(path)
   if path.name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False,model_seed_count=1,source_confusions=4)
   if path.name.endswith('_training.json'):return dict(cost=dict(source_identity_NN=100,prediction_identity_NN=200),curve=[])
   if path.name=='synthetic_smoke.json':return dict(NN=1536,matched_initial_state_per_seed=True)
   if path.name=='real_checkpoint_smoke.json':return dict(NN=2)
   if path.name=='source_decision.json':return dict(source_qualified=False,formal_scores_read=False)
   raise AssertionError(str(path))
  def collect(key,truth):
   self.assertEqual(len(seen),4);pred={v:np.array([0,1,2,0]) if key=='STAGED_B00' else ys.copy() for v in a.VIEWS};views={v:dict(a.base.metric(ys,pred[v]),worst_RX_accuracy=.5,mean_true_margin_over30=.1) for v in a.VIEWS};return dict(views=views,metrics=[dict(views[v],view=v) for v in a.VIEWS],source=dict(cost=dict(source_identity_NN=100,prediction_identity_NN=200),curve=[])),dict(ids=ids,y=ys,pred=pred)
  with tempfile.TemporaryDirectory() as td,patch.object(a,'formal_complete',side_effect=guard),patch.object(a,'read',side_effect=read),patch.object(a.base,'collect_row',side_effect=collect),patch.object(a,'plot_final'):
   q=a.final_analysis(Path(td)/'final',Path('/syntheticsource'),Path('/synthetictruth'));self.assertEqual(q['new_total_NN'],2138);self.assertEqual(q['reused_total_NN'],600);self.assertEqual(q['model_seed_count'],1);self.assertFalse(q['Ag_proved']);self.assertTrue(q['Ag_factorial_interaction_not_estimated']);self.assertFalse(q['goal_complete']);self.assertEqual(len(q['paired_effects']),28);self.assertEqual(len(q['trajectory_x_As_interaction']),7);self.assertEqual(q['trajectory_x_As_interaction'][0]['trajectory_x_As_accuracy_pp'],-25)
   for row in a.rows():self.assertEqual(q['rows'][row['key']]['views']['clean']['arm'],row['arm'])
 def test_sequence_concat_does_not_decode_U(self):
  import gzip
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);old=root/'old';new=root/'new'
   for x in [old/'stage20',old/'source',new/'source']:x.mkdir(parents=True)
   def put(path,values):
    with gzip.open(path,'wt') as f:
     for v in values:f.write(json.dumps(dict(role='U',ids=[['opaque',v]]))+'\n')
   put(old/'stage20/training_sequence.jsonl.gz',[1]);put(old/'source/training_sequence.jsonl.gz',[2]);put(new/'source/training_sequence.jsonl.gz',[1,2]);left=dict(root=old,reused=True);right=dict(root=new,reused=False);self.assertTrue(a.sequence_compare(left,right)['identical']);put(new/'source/training_sequence.jsonl.gz',[1,3]);self.assertEqual(a.sequence_compare(left,right)['mismatched_records'],1)
 def test_macroF1_counts_all6classes(self):self.assertAlmostEqual(a.base.metric([0,1,2,3],[0,1,0,3])['macro_f1'],4/9)
 def test_pipeline_wait_has_verified_liveness_semantics(self):
  class EndWait(Exception):pass
  with tempfile.TemporaryDirectory() as td:
   p.wait_until('SOURCE',[],Path(td),ready=lambda _:True,probe=lambda:(_ for _ in ()).throw(AssertionError('completed needs no liveprocess')))
   for handles in [([123],False),([],True)]:
    with self.assertRaises(EndWait):p.wait_until('SOURCE',[],Path(td),ready=lambda _:False,probe=lambda:handles,failure=lambda:None,sleep=lambda _:(_ for _ in ()).throw(EndWait()))
   with self.assertRaises(p.IncompleteRun):p.wait_until('SOURCE',[],Path(td),ready=lambda _:False,probe=lambda:([],False),failure=lambda:None,sleep=lambda _:None)
 def test_source_and_coverage_never_receive_truth(self):
  for stage in ['source','coverage']:self.assertNotIn('--truth',p.command(stage,Path('/out'),Path('/closedtruth')))
  self.assertIn('--truth',p.command('final',Path('/out'),Path('/closedtruth')))
 def test_full_single_segment_parser(self):
  from types import SimpleNamespace
  a.PARSED.clear();N=6660
  cfg=dict(Ag_mode='NONE',As=True,L_NN_per_step=768,total_L_loss_scale=1.25,noise=dict(SNR_added_db=30.,impulse_probability=.02,impulse_excess_variance=49.))
  r0=dict(optimizer_steps=N,L_NN=N*768,L_As=N*64,Ag_packets=0,Ag_control_NN=N*256,extra_U_NN=0,U_As=0,teacher_train_NN=0,source_RX_holdout=None,student_train_NN=6078780,student_eval_NN=1000,teacher_eval_NN=963900,L_channel=dict(enabled=True,strata=[]))
  done=dict(config=cfg,elapsed_seconds=30);q=dict(status='SOURCE_READOUT_COMPLETE',NN=12600,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False)
  def read(path):
   if path.name=='result.json':return r0
   if path.name=='completion.json':return done
   if path.name=='complete.json':return q
   raise AssertionError(str(path))
  def entries(path):
   if path.name=='training_sequence.jsonl.gz':
    for i in range(1,N+1):
     yield dict(role='L',ids=[[1,(i*128+j)%6300] for j in range(128)])
     if i>13*222:
      n=124 if i%222==0 else 256;start=((i-1)%222)*256;yield dict(role='U',ids=[[1,(start+j)%56700] for j in range(n)])
    return
   if path.name=='epoch_metrics.jsonl':
    for ep in range(1,31):yield dict(epoch=ep,lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(ep-1)/29))/2,epoch_time_s=1,val_tx_acc=99,train_grad_backbone=1,train_grad_total=1)
    return
   for i in range(1,N+1):
    e=(i-1)//222+1;w=.1*min(max((e-5)/10,0),1);loss=.2475+w
    if path.name=='reference_steps.jsonl.gz':
     ds=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ['first','second'] for l in ['time_down','t2.pool']] if w else []
     yield dict(step=i,cumulative_step=i,L_forwards=6,As_packets=64,Ag_packets=0,Ag_mode='NONE',extra_control_packets=128,ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,total_L=loss,base_L_before_shape=loss-.0825,common_CE=.1,common_KL=.01,common_RD=.11,extra_CE=.2,extra_KL=.02,extra_RD=.22,SupCon=1,SupCon_weight=w,ordinary_raw_MSE=.001,crossview_mode='ORD128',new_losses=0,extra_U_NN=0,SupCon_details=ds)
    elif path.name=='shape_steps.jsonl.gz':yield dict(step=i,cumulative_step=i,mode='NONE',NN=256,packets=128,applied=0,weight=.25,same_IQ_each_pair=True,new_boundary_samples=0,relative_noise_RMS=0,noise_peak_fraction_mean=0,SNR_added_db=None,shape_CE=.3,shape_KL=.03,shape_RD=.33)
    elif path.name=='unit_steps.jsonl.gz':yield dict(step=i,cumulative_step=i,physical_clip_calls=i)
    elif path.name=='step_metrics.jsonl.gz':yield dict(step=i,epoch=e,metrics={'concat/L_forwards':4,'concat/As_packets':64,'concat/total_L':loss-.0825,'shape/RD':.33,'train/loss':loss,'train/pseudo_total':(124 if i%222==0 else 256) if e>13 else 0,'train/pseudo_selected':0,'train/pseudo_truth_available':0})
    else:raise AssertionError(str(path))
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'B01-s2026092701';(root.parent/'logs').mkdir();(root.parent/'logs'/(root.name+'.log')).write_text('SYNTHETIC FULL30 LOG\n')
   with patch.object(a,'completed_training',return_value=done),patch.object(a,'read',side_effect=read),patch.object(a,'lines',side_effect=entries),patch.object(a,'raw_telemetry',return_value=dict(records=30)):
    z=a.parse_continuous(root);self.assertEqual(len(z['curve']),30);self.assertEqual(z['full_records']['actual_steps'],6660);self.assertEqual(z['full_records']['loader_packets'],dict(L=852480,U=963900));self.assertEqual(z['full_records']['unique_physical'],dict(L=6300,U=56700));self.assertEqual(z['source_result']['segments'],[r0]);self.assertFalse(z['cost']['own_stage20_is_included'])
    a.PARSED.clear();r0['L_NN']-=1
    with self.assertRaises(ValueError):a.parse_continuous(root)
if __name__=='__main__':unittest.main()
