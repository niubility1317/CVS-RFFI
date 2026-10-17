import tempfile,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cvs_crossviewSupCon_E30_results_20261018 as a
class Cases(unittest.TestCase):
 def values(self):
  c=dict(ORD128=.98,DUP192=.98,SAT192=.98);u={k:[[.8,.6,.5],[.8,.6,.5]] for k in a.ARMS};u['SAT192']=[[.8,.61,.51],[.8,.61,.51]];return c,u
 def relation(self,mode='SAT192',active=True):
  n=128 if mode=='ORD128' else 192;ds=[dict(phase=p,level=l,mode=mode,shape=[n,4,8],total_anchors=n,ordinary_anchors=128,additional_anchors=n-128,same_RX_positive_pairs=0,true_satellite_anchors=64 if mode=='SAT192' else 0,valid_anchors=n,positive_min=2,positive_max=10,cross_view_positive_pairs=40 if n>128 else 0) for p in ('first','second') for l in ('time_down','t2.pool')]
  return dict(SupCon_weight=.1 if active else 0.,SupCon_details=ds if active else [],crossview_mode=mode,new_losses=0,extra_U_NN=0)
 def test_metric_confusion_and_macroF1(self):
  m=a.base.metric([0,1,1,2],[0,0,1,2]);self.assertEqual(m['correct'],3);self.assertAlmostEqual(m['macro_f1'],7/18);self.assertEqual(sum(map(sum,m['confusion'])),4)
 def test_fixed_source_all_controls_and_draws(self):
  c,u=self.values();self.assertTrue(a.independent_source_decision(c,u)['source_qualified']);u['ORD128'][1][2]=.509;self.assertFalse(a.independent_source_decision(c,u)['source_qualified']);u['ORD128'][1][2]=.5;u['SAT192'][0][1]=.6099;self.assertFalse(a.independent_source_decision(c,u)['source_qualified'])
 def test_clean_preservation_and_source_shape(self):
  c,u=self.values();c['SAT192']=.9774;self.assertFalse(a.independent_source_decision(c,u)['source_qualified']);u['SAT192']=[u['SAT192'][0]]
  with self.assertRaises(ValueError):a.independent_source_decision(c,u)
  c,u=self.values();u['DUP192'][0][0]=float('nan')
  with self.assertRaises(ValueError):a.independent_source_decision(c,u)
 def test_original_and_duplicate_not_satellite_evidence(self):
  for mode,total,sat in [('ORD128',512,0),('DUP192',768,0),('SAT192',768,256)]:
   r=a.validate_relation_step(self.relation(mode),mode);self.assertEqual(r['total_anchors'],total);self.assertEqual(r['true_satellite_anchors'],sat)
 def test_reject_wrong_view_rx_or_untagged_entry(self):
  for field,value in [('same_RX_positive_pairs',1),('true_satellite_anchors',0),('total_anchors',128),('mode','DUP192')]:
   r=self.relation();r['SupCon_details'][0][field]=value
   with self.assertRaises(ValueError):a.validate_relation_step(r,'SAT192')
  r=self.relation();r['SupCon_details']=r['SupCon_details'][:3]
  with self.assertRaises(ValueError):a.validate_relation_step(r,'SAT192')
 def test_warmup_has_no_aux_entries(self):
  self.assertEqual(a.validate_relation_step(self.relation(active=False),'SAT192'),{})
  r=self.relation();r['SupCon_weight']=0
  with self.assertRaises(ValueError):a.validate_relation_step(r,'SAT192')
 def test_no_double_count_of_source_readout_or_old_U_check(self):
  counts=dict(student_train_NN=10,student_eval_NN=20,teacher_train_NN=0,teacher_eval_NN=30);proof=dict(status='SOURCE_READOUT_COMPLETE',NN=12600,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False);v=a.source_cost_breakdown(counts,proof);self.assertEqual(v['source_identity_NN'],12660);self.assertEqual(v['first_U_qualification_NN'],0);self.assertEqual(v['extra_U_student_NN'],0)
  proof['NN']=14400
  with self.assertRaises(ValueError):a.source_cost_breakdown(counts,proof)
 def test_formal_truth_closed_before_complete_all7(self):
  def read(p):
   if p.name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False)
   return dict(checks={})
  with tempfile.TemporaryDirectory() as tmp,patch.object(a,'read',side_effect=read),patch.object(a,'completed',side_effect=ValueError('Incomplete all7')),patch.object(a.base,'collect_row') as collect:
   with self.assertRaisesRegex(ValueError,'Incomplete all7'):a.final_analysis(Path(tmp),Path(tmp)/'source',Path(tmp)/'truth.json')
   collect.assert_not_called()
if __name__=='__main__':unittest.main()
