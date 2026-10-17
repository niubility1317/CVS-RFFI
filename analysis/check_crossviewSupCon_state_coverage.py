import unittest
from types import SimpleNamespace
from collections import Counter
import cvs_crossviewSupCon_state_coverage_20261018 as a
class Cases(unittest.TestCase):
 def setUp(self):
  self.meta=dict(source_RX=list(a.RX),source_days=list(a.DAYS),source_TX_count=6,equalized_list=[0,1],metadata_records=[dict(TX=t,RX=r,day=d,eq_index=1,shape=[1000,256,2]) for t in range(6) for r in a.RX for d in a.DAYS]);self.rows=a.layout(self.meta);self.opaque=lambda q:str((q.tx_i,q.rx_i,q.day_i,q.eq_i,q.sig_i))
 def test_layout_matches_native_loop_and_full_base_pool(self):
  self.assertEqual(len(self.rows),90);self.assertEqual(self.rows[-1][1],90000);self.assertEqual(self.rows[15][2],1)
 def test_RX_day_sig_only_is_ambiguous_base_resolves_TX(self):
  allowed={self.opaque(SimpleNamespace(tx_i=t,rx_i=1,day_i=1,eq_i=1,sig_i=3)) for t in range(6)}
  for t in range(6):self.assertEqual(a.decode_L([1,1,1,3,t*15000+3],'L',self.rows,allowed,self.opaque)[0],t)
 def test_hidden_U_rejected_before_label_decode(self):
  def bad(*x):raise AssertionError('Must never evaluate U labels')
  with self.assertRaisesRegex(ValueError,'Never decode'):a.decode_L([1,1,1,3,3],'U',self.rows,set(),bad)
 def test_wrong_meta_or_role_handle_rejected(self):
  for p in [[1,3,1,3,3],[1,1,2,3,3],[1,1,1,4,3],[1,1,1,3,90003],[-1,1,1,3,3]]:
   with self.assertRaises(ValueError):a.decode_L(p,'L',self.rows,set(),self.opaque)
  with self.assertRaisesRegex(ValueError,'L6300'):a.decode_L([1,1,1,3,3],'L',self.rows,set(),self.opaque)
 def test_shape_change_not_silently_assumed(self):
  self.meta['metadata_records'][0]['shape'][0]=999
  with self.assertRaises(ValueError):a.layout(self.meta)
 def test_native_marginals_sum_over_TX_only(self):
  c=Counter({(0,1,2,'mid','LOS'):3,(1,1,2,'mid','LOS'):4,(1,3,2,'mid','LOS'):2});self.assertEqual(a.marginal(c),Counter({(1,2,'mid','LOS'):7,(3,2,'mid','LOS'):2}))
if __name__=='__main__':unittest.main()
