"""Pure synthetic checks;no IQ,checkpoint,training,source or target files."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import cvs_impulse_factorial_E30_results_20261018 as a
p=Path(__file__).resolve().parents[1]/'experiments/cvs_noise_shape_factorial43/criteria.py';s=importlib.util.spec_from_file_location('fixed43criteria',p);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class Checks(unittest.TestCase):
 def metrics(self):
  clean={r:.8 for r in a.ARMS};clean.update(B10_BG=.804,B11_BG=.806);urban={r:[[.8,.5,.4],[.8,.5,.4]] for r in a.ARMS};urban['B01']=[[.8,.6,.5],[.8,.6,.5]];urban['B11_WHITE']=[[.8,.61,.51],[.8,.61,.51]];urban['B11_BG']=[[.8,.63,.53],[.8,.63,.53]];rx={r:{str(k):.8 for k in (1,3,4,6,8)} for r in a.ARMS};return clean,urban,rx
 def test_prospective_equivalence(self):
  rng=np.random.default_rng(4301);cases=[self.metrics()]
  for _ in range(250):cases.append(({r:float(rng.uniform(.7,.99)) for r in a.ARMS},{r:rng.uniform(.3,.95,(2,3)).tolist() for r in a.ARMS},{r:{str(k):float(rng.uniform(.7,.99)) for k in (1,3,4,6,8)} for r in a.ARMS}))
  for z in cases:self.assertEqual(a.independent_source_decision(*z)['checks'],c.decide(*z)['checks'])
  self.assertTrue(a.independent_source_decision(*cases[0])['source_qualified'])
 def test_source_boundary_and_bad_inputs(self):
  for offset in (-2e-12,0,2e-12):
   z=list(self.metrics());z[0]['B10_BG']=z[0]['B00']+.002+offset;self.assertEqual(a.independent_source_decision(*z)['checks'],c.decide(*z)['checks'])
  for change in ('missing','nan','rx'):
   z=list(self.metrics())
   if change=='missing':z[0].pop('B00')
   elif change=='nan':z[1]['B10_BG'][0][1]=float('nan')
   else:z[2]['B00'].pop('1')
   with self.assertRaises(ValueError):a.independent_source_decision(*z)
 def test_effect_formula_known_interaction(self):
  values=dict(B00=.80,B10_WHITE=.82,B10_BG=.83,B01=.90,B11_WHITE=.91,B11_BG=.94);r={k:dict(views={v:dict(accuracy=x,macro_f1=x,worst_RX_accuracy=x) for v in a.base.VIEWS}) for k,x in values.items()};rows=a.effect_table(r);q=next(v for v in rows if v['shape']=='BG' and v['metric']=='accuracy' and v['view']=='clean')
  for key,value in dict(Ag_independent_pp=3,As_independent_pp=10,joint_net_pp=14,As_given_Ag_pp=11,Ag_given_As_pp=4,interaction_pp=1,BG_minus_WHITE_Ag_pp=1,BG_minus_WHITE_joint_pp=3).items():self.assertAlmostEqual(q[key],value,places=10)
  self.assertEqual(len(rows),42)
 def test_truth_stays_closed_until_all_six_frozen(self):
  calls=[];old_read=a.read;old_completed=a.completed;old_collect=a.base.collect_row
  def read(p):
   calls.append(str(p))
   name=Path(p).name
   if name in ('independent_source_decision.json','source_matrix_decision.json'):return dict(checks={})
   if name=='complete.json':return dict(status='SOURCE_ANALYZED',formal_scores_read=False)
   return {}
  def completed(p):
   if p.name.startswith('B11_BG'):raise ValueError('Last required row not frozen')
  def collect(*args):raise AssertionError('Truth reached before allsix guard')
  a.read=read;a.completed=completed;a.base.collect_row=collect
  try:
   with self.assertRaisesRegex(ValueError,'Last required row'):a.final_analysis(Path('/unused-out'),Path('/unused-source'),Path('/forbidden-truth'))
   self.assertFalse(any('forbidden-truth' in p for p in calls))
  finally:a.read=old_read;a.completed=old_completed;a.base.collect_row=old_collect;a.PARSED.clear()
 def test_source_cost_scope(self):
  counter=dict(student_train_NN=10,student_eval_NN=20,teacher_train_NN=0,teacher_eval_NN=30);r=a.source_cost_breakdown(counter,dict(status='SOURCE_READOUT_COMPLETE',NN=12600,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False));self.assertEqual(r['source_identity_NN'],12660);self.assertEqual(r['native_training_and_validation_identity_NN'],60)
 def test_metadata_never_decodes_hidden_U(self):
  import cvs_impulse_factorial_coverage_20261018 as cov
  from types import SimpleNamespace
  rows=cov.old.layout(dict(source_RX=[1,3,4,6,8],source_days=[1,2,3],source_TX_count=6,equalized_list=[0,1],metadata_records=[dict(TX=t,RX=r,day=d,eq_index=1,shape=[1000,256,2]) for t in range(6) for r in (1,3,4,6,8) for d in (1,2,3)]));opaque=lambda q:(q.tx_i,q.rx_i,q.day_i,q.eq_i,q.sig_i);allowed={(5,8,3,1,999)}
  self.assertEqual(cov.old.decode_L([1,8,3,999,89999],'L',rows,allowed,opaque),(5,(5,8,3,1,999)))
  with self.assertRaisesRegex(ValueError,'hidden U'):cov.old.decode_L([1,8,3,999,89999],'U',rows,allowed,opaque)
  with self.assertRaisesRegex(ValueError,'L6300'):cov.old.decode_L([1,8,3,998,89998],'L',rows,allowed,opaque)
 def test_coverage_requires_source_recount_before_output(self):
  import cvs_impulse_factorial_coverage_20261018 as cov
  old=cov.read;cov.read=lambda p:dict(status='INCOMPLETE',formal_scores_read=False)
  try:
   with tempfile.TemporaryDirectory() as td:
    target=Path(td)/'preserved'
    with self.assertRaisesRegex(ValueError,'source recount'):cov.run(target,Path(td)/'source')
    self.assertFalse(target.exists())
  finally:cov.read=old
if __name__=='__main__':unittest.main()
