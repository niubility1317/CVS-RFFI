"""Boundary checks for read-only root routing; no IQ, truth, or checkpoints."""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
from . import merged_analysis as a

class AnalysisBoundary(unittest.TestCase):
 def test_full_audit_preserved_and_compiles(self):
  source=a.audit_script();compile(source,'merged-audit','exec')
  for text in ("len(records)==steps","len(epochs)==e","objective_error<=2e-5","validate_checkpoint(cp,config)","torch.equal(after,perturbed)","record['confusion']==cm.tolist()","all(torch.equal(base[k],other[k])","len(recounted)==324","len(pairs)==27","actual_totals==d.totals()"):
   self.assertIn(text,source)
  self.assertLess(source.index("config,ids=preflight(rid)"),source.index("\ny,rx,days=truth_arrays("))
  self.assertNotIn('subprocess.Popen',source)
  self.assertNotIn('matrix[',source)
 def test_exact_physical_root_routing(self):
  source=a.audit_script();tree=ast.parse(source)
  function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_row')
  rows=a.original_audit.d.rows();new_ids=set(a.recovery.REMAINING)
  old=Path('/original');new=Path('/technical');release=Path('/immutable-model')
  ns=dict(d=SimpleNamespace(ROOT=release),row_base={r['row_id']:new if r['row_id'] in new_ids else old for r in rows},recovery_ids=new_ids,c=dict(recovery_run='new',recovery_owner='new-owner'),original_run='old',original_owner='old-owner',release=release)
  exec(compile(ast.Module(body=[function],type_ignores=[]),'routing','exec'),ns)
  for row in rows:
   expected=row['row_id'] in new_ids;self.assertEqual(ns['select_row'](row['row_id']),new if expected else old)
   self.assertEqual(ns['d'].RUN,'new' if expected else 'old')
   self.assertEqual(ns['d'].OWNER,'new-owner' if expected else 'old-owner')
   self.assertEqual(ns['d'].ROOT,release)
  self.assertEqual(sum(r['row_id'] in new_ids for r in rows),4)
 def test_terminal_guard(self):
  old=dict(live_processes=[],rows=[dict(scored='MODEL_PREDICTED_SCORED') for _ in range(14)]+[dict(scored=None) for _ in range(4)])
  new=dict(completion=dict(status='ARTIFACTS_COMPLETE'),failure=None,dispatcher=None,rows=[dict(process=None,completion=dict(status='MODEL_PREDICTED_SCORED')) for _ in range(4)])
  self.assertTrue(a.ready(old,new))
  old['live_processes']=[dict(pid=1)];self.assertFalse(a.ready(old,new));old['live_processes']=[]
  new['completion']=None;self.assertFalse(a.ready(old,new));new['completion']=dict(status='ARTIFACTS_COMPLETE')
  new['dispatcher']=dict(pid=2);self.assertFalse(a.ready(old,new));new['dispatcher']=None
  new['failure']=dict(error='technical');self.assertFalse(a.ready(old,new));new['failure']=None
  new['rows'][0]['completion']=None;self.assertFalse(a.ready(old,new))
if __name__=='__main__':unittest.main()
