"""Protocol-negative analysis wiring: synthetic files only, no IQ/CP/NN."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import cvs_cycleorder_factorial_E30_results_20261017 as mod
class Stop(Exception):pass
class Cases(unittest.TestCase):
 def test_all_rows_complete_before_any_source_or_truth_analysis(self):
  events=[]
  def completed(root):
   events.append('complete')
   if len(events)==4:raise Stop()
  with tempfile.TemporaryDirectory() as td:
   target=Path(td)/'result'
   with patch.object(sys,'argv',['analysis','--truth',str(Path(td)/'DO_NOT_OPEN_TRUTH'),'--output',str(target),'--commit','test']),patch.object(mod.base,'completed',side_effect=completed),patch.object(mod,'source_readout') as source,patch.object(mod.base,'collect_row') as collect:
    with self.assertRaises(Stop):mod.main()
    source.assert_not_called();collect.assert_not_called();self.assertFalse(target.exists())
 def test_source_rule_precedes_first_truth_consumer(self):
  events=[];source_result={'sentinel':1}
  def source(out):events.append('source');return source_result
  def decide(out,sources):self.assertIs(sources,source_result);events.append('decide');return {'pass':False}
  def collect(*args):events.append('truth');raise Stop()
  with tempfile.TemporaryDirectory() as td:
   with patch.object(sys,'argv',['analysis','--truth',str(Path(td)/'DO_NOT_OPEN_TRUTH'),'--output',str(Path(td)/'result'),'--commit','test']),patch.object(mod.base,'completed',side_effect=lambda root:events.append('complete')),patch.object(mod,'source_readout',side_effect=source),patch.object(mod,'independent_source_rule',side_effect=decide),patch.object(mod.base,'collect_row',side_effect=collect):
    with self.assertRaises(Stop):mod.main()
  self.assertEqual(events,['complete']*4+['source','decide','truth'])
 def test_full_analysis_completes_without_rewriting_source_statistics(self):
  import copy,json,numpy as np
  with tempfile.TemporaryDirectory() as td:
   project=Path(td)/'project';run=project/'runs'/mod.RUN;run.mkdir(parents=True)
   (run/'complete.json').write_text('{}',encoding='utf8');(run/'real_checkpoint_smoke.json').write_text(json.dumps(dict(status='PASS',NN=2,real_IQ=0,target_query=False,training_updates=0)),encoding='utf8')
   launch=Path(str(run)+'-launch');launch.mkdir();(launch/'launch.json').write_text(json.dumps(dict(read_at=0)),encoding='utf8')
   for name in ('20261017-diagnostic-FcrossRX-cycleorder-sourceV-e30-s2026092701-m2-r01','20261017-diagnostic-FcrossRX-cycleorder-placement-sourceV-e30-s2026092701-m2-r01'):
    p=project/'runs'/name;p.mkdir();(p/'completion.json').write_text(json.dumps(dict(NN=0,wall_seconds=0,updates=0,target_access=False)),encoding='utf8')
   ids=np.asarray(['synthetic'+str(i) for i in range(6)]);y=np.arange(6,dtype=np.int64);pred={v:y.copy() for v in mod.base.VIEWS}
   for arm in ('B00','B01'):
    p=project/'runs'/mod.base.RUN30/(arm+'-s2026092701')/'prediction';p.mkdir(parents=True);np.savez(p/'predictions.npz',ids=ids,**pred)
   sources=dict(status='SOURCE_FACTORIAL_READOUT',rows=[],source_clean={a:{} for a in mod.ORDER},source_qualification={a:dict(NN=0,wall_seconds=0) for a in mod.ORDER})
   def source(out):
    with (out/'source_strata.csv').open('x',encoding='utf8') as f:f.write('synthetic source statistics\n')
    return sources
   def collect(arm,truth):
    return dict(views={v:dict(accuracy=.8,macro_f1=.8) for v in mod.base.VIEWS},metrics=[dict(arm=arm,view='clean',accuracy=.8)],source=dict(source_result={},cost={},curve=[dict(epoch=30)])),dict(ids=ids.copy(),y=y.copy(),pred=copy.deepcopy(pred))
   output=Path(td)/'result'
   with patch.object(sys,'argv',['analysis','--truth',str(Path(td)/'DO_NOT_OPEN_TRUTH'),'--output',str(output),'--commit','synthetic']),patch.object(mod.base,'PROJECT',project),patch.object(mod.base,'completed'),patch.object(mod,'source_readout',side_effect=source),patch.object(mod,'independent_source_rule',return_value={}),patch.object(mod.base,'collect_row',side_effect=collect),patch.object(mod.base,'compare_training_sequences',return_value={}),patch.object(mod,'local_trace',return_value={}),patch.object(mod,'native_telemetry',return_value={}),patch.object(mod,'same_local_stream',return_value={}),patch.object(mod.base,'plots',return_value=[]),patch.object(mod,'source_plots',return_value=[]),patch('builtins.print'):
    try:mod.main()
    except FileExistsError as exc:self.fail('Full analysis must not rewrite previously generated source statistics: '+str(exc))
   self.assertEqual((output/'source_strata.csv').read_text(encoding='utf8'),'synthetic source statistics\n')
   self.assertEqual(json.loads((output/'completion.json').read_text())['status'],'FACTORIAL_ANALYSIS_COMPLETE')
 def test_factorial_scale_and_rows(self):
  values={a:{v:x for v in mod.base.VIEWS} for a,x in [('B00',.5),('B10',.6),('B01',.7),('B11',.85)]}
  self.assertTrue(all(abs(x-5.)<1e-10 for x in mod.factorial(values).values()));self.assertTrue(all(run==mod.RUN for run,row in mod.base.ROWS.values()))
  with self.assertRaises(ValueError):mod.factorial({k:v for k,v in values.items() if k!='B00'})
if __name__=='__main__':unittest.main()
