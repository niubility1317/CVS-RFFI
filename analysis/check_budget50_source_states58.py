"""Only wrapper fixtures: fixed seed binding, saved source keys and seed statistics."""
import copy,importlib.util,math,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cvs_budget50_source_states58_20261018 as a


def panel(value):
 return {'panel':{scene:dict(accuracy=value,macro_F1=value-.01) for scene in ('clean',*a.core.SCENES)}}

def summary01():
 return dict(status='SOURCE_ANALYZED',target_scores_read=False,run_id=a.RUN50_01,model_seed_count=1,summary={arm:panel(.1+i*.1) for i,arm in enumerate(a.core.ARMS)})

def summary23():
 return dict(status='SOURCE_ANALYZED',target_scores_read=False,run_id=a.RUN50_23,model_seeds=list(a.SEEDS),summary={arm+'-s'+str(seed):panel(.1*index+.01*i) for index,seed in enumerate(a.SEEDS,1) for i,arm in enumerate(a.core.ARMS)})

def effects_fixture():
 result={}
 for i,seed in enumerate(a.SEEDS):
  effect=[];interaction=[]
  for label,count in [('ALL',4),('empty',0)]:
   context=dict(scene=a.core.SCENES[-1],draw=-1,dimension='overall' if label=='ALL' else 'state_lock',stratum=label,count=count)
   for new,old in a.MAIN_PAIRS:
    effect.append(dict(new=new,control=old,accuracy_delta_pp=[1.,3.,8.][i] if count else None,macro_f1_delta_pp=[.5,1.5,4.][i] if count else None,**context))
   interaction.append(dict(budget_x_As_accuracy_pp=[-1.,0.,1.][i] if count else None,**context))
  result[seed]=dict(effects=effect,interaction=interaction)
 return result

class Checks(unittest.TestCase):
 def test_seed_run_rows_and_restoration(self):
  before=(a.core.SEED,a.core.RUN30,a.core.RUN50)
  for seed in a.SEEDS:
   expected=(a.RUN30_01,a.RUN50_01) if seed==a.SEEDS[0] else (a.RUN30_23,a.RUN50_23)
   with a.bind_seed(seed):
    self.assertEqual((a.core.SEED,a.core.RUN30,a.core.RUN50),(seed,*expected));rows=a.core.rows(Path('fixture'))
    for arm,p in rows.items():
     self.assertEqual(p.name,('B01' if arm.startswith('AS') else 'B00')+'-s'+str(seed));self.assertEqual(p.parent.name,expected[1] if arm.endswith('50') else expected[0])
   self.assertEqual((a.core.SEED,a.core.RUN30,a.core.RUN50),before)
  with self.assertRaisesRegex(RuntimeError,'fixture'):
   with a.bind_seed(a.SEEDS[1]):raise RuntimeError('fixture')
  self.assertEqual((a.core.SEED,a.core.RUN30,a.core.RUN50),before)
  with self.assertRaisesRegex(ValueError,'fixed model seeds'):
   with a.bind_seed(2026092704):pass
 def test_saved_source_key_mapping_exact_seed(self):
  old=a.map_source_summary(summary01(),a.SEEDS[0]);self.assertEqual(set(old['summary']),set(a.core.ARMS));self.assertEqual(old['summary']['AS50']['panel']['clean']['accuracy'],.4)
  saved=summary23()
  for seed in a.SEEDS[1:]:
   mapped=a.map_source_summary(saved,seed)
   for arm in a.core.ARMS:self.assertIs(mapped['summary'][arm],saved['summary'][arm+'-s'+str(seed)])
  with self.assertRaisesRegex(ValueError,'summary/run'):a.map_source_summary(saved,a.SEEDS[0])
  bad=copy.deepcopy(saved);bad['summary']['NOAS30-s2026092704']=bad['summary'].pop('NOAS30-s2026092702')
  with self.assertRaisesRegex(ValueError,'seed-labelled keys'):a.map_source_summary(bad,a.SEEDS[1])
  bad=copy.deepcopy(saved);bad['model_seeds'].reverse()
  with self.assertRaisesRegex(ValueError,'seed-labelled keys'):a.map_source_summary(bad,a.SEEDS[1])
  bad=copy.deepcopy(saved);bad['summary']['AS50-s2026092702']['panel']['clean'].pop('macro_F1')
  with self.assertRaisesRegex(ValueError,'no synthetic fallback'):a.map_source_summary(bad,a.SEEDS[1])
 def test_source_only_summary_required(self):
  for key,value in [('target_scores_read',True),('status','ANALYZED'),('run_id','different-run')]:
   bad=summary23();bad[key]=value
   with self.assertRaisesRegex(ValueError,'source-only'):a.map_source_summary(bad,a.SEEDS[1])
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaisesRegex(a.SourceNotReady,'not ready'):a.check_ready(Path(td))
 def test_three_seed_sample_sd_and_missing_rejection(self):
  value=a.stats3(dict(zip(a.SEEDS,[1.,3.,8.])));self.assertEqual(value['values'],[1.,3.,8.]);self.assertEqual(value['mean'],4.);self.assertAlmostEqual(value['sample_SD'],math.sqrt(13));self.assertEqual(value['valid_model_seed_count'],3)
  for missing in [{a.SEEDS[0]:1.},{a.SEEDS[0]:1.,a.SEEDS[1]:2.,2026092704:3.}]:
   with self.assertRaisesRegex(ValueError,'three fixed seeds'):a.stats3(missing)
  with self.assertRaisesRegex(ValueError,'Incomplete seed'):a.stats3(dict(zip(a.SEEDS,[1.,None,3.])))
  empty=a.stats3(dict.fromkeys(a.SEEDS));self.assertTrue(empty['empty_cell']);self.assertIsNone(empty['mean']);self.assertIsNone(empty['sample_SD']);self.assertEqual(empty['model_seed_count'],3)
 def test_aggregate_paired_means_and_empty_cells(self):
  records=a.aggregate(effects_fixture());self.assertEqual(len(records),6)
  for comparison in ['AS50_minus_NOAS50','AS50_minus_AS30']:
   r=next(v for v in records if v['comparison']==comparison and v['stratum']=='ALL');self.assertEqual(r['observations_per_model_seed'],4);self.assertEqual(r['model_seed_count'],3);self.assertEqual(r['paired_mean_accuracy_delta_pp'],4.);self.assertAlmostEqual(r['paired_sample_SD_accuracy_delta_pp'],math.sqrt(13));self.assertEqual(r['paired_mean_macro_f1_delta_pp'],2.);self.assertAlmostEqual(r['paired_sample_SD_macro_f1_delta_pp'],math.sqrt(13)/2)
   empty=next(v for v in records if v['comparison']==comparison and v['stratum']=='empty');self.assertTrue(empty['empty_cell']);self.assertIsNone(empty['paired_mean_accuracy_delta_pp']);self.assertEqual(empty['observations_per_model_seed'],0)
  inter=next(v for v in records if v['comparison']=='budget_x_As' and v['stratum']=='ALL');self.assertEqual(inter['paired_mean_accuracy_delta_pp'],0.);self.assertEqual(inter['paired_sample_SD_accuracy_delta_pp'],1.)
 def test_missing_seed_context_and_count_rejected(self):
  data=effects_fixture();data.pop(a.SEEDS[1])
  with self.assertRaisesRegex(ValueError,'three-seed'):a.aggregate(data)
  data=effects_fixture();data[a.SEEDS[1]]['effects'].pop()
  with self.assertRaisesRegex(ValueError,'identical state/bin/draw'):a.aggregate(data)
  data=effects_fixture();data[a.SEEDS[1]]['effects'][0]['count']=5
  with self.assertRaisesRegex(ValueError,'observation counts'):a.aggregate(data)
  data=effects_fixture()
  for seed in a.SEEDS:data[seed]['effects']=[]
  with self.assertRaisesRegex(ValueError,'no source strata'):a.aggregate(data)

if __name__=='__main__':unittest.main(verbosity=2)
