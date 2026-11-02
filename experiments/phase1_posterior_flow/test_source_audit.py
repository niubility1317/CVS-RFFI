import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np

class SourceAuditTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir();runs=self.root/'runs';cache=runs/'20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/train';cache.mkdir(parents=True);meta=runs/'20261021-phase1-CRIC-DURA-comparison-m153-r02/data/F0';meta.mkdir(parents=True)
  keys=np.array([f'L{i:04}' for i in range(6300)]);labels=np.arange(6300)%6;np.savez(meta/'L_s_metadata.npz',ids=keys,y=labels);(meta/'source_contract.json').write_text(json.dumps({'role_ids':{'L_s':keys.tolist()}}));phys=np.array([f'L{i:04}' if i<84 else f'U{i:04}' for i in range(672)]);roles=np.array(['L_s' if i<84 else 'U_s' for i in range(672)]);np.save(cache/'physical_ids.npy',phys);np.save(cache/'roles.npy',roles)
  views=['clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban'];ids=np.arange(672);split=np.array([g for g in ('RX4_6','RX8') for v in views for i in range(48)]);scene=np.array([v for g in ('RX4_6','RX8') for v in views for i in range(48)]);rng=np.random.default_rng(5);values=np.zeros((672,8));values[:,0]=np.where(ids%4,.99,.8);values[:,1]=rng.integers(0,6,672)
  for col in (2,4,6):values[:,col]=rng.integers(0,6,672);values[:,col+1]=rng.random(672)
  np.savez(self.source/'source_predictions.npz',indices=ids,split=split,scene=scene,values=values);(self.source/'checkpoint.pt').write_bytes(b'checkpoint');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();h=sha(self.source/'source_predictions.npz');manifest={'status':'SOURCE_PREDICTIONS_FIXED','commit':'b1170e6ad3620c2c821bbcbcdca2f5a3482e3285','count':672,'prediction_sha256':h,'checkpoint_sha256':sha(self.source/'checkpoint.pt')};(self.source/'prediction_complete.json').write_text(json.dumps(manifest));results={}
  for g in ('RX4_6','RX8'):
   results[g]={}
   for view in views+['all_impaired']:
    ix=[i for i in ids if split[i]==g and (scene[i]!='clean' if view=='all_impaired' else scene[i]==view)];ps=[i for i in ix if values[i,0]>=.95];ls=[i for i in ix if i<84];cell={'count':len(ix),'pseudo_count':len(ps),'L_s_count':len(ls)}
    for name,col in [('raw',2),('physical',4),('synthesis',6)]:
     item={'iq_error_mean':sum(float(values[i,col+1]) for i in ix)/len(ix)}
     for key,subset in [('pseudo',ps),('L_s',ls)]:
      target=lambda i:int(values[i,1]) if key=='pseudo' else i%6;good={i for i in subset if values[i,col]==target(i)};base={i for i in subset if values[i,4]==target(i)};item[key+'_correct']=len(good);item[key+'_repairs_vs_physical']=len(good-base);item[key+'_damages_vs_physical']=len(base-good)
     cell[name]=item
    results[g][view]=cell
  self.score={'status':'SOURCE_SCORE_COMPLETE','prediction_sha256':h,'fixed_predictions_before_truth':True,'L_s_scored':84,'U_s_truth_reads':0,'V_reads':0,'test_reads':0,'results':results};self.score_path=self.root/'score.json';self.score_path.write_text(json.dumps(self.score))
 def tearDown(self):self.tmp.cleanup()
 def audit(self):
  return subprocess.run([sys.executable,str(Path(__file__).with_name('audit_nonlocal_source.py')),'--root',str(self.root),'--source',str(self.source),'--score',str(self.score_path),'--output',str(self.root/'audit.json')],capture_output=True,text=True)
 def test_valid_report_and_tampered_count(self):
  result=self.audit();self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertEqual(json.loads((self.root/'audit.json').read_text())['fields_recomputed'],384);(self.root/'audit.json').unlink();self.score['results']['RX4_6']['clean']['synthesis']['L_s_correct']+=1;self.score_path.write_text(json.dumps(self.score));self.assertNotEqual(self.audit().returncode,0)
if __name__=='__main__':unittest.main()
