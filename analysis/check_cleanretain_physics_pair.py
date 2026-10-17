import gzip,json,tempfile,unittest
from pathlib import Path
from cvs_cleanretain_physics_pair_20261017 import compare
class Cases(unittest.TestCase):
 def files(self,path,corrupt=False,bad_slots=False):
  aa=path/'actual.gz';bb=path/'reference.gz'
  with gzip.open(aa,'wt',encoding='utf8') as a,gzip.open(bb,'wt',encoding='utf8') as b:
   for i in range(1,3):
    records=[dict(sample_id='source:'+str(j),signal_to_added_noise_db=float(j),receiver=dict(phase=1.)) for j in range(128)];positions=[j for j in range(128) if (j+i)%2==1];selected=[dict(records[j]) for j in positions]
    if corrupt and i==2:selected[0]['signal_to_added_noise_db']+=1
    a.write(json.dumps(dict(step=i,scene='practical_low_urban',positions=positions[::-1] if bad_slots else positions,records=selected))+'\n');b.write(json.dumps(dict(step=i,scene='practical_low_urban',records=records))+'\n')
  return aa,bb
 def test_same_selected_realizations(self):
  with tempfile.TemporaryDirectory() as t:r=compare(*self.files(Path(t)),expected_steps=2)
  self.assertEqual(r['identical_records'],128);self.assertTrue(r['all_selected_realizations_identical'])
 def test_numeric_change_reported_without_discard(self):
  with tempfile.TemporaryDirectory() as t:r=compare(*self.files(Path(t),corrupt=True),expected_steps=2)
  self.assertEqual(r['pairs'],128);self.assertEqual(r['record_mismatches'],1);self.assertEqual(r['field_mismatches'],dict(signal_to_added_noise_db=1))
 def test_order_and_incomplete_budget_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):compare(*self.files(Path(t),bad_slots=True),expected_steps=2)
   with self.assertRaises(ValueError):compare(*self.files(Path(t)),expected_steps=3)
if __name__=='__main__':unittest.main()
