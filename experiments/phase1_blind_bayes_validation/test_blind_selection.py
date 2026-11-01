import ast,math,unittest
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
tree=ast.parse(Path(__file__).with_name('blind_bayes_source_validation.py').read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select');env={'math':math,'VIEWS':VIEWS};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(Path(__file__)), 'exec'),env);select=env['select']
def cells():return {f'{part}:{rx}:{v}':dict(count=100,valid_count=100,baseline_correct=90,candidate_correct=90 if v=='clean' else 92,baseline_iq=.1,candidate_iq=.1) for part,receivers in ((1,(4,6)),(2,(8,))) for rx in receivers for v in VIEWS}
class SelectionTests(unittest.TestCase):
 def test_holdout_not_used_and_improvement_selected(self):
  a=cells();a['2:8:practical_low_urban']['candidate_correct']=0;self.assertEqual(select({'0.8':a},True),dict(kind='blind_bayes',threshold=.8,validation_net=24))
 def test_clean_and_per_scene_protection(self):
  a=cells();a['1:4:clean']['candidate_iq']=.11;self.assertEqual(select({'0.8':a},False)['kind'],'dura');a=cells();a['1:4:practical_low_urban']['candidate_correct']=88;self.assertEqual(select({'0.8':a},False)['kind'],'dura')
 def test_impaired_iq_guard_separate(self):
  a=cells();a['1:4:practical_low_urban']['candidate_iq']=.13;self.assertEqual(select({'0.8':a},True)['kind'],'dura');self.assertEqual(select({'0.8':a},False)['kind'],'blind_bayes')
if __name__=='__main__':unittest.main()
