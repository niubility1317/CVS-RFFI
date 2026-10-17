import unittest
from .criteria import decide
class Cases(unittest.TestCase):
 def test_preserved_both_draws_and_actual_failure(self):
  c=dict(C0=.98,R25=.979);a=[[.7,.6],[.7,.6]];b=[[.71,.61],[.71,.61]];self.assertTrue(decide(c,a,b)['source_protection_evidence']);c['R25']=.9774;self.assertFalse(decide(c,a,b)['source_protection_evidence']);c['R25']=.98;a[1][1]=.58;self.assertFalse(decide(c,a,b)['source_protection_evidence'])
 def test_missing_draw_rejected(self):
  with self.assertRaises(ValueError):decide(dict(C0=.98,R25=.98),[[.7,.6]],[[.7,.6]])
if __name__=='__main__':unittest.main()
