import unittest
from cvs_cleanretain_delivery_report_20261017 import formal_rows,pp
class Cases(unittest.TestCase):
 def test_explicit_reference_values_and_percentage_points(self):
  views=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');s=dict(rows={a:dict(views={v:dict(accuracy=x) for v in views}) for a,x in [('C0',.8),('AS50',.5),('R25',.7)]});r=formal_rows(s);self.assertEqual(len(r),9);self.assertEqual(r[2],'|clean|80.000000|50.000000|70.000000|');self.assertAlmostEqual(pp(.7,.5),20)
 def test_incomplete_view_matrix_rejected(self):
  with self.assertRaises(ValueError):formal_rows(dict(rows=dict(C0=dict(views={}),AS50=dict(views={}),R25=dict(views={}))))
if __name__=='__main__':unittest.main()
