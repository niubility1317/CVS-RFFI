import math
import unittest
from experiments.cric_dura_comparison.analyze_main import metrics,moments


class MainStatisticsTests(unittest.TestCase):
    def test_registered_ten_class_macro_f1(self):
        cm=[[0]*10 for _ in range(10)]
        for i in range(10):cm[i][i]=2
        cm[0][0]=1;cm[0][1]=1
        row=dict(confusion=cm,count=20,accuracy=.95,
                 macro_f1=(2/3+4/5+8)/10)
        self.assertAlmostEqual(metrics(row,10)['macro_f1'],row['macro_f1'])
        with self.assertRaises(ValueError):metrics(row,6)
        row['accuracy']=1.
        with self.assertRaises(ValueError):metrics(row,10)

    def test_three_seed_sd_keeps_negative_effects(self):
        v=moments([-2.,-1.,0.])
        self.assertEqual(v['mean'],-1.)
        self.assertEqual(v['sd'],1.)
        self.assertEqual(v['positive_seeds'],0)
        with self.assertRaises(ValueError):moments([1.,2.])


if __name__=='__main__':unittest.main()
