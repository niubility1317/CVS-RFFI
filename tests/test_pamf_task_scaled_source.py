import unittest
from experiments.pamf_channel_pl.task_scaled_source import decide,VARIANTS


class WholeModelDoseTests(unittest.TestCase):
    def test_common_dose_uses_largest_variant_norm(self):
        row=dict(primary_norm=1.,**{v+'_norm':float(i+1) for i,v in enumerate(VARIANTS)},**{v+'_cosine':.2 for v in VARIANTS})
        result=decide([row]*12);self.assertAlmostEqual(result['selected_weight'],.1/3)
        self.assertTrue(all(v<=.1 for v in result['weighted_median_norm_ratio'].values()))

    def test_partial_zero_and_nonfinite_sources_rejected(self):
        row=dict(primary_norm=1.,**{v+'_norm':1. for v in VARIANTS},**{v+'_cosine':.2 for v in VARIANTS})
        with self.assertRaises(ValueError):decide([row]*11)
        with self.assertRaises(ValueError):decide([dict(row,primary_norm=0.)]*12)
        with self.assertRaises(ValueError):decide([dict(row,**{VARIANTS[0]+'_norm':float('nan')})]*12)
        one_bad=[dict(row) for _ in range(12)];one_bad[0][VARIANTS[0]+'_norm']=float('nan')
        with self.assertRaises(ValueError):decide(one_bad)
        one_bad[0][VARIANTS[0]+'_norm']=0.
        with self.assertRaises(ValueError):decide(one_bad)


if __name__=='__main__':unittest.main()
