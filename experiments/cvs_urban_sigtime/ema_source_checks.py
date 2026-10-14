import unittest
import torch
from .ema_source import summarize
from .publish import configuration


class SourceCalibrationChecks(unittest.TestCase):
    def setUp(self):
        self.p = torch.tensor([[.96,.008,.008,.008,.008,.008],
            [.99,.002,.002,.002,.002,.002], [.93,.014,.014,.014,.014,.014],
            [.3,.14,.14,.14,.14,.14]])
        self.w = self.p.clone()
        self.w[1] = torch.tensor([.002,.99,.002,.002,.002,.002])
        self.y = torch.tensor([0,1,0,0]);self.rx=torch.tensor([1,1,3,3])

    def test_confidence_is_not_precision_and_same_sample_has_its_own_count(self):
        value=summarize(self.p,self.y,self.rx,self.w)
        clean=next(r for r in value['criteria'] if r['threshold']==.95 and r['mode']=='clean_confidence')
        paired=next(r for r in value['criteria'] if r['threshold']==.95 and r['mode']=='same_sample_weak_agreement')
        self.assertEqual((clean['selected'],clean['precision']), (2,.5))
        self.assertEqual((paired['selected'],paired['precision']), (1,1.))
        self.assertEqual(sum(g['selected'] for g in paired['RX_TX']),1)

    def test_empty_selection_is_missing_precision(self):
        value=summarize(self.p,self.y,self.rx,self.w)
        paired=next(r for r in value['criteria'] if r['threshold']==.97 and r['mode']=='same_sample_weak_agreement')
        self.assertEqual(paired['selected'],0);self.assertIsNone(paired['precision'])

    def test_hidden_U_sentinels_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError,'legal labeled source V'):
            summarize(self.p,torch.full_like(self.y,-1),self.rx,self.w)

    def test_nonfinite_teacher_is_rejected(self):
        p=self.p.clone();p[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            summarize(p,self.y,self.rx,self.w)

    def test_source_release_is_separate_from_training_and_old_analysis(self):
        c=configuration('ema')
        self.assertTrue(c['analysis']);self.assertEqual(c['analysis_output'],'source_ema_vcal')
        self.assertNotEqual(c['release'],configuration(True)['release'])
        self.assertNotEqual(c['release'],configuration(False)['release'])


if __name__=='__main__':
    unittest.main()
