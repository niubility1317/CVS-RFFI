import json
import unittest
import torch
from cvsrffi.evidence_decision import ConditionAwareCalibrator, selective_metrics


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[2.,0.]]*8)
        self.mahal = torch.tensor([[1.,2.]]*4+[[10.,20.]]*4)
        self.dims = torch.ones(8,dtype=torch.long)
        self.quality = torch.tensor([[.1]]*4+[[.9]]*4)
        self.labels = torch.zeros(8,dtype=torch.long)

    def fitted(self):
        return ConditionAwareCalibrator(quality_bins=2,min_group_samples=4).fit(self.logits,self.mahal,self.dims,self.quality,self.labels)

    def test_condition_groups_and_missing(self):
        cal = self.fitted()
        out = cal.predict(torch.tensor([[2.,0.]]*4),torch.tensor([[5.,9.]]*4),torch.tensor([1,1,0,2]),torch.tensor([[.1],[.9],[.1],[.1]]))
        self.assertEqual(out['status'],['model_mismatch_candidate','identify','defer','defer'])
        self.assertEqual(out['calibration_available'].tolist(),[True,True,False,False])
        self.assertFalse(out['evidence_sufficiency'][2])

    def test_role_and_freezing(self):
        for role in ('query','source_train','train','T'):
            with self.assertRaises(ValueError):
                ConditionAwareCalibrator().fit(self.logits,self.mahal,self.dims,self.quality,self.labels,role=role)
        cal = self.fitted()
        with self.assertRaises(RuntimeError):
            cal.fit(self.logits,self.mahal,self.dims,self.quality,self.labels)

    def test_serialization_batching_no_updates(self):
        cal = self.fitted()
        state = json.dumps(cal.state_dict(),sort_keys=True)
        restored = ConditionAwareCalibrator.from_state_dict(json.loads(state))
        whole = restored.predict(self.logits,self.mahal,self.dims,self.quality)
        for i in range(8):
            row = restored.predict(self.logits[i:i+1],self.mahal[i:i+1],self.dims[i:i+1],self.quality[i:i+1])
            self.assertEqual(row['status'][0],whole['status'][i])
            torch.testing.assert_close(row['probabilities'][0],whole['probabilities'][i])
        self.assertEqual(json.dumps(restored.state_dict(),sort_keys=True),state)
        self.assertTrue(all(not isinstance(v,torch.Tensor) for v in restored.__dict__.values()))

    def test_rejects_wrong_and_curve_ties(self):
        logits = torch.tensor([[2.,0.],[2.,0.],[0.,2.]])
        labels = torch.tensor([0,1,1])
        out = selective_metrics(logits,labels,torch.tensor([True,False,False]))
        self.assertAlmostEqual(out['accuracy'],1/3)
        self.assertAlmostEqual(out['coverage'],1/3)
        self.assertEqual(out['accepted_risk'],0)
        self.assertEqual(len(out['risk_coverage']),2)
        self.assertGreater(out['nll'],0)
        self.assertGreater(out['brier'],0)
        empty = selective_metrics(logits,labels,torch.zeros(3,dtype=torch.bool))
        self.assertIsNone(empty['accepted_risk'])


if __name__ == '__main__':
    unittest.main()
