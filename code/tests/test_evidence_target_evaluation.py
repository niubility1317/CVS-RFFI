import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import Dataset
from cvsrffi.evidence_target_evaluation import seal_target,score_target,opaque_record_id,SCENES


class TargetIQ(Dataset):
    role='R_t'
    def __init__(self,order=(0,1,2,3)):
        self.index=[SimpleNamespace(tx_i=i%2,rx_i=i//2,day_i=0,sig_i=i,eq_i=1) for i in order]
    def __len__(self): return len(self.index)
    def __getitem__(self,k):
        row=self.index[k]
        t=torch.arange(32).float()
        x=torch.stack((torch.cos(t*.1)+row.tx_i*.2,torch.sin(t*.1)))
        return x, row.tx_i, {'receiver':row.rx_i}


class Predict(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,x,**kwargs):
        assert kwargs['y_tx'] is None
        return {'tx_logits':x.mean(-1)*self.scale}


class TargetTests(unittest.TestCase):
    def test_batch_permutation_seal_and_independent_score(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a,b=root/'a',root/'b'
            data=TargetIQ(); model=Predict()
            seal_target(model,data,'cpu',a,batch_size=1)
            seal_target(model,TargetIQ((3,1,0,2)),'cpu',b,batch_size=3)
            self.assertTrue(model.training)
            for scene in SCENES:
                left=torch.load(a/(scene+'.pt'),weights_only=True)
                right=torch.load(b/(scene+'.pt'),weights_only=True)
                self.assertTrue(all(torch.is_tensor(v) for v in left.values()))
                self.assertNotIn('labels',left)
                indices={bytes(row.tolist()).hex():i for i,row in enumerate(right['ids'])}
                order=[indices[bytes(row.tolist()).hex()] for row in left['ids']]
                torch.testing.assert_close(left['scores'],right['scores'][order],atol=1e-6,rtol=1e-6)
            truth={'rows':{opaque_record_id(row):dict(label=row.tx_i,receiver=row.rx_i,day=row.day_i) for row in data.index}}
            path=root/'truth.json';path.write_text(json.dumps(truth),encoding='utf-8')
            report=score_target(a,path,root/'scores.json')
            self.assertEqual(report['scenes']['clean']['overall']['count'],4)
            self.assertEqual(len(report['scenes']['clean']['by_receiver']),2)
            self.assertEqual(len(report['scenes']['clean']['by_class']),2)
            self.assertEqual(len(report['scenes']['clean']['by_receiver_class']),4)
            self.assertGreater(len(report['scenes']['clean']['risk_coverage']),1)
            self.assertIn('receiver_class_accuracy_floor',report['scenes']['clean'])
            self.assertGreater(report['resources']['total_artifact_bytes'],0)
            self.assertGreater(report['resources']['total_prediction_seconds'],0)
            for metrics in report['scenes']['clean']['by_receiver_class'].values():
                self.assertTrue(torch.isfinite(torch.tensor(metrics['nll'])))
                self.assertTrue(torch.isfinite(torch.tensor(metrics['accepted_risk'])))
            self.assertFalse(report['scientific_promotion_claim'])
            with self.assertRaises(FileExistsError): seal_target(model,data,'cpu',a)

    def test_scorer_requires_full_seal_before_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'seal.json').write_text(json.dumps(dict(status='INCOMPLETE',scenes={})))
            with self.assertRaisesRegex(ValueError,'four'):
                score_target(root,root/'truth-does-not-exist.json',root/'out.json')


if __name__=='__main__': unittest.main()
