import json
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from cvsrffi.evidence_source_evaluation import evaluate_source


class ToyModel(nn.Module):
    def __init__(self,evidence):
        super().__init__()
        self.weight=nn.Parameter(torch.eye(2))
        self.evidence=evidence
    def forward(self,x,**kwargs):
        z=x@self.weight
        out=dict(tx_logits=z*3,z_id=z)
        if self.evidence:
            out['evidence']=dict(z=z,mahalanobis=(z[:,None,:]-torch.eye(2)[None]).square().sum(-1),
                observed_count=torch.full((len(z),),2,dtype=torch.long),quality=torch.ones(len(z),2)*.5,
                state=z,observed=torch.ones_like(z,dtype=torch.bool),state_in_domain=torch.ones(len(z),dtype=torch.bool))
        return out


class SourceEvaluationTests(unittest.TestCase):
    def loader(self,role):
        y=torch.arange(40)%2
        data=TensorDataset(torch.eye(2)[y],y)
        data.role=role
        return DataLoader(data,batch_size=7)
    def test_frozen_evidence_artifacts(self):
        model=ToyModel(True)
        before=model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as out:
            summary=evaluate_source(model,self.loader('L_s'),self.loader('V'),'cpu',out)
            self.assertTrue(model.training)
            torch.testing.assert_close(model.weight,before)
            self.assertEqual(summary['count_val'],40)
            self.assertEqual(summary['calibrated_head']['accuracy'],1.)
            self.assertEqual(len(summary['baselines']),3)
            self.assertIn('quality_TX',summary['condition_probes'])
            self.assertFalse(summary['resources']['covariance_retained'])
            self.assertTrue(Path(out,'source_calibrator.json').is_file())
            self.assertFalse(Path(out,'source_readouts.pt').exists())
            scores=torch.load(Path(out,'source_scores.pt'),weights_only=True)
            self.assertNotIn('covariance',scores)
            self.assertEqual(len(scores['labels']),40)
            self.assertFalse(json.loads(Path(out,'source_evaluation.json').read_text())['target_access'])
    def test_h0_and_role_rejection(self):
        with tempfile.TemporaryDirectory() as out:
            summary=evaluate_source(ToyModel(False),self.loader('L_s'),self.loader('V'),'cpu',out)
            self.assertNotIn('calibrated_head',summary)
            self.assertFalse(Path(out,'source_calibrator.json').exists())
            readouts=torch.load(Path(out,'source_readouts.pt'),weights_only=True)
            rows=torch.load(Path(out,'source_scores.pt'),weights_only=True)
            z=torch.eye(2)[torch.arange(40)%2]
            for kind,control in readouts['controls'].items():
                self.assertEqual(set(control),{'weight','bias','normalize_input','classes'})
                features=torch.nn.functional.normalize(z,dim=-1) if control['normalize_input'] else z
                logits=features@control['weight'].T+control['bias']
                # Shared Gaussian terms cancel in softmax; every control is
                # equivalent to the fitted baseline classifier on identical z.
                torch.testing.assert_close(logits.softmax(-1),rows['baseline_'+kind].softmax(-1))
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError,'expected V'):
                evaluate_source(ToyModel(False),self.loader('L_s'),self.loader('query'),'cpu',out)


if __name__=='__main__':
    unittest.main()
