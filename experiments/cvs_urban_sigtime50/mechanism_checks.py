import json,tempfile,unittest
from pathlib import Path
import torch
from . import design as d
from . import mechanism as m


class SourceMechanismChecks(unittest.TestCase):
    def test_source_analysis_rejects_missing_or_nonfinal_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with self.assertRaises(FileNotFoundError):m.frozen_rows(base)
            for row in d.rows():
                src=base/row['row_id']/'source';src.mkdir(parents=True)
                (src/'completion.json').write_text(json.dumps(dict(epoch=50,optimizer_steps=11100,head_fit_updates=0,target_access=False)))
                (src/'source_frozen.json').write_text(json.dumps(dict(epoch=50,selection='final_only',target_access=False)))
                (src/'training_mechanism_complete.json').write_text('{}')
            self.assertEqual(m.frozen_rows(base),d.rows())
            (src/'completion.json').write_text(json.dumps(dict(epoch=30,optimizer_steps=6660,head_fit_updates=0,target_access=False)))
            with self.assertRaisesRegex(ValueError,'E50'):m.frozen_rows(base)

    def test_branch_observation_uses_actual_path_without_fitting(self):
        from .model import build
        model=build(d.config(d.rows()[0]),'cpu').eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(2,2,256)
        with torch.no_grad():
            expected=model.forward_identity_only(x)['tx_logits']
            with m.observe(model) as values:actual,branches=m.forward(model,x,values)
        self.assertTrue(torch.equal(expected,actual['tx_logits']))
        self.assertEqual(set(branches),{'time','frequency','PA','reference','signature'})
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
