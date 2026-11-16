import unittest
from pathlib import Path
import runpy

import torch
from experiments.cric_dura_comparison import f_optimizer
from experiments.pamf_ssl_research.optimizer_step_guard import OptimizerStepGuard,installed


class StepGuardTests(unittest.TestCase):
    def test_full_L_only_with_existing_adam_moments(self):
        model=runpy.run_path(str(Path('tests/test_pamf_tri_adamw_diagnostic.py')))['model']()
        named=list(model.named_parameters())
        opt=f_optimizer.build(model,[p for _,p in named],'CENTER_UNITS',lr=2e-4,weight_decay=1e-4)
        for _ in range(2):
            for _,p in named:p.grad=torch.ones_like(p)
            f_optimizer.clip_physical(opt,[p for _,p in named],5.);opt.step()
        full=tuple(torch.ones_like(p) for _,p in named);empty=tuple(None for _ in named)
        guard=OptimizerStepGuard()
        with installed():
            guard.capture(model,named,full,empty,empty)
            for _,p in named:p.grad=-torch.ones_like(p)
            f_optimizer.clip_physical(opt,[p for _,p in named],5.);opt.step()
            self.assertEqual(guard.calls,1)
            for details in guard.last_details.values():
                self.assertTrue(details['actual_parameter_check'] and details['norm_nonexpansion'])
                cosine=details['anchor_cosines_after']['L_full']
                self.assertTrue(cosine is None or cosine>=-1.000001e-6)
                self.assertIsNone(details['anchor_cosines_after']['L_clean'])
                self.assertIsNone(details['anchor_cosines_after']['L_As'])
            for parameter,state in opt.state.items():self.assertEqual(int(state['step']),3)
        self.assertIsNone(guard.hook)


if __name__=='__main__':unittest.main()
