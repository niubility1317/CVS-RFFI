import copy
import unittest
from experiments.pamf_u_history.receiver_probe import RUN,RELEASE,RECIPE,validate


class ProbeBoundaryTest(unittest.TestCase):
    def config(self):
        return dict(run_id=RUN,recipe=copy.deepcopy(RECIPE),project='/source/project',
            run_root='/source/project/runs/'+RUN,release_root='/source/project/releases/'+RELEASE,allowed_gpus=list(range(8)))

    def test_fixed_source_plan_rejects_target_or_altered_budget(self):
        c=self.config();validate(c)
        c['truth']='foreign'
        with self.assertRaises(ValueError):validate(c)
        c=self.config();c['recipe']['optimizer_updates']=1
        with self.assertRaises(ValueError):validate(c)

    def test_foreign_run_root_cannot_receive_outputs(self):
        c=self.config();c['run_root']='/source/project/runs/another-owner'
        with self.assertRaises(ValueError):validate(c)


if __name__=='__main__':unittest.main()
