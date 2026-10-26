import unittest
import numpy as np
from experiments.pamf_channel_pl.restoration_labeler_policy import certify,quality_pass


class RestorationLabelerPolicyTests(unittest.TestCase):
    def probabilities(self,labels,peak=.99):
        p=np.full((len(labels),6),(1-peak)/5,dtype=np.float64)
        p[np.arange(len(labels)),labels]=peak
        return np.log(p)

    def test_fixed_original_hard_pool_and_two_adapted_labelers(self):
        raw=self.probabilities([0,1,2,3]);g=np.stack([self.probabilities([0,1,4,3])]*12)
        result=certify(raw,g,np.array([.90,.99,.90,.94]))
        np.testing.assert_array_equal(result['accepted'],[True,False,False,True])
        np.testing.assert_array_equal(result['label'],[0,1,2,3])

    def test_probabilities_are_averaged_before_certification(self):
        raw=self.probabilities([0]);g=np.stack([self.probabilities([0])]*12)
        g[:6]=self.probabilities([1])
        result=certify(raw,g,np.array([.80]))
        self.assertFalse(result['accepted'][0]);self.assertLess(result['G_confidence'][0],.95)

    def test_batch_and_class_permutation_equivariance(self):
        raw=self.probabilities([0,3,5]);g=np.stack([raw]*12);old=np.array([.8,.9,.94]);p=np.array([2,5,1,4,0,3])
        a=certify(raw,g,old);b=certify(raw[:,p],g[:,:,p],old)
        np.testing.assert_array_equal(a['accepted'],b['accepted'])
        np.testing.assert_array_equal(p[b['label']],a['label'])
        for i in range(3):
            c=certify(raw[i:i+1],g[:,i:i+1],old[i:i+1])
            self.assertEqual(bool(c['accepted'][0]),bool(a['accepted'][i]))

    def test_quality_requires_coverage_precision_and_new_corrections(self):
        self.assertTrue(quality_pass(200,199,100))
        self.assertFalse(quality_pass(199,199,100))
        self.assertFalse(quality_pass(200,198,100))
        self.assertFalse(quality_pass(500,500,99))

    def test_invalid_or_unregistered_inputs_rejected(self):
        raw=self.probabilities([0]);g=np.stack([raw]*12)
        for a,b,c in [(raw,g[:3],np.array([.9])),(raw[:,:5],g,np.array([.9])),(raw,g,np.array([1.1])),(raw*np.nan,g,np.array([.9]))]:
            with self.assertRaises(ValueError):certify(a,b,c)

    def test_runtime_rejects_target_handles_and_foreign_output(self):
        from experiments.pamf_channel_pl.restoration_labeler import RUN,RELEASE,validate_runtime
        c=dict(run_id=RUN,project='/project',python='python',release_root='/project/releases/'+RELEASE,run_root='/project/runs/'+RUN,
            context='context',runtime_file='runtime',code_commit='commit',frozen_sources={},allowed_gpus=list(range(8)),launch_owner='owner')
        validate_runtime(c)
        for extra in (dict(truth='labels'),dict(target_inputs='query'),dict(run_root='/foreign'),dict(allowed_gpus=[0])):
            with self.subTest(extra=extra),self.assertRaises(ValueError):validate_runtime(dict(c,**extra))

    def test_backward_probe_is_frozen_for_state_check_without_hiding_state_changes(self):
        import torch
        from experiments.pamf_channel_pl.restoration_labeler import finish_probe
        from experiments.pamf_channel_pl.restored_source import snapshot,unchanged
        model=torch.nn.Linear(2,6);model.register_buffer('witness',torch.tensor([1.]))
        before=snapshot(model);model(torch.ones(3,2)).square().mean().backward()
        finish_probe(model,before)
        self.assertTrue(unchanged(model,before))
        model.witness.add_(1)
        with self.assertRaises(ValueError):finish_probe(model,before)


if __name__=='__main__':unittest.main()
