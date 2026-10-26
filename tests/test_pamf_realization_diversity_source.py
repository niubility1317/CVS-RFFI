import os
import unittest
from unittest.mock import patch
import torch
from experiments.pamf_channel_pl.realization_diversity_source import geometry,views,namespace,validate_runtime,RUN,RELEASE


class RealizationDiversityTests(unittest.TestCase):
    def test_fixed_namespace_and_distinct_registered_views(self):
        names=views();self.assertEqual(len(names),25);self.assertEqual(len(set(names)),25)
        self.assertEqual(namespace('fixed'),'PAMF/restored_dose/U/fixed')
        self.assertEqual(len({namespace(v) for v in ('fixed','fresh1','fresh2','fresh3')}),4)
        with self.assertRaises(ValueError):namespace('other')
    def test_repeat_and_changed_geometry(self):
        x=torch.randn(4,2,256);g=geometry(x,x,x,x,x)
        torch.testing.assert_close(g[:,0],torch.zeros(4));torch.testing.assert_close(g[:,1],torch.ones(4))
        changed=geometry(x,x,x*1.1,x,x*1.2)
        torch.testing.assert_close(changed[:,0],torch.full((4,),.2));self.assertTrue(torch.isfinite(changed).all())
        with self.assertRaises(ValueError):geometry(x,x,x,x,x[:2])
    def test_source_runtime_rejects_target_and_foreign_output_before_io(self):
        c=dict(run_id=RUN,project='/project',python='python',release_root='/project/releases/'+RELEASE,run_root='/project/runs/'+RUN,
            context='/project/releases/'+RELEASE+'/context.json',runtime_file='/project/releases/'+RELEASE+'/runtime.json',code_commit='commit',
            frozen_sources={},allowed_gpus=list(range(8)),launch_owner='owner',entrypoint='experiments.pamf_channel_pl.realization_diversity_source')
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='5'):
            validate_runtime(c)
            with self.assertRaises(ValueError):validate_runtime(dict(c,target_inputs='query'))
            with self.assertRaises(ValueError):validate_runtime(dict(c,run_root='/other'))
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'):
            with self.assertRaises(ValueError):validate_runtime(c)


if __name__=='__main__':unittest.main()
