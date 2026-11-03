import importlib
import unittest
import numpy as np
from unittest.mock import patch


class CrossfitSelectionTest(unittest.TestCase):
    def adapter(self):
        try:return importlib.import_module('experiments.phase1_dura_crossfit_blend.train_source')
        except ModuleNotFoundError as exc:
            if exc.name=='experiments.phase1_dura_crossfit_blend.train_source':self.fail('disjoint fusion training is not implemented')
            raise

    def test_selection_is_deterministic_unique_and_disjoint_from_both_fits(self):
        f=self.adapter().select_disjoint_fit;training=np.arange(30000);prior=np.arange(8192);expert=np.arange(8192,16384)
        a=f(training,prior,expert,8192);b=f(training,prior,expert,8192)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(len(np.unique(a)),8192)
        self.assertTrue(np.all(a[:-1]<a[1:]));self.assertEqual(len(np.intersect1d(a,prior)),0);self.assertEqual(len(np.intersect1d(a,expert)),0)

    def test_selection_fails_if_disjoint_pool_is_too_small(self):
        f=self.adapter().select_disjoint_fit
        with self.assertRaises(ValueError):f(np.arange(20000),np.arange(8192),np.arange(8192,16384),8192)

    def test_selection_rejects_expert_ids_outside_training_pool(self):
        f=self.adapter().select_disjoint_fit
        with self.assertRaises(ValueError):f(np.arange(30000),np.arange(8192),np.arange(25000,33192),8192)

    def test_adapter_installs_crossfit_selection_and_restores(self):
        a=self.adapter();base=a.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA','load_components','input_noise','predict_batch','extra_frozen_components','extra_frozen_model_components','select_fit')
        old=tuple(getattr(base,k) for k in names)
        def fail():
            self.assertIs(base.select_fit,a.select_crossfit);self.assertTrue(base.USES_DURA)
            raise RuntimeError('synthetic failure')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'synthetic failure'):a.main()
        self.assertEqual(old,tuple(getattr(base,k) for k in names))


if __name__=='__main__':unittest.main()
