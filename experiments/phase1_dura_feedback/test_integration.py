import unittest
from unittest.mock import patch
from experiments.phase1_dura_feedback import train_source,ground
from experiments.phase1_dura_feedback.model import FeedbackRecovery
from experiments.phase1_dura_feedback.pipeline import predict_batch

class FeedbackAdapterTest(unittest.TestCase):
    def test_training_installs_and_freezes_both_components(self):
        base=train_source.base
        names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA',
               'load_components','input_noise','predict_batch','extra_frozen_components')
        before=tuple(getattr(base,n) for n in names)
        def fail():
            self.assertIs(base.DynamicVolterra,FeedbackRecovery)
            self.assertIs(base.source_objective,train_source.feedback_source_objective)
            self.assertTrue(base.USES_DURA)
            a,b=object(),object()
            self.assertEqual(base.extra_frozen_components(a,b),(a,b))
            raise RuntimeError('expected')
        with patch.object(base,'main',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'expected'):train_source.main()
        self.assertEqual(tuple(getattr(base,n) for n in names),before)
    def test_ground_uses_same_model_and_feedback_pipeline(self):
        self.assertIs(ground.DynamicVolterra,FeedbackRecovery)
        self.assertIs(ground.predict_batch,predict_batch)
        self.assertEqual(ground.SOURCE,'20261103-phase1-dura-feedback-source-s2026092701-r01')

if __name__=='__main__':unittest.main()
