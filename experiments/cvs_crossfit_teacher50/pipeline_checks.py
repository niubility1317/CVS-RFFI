import copy,unittest
from pathlib import Path
from unittest.mock import patch
from . import design as d,model,publish,evaluate


class Checks(unittest.TestCase):
    def test_teacher_U_or_unfinished_mechanism_rejected_before_checkpoint_load(self):
        c=d.config(d.rows()[1]);root=Path(c['output_root'])
        values={
            'completion.json':dict(config=c,epoch=50,optimizer_steps=11100,target_access=False,head_fit_updates=0),
            'source_frozen.json':dict(config=c,selection='final_only',target_access=False),
            'initialization.json':dict(scratch_only=True,checkpoint_sources=[],ancestors=[],target_contact=False),
            'inner_source_contract.json':dict(train_receivers=[3,4,6,8],counts=dict(L_train=4464,L_dev=720,L_unused=1116,U_s=56700,V=27000),
                L_dev_updates=False,fold_seed=2026101503,subset_seed=d.SUBSET_SEED,arm='EXCLUDE_RX1'),
            'training_mechanism_complete.json':dict(status='VERIFIED',optimizer_updates=11100,training_epochs=50,unique_train_samples=4464,
                forward_packets=dict(train=1420800),head_fit_updates=0,U_forward_packets=0,U_truth_read=True,EMA_updates=0,train_receivers=[3,4,6,8])}
        with patch.object(d,'read',lambda p:copy.deepcopy(values.get(Path(p).name,{}))),\
             patch('experiments.cvs_phase1_stack.recover.verify_contract'),\
             patch.object(model.torch,'load',side_effect=AssertionError('Must reject before CP')):
            with self.assertRaisesRegex(ValueError,'L-only teacher'):model.own_model(c,'cpu')

    def test_source_thresholds_cannot_trade_rejected_damage_or_highconfidence_errors_for_net_gain(self):
        primary=dict(count=144,net=3,sample_highconfidence_wrong=.1,excluded_highconfidence_wrong=.1)
        rejected=dict(count=100,net=0)
        self.assertTrue(evaluate.source_decision(primary,rejected)['passed'])
        for p,r in ((dict(primary,net=2),rejected),(primary,dict(rejected,net=-1)),
                (dict(primary,excluded_highconfidence_wrong=.13),rejected)):
            self.assertFalse(evaluate.source_decision(p,r)['passed'])
        with self.assertRaises(ValueError):evaluate.source_decision(dict(primary,count=720),rejected)

    def test_source_only_publisher_uses_new_training_dispatch_and_restores_transport(self):
        from experiments.cvs_urban_bridge50 import publish as previous
        old_d,old_remote=previous.d,previous.REMOTE
        def inspect(output):
            self.assertIs(previous.d,d);self.assertIn('experiments.cvs_crossfit_teacher50.dispatch',previous.REMOTE)
            self.assertIn('experiments/cvs_crossfit_teacher50/registration.json',previous.REMOTE)
            return dict(output=output)
        with patch.object(previous,'publish',inspect):self.assertEqual(publish.publish('no-mutation'),dict(output='no-mutation'))
        self.assertIs(previous.d,old_d);self.assertEqual(previous.REMOTE,old_remote)


if __name__=='__main__':unittest.main()
