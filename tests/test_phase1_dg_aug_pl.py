import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.phase1_external_dg_aug.plan import make_plan, validate_plan
from experiments.phase1_external_dg_aug.pl import pseudo_targets, masked_pl_loss
from experiments.phase1_external_dg_aug.augmentation import DefaultConcat


class AugmentedMatrixTests(unittest.TestCase):
    def test_all_pl_rows_keep_satellite_and_matched_supervised_control(self):
        plan = make_plan()
        self.assertEqual(len(plan['rows']), 219)
        self.assertEqual(plan['allowed_gpus'], [2, 3])
        bad_gpu=copy.deepcopy(plan);bad_gpu['allowed_gpus']=[0,1]
        with self.assertRaises(ValueError):validate_plan(bad_gpu)
        self.assertEqual(sum(r['U_s_use'] for r in plan['rows']), 27)
        self.assertTrue(all(r['satellite_train_augmentation'] for r in plan['rows']))
        self.assertTrue(all(r['stages']==(2 if r['family']=='ibrff' else 1) for r in plan['rows']))
        for changed in ('satellite_train_augmentation', 'default_head_joint_training'):
            bad = copy.deepcopy(plan); bad['rows'][0][changed] = False
            with self.assertRaises(ValueError): validate_plan(bad)
        bad = copy.deepcopy(plan)
        next(r for r in bad['rows'] if r['U_s_use'])['lr'] *= 2
        with self.assertRaises(ValueError): validate_plan(bad)

    def test_pl_acceptance_does_not_update_batchnorm_or_require_labels(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 2))
        model.train(); before = {k: v.clone() for k, v in model.state_dict().items()}
        labels, accepted, confidence = pseudo_targets(model, model, torch.ones(4, 2))
        self.assertTrue(model.training)
        self.assertFalse(confidence.requires_grad)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertEqual(labels.shape, accepted.shape)

    def test_unaccepted_pseudo_labels_have_zero_gradient(self):
        logits = torch.tensor([[4., 0.], [0., 4.]], requires_grad=True)
        loss = masked_pl_loss(logits, torch.tensor([0, 1]), torch.tensor([True, False]))
        loss.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)
        empty = torch.randn(3, 2, requires_grad=True)
        zero = masked_pl_loss(empty, torch.tensor([0, 1, 0]), torch.zeros(3, dtype=torch.bool))
        zero.backward(); self.assertEqual(float(zero), 0.)
        self.assertEqual(float(empty.grad.abs().sum()), 0.)

    def test_real_default_head_updates_with_concat_and_pl_in_same_step(self):
        from experiments.phase1_external_dg_aug.models import build
        from experiments.phase1_external_dg_aug.objectives import step
        from experiments.phase1_dg_nativebudget.pipeline import seed
        row = next(r for r in make_plan()['rows'] if r['group']=='F0-pl' and r['family']=='drift' and r['variant']=='ce')
        c = dict(row, num_classes=6, num_domains=5)
        seed(c['model_seed']); net = build(c)
        parameters = {name: p.detach().clone() for name, p in net.named_parameters()}
        optimizer = torch.optim.Adam(net.parameters(), lr=c['lr'])
        x = torch.randn(6, 2, 256)
        result = step(net, c, x, torch.arange(6), torch.arange(6)%5, optimizer,
            DefaultConcat(), .5, 1, unlabeled=torch.randn(6, 2, 256))
        self.assertTrue(result['pl_enabled'])
        self.assertEqual(result['pl_seen'], 6)
        self.assertEqual(result['satellite_samples'], 6)
        self.assertEqual(result['satellite_weight'], .68)
        changed = [name for name, p in net.named_parameters() if not torch.equal(p.detach(), parameters[name])]
        self.assertTrue(any(name.startswith('encoder.') for name in changed))
        self.assertTrue(any(name.startswith('tx_classifier.') for name in changed))
        self.assertFalse(any(name.startswith('rx_classifier.') for name in changed))
        with self.assertRaises(ValueError):
            step(net, c, x, torch.arange(6), torch.arange(6)%5, optimizer, DefaultConcat(), .5, 2)

    def test_default_scene_stages_scale_to_fixed_progress(self):
        aug = DefaultConcat()
        self.assertEqual(aug.generator.stage_for_epoch(40)[0], 0)
        self.assertEqual(aug.generator.stage_for_epoch(41)[0], 1)
        self.assertEqual(aug.generator.stage_for_epoch(90)[0], 1)
        self.assertEqual(aug.generator.stage_for_epoch(91)[0], 2)
        x = torch.randn(4, 2, 256)
        for batch in range(10):
            transformed, meta = aug(x, 0.5, batch)
            if meta['applied']: break
        self.assertTrue(meta['applied'])
        self.assertFalse(torch.equal(transformed, x))
        self.assertEqual(transformed.shape, x.shape)
        self.assertEqual(meta['weight'], .68)
        self.assertTrue(torch.isfinite(transformed).all())

    def test_both_native_alternating_and_adversarial_objectives_keep_joint_pl(self):
        from experiments.phase1_external_dg_aug.models import build
        from experiments.phase1_external_dg_aug.objectives import step
        from experiments.phase1_dg_nativebudget.pipeline import seed
        for family in ('riei', 'drift'):
            row = next(r for r in make_plan()['rows'] if r['group']=='F0-pl' and r['family']==family and r['variant']=='full')
            c = dict(row, num_classes=6, num_domains=5)
            seed(c['model_seed']); net = build(c)
            fed = None
            if family=='riei':
                optimizer=torch.optim.SGD(list(net.ec.parameters())+list(net.rc.parameters()),lr=c['lr'])
                fed=torch.optim.SGD(net.fed.parameters(),lr=c['lr'])
            else: optimizer=torch.optim.Adam(net.parameters(),lr=c['lr'])
            result=step(net,c,torch.randn(6,2,256),torch.arange(6),torch.arange(6)%5,
                optimizer,DefaultConcat(),.5,1,fed_optimizer=fed,
                unlabeled=torch.randn(6,2,256),unlabeled_receivers=torch.arange(6)%5)
            self.assertEqual(result['pl_seen'],6)
            self.assertEqual(sum(v[0] for v in result['pl_receiver_counts'].values()),6)
            self.assertEqual(result['optimizer_steps'],3 if family=='riei' else 1)
            self.assertTrue(all(p.requires_grad for p in net.parameters()))

    def test_source_rejects_target_and_old_no_augmentation_config_before_open(self):
        from experiments.phase1_external_dg_aug.source import check_training
        row=copy.deepcopy(make_plan()['rows'][0])
        with self.assertRaises(ValueError):check_training(dict(row,target_truth='forbidden'))
        row['satellite_train_augmentation']=False
        with self.assertRaises(ValueError):check_training(row)

    def test_cost_matrix_covers_both_augmented_regimes_without_duplicate_seed_groups(self):
        from collections import defaultdict
        from experiments.phase1_external_dg_aug.cost import tasks, policy
        selected=tasks();self.assertEqual(len(selected),96)
        groups=defaultdict(list)
        for t in selected:groups[(t['origin'],t['protocol_id'],t['family'],t['variant'])].append(t['model_seed'])
        self.assertEqual(len(groups),32)
        self.assertTrue(all(sorted(v)==[2026092701,2026092702,2026092703] for v in groups.values()))
        with self.assertRaises(ValueError):
            policy(dict(allowed_gpus=[2,3],batches=[1,128],warmup=100,repeats=1000,truth='forbidden'))


if __name__ == '__main__': unittest.main()
