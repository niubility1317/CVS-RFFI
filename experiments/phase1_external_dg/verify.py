"""Focused correctness checks for non-PAMF roles, matrix and regularizers."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .plan import make_plan, protocol, validate_plan
from .data import iq_record, sample_id, selections, build_source, make_wrapper, read
from .models import (RoutedResNet, balanced_indices, linear_ce_sample_gradients,
                     FishrVariance, source_coral, mix_statistics)


def fake_dataset():
    generator = np.random.default_rng(9)
    return dict(tx_list=['a', 'b'], rx_list=['r0', 'r1', 'r2'],
        capture_date_list=['d0', 'd1', 'd2', 'd3'], equalized_list=[0, 1],
        data=[[[[None, generator.normal(size=(100, 256, 2)).astype(np.float32)]
                 for _ in range(4)] for _ in range(3)] for _ in range(2)])


class RolesAndMatrix(unittest.TestCase):
    def test_complete_fixed_scope_and_seed_counts(self):
        plan = make_plan()
        self.assertEqual(len(plan['rows']), 168)
        self.assertEqual(sum(x['budget_kind'] != 'frozen_eval' for x in plan['rows']), 162)
        self.assertTrue(plan['exclude_pamf'])
        self.assertTrue(all('pamf' not in r['family'].lower() for r in plan['rows']))
        broken = json.loads(json.dumps(plan)); broken['allowed_gpus'] = [0, 1, 6]
        with self.assertRaises(ValueError): validate_plan(broken)
        broken = json.loads(json.dumps(plan)); broken['rows'].pop()
        with self.assertRaises(ValueError): validate_plan(broken)

    def test_role_counts_and_no_hidden_label_export(self):
        data = fake_dataset()
        p = protocol('test', [0, 1]); p['target_rxs'] = [2]
        roles = selections(data, p)
        self.assertEqual({k: len(v) for k, v in roles.items()}, dict(L_s=84, U_s=756, V=360))
        sparse = dict(p, label_per_thousand=7)
        self.assertEqual(len(selections(data, sparse)['L_s']), 0)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'source'; path = build_source(data, p, out)
            self.assertFalse((out / 'U_s_x.npy').exists())
            meta = read(out / 'metadata.json')
            self.assertFalse(meta['target_constructed'])
            self.assertEqual(meta['source_contract'], read(path))
            with np.load(out / 'L_s_index.npz') as index:
                self.assertEqual(set(index['rx'].tolist()), {0, 1})
                self.assertEqual(set(index['y'].tolist()), {0, 1})

    def test_forbidden_role_and_label_changes(self):
        p = protocol('test', [0, 1]); p['target_rxs'] = [1, 2]
        with self.assertRaises(ValueError): selections(fake_dataset(), p)
        p['target_rxs'] = [2]; p['label_per_thousand'] = 200
        with self.assertRaises(ValueError): selections(fake_dataset(), p)

    def test_id_namespace_and_normalization(self):
        self.assertNotEqual(sample_id('ManySig', 0, 0, 0, 1, 0),
                            sample_id('ManyRx', 0, 0, 0, 1, 0))
        x = iq_record(np.ones((256, 2), dtype=np.float32))
        self.assertAlmostEqual(float(np.mean(x[0] ** 2 + x[1] ** 2)), 1., places=6)
        with self.assertRaises(ValueError): iq_record(np.zeros((256, 2)))

    def test_target_source_overlap_is_rejected(self):
        data = fake_dataset(); p = protocol('test', [0]); p['target_rxs'] = [2]
        roles = selections(data, p)
        record = roles['L_s'][0]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / 'source.json'
            path.write_text(json.dumps(dict(role_ids={r: [x[-1] for x in rows]
                                                      for r, rows in roles.items()})))
            spoofed = (0, 2, 0, 1, 0, record[-1])
            with self.assertRaises(ValueError):
                make_wrapper(data, p, path, folder / 'union', [spoofed], folder / 'target')


class Mechanisms(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91); torch.set_num_threads(2)

    def test_default_head_gradients_equal_per_example_autograd(self):
        head = torch.nn.Linear(5, 3).double()
        z = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
        y = torch.arange(8) % 3; logits = head(z)
        analytical = linear_ce_sample_gradients(z, logits, y)
        direct = []
        for i in range(len(z)):
            grads = torch.autograd.grad(F.cross_entropy(logits[i:i+1], y[i:i+1]),
                tuple(head.parameters()), create_graph=True, retain_graph=True)
            direct.append(torch.cat([g.flatten() for g in grads]))
        torch.testing.assert_close(analytical, torch.stack(direct), rtol=1e-12, atol=1e-12)
        penalty = FishrVariance(2)(z, logits, y, torch.arange(8) // 4)
        grad = torch.autograd.grad(penalty, z)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_fishr_centering_and_ema_correction(self):
        z = torch.randn(4, 3, requires_grad=True); head = torch.nn.Linear(3, 2)
        zz = torch.cat((z, z)); y = torch.tensor([0, 1, 0, 1] * 2)
        rx = torch.tensor([0] * 4 + [1] * 4)
        state = FishrVariance(2)
        self.assertAlmostEqual(float(state(zz, head(zz), y, rx)), 0., places=12)
        self.assertTrue(all(not x.requires_grad for x in state.previous))

    def test_sampler_and_mixstyle_identity_protection(self):
        labels = torch.arange(3).repeat(12); rx = torch.arange(3).repeat_interleave(12)
        generator = torch.Generator().manual_seed(2)
        ix = balanced_indices(labels, rx, 3, 3, 18, generator)
        for domain in range(3):
            for label in range(3):
                self.assertEqual(int(((labels[ix] == label) & (rx[ix] == domain)).sum()), 2)
        with self.assertRaises(ValueError): balanced_indices(labels, rx, 3, 3, 17, generator)
        x = torch.randn(6, 4, 16)
        # No same-TX cross-RX partners: transformation is identity.
        result, count = mix_statistics(x, torch.arange(6), torch.arange(6) % 2,
            identity_protected=True, probability=1.)
        self.assertEqual(count, 0)
        torch.testing.assert_close(result, x, rtol=1e-6, atol=1e-6)

    def test_coral_zero_for_identical_domains_and_backprop(self):
        a = torch.randn(5, 7, requires_grad=True)
        z = torch.cat((a, a)); rx = torch.tensor([0] * 5 + [1] * 5)
        self.assertAlmostEqual(float(source_coral(z, rx, 2)), 0., places=12)
        shifted = torch.cat((a, a + 1.))
        penalty = source_coral(shifted, rx, 2)
        self.assertGreater(float(penalty), 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(penalty, a)[0]).all())

    def test_route_has_same_default_head_and_checkpoint_keys(self):
        a = RoutedResNet(route='erm')
        b = RoutedResNet(route='mixstyle_id')
        b.load_state_dict(a.state_dict(), strict=True)
        a.eval(); b.eval(); x = torch.randn(2, 2, 256)
        with torch.no_grad():
            torch.testing.assert_close(a(x)[1], b(x)[1], rtol=0, atol=0)
        self.assertEqual(a.classifier.out_features, 6)
        self.assertEqual(RoutedResNet(classes=10).classifier.out_features, 10)
        self.assertFalse(any('probe' in k or 'diagnostic' in k for k in a.state_dict()))


class TrainingProtocol(unittest.TestCase):
    def fixture(self, folder):
        from .source import check_training
        row = next(r for r in make_plan()['rows'] if r['group'] == 'generic')
        contract = dict(source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3],
            num_classes=6, counts=dict(L_s=6300, U_s=56700, V=27000),
            split_seed=392005, role_ids={}, target_access_before_freeze=False)
        classes = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
        meta = dict(source_contract=contract, classes=classes,
            U_s_exported=False, target_constructed=False)
        folder = Path(folder)
        (folder / 'source_contract.json').write_text(json.dumps(contract))
        (folder / 'metadata.json').write_text(json.dumps(meta))
        row = dict(row, source_cache=str(folder), source_contract_path=str(folder / 'source_contract.json'),
            classes=classes, num_classes=6, num_domains=5)
        return row, meta, check_training

    def test_matching_source_and_forbidden_target_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            row, meta, check = self.fixture(folder)
            self.assertEqual(check(row), meta)
            for key in ('truth', 'target_inputs', 'dataset', 'checkpoint', 'tests'):
                with self.assertRaises(ValueError): check(dict(row, **{key: 'forbidden'}))

    def test_budget_and_group_miswiring_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            row, meta, check = self.fixture(folder)
            with self.assertRaises(ValueError): check(dict(row, updates=20))
            with self.assertRaises(ValueError): check(dict(row, fishr_lambda=0))
            contract = meta['source_contract']; contract['source_rxs'] = [0, 2, 5, 7, 9]
            Path(row['source_contract_path']).write_text(json.dumps(contract))
            (Path(folder) / 'metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError): check(row)

    def test_native_default_heads_follow_dataset_cardinality(self):
        from .source import build, forward
        torch.set_num_threads(2)
        for family in ('riei', 'drift'):
            for classes, domains in [(6, 5), (10, 16)]:
                c = dict(family=family, num_classes=classes, num_domains=domains)
                model = build(c).eval()
                with torch.no_grad(): output = forward(model, c, torch.randn(2, 2, 256))
                self.assertEqual(tuple(output.shape), (2, classes))
                self.assertTrue(torch.isfinite(output).all())

    def test_resnet_default_head_and_backbone_both_receive_gradients(self):
        torch.set_num_threads(2)
        net = RoutedResNet(); x = torch.randn(6, 2, 256); labels = torch.arange(6)
        loss = F.cross_entropy(net(x)[1], labels); loss.backward()
        self.assertGreater(float(net.classifier.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(net.features[0].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
