"""Focused correctness checks for non-PAMF roles, matrix and regularizers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
    def test_fixed_budget_balances_TX_RX_and_dates(self):
        class CountCell:
            def __len__(self): return 1000
        data = dict(tx_list=list(range(6)), equalized_list=[0, 1],
            data=[[[[None, CountCell()] for _ in range(4)] for _ in range(12)] for _ in range(6)])
        p = protocol('count-test', [1, 3, 4], fixed_labels=1260)
        records = selections(data, p)['L_s']
        self.assertEqual(len(records), 1260)
        for tx in range(6):
            for rx in [1, 3, 4]:
                self.assertEqual(sum(r[0] == tx and r[1] == rx for r in records), 70)
            for day in [1, 2, 3]:
                self.assertEqual(sum(r[0] == tx and r[2] == day for r in records), 70)

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


class EvaluationProtocol(unittest.TestCase):
    def fixtures(self, folder):
        from . import evaluation
        root = Path(folder); views = root / 'views'; views.mkdir()
        classes = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
        ids = np.asarray(['id' + str(i) for i in range(8)])
        np.savez(views / 'index.npz', ids=ids)
        rng = np.random.default_rng(4)
        x = rng.normal(size=(8, 2, 256)).astype(np.float32)
        for view in evaluation.VIEWS: np.save(views / (view + '.npy'), x)
        manifest = dict(status='VALIDATED_ONCE', classes=classes, count=8,
            scenes=list(evaluation.VIEWS[1:]), source_target_disjoint=True,
            clean_ref=str(views / 'clean.npy'))
        (views / 'manifest.json').write_text(json.dumps(manifest))
        row = next(r for r in make_plan()['rows'] if r['group'] == 'generic')
        pc = dict(row_id=row['row_id'], model_seed=row['model_seed'], source_output='unused',
            expected_source_contract='unused', output_root=str(root / 'prediction'),
            tests=dict(main=dict(views_root=str(views), kind='unseen_receiver')))
        sc = dict(row_id=row['row_id'], model_seed=row['model_seed'],
            prediction_output=pc['output_root'], output_root=str(root / 'scores'), classes=classes,
            tests=dict(main=dict(views_root=str(views), truth=str(root / 'truth.json'))))
        model = RoutedResNet().eval()
        metadata = dict(row_id=row['row_id'], model_seed=row['model_seed'], family='resnet18',
            num_classes=6, classes=classes)
        return evaluation, pc, sc, model, metadata, ids

    def test_predictor_has_no_truth_and_keeps_model_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluation, pc, sc, model, metadata, ids = self.fixtures(folder)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with patch.object(evaluation, 'load_model', return_value=(model, metadata)):
                evaluation.predict(pc, device='cpu')
            self.assertFalse(Path(sc['tests']['main']['truth']).exists())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                evaluation.predict(dict(pc, truth='forbidden'), device='cpu')
            prepared = evaluation.score_preflight(sc)
            self.assertEqual(set(prepared), {'main'})
            self.assertEqual(len(prepared['main'][0]), 8)

    def test_swapped_prediction_ids_keep_truth_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluation, pc, sc, model, metadata, ids = self.fixtures(folder)
            with patch.object(evaluation, 'load_model', return_value=(model, metadata)):
                evaluation.predict(pc, device='cpu')
            path = Path(pc['output_root']) / 'main/predictions.npz'
            with np.load(path) as p: values = {v: p[v].copy() for v in evaluation.VIEWS}
            np.savez(path, ids=ids[::-1], **values)
            with self.assertRaises(ValueError): evaluation.score(sc)
            self.assertFalse(Path(sc['tests']['main']['truth']).exists())

    def test_missing_chrono_second_task_keeps_truth_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluation, pc, sc, model, metadata, ids = self.fixtures(folder)
            with patch.object(evaluation, 'load_model', return_value=(model, metadata)):
                evaluation.predict(pc, device='cpu')
            row = next(r for r in make_plan()['rows'] if r['group'] == 'chrono')
            sc['row_id'] = row['row_id']
            sc['tests']['source_unseen_date'] = dict(sc['tests']['main'])
            path = Path(pc['output_root']) / 'complete.json'
            done = json.loads(path.read_text()); done['row_id'] = row['row_id']
            done['tests']['source_unseen_date'] = dict(done['tests']['main'])
            path.write_text(json.dumps(done))
            with self.assertRaises(FileNotFoundError): evaluation.score(sc)
            self.assertFalse(Path(sc['tests']['main']['truth']).exists())

    def test_complete_prediction_scores_all_views_and_date_layers(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluation, pc, sc, model, metadata, ids = self.fixtures(folder)
            with patch.object(evaluation, 'load_model', return_value=(model, metadata)):
                evaluation.predict(pc, device='cpu')
            truth = {sid: dict(label=i % 6, receiver='rx' + str(i % 2),
                day='2021_03_01' if i % 2 else '2021_03_08') for i, sid in enumerate(ids)}
            Path(sc['tests']['main']['truth']).write_text(json.dumps(truth))
            evaluation.score(sc)
            results = read(Path(sc['output_root']) / 'scores.json')
            self.assertTrue(results['truth_last'])
            overall = [x for x in results['results'] if x['dimension'] == 'overall']
            self.assertEqual(len(overall), 7)
            for x in overall: self.assertEqual(sum(map(sum, x['confusion'])), 8)
            layers = {x['stratum'] for x in results['results'] if x['dimension'] == 'date_visibility'}
            self.assertEqual(layers, {'seen_dates', 'unseen_dates'})
            for x in results['results']:
                if x['dimension'] == 'transmitter':
                    self.assertIn('recall', x); self.assertNotIn('macro_f1', x)


class QueueAndAudit(unittest.TestCase):
    def test_parent_must_be_terminal_and_actual_process_gone(self):
        from .driver import parent_state
        with tempfile.TemporaryDirectory() as folder:
            state = dict(status='ANALYZED', rows={str(i): dict(status='ANALYZED') for i in range(24)})
            path = Path(folder) / 'pipeline_state.json'; path.write_text(json.dumps(state))
            rt = dict(parent_run_root=folder, parent_pid=1, parent_release_root='parent-owner')
            with patch('experiments.phase1_external_dg.driver.process_alive', return_value=True):
                self.assertEqual(parent_state(rt)[0], 'LIVE')
            with patch('experiments.phase1_external_dg.driver.process_alive', return_value=False):
                self.assertEqual(parent_state(rt)[0], 'DONE')
                state['status'] = 'RUNNING'; path.write_text(json.dumps(state))
                self.assertEqual(parent_state(rt)[0], 'NEEDS_RECOVERY')

    def test_systemic_failure_matching_does_not_use_scores(self):
        from .driver import failure_fingerprint
        with tempfile.TemporaryDirectory() as folder:
            fingerprints = []
            for rid in ['method-s1', 'method-s2']:
                p = Path(folder) / (rid + '.log')
                p.write_text('Traceback:\nValueError: wrong contract in /run/' + rid + '/source\n')
                fingerprints.append(failure_fingerprint(p, rid))
            self.assertEqual(fingerprints[0], fingerprints[1])

    def test_audit_distinguishes_main_and_diagnostic_CMs(self):
        from .analysis import extend_frozen_scores
        from .evaluation import metrics
        truth = dict(a=dict(label=9, receiver='R', day='2021_03_23'),
                     b=dict(label=0, receiver='R', day='2021_03_08'))
        ids = np.asarray(['a', 'b']); predicted = np.asarray([9, 1])
        original = [dict(test='main', view='clean', dimension='overall',
            **metrics(np.asarray([9, 0]), predicted, 10)),
            dict(test='source_unseen_date', view='clean', dimension='overall',
            **metrics(np.asarray([9]), np.asarray([9]), 10))]
        result = extend_frozen_scores(np.asarray(['a']), dict(clean=np.asarray([9])), truth, 10,
            {'2021_03_08'}, test='source_unseen_date', original=original)
        overall = next(x for x in result if x['dimension'] == 'overall')
        self.assertEqual(overall['accuracy'], 1.)
        self.assertEqual(np.asarray(overall['confusion']).shape, (10, 10))
        with self.assertRaises(ValueError):
            extend_frozen_scores(ids, dict(clean=np.asarray([0, 0])), truth, 10,
                {'2021_03_08'}, original=original)

    def test_three_seed_pairing_and_no_duplicate_seed_inflation(self):
        from .analysis import summary_records
        rows = []
        for variant, offset in [('full', .02), ('ce', 0.)]:
            for seed in [1, 2, 3]:
                value = .6 + offset + seed / 100
                rows.append(dict(config=dict(group='G', family='ibrff', variant=variant, model_seed=seed, row_id=variant+str(seed)),
                    results=[dict(test='main', view='clean', dimension='overall', accuracy=value, macro_f1=value),
                             dict(test='main', view='clean', dimension='receiver', accuracy=value-.1),
                             dict(test='main', view='clean', dimension='receiver', accuracy=value+.2)]))
        summary, paired = summary_records(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(paired[0]['actual_pairs'], 3)
        self.assertAlmostEqual(paired[0]['mean'], .02)
        self.assertAlmostEqual(paired[0]['equal_rx_mean'], .02)
        full = next(r for r in summary if r['version'] == 'full')
        self.assertAlmostEqual(full['accuracy_mean'], .64)
        self.assertAlmostEqual(full['equal_rx_accuracy_mean'], .69)
        with self.assertRaises(ValueError): summary_records(rows + [rows[0]])


    def test_joint_receiver_date_counts_and_confusion_cover_the_overall(self):
        from .analysis import extend_frozen_scores
        ids = np.asarray(['a', 'b', 'c', 'd'])
        truth = {name: dict(label=label, receiver=rx, day=day) for name, label, rx, day in
            [('a', 0, 'R1', 'D1'), ('b', 1, 'R1', 'D2'),
             ('c', 0, 'R2', 'D1'), ('d', 1, 'R2', 'D2')]}
        records = extend_frozen_scores(ids, {'clean': np.asarray([0, 1, 1, 1])},
            truth, 2, {'D1'})
        joint = [x for x in records if x['dimension'] == 'receiver_date']
        overall = next(x for x in records if x['dimension'] == 'overall')
        self.assertEqual({x['stratum'] for x in joint}, {'R1|D1', 'R1|D2', 'R2|D1', 'R2|D2'})
        self.assertEqual(sum(x['count'] for x in joint), 4)
        self.assertEqual(np.asarray([x['confusion'] for x in joint]).sum(0).tolist(), overall['confusion'])

    def test_generic_routes_pair_with_same_seed_erm_and_keep_negative_differences(self):
        from .analysis import summary_records
        rows = []
        for route, delta in [('erm', 0.), ('fishr', .01), ('mixstyle', -.02)]:
            for seed in [1, 2, 3]:
                accuracy = .6 + seed * .001 + delta
                rows.append(dict(config=dict(group='generic', family='resnet18', variant='ce',
                    route=route, model_seed=seed, row_id=route+str(seed)), results=[
                    dict(test='main', view='clean', dimension='overall', accuracy=accuracy, macro_f1=accuracy),
                    dict(test='main', view='clean', dimension='receiver', accuracy=accuracy)]))
        _, paired = summary_records(rows)
        indexed = {x['route']: x for x in paired}
        self.assertAlmostEqual(indexed['fishr']['mean'], .01)
        self.assertAlmostEqual(indexed['mixstyle']['mean'], -.02)
        self.assertEqual(indexed['mixstyle']['positive_pairs'], 0)
        self.assertTrue(all(x['actual_pairs'] == 3 and x['baseline'] == 'erm' for x in paired))

    def test_frozen_report_reanalysis_preserves_original_output_and_state(self):
        from .analysis import analyze
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            state = dict(status='ANALYZED', owner='original')
            (source / 'pipeline_state.json').write_text(json.dumps(state), encoding='utf-8')
            (source / 'summary.json').write_text('preserve original', encoding='utf-8')
            output = Path(folder) / 'new-report'
            with patch('experiments.phase1_external_dg.analysis.make_plan', return_value=dict(rows=[], protocols={})), \
                    patch('experiments.phase1_external_dg.analysis.parent_audit', return_value=[]):
                analyze(dict(run_root=str(source)), str(output))
                with self.assertRaises(ValueError):
                    analyze(dict(run_root=str(source)), str(source / 'nested'))
                with self.assertRaises(ValueError):
                    analyze(dict(run_root=str(source)), str(output))
            self.assertEqual(json.loads((source / 'pipeline_state.json').read_text()), state)
            self.assertEqual((source / 'summary.json').read_text(), 'preserve original')
            self.assertTrue((output / 'analysis_state.json').exists())
            self.assertFalse((source / 'nested').exists())


class LegacyProvenance(unittest.TestCase):
    def test_legacy_default_checkpoint_load_and_forbidden_ancestor(self):
        from .evaluation import legacy_model
        from experiments.standard_ce_baselines.model import build
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); model = build('resnet18'); count = sum(p.numel() for p in model.parameters())
            seed = 2026092701
            contract = dict(source_rxs=[1,3,4,6,8], source_days=[1,2,3], equalized=1,
                out_len=256, normalize=True, classes=['a','b','c','d','e','f'])
            initial = dict(scratch_only=True, checkpoint=None, ancestors=[], checkpoint_sources=[],
                target_access=False, target_contact=False, model_seed=seed, physical_roles='EXACT_MATCH',
                selection='fixed_last_epoch')
            resolved = dict(method='standard_ce_baselines', variant='resnet18', model_seed=seed,
                epochs=200, batch_size=128, lr=.0002, lr_min=1e-6, weight_decay=.0001,
                drop_last=False, augmentation=False, domain_backbone=False, extra_losses=[],
                selection='fixed_last_epoch', split_seed=392005, trainable_parameters=count)
            freeze = dict(status='FROZEN', epoch=200, variant='resnet18', model_seed=seed,
                checkpoint=str(root/'last.pt'), target_access=False, target_score_used=False)
            done = dict(status='SOURCE_TRAINED', epoch=200, steps=10000, target_access=False)
            for name, content in [('source_contract.json',contract),('initialization.json',initial),
                ('resolved_config.json',resolved),('freeze.json',freeze),('completion.json',done)]:
                (root/name).write_text(json.dumps(content))
            (root/'epoch_metrics.jsonl').write_text(json.dumps(dict(epoch=200,
                optimizer_steps_total=10000, gradient_used_parameters=count))+'\n')
            torch.save(dict(method='standard_ce_baselines', variant='resnet18', epoch=200,
                selection='fixed_last_epoch', config=resolved, initialization=initial,
                source_contract=contract, model=model.state_dict()),root/'last.pt')
            c=dict(row_id='legacy-resnet18-s2026092701', model_seed=seed, source_output=folder)
            net, metadata=legacy_model(c,contract,torch.device('cpu'))
            self.assertEqual(metadata['num_classes'],6)
            self.assertEqual(net.classifier.out_features,6)
            initial['ancestors']=['unknown-parent'];(root/'initialization.json').write_text(json.dumps(initial))
            with self.assertRaises(ValueError):legacy_model(c,contract,torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
