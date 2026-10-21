"""Focused scientific-permission and scheduling regressions for the source adapter."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.cric_dura_comparison.f_data import context, SourceDataset
from experiments.cric_dura_comparison.f_model import contrast
from experiments.cric_dura_comparison.source_queue import allow_shared, row_configs
from experiments.cric_dura_comparison.plan import make_plan
from experiments.cric_dura_comparison.data import bind_classes
from experiments.cric_dura_comparison.smoke import forward_source


class SourceContracts(unittest.TestCase):
    def test_native_source_smoke_retains_tensor_version_counters(self):
        class Versioned(torch.nn.Module):
            def forward(self, x):
                value = x + 1
                _ = value._version
                return value
        net=Versioned()
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'version counter'):
                net(torch.zeros(2))
        result=forward_source(net,torch.zeros(2,requires_grad=True))
        self.assertTrue(torch.equal(result,torch.ones(2)))
        self.assertFalse(result.requires_grad)

    def test_legacy_class_map_is_taken_from_validated_cache_metadata(self):
        original = dict(num_classes=2, role_ids={'L_s':['a'], 'U_s':['b'], 'V':['c']})
        normalized = bind_classes(original, dict(classes=['TX-A','TX-B']))
        self.assertEqual(normalized['classes'], ['TX-A','TX-B'])
        self.assertEqual(normalized['role_ids'], original['role_ids'])
        self.assertNotIn('classes', original)
        with self.assertRaisesRegex(ValueError, 'class'):
            bind_classes(original, dict(classes=['TX-A']))
        with self.assertRaisesRegex(ValueError, 'class'):
            bind_classes(dict(original, classes=['TX-B','TX-A']), dict(classes=['TX-A','TX-B']))

    def test_supervised_does_not_open_U(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = dict(counts={'L_s': 128, 'V': 4, 'U_s': 8}, source_rxs=[1], source_days=[1],
                role_ids={'L_s': ['L%d' % i for i in range(128)], 'V': ['V%d' % i for i in range(4)]}, classes=list('abcdef'))
            for role in ('L_s', 'V'):
                n = contract['counts'][role]
                np.save(root / (role + '_x.npy'), np.zeros((n, 2, 256), np.float32))
                meta = {k: np.ones(n, np.int64) for k in ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain', 'y')}
                np.savez(root / (role + '_metadata.npz'), ids=np.asarray(contract['role_ids'][role]), **meta)
            path = root / 'contract.json'; path.write_text(json.dumps(contract), encoding='utf-8')
            c = dict(source_contract=str(path), source_cache=folder, source_rxs=[1], source_days=[1], regime='matched_sat_supervised', model_seed=7)
            self.assertIsNone(context(c)['unlabeled_loader'])
            with self.assertRaisesRegex(ValueError, 'rejects'):
                context(dict(c, target_truth='forbidden'))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                context(dict(c, source_rxs=[3]))

    def test_U_truth_rejected_before_IQ(self):
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder) / 'U_s_metadata.npz', y=np.asarray([0]))
            with self.assertRaisesRegex(ValueError, 'U truth'):
                SourceDataset(folder, 'U_s', {})

    def test_single_physical_RX_has_no_positive(self):
        h = torch.randn(8, 4, 16, requires_grad=True)
        value, stats = contrast({'shallow': h}, torch.arange(8).long() % 2, torch.ones(8).long())
        value.backward()
        self.assertEqual(float(value), 0.)
        self.assertEqual(stats[0]['valid_anchors'], 0)
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_fixed_matrix_no_new_five_source_sweep(self):
        plan = make_plan()
        handles = {pid: dict(classes=list('abcdef') if p['dataset']=='ManySig' else list(range(10)),
            source_cache='source/' + pid, source_contract='contract/' + pid) for pid,p in plan['protocols'].items()}
        rows = row_configs(dict(run_root='/project/runs/main', project='/project', code_commit='fixed', frozen_sources={}), plan, handles)
        self.assertEqual(len(rows), 195)
        self.assertEqual(sum(bool(r['ablation']) for r in rows), 45)
        self.assertFalse(any(r['protocol_id'].startswith('m5-') for r in rows))
        self.assertFalse(any(r['protocol_id']=='F0' and r['ablation'] is None and r['regime']=='native_cric' for r in rows))

    def test_shared_capacity_keeps_two_process_limit(self):
        self.assertTrue(allow_shared(1, 12000))
        self.assertTrue(allow_shared(0, 20000))
        self.assertFalse(allow_shared(2, 20000))
        self.assertFalse(allow_shared(1, 11999))


if __name__ == '__main__':
    unittest.main()
