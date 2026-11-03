import unittest
from types import SimpleNamespace
import torch
from experiments.phase1_iterative_margin_fullsource.replay import model_for_schema, paired_outputs, validate_source_manifest
from experiments.phase1_dura_iterative_timefreq.model import IterativeDura


class PairedModelTest(unittest.TestCase):
    def test_only_registered_schemas_map_to_dura_model(self):
        first = model_for_schema('phase1_dura_iterative_timefreq_v1', factory=lambda: SimpleNamespace(kind='DURA'))
        second = model_for_schema('phase1_dura_iterative_margin_v1', factory=lambda: SimpleNamespace(kind='DURA'))
        self.assertEqual(first.kind, 'DURA')
        self.assertEqual(second.kind, 'DURA')
        self.assertIs(type(model_for_schema('phase1_dura_iterative_margin_v1')), IterativeDura)
        with self.assertRaises(ValueError):
            model_for_schema('phase1_iterative_timefreq_v1')

    def test_both_inputs_must_have_frozen_dura_and_no_truth_reads(self):
        meta = dict(status='SOURCE_PREDICTIONS_FIXED', commit='x', count=672, fit_ids=8192,
                    F_frozen=True, DURA_loaded=True, DURA_frozen=True, TX_labels_read=0,
                    U_s_truth_reads=0, V_reads=0, test_reads=0, model_updates_after_prediction=0)
        validate_source_manifest(meta, 'x')
        for field in ('DURA_loaded', 'DURA_frozen', 'F_frozen'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source_manifest(dict(meta, **{field: False}), 'x')
        with self.assertRaises(ValueError):
            validate_source_manifest(dict(meta, U_s_truth_reads=1), 'x')

    def test_both_models_use_dura_anchor_and_closed_gate_preserves_it(self):
        x = torch.randn(3, 2, 256)
        anchor = torch.randn_like(x)
        active = torch.tensor([False, True, True])
        calls = []
        def blind(z):
            calls.append(len(z))
            return torch.zeros(len(z), 4), torch.ones(len(z))
        def first(z, start, lv):
            self.assertTrue(torch.equal(start, anchor[active]))
            return start + 1
        def second(z, start, lv):
            self.assertTrue(torch.equal(start, anchor[active]))
            return start + 2
        a, b = paired_outputs(x, anchor, active, [first, second], blind)
        self.assertEqual(calls, [2])
        self.assertTrue(torch.equal(a[~active], anchor[~active]))
        self.assertTrue(torch.equal(b[~active], anchor[~active]))
        self.assertTrue(torch.equal(a[active], anchor[active] + 1))
        self.assertTrue(torch.equal(b[active], anchor[active] + 2))
        calls.clear()
        paired_outputs(x, anchor, torch.zeros(3, dtype=torch.bool), [first, second], blind)
        self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
