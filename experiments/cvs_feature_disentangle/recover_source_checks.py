"""Focused recovery provenance and complete-budget regression checks."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import recover_source as r
from . import design as d


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.old = d.config(d.rows()[0])
        self.old['run_id'] = r.OLD_RUN
        self.old['output_root'] = str(self.path / 'old')
        self.new = deepcopy(self.old)
        self.new.update(run_id='diagnostic-recovery-test', output_root=str(self.path / 'new'),
            checkpoint_origin=dict(config=deepcopy(self.old), commit=r.OLD_COMMIT, mode=r.MODE))

    def logs(self, steps=44400, bad_step=None, epochs=200):
        self.path.mkdir(exist_ok=True)
        with gzip.open(self.path / 'step_metrics.jsonl.gz', 'wt', encoding='utf-8') as handle:
            for step in range(1, steps + 1):
                value = dict(step=step, epoch=(step - 1) // 222 + 1, metrics={'train/loss': 1.})
                if step == bad_step: value['metrics']['train/skipped_nonfinite_grad'] = 1
                handle.write(json.dumps(value) + '\n')
        (self.path / 'epoch_metrics.jsonl').write_text(''.join(json.dumps(dict(epoch=i, epoch_time_s=2.)) + '\n'
            for i in range(1, epochs + 1)), encoding='utf-8')

    def test_complete_budget_and_measured_time(self):
        self.logs()
        value = r.inspect_logs(self.path)
        self.assertEqual(value['logged_steps'], 44400)
        self.assertEqual(value['training_epoch_seconds'], 400.)

    def test_missing_update_rejected(self):
        self.logs(44399)
        with self.assertRaisesRegex(ValueError, 'Incomplete'): r.inspect_logs(self.path)

    def test_skipped_update_rejected(self):
        self.logs(bad_step=19)
        with self.assertRaisesRegex(ValueError, 'Failed optimizer'): r.inspect_logs(self.path)

    def test_incomplete_epoch_rejected(self):
        self.logs(epochs=199)
        with self.assertRaisesRegex(ValueError, 'Incomplete'): r.inspect_logs(self.path)

    def test_config_pair_exact_and_output_exclusive(self):
        with patch.object(d, 'validate', lambda c: c):
            r.validate_pair(self.new, self.old)
            changed = deepcopy(self.new); changed['feature_recipe']['style_eta'] = .9
            with self.assertRaises(ValueError): r.validate_pair(changed, self.old)
            Path(self.new['output_root']).mkdir()
            with self.assertRaises(FileExistsError): r.validate_pair(self.new, self.old)

    def test_wrong_run_and_contaminated_origin_rejected(self):
        with patch.object(d, 'validate', lambda c: c):
            changed = deepcopy(self.old); changed['run_id'] = 'unregistered-origin'
            new = deepcopy(self.new); new['checkpoint_origin']['config'] = changed
            with self.assertRaises(ValueError): r.validate_pair(new, changed)
            changed = deepcopy(self.old); changed['target_access'] = True
            new = deepcopy(self.new); new['target_access'] = True; new['checkpoint_origin']['config'] = changed
            with self.assertRaisesRegex(ValueError, 'CONTAMINATED'): r.validate_pair(new, changed)

    def test_missing_mechanism_not_promoted(self):
        c = d.config(next(row for row in d.rows() if row['arm'] == 'S_dir5'))
        execution = dict(fit_steps={}, attempted_identity_steps={}, exposure_totals={},
            feature_counts=dict(relation_calls=5550, fishr_statistics_updates=5550, fishr_positive_weight_calls=4995),
            fishr_calibration=dict(lambda_max=.1))
        self.assertEqual(r.mechanism_audit(c, execution)['status'], 'VERIFIED')
        execution['feature_counts']['fishr_positive_weight_calls'] = 4994
        self.assertIn('fishr_positive_expected_4995', r.mechanism_audit(c, execution)['missing'])

    def artifacts(self):
        import torch
        source = Path(self.old['output_root']); source.mkdir()
        original = self.path; self.path = source
        try: self.logs()
        finally: self.path = original
        args = dict(baseline_ckpt='', teacher_ckpt='', from_scratch=True,
            output_dir=str(source), candidate_id=self.old['row_id'], run_id=r.OLD_RUN,
            seed=self.old['model_seed'], wisig_pkl=self.old['dataset'],
            use_unlabeled=False, use_ema_teacher=False, use_concat_sat_channel_aug=False,
            label_smoothing=0., label_epochs=200, pseudo_epochs=0, a1_source_screen_only=True,
            a1_periodic_target_inputs='', a1_periodic_target_truth='', a1_periodic_target_start=0,
            a1_periodic_target_interval=0, a1_final_weak_reference=False)
        ck = dict(epoch=200, candidate_id=self.old['row_id'], run_id=r.OLD_RUN,
            checkpoint_selection='final_only', args=args, a1_ema_successful_updates=44400)
        auxiliary = dict(epoch=200, config=self.old, target_access=False,
            mechanism_execution=dict(fit_steps={}, attempted_identity_steps={}, exposure_totals={}, feature_counts={}))
        torch.save(ck, source / 'final_ssdg.pth'); torch.save(auxiliary, source / 'auxiliary_final.pth')
        contract = dict(role_ids={'L': [1], 'V': [2]}, class_txs=[0, 1, 2, 3, 4, 5])
        self.contract = original / 'expected_contract.json'
        self.contract.write_text(json.dumps(contract), encoding='utf-8')
        for name, value in [('source_contract.json', dict(contract, native_role_comparison='EXACT_MATCH')),
            ('resolved_config.json', dict(self.old, commit=r.OLD_COMMIT)), ('resolved_native_args.json', args),
            ('initialization.json', dict(scratch_only=True, checkpoint_sources=[], ancestors=[], target_contact=False,
                                       source_roles='EXACT_MATCH', seed=self.old['model_seed']))]:
            (source / name).write_text(json.dumps(value), encoding='utf-8')
        (source / 'epoch_metrics.csv').write_text('epoch\n', encoding='utf-8')
        return source, ck, auxiliary

    def test_actual_checkpoint_identity_and_auxiliary_contract(self):
        import torch
        source, ck, auxiliary = self.artifacts()
        with patch.object(d, 'SOURCE', str(self.contract)):
            self.assertEqual(r.inspect_artifacts(self.old)[2]['logged_steps'], 44400)
            ck['run_id'] = 'wrong-run'; torch.save(ck, source / 'final_ssdg.pth')
            with self.assertRaisesRegex(ValueError, 'Checkpoint actual'): r.inspect_artifacts(self.old)
            ck['run_id'] = r.OLD_RUN; torch.save(ck, source / 'final_ssdg.pth')
            auxiliary['target_access'] = True; torch.save(auxiliary, source / 'auxiliary_final.pth')
            with self.assertRaisesRegex(ValueError, 'Auxiliary final-state'): r.inspect_artifacts(self.old)

    def test_actual_source_contract_and_initialization_rejected(self):
        source, _, _ = self.artifacts()
        with patch.object(d, 'SOURCE', str(self.contract)):
            contract = r.read(source / 'source_contract.json')
            contract['role_ids']['L'] = [3]
            (source / 'source_contract.json').write_text(json.dumps(contract), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'DATA_CONTRACT'): r.inspect_artifacts(self.old)
            contract['role_ids']['L'] = [1]
            (source / 'source_contract.json').write_text(json.dumps(contract), encoding='utf-8')
            init = r.read(source / 'initialization.json'); init['target_contact'] = True
            (source / 'initialization.json').write_text(json.dumps(init), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'PROVENANCE'): r.inspect_artifacts(self.old)


if __name__ == '__main__': unittest.main()
