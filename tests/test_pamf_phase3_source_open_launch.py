import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class SourceOpenLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_open_launch')

    def runtime(self, root):
        return dict(run_id='source-open-pilot', run_root=str(root / 'source-open-pilot'),
            release_root=str(root), python='python', code_commit='a' * 40, launch_owner='source-open-owner',
            allowed_gpus=[2, 3], max_added_workers_per_gpu=2, minimum_free_mb=4096,
            export=dict(model_seed=2026092701, base_release=str(root / 'base'),
                frozen_sources={name: str(root / name) for name in ['F', 'AF', 'BASE', 'FOLD', 'G']},
                source_output=str(root / 'full-six-class'), source_contract=str(root / 'source-contract.json'),
                registered_classes=['old%d' % i for i in range(6)], batch_size=128,
                source_cache=str(root / 'source-cache'), retention=.99, oe_data_root=str(root / 'data/source_oe')))

    def markers(self):
        common = dict(source_only=True, target_reads=0)
        names = ['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe']
        return dict(export=dict(common, status='SOURCE_OPEN_FEATURES_EXPORTED', head_changed=False,
                    model_classes=6, parameters=232993, optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0,
                    counts=dict(source_L=6300, source_V=27000, exposure_train=4200, novel_validation=3150)),
            fit=dict(common, status='SOURCE_OPEN_READOUTS_FROZEN', encoder_updates=0, identity_head_updates=0,
                readout_steps_per_model=200, source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0,
                method_names=names, learned_method_names=['linear_oe', 'mlp16_oe'],
                parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633}),
            predict=dict(common, status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', known_count=27000, novel_count=3150,
                count=30150, method_names=names, truth_read=False, query_fit=False,
                encoder_updates=0, identity_head_updates=0, readout_updates=0, source_L_reads=0, exposure_train_reads=0,
                source_V_labels_read=False, OE_truth_reads=0),
            score=dict(common, status='SCORED_SOURCE_OPEN_COMPLETE', truth_last=True,
                eligible_methods=[], selected_method=None, next_step='NO_PROMOTION', final_phase3_complete=False))

    def child_factory(self, rt, markers=None, fail=None, exit_stage=None):
        root = Path(rt['run_root']); seen, active = [], [0, 0]
        stages = self.markers() if markers is None else markers
        class Child:
            pid = 2345
            def __init__(child, command, **kwargs):
                if command[4].endswith('.source_open_export'): kind = 'export'
                elif command[4].endswith('.source_open_run'): kind = command[5]
                elif command[4].endswith('.source_open_scoring'): kind = 'score'
                else: raise AssertionError('Unexpected command: ' + str(command))
                child.kind = kind
                self.assertEqual(len(seen), ['export', 'fit', 'predict', 'score'].index(kind))
                active[0] += 1; active[1] = max(active)
                seen.append((kind, command, kwargs))
                kwargs['stdout'].write('simulated ' + kind + '\n')
                folder = root / kind; folder.mkdir()
                marker = dict(stages[kind])
                if fail == kind: marker['status'] = 'INCOMPLETE'
                (folder / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
            def wait(child):
                active[0] -= 1; return 9 if child.kind == exit_stage else 0
        return Child, seen, active

    def test_full_four_stage_pipeline_and_no_promotion_complete(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rt = self.runtime(root); child, seen, active = self.child_factory(rt)
            with patch.object(module.subprocess, 'Popen', child), patch.object(module, 'choose_gpu', return_value=3) as gpu:
                state = module.dispatch(rt)
            self.assertEqual([kind for kind, _, _ in seen], ['export', 'fit', 'predict', 'score'])
            self.assertEqual(gpu.call_count, 1); self.assertEqual(active[1], 1)
            self.assertEqual([kwargs['env']['CUDA_VISIBLE_DEVICES'] for _, _, kwargs in seen], ['3', '', '', ''])
            self.assertTrue(all(kwargs['cwd'] == rt['release_root'] for _, _, kwargs in seen))
            run = Path(rt['run_root'])
            self.assertIn(str(run / 'configs/export.json'), seen[0][1])
            self.assertEqual(seen[0][1][-2:], ['--device', 'cuda:0'])
            self.assertIn(str(run / 'export'), seen[1][1]); self.assertIn('--model-seed', seen[1][1])
            self.assertIn(str(run / 'fit'), seen[2][1]); self.assertIn(str(run / 'predict/predictions.npz'), seen[3][1])
            self.assertIn(str(run / 'export/source_V/features.npz'), seen[3][1])
            self.assertIn(str(root / 'data/source_oe/novel_validation/truth/truth.npz'), seen[3][1])
            config = json.loads((run / 'configs/export.json').read_text())
            self.assertEqual(config['row_id'], 'source-open-readout-s2026092701')
            self.assertEqual(config['oe_data_root'], rt['export']['oe_data_root'])
            self.assertEqual(state['status'], 'ANALYZED_SOURCE_ONLY'); self.assertEqual(state['next_step'], 'NO_PROMOTION')
            self.assertIs(state['final_phase3_complete'], False); self.assertEqual(state['foreign_processes_stopped'], 0)
            self.assertEqual(json.loads((run / 'pipeline_state.json').read_text()), state)
            with self.assertRaises(FileExistsError): module.dispatch(rt)
            self.assertEqual(len(seen), 4)

    def test_wrong_stage_markers_or_exit_stop_and_preserve_outputs(self):
        module = self.module()
        for bad in ['export', 'fit', 'predict', 'score', 'fit-exit']:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                rt = self.runtime(Path(tmp)); stage = 'fit' if bad == 'fit-exit' else bad
                child, seen, _ = self.child_factory(rt, fail=None if bad == 'fit-exit' else stage,
                    exit_stage='fit' if bad == 'fit-exit' else None)
                with patch.object(module.subprocess, 'Popen', child), patch.object(module, 'choose_gpu', return_value=2), \
                     self.assertRaises((ValueError, RuntimeError)):
                    module.dispatch(rt)
                run = Path(rt['run_root']); state = json.loads((run / 'pipeline_state.json').read_text())
                self.assertEqual(state['status'], 'STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE')
                self.assertEqual(state['jobs'][stage]['status'], 'FAILED')
                self.assertEqual(len(seen), ['export', 'fit', 'predict', 'score'].index(stage) + 1)
                self.assertTrue((run / 'logs/failure.log').is_file()); self.assertTrue((run / 'logs' / (stage + '.log')).is_file())
                self.assertEqual(state['foreign_processes_stopped'], 0)

    def test_stage_source_boundaries_and_static_controls_cannot_promote(self):
        module = self.module()
        cases = [('export', 'model_classes', 4), ('export', 'optimizer_updates', 1),
            ('export', 'original_U_s_reads', 1), ('fit', 'source_validation_reads', 1),
            ('fit', 'novel_validation_reads', 1), ('fit', 'readout_steps_per_model', 201),
            ('predict', 'truth_read', True), ('predict', 'query_fit', True),
            ('predict', 'source_V_labels_read', True), ('predict', 'known_count', 26999),
            ('score', 'eligible_methods', ['cosine']), ('score', 'selected_method', 'linear_oe'),
            ('export', 'target_reads', False)]
        for stage, key, value in cases:
            with self.subTest(stage=stage, key=key):
                markers = self.markers(); markers[stage][key] = value
                with self.assertRaises(ValueError): module._validate_marker(stage, markers[stage])
        score = self.markers()['score']
        for eligible, selected in [(['linear_oe'], 'linear_oe'), (['mlp16_oe'], 'mlp16_oe'), (['linear_oe', 'mlp16_oe'], 'linear_oe')]:
            module._validate_marker('score', dict(score, eligible_methods=eligible, selected_method=selected,
                next_step='SOURCE_OPEN_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'))
        with self.assertRaises(ValueError):
            module._validate_marker('score', dict(score, eligible_methods=['linear_oe', 'mlp16_oe'], selected_method='mlp16_oe',
                next_step='SOURCE_OPEN_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION'))

    def test_runtime_whitelist_seed_capacity_and_paths(self):
        module = self.module()
        for fault in ['runtime_target', 'export_target', 'cap', 'gpu_bool', 'seed', 'four_classes', 'commit']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                rt = self.runtime(Path(tmp))
                if fault == 'runtime_target': rt['target_root'] = 'forbidden'
                elif fault == 'export_target': rt['export']['truth'] = 'forbidden'
                elif fault == 'cap': rt['max_added_workers_per_gpu'] = 3
                elif fault == 'gpu_bool': rt['allowed_gpus'] = [True]
                elif fault == 'seed': rt['export']['model_seed'] += 1
                elif fault == 'four_classes': rt['export']['registered_classes'] = ['a', 'b', 'c', 'd']
                else: rt['code_commit'] = 'not-a-commit'
                with self.assertRaises(ValueError): module.validate_runtime(rt)
                self.assertFalse(Path(rt['run_root']).exists())

    def test_foreign_processes_do_not_exclude_card_or_trigger_kill(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); commands = []
            def output(command, **kwargs):
                commands.append(command); return '2, 6000\n3, 5000\n'
            with patch.object(module.subprocess, 'check_output', side_effect=output), \
                 patch.object(module.os, 'kill', side_effect=AssertionError('Never stop foreign work')):
                self.assertEqual(module.choose_gpu(rt), 2)
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][:2], ['nvidia-smi', '--query-gpu=index,memory.free'])


if __name__ == '__main__':
    unittest.main()
