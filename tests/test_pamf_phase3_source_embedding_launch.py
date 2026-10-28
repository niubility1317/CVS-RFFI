import importlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


class EmbeddingLaunchTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_launch.py'
        self.assertTrue(path.is_file(), 'Two-row source embedding launcher is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_launch')

    def runtime(self, root):
        return dict(run_id='20261029-phase3-source-embedding-increment-s2026092701-r01',
            run_root=str(root / '20261029-phase3-source-embedding-increment-s2026092701-r01'),
            release_root=str(root / 'release'), python=str(root / 'python'), code_commit='a' * 40,
            launch_owner='own-source-embedding', parent_run_root=str(root / '20261029-phase3-source-open-readout-s2026092701-r01'),
            parent_oe_data_root=str(root / 'data/source_oe'), diverse_data_root=str(root / 'data/source_oe_diverse'),
            allowed_gpus=[0, 1, 2], max_added_workers_per_gpu=2, minimum_free_mb=4096, max_concurrent_rows=2,
            model=dict(model_seed=2026092701, batch_size=128, source_output=str(root / 'parent-E50'),
                source_contract=str(root / 'source_contract.json'), source_cache=str(root / 'source-cache'),
                base_release=str(root / 'base'), frozen_sources={key: str(root / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')},
                registered_classes=['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']))

    def marker(self, stage, arm, eligible=False):
        module = self.module()
        if stage == 'train':
            return dict(status='SOURCE_OPEN_EMBEDDING_FROZEN', arm=arm, source_only=True, updates=222,
                parent_updates=11100, cumulative_updates=11322, optimizer_reset=True, default_head_joint=True,
                identity_head_replaced=False, model_classes=6, parameters=232993,
                U_reads=0, validation_reads=0, novel_validation_reads=0, target_reads=0, OE_truth_reads=0)
        if stage == 'eval':
            return dict(status='SOURCE_EMBEDDING_PREDICTIONS_COMPLETE', arm=arm, source_only=True,
                target_reads=0, updates=222, encoder_updates=222, identity_head_updates=222,
                default_head_replaced=False, method_names=list(module.METHODS), count=30150,
                known_count=27000, novel_count=3150, source_V_labels_read=True,
                source_known_label_calibration=True, unknown_truth_read=False, query_fit=False)
        return dict(status='SCORED_SOURCE_EMBEDDING_COMPLETE', arm=arm, source_only=True,
            target_reads=0, truth_last=True, final_phase3_complete=False,
            eligible_methods=['msp'] if eligible else [], selected_method='msp' if eligible else None,
            next_step='SOURCE_EMBEDDING_FIRST_FOLD_PASS_REQUIRES_CONFIRMATION' if eligible else 'NO_PROMOTION')

    def child(self, fail_arm=None, bad_marker=False, eligible=False):
        seen, active, lock = [], [0, 0], threading.Lock()
        barrier = threading.Barrier(2)
        owner = self
        class Child:
            pid = 4567
            def __init__(child, command, **kwargs):
                if command[4].endswith('source_open_embedding_train'):
                    stage = 'train'
                elif command[4].endswith('source_embedding_eval'):
                    stage = 'eval'
                else:
                    stage = 'score'
                if stage == 'score':
                    output = Path(command[command.index('--output') + 1])
                else:
                    cfg = json.loads(Path(command[command.index('--config') + 1]).read_text())
                    output = Path(cfg['output_root'])
                child.stage, child.arm, child.output = stage, output.parent.name, output
                with lock:
                    active[0] += 1; active[1] = max(active)
                    seen.append((child.arm, stage, command, kwargs))
                kwargs['stdout'].write(stage + '\n')
            def wait(child):
                if child.stage == 'train':
                    barrier.wait(timeout=5)
                time.sleep(.002)
                child.output.mkdir()
                failed = child.arm == fail_arm and child.stage == 'train'
                if failed and not bad_marker:
                    (child.output / 'partial.pt').write_text('keep failed artifacts')
                    result = 7
                else:
                    marker = owner.marker(child.stage, child.arm, eligible and child.arm == 'uniform_oe')
                    if failed:
                        marker['U_reads'] = 1
                    (child.output / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
                    if child.stage == 'score':
                        metric = dict(unknown_macro_rejection=.8 if child.arm == 'supervised_only' else .9,
                            known_macro_closed_accuracy=.99, known_macro_open_accuracy=.98,
                            known_macro_frr=.01, encoder_closed_loss=.001, total_open_loss=.011)
                        result_body = dict(methods={'msp': {'overall': metric,
                            'by_receiver': {str(rx): metric for rx in (1, 3, 4, 6, 8)}}})
                        (child.output / 'results.json').write_text(json.dumps(result_body), encoding='utf-8')
                    result = 0
                with lock:
                    active[0] -= 1
                return result
        return Child, seen, active

    def test_two_rows_finish_gpu_train_eval_cpu_score_and_scientific_failure_is_complete(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            rt = self.runtime(Path(td)); child, seen, active = self.child()
            with patch.object(module, 'select_gpus', return_value=[1, 2]), \
                    patch.object(module.subprocess, 'Popen', child), \
                    patch.object(module.os, 'kill', side_effect=AssertionError('Never stop foreign work')):
                state = module.dispatch(rt)
            self.assertEqual(active[1], 2)
            for arm, gpu in (('supervised_only', '1'), ('uniform_oe', '2')):
                stages = [(stage, cmd, kw) for row, stage, cmd, kw in seen if row == arm]
                self.assertEqual([stage for stage, _, _ in stages], ['train', 'eval', 'score'])
                self.assertEqual([kw['env']['CUDA_VISIBLE_DEVICES'] for _, _, kw in stages], [gpu, gpu, ''])
                self.assertNotIn('--novel-truth', stages[0][1])
                self.assertNotIn('--novel-truth', stages[1][1])
                train = json.loads((Path(rt['run_root']) / ('configs/' + arm + '.train.json')).read_text())
                evaluation = json.loads((Path(rt['run_root']) / ('configs/' + arm + '.eval.json')).read_text())
                self.assertEqual('diverse_data_root' in train, arm == 'uniform_oe')
                self.assertEqual(set(evaluation), module.eval_module.KEYS)
                self.assertNotIn('truth', json.dumps(train))
                self.assertNotIn('truth', json.dumps(evaluation))
                self.assertIn(str(Path(rt['parent_oe_data_root']) / 'novel_validation/truth/truth.npz'), stages[2][1])
            self.assertEqual(state['status'], 'ANALYZED_SOURCE_ONLY')
            self.assertEqual(state['next_step'], 'NO_PROMOTION')
            self.assertEqual(state['foreign_processes_stopped'], 0)
            self.assertFalse(state['final_phase3_complete'])
            paired = json.loads((Path(rt['run_root']) / 'paired_source_metrics.json').read_text())
            self.assertAlmostEqual(paired['difference_uniform_oe_minus_supervised_only']['overall']['unknown_macro_rejection'], .1)
            self.assertEqual(json.loads((Path(rt['run_root']) / 'pipeline_state.json').read_text())['status'], state['status'])

    def test_one_row_technical_failure_preserves_other_row_and_blocks_global_promotion(self):
        module = self.module()
        for bad_marker in (False, True):
            with self.subTest(bad_marker=bad_marker), tempfile.TemporaryDirectory() as td:
                rt = self.runtime(Path(td)); child, seen, _ = self.child('supervised_only', bad_marker, eligible=True)
                with patch.object(module, 'select_gpus', return_value=[0, 0]), patch.object(module.subprocess, 'Popen', child):
                    state = module.dispatch(rt)
                self.assertEqual([stage for row, stage, _, _ in seen if row == 'supervised_only'], ['train'])
                self.assertEqual([stage for row, stage, _, _ in seen if row == 'uniform_oe'], ['train', 'eval', 'score'])
                self.assertEqual(state['status'], 'PARTIAL_SOURCE_ONLY')
                self.assertEqual(state['eligible_candidates'], [])
                self.assertFalse(state['final_phase3_complete'])
                self.assertTrue((Path(rt['run_root']) / 'rows/supervised_only/logs/failure.log').is_file())
                self.assertTrue((Path(rt['run_root']) / 'rows/uniform_oe/score/complete.json').is_file())

    def test_capacity_uses_free_memory_only_distinct_cards_then_two_slots_on_one(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            rt = self.runtime(Path(td))
            with patch.object(module.subprocess, 'check_output', return_value='0, 9000\n1, 12000\n2, 0\n') as query:
                self.assertEqual(module.select_gpus(rt), [1, 0])
            self.assertEqual(query.call_args.args[0], ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'])
            with patch.object(module.subprocess, 'check_output', return_value='0, 5000\n1, 0\n2, 0\n'):
                self.assertEqual(module.select_gpus(rt), [0, 0])
            with patch.object(module.subprocess, 'check_output', side_effect=['0, 0\n1, 0\n2, 0\n', '0, 5000\n1, 0\n2, 0\n']), \
                    patch.object(module.time, 'sleep') as sleep:
                self.assertEqual(module.select_gpus(rt), [0, 0])
                sleep.assert_called_once_with(20)

    def test_runtime_output_and_promotion_boundaries(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            rt = self.runtime(Path(td)); module.validate_runtime(rt)
            for extra in ({'max_added_workers_per_gpu': 3}, {'max_concurrent_rows': 3}, {'target_truth': 'no'}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    module.validate_runtime(dict(rt, **extra))
            with self.assertRaises(ValueError):
                module.validate_runtime(dict(rt, model=dict(rt['model'], retention=.99)))
            Path(rt['run_root']).mkdir()
            with patch.object(module, 'select_gpus', side_effect=AssertionError('No resource IO')):
                with self.assertRaises(FileExistsError):
                    module.dispatch(rt)
        for arm, method in (('supervised_only', 'msp'), ('uniform_oe', 'energy')):
            marker = self.marker('score', arm, True)
            marker['eligible_methods'] = [method]; marker['selected_method'] = method
            with self.subTest(arm=arm, method=method), self.assertRaises(ValueError):
                module._validate_marker('score', marker, arm)


if __name__ == '__main__':
    unittest.main()
