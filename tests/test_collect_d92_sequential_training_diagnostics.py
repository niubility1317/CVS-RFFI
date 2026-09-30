"""Tiny synthetic full-log fixtures; never load experiment logs or run training."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import collect_d92_sequential_training_diagnostics as collect


ALGORITHM = dict(steps=64, learning_rate=.01, beta1=.9, beta2=.999,
                 adam_epsilon=1e-8, global_gradient_norm_clip=1.)


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')


def make_fixture(root):
    lane = root/'row-a'/'probe'; lane.mkdir(parents=True)
    stages, full, compact, text = [], [], [], []
    for index, (scope, name, k) in enumerate((('support_oof', 'B', 3), ('support_oneshot_proxy', 'C_seq', 1))):
        identity = dict(split_id='parent-'+str(index), scope=scope, fold=0 if index == 0 else None,
            trial=0 if index else None, state=name, train_k=k)
        base = dict(event='BASE_FIT', state='B0', split_id=identity['split_id'], fit_seconds=.001)
        stages.append(base); text.append(json.dumps(base))
        text.append(json.dumps(dict(identity, event='RESIDUAL_STAGE_START', config=ALGORITHM)))
        for step in range(1, 65):
            event = dict(identity, step=step, optimizer_steps=step,
                state_timing='loss_gradient_pre_update_parameters_post_update', inherited=index == 1,
                optimizer_state_inherited=False, q=2., source_validation=None,
                loss_ce=1.-step/100., loss_proximal_U=.01, loss_proximal_V=.02,
                loss_total=1.03-step/100., training_accuracy=step/100.,
                residual_rms=step/10., base_logits_rms=2., post_U_anchor_distance=.1*step,
                post_V_anchor_distance=.2*step, gradient_clip_scale=.5 if step % 2 == 0 else 1.,
                gradient_zero=step == 64, training_physical_ids=['physical-a', 'physical-b'],
                elapsed_seconds=step/10., **{key: value for key, value in ALGORITHM.items() if key != 'steps'})
            full.append(event); compact.append(collect.scalars(event))
            text.append('SUPPORT_RESIDUAL_STEP '+json.dumps(compact[-1]))
        final = dict(identity, event='RESIDUAL_FIT', config=ALGORITHM, optimizer_steps=64,
            final_state_timing='after_64_updates', source_validation=None, held_used_for_training=False,
            optimizer_state_inherited=False, inherited=index == 1, q=2., fit_seconds=.5,
            class_count=2, train_physical_count=2*k, loss_ce=.35, loss_total=.38,
            training_accuracy=.65, residual_rms=6.5, base_logits_rms=2.,
            post_U_anchor_distance=6.4, post_V_anchor_distance=12.8)
        stages.append(final); text.append(json.dumps(final))
    jsonl(lane/'fit_stages.jsonl', stages)
    jsonl(lane/'training_steps.jsonl', full)
    jsonl(lane/'training_steps_compact.jsonl', compact)
    for name, rows in (('fit_stages', stages), ('training_steps_compact', compact)):
        fields = sorted({key for row in rows for key in row})
        with (lane/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({key: collect.csv_value(row.get(key)) for key in fields} for row in rows)
    (lane/'training.log').write_text('\n'.join(text)+'\nWarning: final synthetic warning\n', encoding='utf-8')
    (lane.parent/'probe.log').write_text(json.dumps(dict(event='STARTUP', argv=['synthetic']))+'\n'+
        '\n'.join(text)+'\n'+json.dumps(dict(event='SUPPORT_PARENT_COMPLETE', oof={'ignored_held_metric': .99}))+'\n', encoding='utf-8')
    return lane, stages, full, compact


class CollectTests(unittest.TestCase):
    def test_full_scan_preserves_stage_identity_and_pre_vs_post_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, full, compact = make_fixture(Path(directory))
            result = collect.collect_lane(lane, 'row-a', ALGORITHM)
        self.assertEqual(result['coverage']['optimizer_steps'], 128)
        self.assertEqual(result['coverage']['residual_training_stages'], 2)
        self.assertEqual(result['coverage']['base_fit_stages'], 2)
        self.assertEqual(result['coverage']['structured_nonfinite_count'], 0)
        self.assertEqual(len(result['curves']), 128)
        stage = result['stages'][0]
        self.assertEqual(stage['identity'], dict(row_id='row-a', split_id='parent-0', scope='support_oof',
            fold=0, trial=None, state='B', train_k=3))
        self.assertAlmostEqual(stage['first']['loss_ce'], .99)
        self.assertAlmostEqual(stage['last']['loss_ce'], .36)
        self.assertAlmostEqual(stage['final']['loss_ce'], .35)
        self.assertEqual(stage['clip_count'], 32)
        self.assertEqual(stage['zero_count'], 1)
        self.assertIsNone(stage['final']['gradient_norm_pre_clip'])
        self.assertEqual(stage['last']['residual_to_base_rms'], 3.2)
        self.assertEqual(result['text_logs'][0]['marker_line_counts']['warning'], 1)
        self.assertEqual(result['text_logs'][1]['events']['SUPPORT_PARENT_COMPLETE'], 1)
        self.assertNotIn('ignored_held_metric', json.dumps(result['stages']))

    def test_csv_disagreement_and_missing_steps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, full, compact = make_fixture(Path(directory))
            path = lane/'training_steps_compact.csv'
            content = path.read_text(encoding='utf-8')
            path.write_text(content.replace('0.99', '0.98', 1), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'CSV/JSON mismatch'):
                collect.collect_lane(lane, 'row-a', ALGORITHM)
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, full, compact = make_fixture(Path(directory))
            jsonl(lane/'training_steps.jsonl', full[:-1])
            with self.assertRaisesRegex(ValueError, 'length mismatch'):
                collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_wrong_text_tail_not_hidden_by_structured_success(self):
        with tempfile.TemporaryDirectory() as directory:
            lane, stages, full, compact = make_fixture(Path(directory))
            with (lane/'training.log').open('a', encoding='utf-8') as stream:
                stream.write('SUPPORT_RESIDUAL_STEP '+json.dumps(compact[-1])+'\n')
            with self.assertRaisesRegex(ValueError, 'Text/step stream mismatch'):
                collect.collect_lane(lane, 'row-a', ALGORITHM)

    def test_missing_metrics_stay_null_and_zero_base_ratio_is_undefined(self):
        values = collect.metrics(dict(loss_ce=.3, residual_rms=1., base_logits_rms=0.))
        self.assertIsNone(values['residual_to_base_rms'])
        self.assertIsNone(values['post_V_norm'])
        stats = {}; collect.add_stats(stats, values); collect.add_stats(stats, dict(loss_ce=None))
        self.assertEqual(collect.finish_stats(stats)['loss_ce'],
            dict(count=1, missing_count=1, mean=.3, minimum=.3, maximum=.3))

    def test_stage_key_separates_rows_folds_anchors_and_stages(self):
        record = dict(split_id='same', scope='support_oof', fold=0, trial=None, state='B', train_k=1)
        first = collect.stage_key('row1', record)
        self.assertNotEqual(first, collect.stage_key('row2', record))
        self.assertNotEqual(first, collect.stage_key('row1', dict(record, fold=1)))
        self.assertNotEqual(first, collect.stage_key('row1', dict(record, state='C_reset')))
        self.assertNotEqual(first, collect.stage_key('row1', dict(record, scope='support_oneshot_proxy', fold=None, trial=0)))

    def test_output_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): collect.write_outputs(directory, {})

    def test_ssh_config_argv_and_remote_readonly_stdin(self):
        result = dict(status='COMPLETE_TRAINING_LOG_SCAN_VERIFIED', coverage={'optimizer_steps': 292864})
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'new-local-output')
            argv = ['collector', '--run-root', '/remote/run root', '--run-log', '/remote/release/run.log',
                '--output', output, '--ssh-host', 'n607', '--ssh-config', 'E:/type10-7/tools/n607_ssh_config',
                '--remote-python', '/remote/python']
            response = SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr='')
            with patch.object(sys, 'argv', argv), patch.object(collect.subprocess, 'run', return_value=response) as ssh, \
                 patch.object(collect, 'write_outputs') as write, patch('builtins.print'):
                collect.main()
            command = ssh.call_args.args[0]
            self.assertEqual(command[:-1], ['ssh', '-F', 'E:/type10-7/tools/n607_ssh_config',
                '-T', '-o', 'BatchMode=yes', 'n607'])
            self.assertEqual(collect.shlex.split(command[-1]), ['/remote/python', '-', '--run-root',
                '/remote/run root', '--collect-stdout', '--run-log', '/remote/release/run.log'])
            self.assertNotIn('--output', command[-1])
            self.assertEqual(ssh.call_args.kwargs['input'], Path(collect.__file__).read_text(encoding='utf-8'))
            self.assertTrue(ssh.call_args.kwargs['text'])
            write.assert_called_once_with(output, result)


if __name__ == '__main__': unittest.main()
