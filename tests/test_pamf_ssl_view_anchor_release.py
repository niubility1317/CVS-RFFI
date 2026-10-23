import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

from experiments.pamf_ssl_research.control import validate_rows
from experiments.pamf_ssl_research.plan import configuration, SEEDS, GPU_BY_SEED, VIEWS
from experiments.pamf_ssl_research.view_anchor import RUN, PARENT_RUN
from experiments.pamf_ssl_research.view_anchor_publish import prepare
from tests.test_pamf_ssl_research import template


class ViewAnchorReleaseTest(unittest.TestCase):
    def test_busy_smoke_card_queues_without_touching_other_jobs(self):
        from experiments.pamf_ssl_research.control import run
        class SmokeReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = dict(methods=['VIEWANCHOR'], run_root=str(root/'new_run'), release_root=str(root),
                runtime_file=str(root/'runtime.json'), code_commit='c',
                rows=[dict(row_id='VIEWANCHOR-s'+str(seed),model_seed=seed,method='VIEWANCHOR',gpu=GPU_BY_SEED[seed])
                      for seed in SEEDS])
            def stop_at_smoke(command, **kwargs):
                self.assertIn('experiments.pamf_ssl_research.view_anchor_worker',command)
                raise SmokeReached
            with patch('experiments.pamf_ssl_research.control.capacity',side_effect=[(3,16000),(1,16000)]), \
                 patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
                 patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=stop_at_smoke), \
                 patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
                with self.assertRaises(SmokeReached):
                    run(rt)
                sleep.assert_called_once_with(20)
                launch.assert_not_called()
            state=json.loads((root/'new_run/state.json').read_text())
            self.assertEqual(state['status'],'SMOKE')
            self.assertEqual({row['status'] for row in state['jobs'].values()},{'QUEUED'})

    def test_continuation_keeps_real_names_parents_and_separate_truth_side(self):
        context = {'_rxpsm_parents': {}, '_parent_timing': {}}
        for seed in SEEDS:
            base = dict(template(), source_contract='/p/source_contract.json', frozen_sources={'F': '/f'},
                        output_root='/p/baseline/' + str(seed) + '/source')
            score = dict(output_root='/p/baseline/' + str(seed), tests={'main': {'truth': '/private/truth'}})
            prediction = dict(tests={'main': {'truth': '/private/truth', 'iq': '/target/iq'}}, views=list(VIEWS))
            context[str(seed)] = {arm:dict(train=base, score=score, predict=prediction) for arm in ('F20', 'F21')}
            context['_rxpsm_parents'][str(seed)] = configuration(base, 'RXPSM', seed, '/p', 'old')
            context['_parent_timing'][str(seed)] = 5000.
        rt, blobs = prepare(context, '/p', '/p/releases/' + RUN, 'new')
        self.assertEqual(len(rt['rows']), 3)
        self.assertEqual(rt['methods'], ['VIEWANCHOR'])
        self.assertEqual(rt['max_total_workers'], 3)
        for row in rt['rows']:
            seed = row['model_seed']
            row_id = 'VIEWANCHOR-s' + str(seed)
            self.assertEqual(row['row_id'], row_id)
            self.assertEqual(row['gpu'], GPU_BY_SEED[seed])
            train = blobs['configs/' + row_id + '-train.json']
            prediction = blobs['configs/' + row_id + '-predict.json']
            score = blobs['configs/' + row_id + '-score.json']
            self.assertEqual(train['fork_parent_source'], '/p/runs/' + PARENT_RUN + '/RXPSM-s' + str(seed) + '/source')
            self.assertEqual(train['optimizer_updates'], 11100)
            self.assertEqual(train['parent_prefix_training_seconds'], 5000.)
            self.assertEqual(prediction['source_output'], train['output_root'])
            self.assertEqual(prediction['readouts'], {'raw': {}})
            self.assertNotIn('truth', prediction['tests']['main'])
            self.assertEqual(score['tests']['main']['truth'], '/private/truth')
            self.assertEqual(score['prediction_output'], prediction['output_root'])
            self.assertEqual(score['row_id'], row_id)
        self.assertFalse(any('PSM-s' in key for key in blobs))

    def test_controller_rejects_missing_duplicate_and_unregistered_matrices(self):
        rows = [dict(row_id='VIEWANCHOR-s' + str(seed), model_seed=seed, method='VIEWANCHOR',
                     gpu=GPU_BY_SEED[seed]) for seed in SEEDS]
        validate_rows(rows, ('VIEWANCHOR',))
        for wrong in (rows[:-1], rows + rows[:1], [dict(rows[0]) for _ in SEEDS]):
            with self.assertRaises(ValueError):
                validate_rows(wrong, ('VIEWANCHOR',))
        with self.assertRaises(ValueError):
            validate_rows(rows)
        with self.assertRaises(ValueError):
            validate_rows(rows, ('PSM', 'VIEWANCHOR'))


if __name__ == '__main__':
    unittest.main()
