"""Seven focused pure-array cases; no Torch, IQ, checkpoint or remote access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import readout as r


def metadata():
    return dict(status='SOURCE_READOUT_COMPLETE', run_id=r.RUN, model_seed=r.SEED,
        checkpoint_run=r.PARENT_RUN, checkpoint_row=r.PARENT_ROW, checkpoint_stage='stage20',
        checkpoint_key='model', checkpoint_epoch=20, checkpoint_optimizer_steps=4440,
        checkpoint_selection='final_only', source_role='V', source_unique_V=1800,
        channel_draws=2, scenes=list(r.SCENES), condition='full', condition_index=0,
        NN=12602, source_NN=12600, smoke_NN=2, training_updates=0, head_fit_updates=0,
        teacher_forwards=0, teacher_bank_created=False, U_truth_read=False, U_records_used=0,
        target_access=False, formal_scores_read=False, query_access=False, new_channel_calls=0,
        model_state_unchanged=True, all_parameters_frozen=True, model_parameters=900722,
        model_state_keys=206, default_head_shape=[6, 160], default_head_scale=30.,
        eval_counter=12602, train_counter=0, source_role_counts=dict(L_s=6300, U_s=56700, V=27000),
        checkpoint='toy/AS_CE20/model', source_contract='toy/contract', source_panel='toy/V',
        source_observations='toy/full', wall_seconds=1., smoke_wall_seconds=.1, source_wall_seconds=.9)


def predictions():
    cells = [(t, rx, day) for t in range(6) for rx in r.RX for day in r.DAYS]
    y = np.asarray([v[0] for v in cells], np.int64)
    z = np.eye(6, dtype=np.float32)[y] * 30
    return dict(ids=np.asarray(['toy' + str(i) for i in range(90)]), TX=y,
        RX=np.asarray([v[1] for v in cells], np.int64), day=np.asarray([v[2] for v in cells], np.int64),
        clean=z.copy(), satellite=np.broadcast_to(z, (2, 3, 90, 6)).copy())


class Checks(unittest.TestCase):
    def test_actual_ASCE_student_not_EMA(self):
        r.validate_metadata(metadata())
        with self.assertRaises(ValueError):
            r.validate_metadata(dict(metadata(), checkpoint_key='ema_model'))

    def test_P43_parent_and_wrong_budget_rejected(self):
        for change in (dict(checkpoint_run='P43'), dict(checkpoint_row='B01-s2026092701'),
                dict(checkpoint_epoch=30), dict(NN=12600)):
            with self.assertRaises(ValueError):
                r.validate_metadata(dict(metadata(), **change))

    def test_no_training_target_teacher_or_channel_draw(self):
        for change in (dict(training_updates=1), dict(target_access=True),
                dict(teacher_forwards=1), dict(new_channel_calls=1)):
            with self.assertRaises(ValueError):
                r.validate_metadata(dict(metadata(), **change))

    def test_unique_finite_source_logits(self):
        p = predictions()
        r.validate_predictions(p, 90)
        p['ids'][0] = p['ids'][1]
        with self.assertRaises(ValueError):
            r.validate_predictions(p, 90)

    def test_order_mismatch_rejected(self):
        parent = predictions()
        current = copy.deepcopy(parent)
        current['ids'] = current['ids'][::-1]
        with self.assertRaises(ValueError):
            r.cohort_metrics(parent, current)

    def test_actual_parent_cohort_retention_and_empty_strata(self):
        parent = predictions()
        parent['clean'][0] = np.eye(6, dtype=np.float32)[1] * 30
        parent['satellite'][:, :, 1] = np.eye(6, dtype=np.float32)[1] * 30
        current = copy.deepcopy(parent)
        current['satellite'][:, :, 1] = np.eye(6, dtype=np.float32)[0] * 30
        current['satellite'][:, :, 2] = np.eye(6, dtype=np.float32)[1] * 30
        physics = dict(state=np.zeros((2, 3, 90), np.int8), lock=np.zeros((2, 3, 90), np.int8))
        result = r.cohort_metrics(parent, current, physics)
        self.assertEqual(result['parent_clean_correct_count'], 89)
        pooled = next(v for v in result['records'] if v['scene'] == r.SCENES[-1]
            and v['draw'] == -1 and v['dimension'] == 'overall')
        self.assertEqual(pooled['parent_clean_correct_count'], 178)
        self.assertEqual(pooled['repairs_within_parent_cohort'], 2)
        self.assertEqual(pooled['new_errors_within_parent_cohort'], 2)
        self.assertEqual(pooled['retention_delta_pp'], 0.)
        empty = next(v for v in result['records'] if v['scene'] == r.SCENES[-1]
            and v['draw'] == 0 and v['dimension'] == 'state' and v['stratum'] == 'BLOCKED')
        self.assertIsNone(empty['current_retention'])
        self.assertTrue(empty['empty_cohort'])
        self.assertTrue(all(v['cohort_reference'] == r.COHORT_REFERENCE for v in result['records']))

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.json'
            r.write(path, dict(value=1))
            with self.assertRaises(FileExistsError):
                r.write(path, dict(value=2))
            self.assertEqual(json.loads(path.read_text()), dict(value=1))


def run():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise AssertionError('AS_CE20 parent source-only toy checks failed')
    return dict(status='PASS', tests=result.testsRun, NN=0, IQ_reads=0, checkpoint_reads=0,
        teacher_forwards=0, updates=0, target_access=False, remote_calls=0)


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
