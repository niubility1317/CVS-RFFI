"""Acceptance boundaries use paired counts, not rounded percentage arithmetic."""
import json
from pathlib import Path
import tempfile
import unittest
from experiments.pamf_ssl_research.analyze import summarize
from experiments.pamf_ssl_research.plan import SEEDS, METHODS, VIEWS

N = 168000
CLEAN = (142569, 142603, 143136)
SIX = ((136552, 132371, 127916, 130771, 106820, 87642),
       (132975, 128791, 124689, 127221, 103746, 85798),
       (133891, 129791, 125794, 128198, 105039, 86824))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def score(folder, run_id, row_id, seed, counts):
    rows = []
    for view, correct in zip(VIEWS, counts):
        cm = [[0] * 6 for _ in range(6)]
        for cls in range(6):
            diagonal = correct // 6 + int(cls < correct % 6)
            cm[cls][cls] = diagonal
            cm[cls][(cls + 1) % 6] = N // 6 - diagonal
        rows.append(dict(test='main', readout='raw', view=view, dimension='overall',
                         stratum='ALL', count=N, accuracy=correct / N, confusion=cm))
    save(folder / 'scores.json', dict(status='SCORED', truth_last=True, run_id=run_id,
                                     row_id=row_id, model_seed=seed, results=rows))


def fixture(root, clean_added, six_extra):
    state = root / 'state'
    save(state / 'state.json', dict(status='ALL_SCORED'))
    rt = dict(run_root=str(state), baseline_scores={}, rows=[])
    for i, seed in enumerate(SEEDS):
        rt['baseline_scores'][str(seed)] = {}
        for arm in ('F20', 'F21'):
            row = arm + '-s' + str(seed)
            folder = root / 'baseline' / row / 'scores'
            score(folder, 'baseline', row, seed, (CLEAN[i], *SIX[i]))
            rt['baseline_scores'][str(seed)][arm] = str(folder)
        for method in METHODS:
            row = method + '-s' + str(seed)
            folder = root / 'current' / row / 'scores'
            counts = [CLEAN[i] + clean_added[i], *(v + 2520 for v in SIX[i])]
            if i == 0:
                counts[1] += six_extra
            score(folder, 'current', row, seed, counts)
            config = root / (row + '.json')
            save(config, dict(output_root=str(folder), run_id='current', row_id=row, model_seed=seed))
            rt['rows'].append(dict(method=method, model_seed=seed, score_config=str(config)))
    return rt


class GoalCountBoundaryTest(unittest.TestCase):
    def test_exactly_one_point_five_pp_does_not_satisfy_strict_six_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(1700, 1700, 1700), six_extra=0)
            summarize(rt, root / 'analysis.json')
            self.assertFalse(json.loads((root / 'analysis.json').read_text())['goal_met'])

    def test_exact_one_pp_clean_and_one_correct_above_six_boundary_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(4, 1680, 3356), six_extra=1)
            summarize(rt, root / 'analysis.json')
            self.assertTrue(json.loads((root / 'analysis.json').read_text())['goal_met'])

    def test_accuracy_conflicting_with_confusion_counts_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(1700, 1700, 1700), six_extra=1)
            config = json.loads(Path(rt['rows'][0]['score_config']).read_text())
            path = Path(config['output_root']) / 'scores.json'
            value = json.loads(path.read_text())
            value['results'][0]['accuracy'] = .99
            save(path, value)
            with self.assertRaisesRegex(ValueError, 'integer correct counts'):
                summarize(rt, root / 'analysis.json')

    def test_score_from_another_model_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(1700, 1700, 1700), six_extra=1)
            config = json.loads(Path(rt['rows'][0]['score_config']).read_text())
            path = Path(config['output_root']) / 'scores.json'
            value = json.loads(path.read_text())
            value['model_seed'] = 999
            save(path, value)
            with self.assertRaisesRegex(ValueError, 'model seed'):
                summarize(rt, root / 'analysis.json')

    def test_perfect_source_validation_scores_cannot_replace_target_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rt = fixture(root, clean_added=(1700, 1700, 1700), six_extra=1)
            config = json.loads(Path(rt['rows'][0]['score_config']).read_text())
            path = Path(config['output_root']) / 'scores.json'
            value = json.loads(path.read_text())
            for row in value['results']:
                row['count'] = 27000
                row['accuracy'] = 1.
                row['confusion'] = [[4500 if i == j else 0 for j in range(6)] for i in range(6)]
            save(path, value)
            with self.assertRaisesRegex(ValueError, 'seven-view coverage'):
                summarize(rt, root / 'analysis.json')


if __name__ == '__main__':
    unittest.main()
