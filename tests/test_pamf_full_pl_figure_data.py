"""Focused checks for score-only publication-figure statistics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/figures/plot_pamf_full_pl_comparison.py'
spec = importlib.util.spec_from_file_location('full_pl_figure', SCRIPT)
figure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figure)


def fixture():
    records = []
    for family in ('riei', 'drift', 'ibrff', 'sdg', 'pamf'):
        for regime in (('pamf',) if family == 'pamf' else ('full', 'full_pl')):
            for number, seed in enumerate(figure.SEEDS):
                for scene_index, scene in enumerate(figure.SCENES):
                    records.append(dict(family=family, regime=regime, seed=seed,
                        scene=scene, accuracy_pct=60 + number + scene_index,
                        macro_f1_pct=59 + number + scene_index, count=168000,
                        source_run='unit-fixture', source_row='unit-fixture'))
    return records


class FigureDataTests(unittest.TestCase):
    def test_six_view_mean_excludes_clean_before_seed_sd(self):
        values = np.array([[99, 10, 20, 30, 40, 50, 60],
                           [99, 20, 30, 40, 50, 60, 70],
                           [99, 30, 40, 50, 60, 70, 80]], float)
        samples = figure.metric(values, 'six')
        self.assertEqual(samples.tolist(), [35, 45, 55])
        center, sd = figure.describe(samples)
        self.assertEqual(center, 45)
        self.assertEqual(sd, 10)

    def test_all_nine_series_have_three_seeds_seven_scenes(self):
        arrays = figure.arrays_from_records(fixture())
        self.assertEqual(len(arrays), 9)
        self.assertTrue(all(value.shape == (3, 7) for value in arrays.values()))

    def test_ce_series_are_rejected_not_silently_plotted(self):
        records = fixture()
        records[0]['regime'] = 'ce'
        with self.assertRaises(ValueError):
            figure.arrays_from_records(records)

    def test_missing_seed_scene_is_rejected(self):
        with self.assertRaises(ValueError):
            figure.arrays_from_records(fixture()[1:])

    def test_duplicate_seed_scene_is_rejected(self):
        records = fixture()
        with self.assertRaises(ValueError):
            figure.arrays_from_records(records + [records[0]])

    def test_wrong_percent_scale_and_nonfinite_values_are_rejected(self):
        for value in (104, float('nan')):
            records = fixture()
            records[0]['accuracy_pct'] = value
            with self.assertRaises(ValueError):
                figure.arrays_from_records(records)

    def test_pl_change_is_paired_with_same_seed(self):
        values = np.array([[0, 10, 10, 10, 10, 10, 10],
                           [0, 20, 20, 20, 20, 20, 20],
                           [0, 30, 30, 30, 30, 30, 30]], float)
        pl = values + np.array([[-1], [2], [5]])
        delta = figure.pl_delta({('sdg', 'full'): values, ('sdg', 'full_pl'): pl}, 'sdg')
        self.assertEqual(delta.tolist(), [-1, 2, 5])
        self.assertEqual(figure.describe(delta), (2, 3))

    def test_every_figure_contains_one_data_chart(self):
        arrays = figure.arrays_from_records(fixture())
        for factory in (figure.accuracy_figure, figure.balance_figure,
                        figure.pl_effect_figure, figure.scenario_figure):
            with self.subTest(figure=factory.__name__):
                plot = factory(arrays)
                try:
                    self.assertEqual(figure.single_chart_count(plot), 1)
                finally:
                    figure.plt.close(plot)

    def test_scene_bars_show_four_pairs_and_one_pamf_with_correct_heights(self):
        arrays = {key: np.array([[20 + 5 * i + seed + scene for scene in range(7)]
                                for seed in range(3)], float)
                  for i, key in enumerate(figure.KEYS)}
        for scene_index, scene in enumerate(figure.SCENES):
            with self.subTest(scene=scene):
                plot = figure.grouped_scene_figure(arrays, scene)
                try:
                    self.assertEqual(figure.single_chart_count(plot), 1)
                    bars = {bar.get_gid(): bar for bar in plot.axes[0].patches}
                    self.assertEqual(len(bars), 9)
                    self.assertEqual(set(bars), {f'{family}-{regime}' for family, regime in figure.KEYS})
                    for key in figure.KEYS:
                        bar = bars['-'.join(key)]
                        self.assertAlmostEqual(bar.get_height(), arrays[key][:, scene_index].mean())
                    for family in figure.FAMILIES:
                        left = bars[f'{family}-full']
                        right = bars[f'{family}-full_pl']
                        self.assertLess(left.get_x() + left.get_width(), right.get_x())
                        self.assertEqual(left.get_hatch(), '//')
                        self.assertFalse(right.get_hatch())
                finally:
                    figure.plt.close(plot)

    def test_all_seven_bar_figures_share_zero_based_percentage_axis(self):
        arrays = figure.arrays_from_records(fixture())
        for scene in figure.SCENES:
            plot = figure.grouped_scene_figure(arrays, scene)
            try:
                self.assertEqual(tuple(plot.axes[0].get_ylim()), (0, 100))
            finally:
                figure.plt.close(plot)


if __name__ == '__main__':
    unittest.main()
