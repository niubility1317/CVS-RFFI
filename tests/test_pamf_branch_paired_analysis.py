import pytest
import csv
import json
from experiments.pamf_branch_training.paired_analysis import cells, merge_disjoint, run


def test_combined_target_macro_f1_uses_summed_confusion_not_weighted_f1():
    a = dict(count=3, confusion=[[2, 0], [1, 0]])
    b = dict(count=2, confusion=[[0, 1], [0, 1]])
    result = merge_disjoint(a, b)
    assert result['count'] == 5
    assert result['accuracy'] == pytest.approx(.6)
    assert result['macro_f1'] == pytest.approx((4 / 6 + 2 / 4) / 2)


def test_comparison_refuses_unscored_or_duplicate_cells():
    with pytest.raises(ValueError):
        cells({'status': 'SCORED', 'truth_last': False, 'results': []})
    result = dict(dimension='overall', test='cross_rx', readout='raw', view='clean')
    with pytest.raises(ValueError, match='Duplicate'):
        cells({'status': 'SCORED', 'truth_last': True, 'results': [result, result]})


def test_explicit_physical_control_comparison(tmp_path):
    def source(name, accuracy):
        record = dict(dimension='overall', test='combined_target', readout='raw', view='leo_clear_weak',
                      count=10, accuracy=accuracy, macro_f1=accuracy)
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps(dict(status='SCORED', truth_last=True, results=[record])))
        return str(path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'candidate': {'weak': source('candidate', .8)},
                                    'control': {'weak': source('control', .7)},
                                    '_comparisons': [['candidate', 'control']]}))
    output = tmp_path / 'paired.csv'
    assert run(manifest, output) == 1
    with output.open(newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert float(row['delta_accuracy_pp']) == pytest.approx(10.)
