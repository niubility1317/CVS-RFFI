import json
import numpy as np
import pytest

from experiments.pamf_lora_main import shared_target_screen as pilot


def frozen(root, dataset, n):
    folder = root/dataset
    folder.mkdir()
    mark = dict(status='PREDICTIONS_FROZEN', dataset=dataset, views=list(pilot.VIEWS),
                modes=list(pilot.modes_for(dataset)), count=n, truth_read=False, target_fit=False, candidate=pilot.PARENT,
                receiver=pilot.LORA_RX if dataset == 'lora' else pilot.WISIG_RX)
    (folder/'frozen.json').write_text(json.dumps(mark), encoding='utf-8')
    np.savez(folder/'predictions.npz', ids=np.arange(n),
             **{v: np.zeros((len(pilot.MODES), n), dtype=np.int16) for v in pilot.VIEWS})


def test_truth_closed_until_both_datasets_and_views_frozen(tmp_path):
    frozen(tmp_path, 'lora', pilot.LORA_N)
    with pytest.raises(FileNotFoundError):
        pilot.frozen_pair(tmp_path)
    frozen(tmp_path, 'wisig', pilot.WISIG_N)
    assert set(pilot.frozen_pair(tmp_path)) == {'lora', 'wisig'}
    mark = tmp_path/'wisig'/'frozen.json'
    value = json.loads(mark.read_text(encoding='utf-8'))
    value['truth_read'] = True
    mark.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        pilot.frozen_pair(tmp_path)


def test_paired_scoring_counts_repairs_and_clean_loss():
    n = 100
    labels = np.zeros(n, dtype=np.int64)
    base = np.zeros((len(pilot.MODES), n), dtype=np.int16)
    low = base.copy()
    low[0, :3] = 1
    low[-1, 3:5] = 1
    result = pilot.score_values(np.arange(n), {'clean':base, 'practical_low_urban':low},
                                labels, 0, np.full(n, 'day0'), 'wisig')
    pair = result['views']['practical_low_urban']['pairs']['direct']
    assert (pair['repaired'], pair['harmed'], pair['net']) == (3, 2, 1)
    assert result['clean_net']['direct'] == 0
