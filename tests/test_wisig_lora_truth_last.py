import json

import numpy as np
import pytest

from experiments.pamf_unrolled_restore.wisig_lora_score import read_candidate


def test_missing_prediction_closure_blocks_truth_scoring(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        read_candidate(tmp_path, 'expanded', 'frozen', 'full', 'cross_rx', [], [], [])


def test_candidate_closure_and_physical_order_are_required(tmp_path):
    folder = tmp_path / 'frozen' / 'guard' / 'full' / 'cross_rx'
    folder.mkdir(parents=True)
    ids, seeds, views = np.array([5, 7]), [2], ['clean']
    payload = dict(status='PREDICTIONS_FROZEN', arm='guard', mode='frozen', route='full',
                   split='cross_rx', count=2, seeds=seeds, views=views, truth_read=False,
                   active={'clean': 1})
    (folder / 'complete.json').write_text(json.dumps(payload), encoding='utf-8')
    np.savez(folder / 'decisions.npz', ids=ids, clean=np.array([[1, 2]], dtype=np.uint8))
    values, active = read_candidate(tmp_path, 'guard', 'frozen', 'full', 'cross_rx', ids, seeds, views)
    assert values.shape == (1, 1, 2) and active['clean'] == 1
    with pytest.raises(ValueError, match='physical IDs'):
        read_candidate(tmp_path, 'guard', 'frozen', 'full', 'cross_rx', ids[::-1], seeds, views)
