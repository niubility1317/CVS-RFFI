from pathlib import Path

import numpy as np
import pytest

from experiments.pamf_lora_main.w24_formal import PEER_SPECTRAL_KIND
from experiments.pamf_lora_main import w24_formal_score as scorer
from experiments.pamf_lora_main.w24_full_ablation_audit import (
    BASELINES, audit, summarize,
)


def test_every_candidate_has_paired_mechanism_and_baseline_counts(monkeypatch):
    monkeypatch.setattr(scorer, 'TARGET', ('r0', 'r1'))
    monkeypatch.setattr(scorer, 'LIMIT', 2)
    monkeypatch.setattr(scorer, 'DRAWS', 8)
    truth = np.array([[0, 1], [0, 1]])
    direct = np.ones((2, 7, 1, 2), dtype=bool)
    direct[0, -1, 0, 0] = False
    within = direct.copy()
    within[0, -1, 0, 0] = True
    cross = within.copy()
    cross[1, -1, 0, 1] = False
    hits = dict.fromkeys(BASELINES, direct) | {'within': within, 'cross': cross}

    result = summarize(hits, ('within', 'cross'), truth)

    assert set(result) == {'within', 'cross'}
    for base in BASELINES:
        pair = result['within']['paired'][base]['practical_low_urban']
        assert (pair['repairs'], pair['harms'], pair['net']) == (1, 0, 1)
    pair = result['cross']['paired']['within']['practical_low_urban']
    assert (pair['repairs'], pair['harms'], pair['net']) == (0, 1, -1)
    assert result['cross']['by_tx']['1']['paired']['within']['practical_low_urban']['harms'] == 1


def test_incomplete_formal_run_is_rejected_before_truth_access(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        audit(tmp_path/'current', tmp_path/'practical', PEER_SPECTRAL_KIND,
              tmp_path/'missing-dataset', tmp_path/'output.json')
    assert Path(error.value.filename) == tmp_path/'current'/'scores'/'scores.json'
    assert not (tmp_path/'output.json').exists()
