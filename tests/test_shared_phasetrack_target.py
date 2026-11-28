import json
import numpy as np
import pytest

from experiments.pamf_lora_main import shared_phasetrack_target as pilot


def test_phase_target_requires_both_datasets_frozen(tmp_path):
    for dataset, n in [('lora', pilot.LORA_N), ('wisig', pilot.WISIG_N)]:
        folder = tmp_path/dataset
        folder.mkdir()
        mark = dict(status='PREDICTIONS_FROZEN', dataset=dataset, views=list(pilot.VIEWS),
            modes=list(pilot.modes_for(dataset)), count=n, truth_read=False, target_fit=False,
            candidate=pilot.PARENT, receiver=pilot.LORA_RX if dataset == 'lora' else pilot.WISIG_RX)
        (folder/'frozen.json').write_text(json.dumps(mark), encoding='utf-8')
        np.savez(folder/'predictions.npz', ids=np.arange(n),
                 **{v:np.zeros((len(pilot.MODES), n), dtype=np.int16) for v in pilot.VIEWS})
    assert set(pilot.frozen_pair(tmp_path)) == {'lora', 'wisig'}
    mark_path = tmp_path/'wisig'/'frozen.json'
    mark = json.loads(mark_path.read_text(encoding='utf-8'))
    mark['candidate'] = 'other'
    mark_path.write_text(json.dumps(mark), encoding='utf-8')
    with pytest.raises(ValueError):
        pilot.frozen_pair(tmp_path)


def test_predeclared_same_receiver_and_candidate():
    assert pilot.WISIG_RX == 0 and pilot.WISIG_RX_LABEL == '1-1'
    assert pilot.PARENT == '20261128-phase1-shared-phasetrack-gain-s2026112957-r01'
    assert pilot.candidate.SharedPhaseTrackGain.__name__ == 'SharedPhaseTrackGain'
