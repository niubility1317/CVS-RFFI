import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import dura_private_projection_target as candidate
from experiments.pamf_lora_main import fixed_mid_target as predict
from experiments.pamf_lora_main import fixed_mid_target_score as score


def test_target_selects_the_frozen_source_readouts_in_registered_order(monkeypatch):
    wave = torch.zeros(1, 6, 2, 256)
    outputs = tuple(wave+i for i in range(len(candidate.SOURCE_MODES)-2))
    monkeypatch.setattr(candidate, 'readouts', lambda *_: (outputs,1))
    selected, active = candidate.selected_readouts(wave, None, None, 0.)
    assert tuple(x[0,0,0,0].item() for x in selected) == (0,1,5,4)
    assert active == 1


def test_truth_reader_requires_every_registered_mode_from_both_routes(tmp_path, monkeypatch):
    for module in (predict, score):
        monkeypatch.setattr(module, 'LIMIT', 2)
        monkeypatch.setattr(module, 'TARGET', ('rx',))
        monkeypatch.setattr(module, 'VIEWS', ('clean',))
    controls = []
    for route in ('full','residual'):
        history = tmp_path / route
        history.mkdir()
        controls.append(history)
        summary = predict.freeze_metadata(route, candidate.MODES)
        summary['modes'] = list(predict.CONTROL_MODES[route])
        if route == 'residual':
            summary['practical'] = {'processing_route':'residual'}
        (history/'complete.json').write_text(json.dumps(summary))
        for mode in predict.CONTROL_MODES[route]:
            np.save(predict.output_path(history, 'rx', 'clean', mode), np.array([0,1]))
        output = tmp_path/'predictions'/route
        output.mkdir(parents=True)
        (output/'complete.json').write_text(json.dumps(predict.freeze_metadata(route, candidate.MODES)))
        (output/'activity.json').write_text(json.dumps({'rx':{'clean':0}}))
        for mode in candidate.MODES:
            np.save(predict.output_path(output, 'rx', 'clean', mode), np.array([0,1]))
    result,_ = score.read_matrix(tmp_path, tuple(controls), candidate.MODES)
    assert tuple(result['full'])[-len(candidate.MODES):] == candidate.MODES
    missing = predict.output_path(tmp_path/'predictions'/'residual', 'rx', 'clean', candidate.PRIMARY)
    missing.unlink()
    with pytest.raises(ValueError, match='Incomplete frozen residual'):
        score.read_matrix(tmp_path, tuple(controls), candidate.MODES)
