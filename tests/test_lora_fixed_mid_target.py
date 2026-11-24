import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import fixed_mid_target as target
from experiments.pamf_lora_main.fixed_mid_target_score import interval, paired_counts


def test_target_reuses_exact_frozen_source_midpoint(monkeypatch):
    iq = torch.zeros((2, 6, 2, 256))
    chosen = iq + 3
    controls = ("tracker", "models", .5)

    def source_readout(wave, tracker, models, threshold):
        assert wave is iq and (tracker, models, threshold) == controls
        return (iq + 1, iq + 2, chosen, iq + 4, iq + 5), 2

    monkeypatch.setattr(target, 'readouts', source_readout)
    actual, active = target.fixed_readout(iq, *controls)
    assert actual is chosen and active == 2


@pytest.mark.parametrize('route', ('full', 'residual'))
def test_target_view_contract_fails_closed_on_truth_or_wrong_route(tmp_path, route):
    manifest = dict(status='TARGET_READY', protocol=target.PROTOCOL, target_truth_read=False,
                    receivers=list(target.TARGET), per_receiver=target.LIMIT, views=list(target.VIEWS))
    if route == 'residual':
        manifest['processing_route'] = 'residual'
    p = tmp_path/'complete.json'
    runtime = {'target_views':{route:str(tmp_path)}}
    p.write_text(json.dumps(manifest),encoding='utf-8')
    assert target.blind_views(runtime,route) == tmp_path
    p.write_text(json.dumps(dict(manifest,target_truth_read=True)),encoding='utf-8')
    with pytest.raises(ValueError, match='blind'):
        target.blind_views(runtime,route)
    p.write_text(json.dumps(dict(manifest,processing_route='other')),encoding='utf-8')
    with pytest.raises(ValueError, match='blind'):
        target.blind_views(runtime,route)


def test_pair_interval_uses_signed_changes_and_same_packets():
    candidate = np.array([[True,False],[True,True]])
    reference = np.array([[False,True],[False,True]])
    assert paired_counts(candidate, reference)['net'] == 1
    draws = np.array([[[0,0],[1,1]],[[0,1],[1,1]]])
    assert interval(candidate, reference, draws) == pytest.approx([-1.875,2.875])


def test_update_preserves_bypass_and_does_not_use_tensor_numpy_bridge(monkeypatch):
    def reject_numpy(*_):
        pytest.fail('Torch cannot create an ndarray for this interpreter')

    monkeypatch.setattr(torch.Tensor, 'numpy', reject_numpy)
    predictions = np.asarray([9, 5, 7], dtype=np.int16)
    target.update_predictions(predictions, torch.tensor([True, False, True]),
                              torch.tensor([[0., 4., 1.], [3., 1., 0.]]))
    assert predictions.tolist() == [1, 5, 0]
