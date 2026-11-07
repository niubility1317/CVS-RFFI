import json
import pytest
from experiments.phase1_dura_orthogonal import direct_skip_full_parity as replay


def test_wait_stops_on_failed_original_and_only_returns_on_fixed_predictions(tmp_path,monkeypatch):
    reference=tmp_path/'runs'/'ground'/'flow'/'test_prediction'
    controller=tmp_path/'runs'/'source-controller'
    reference.mkdir(parents=True)
    controller.mkdir(parents=True)
    state=controller/'status.json'
    state.write_text(json.dumps({'state':'FAILED'}),encoding='utf8')
    with pytest.raises(RuntimeError,match='failed'):
        replay.wait_for_fixed(tmp_path,'source','ground')
    state.write_text(json.dumps({'state':'RUNNING_SEVEN_VIEW_PREDICTION'}),encoding='utf8')
    def complete(_):
        (reference/'complete.json').write_text('{}',encoding='utf8')
    monkeypatch.setattr(replay.time,'sleep',complete)
    assert replay.wait_for_fixed(tmp_path,'source','ground')==reference
