import json
import numpy as np
import pytest
from experiments.pamf_phase3_restoration.covariance_parameter_source import parameter_pairs,spare_card,summarize


def test_parameter_attribution_changes_only_the_named_quantity():
    predicted_action=object();predicted_noise=object();true_action=object();true_noise=object()
    result=parameter_pairs(predicted_action,predicted_noise,true_action,true_noise)
    assert result=={'blind':(predicted_action,predicted_noise),'true_noise':(predicted_action,true_noise),
        'true_action':(true_action,predicted_noise),'true_both':(true_action,true_noise)}


def test_diagnostic_waits_until_last_parent_gpu_stage_and_stops_on_failure(tmp_path):
    from experiments.pamf_phase3_restoration import covariance_trial as parent
    folder=tmp_path/'runs'/parent.RUN;folder.mkdir(parents=True)
    assert spare_card(tmp_path)==()
    (folder/f'terminal-{parent.ARM}-cross_rx.json').write_text(json.dumps(dict(exit_code=0)))
    assert spare_card(tmp_path)==()
    (folder/f'terminal-{parent.ARM}-cross_both.json').write_text(json.dumps(dict(exit_code=0)))
    assert spare_card(tmp_path)==(4,6)
    (folder/f'terminal-{parent.ARM}-train.json').write_text(json.dumps(dict(exit_code=1)))
    with pytest.raises(RuntimeError,match='Parent technical failure'):spare_card(tmp_path)


def test_incomplete_source_attribution_is_rejected():
    with pytest.raises(ValueError,match='All fixed source'):
        summarize({'epoch':np.ones(512,dtype=int)})
