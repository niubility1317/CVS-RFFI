import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_sourceV_waits_for_finished_latency_without_reading_quality(tmp_path):
    from experiments.pamf_phase3_restoration.covariance_detail_source_validation import dependency_ready
    from experiments.pamf_phase3_restoration.covariance_detail_latency import RUN
    folder=tmp_path/'runs'/RUN;folder.mkdir(parents=True)
    assert not dependency_ready(tmp_path)
    (folder/'complete.json').write_text(json.dumps(dict(status='ARTIFACTS_COMPLETE')))
    assert dependency_ready(tmp_path)
    (folder/'terminal.json').write_text(json.dumps(dict(exit_code=1)))
    with pytest.raises(RuntimeError):dependency_ready(tmp_path)


def test_cached_baselines_must_match_original_F0_and_all_source_views(tmp_path):
    from experiments.pamf_phase3_restoration.covariance_detail_source_validation import baseline_reference
    with pytest.raises(FileNotFoundError,match='predictions_complete'):baseline_reference(tmp_path,{})
    (tmp_path/'predictions_complete.json').write_text(json.dumps(dict(status='SOURCE_V_PREDICTIONS_COMPLETE',code_commit='other')))
    with pytest.raises(ValueError):baseline_reference(tmp_path,{})


def test_unknown_comparison_is_rejected_before_prediction_or_truth_reads(tmp_path):
    from experiments.pamf_phase3_restoration import covariance_guidance_source as base
    with pytest.raises(ValueError,match='comparison'):
        base.score(SimpleNamespace(root=tmp_path,inputs=tmp_path/'must-not-open',commit='test'),comparison='invalid')
