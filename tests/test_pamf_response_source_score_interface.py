from types import SimpleNamespace
import numpy as np
import pytest
from experiments.pamf_response_restore import evaluation


def test_actual_three_value_source_reference_reaches_raw_prediction_validation_before_truth(monkeypatch,tmp_path):
    class RawPredictionValidation(RuntimeError):pass
    args=SimpleNamespace(parent_action=tmp_path/'action',source_cache=tmp_path/'source',output=tmp_path/'new',row='pair_aug_control')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    ids=np.arange(2)
    monkeypatch.setattr(evaluation,'source_reference',lambda *args:(ids,None,None))
    def before_truth(root,actual_ids,row):
        assert np.array_equal(actual_ids,ids) and row=='pair_aug_control'
        raise RawPredictionValidation('Response arrays validated before any source labels')
    monkeypatch.setattr(evaluation.response,'load_predictions',before_truth)
    with pytest.raises(RawPredictionValidation):evaluation.score(args,'source')
