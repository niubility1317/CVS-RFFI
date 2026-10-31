"""Fixed tied-density main, unchanged protection and no partial-Q truth join."""
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_tied_density_score')


def test_fixed_main_thresholds_and_nonregression_are_not_control_selection():
    from test_pamf_phase3_star_class_density_score import target
    s=api()
    assert s.METHODS==('pooled_ratio','predicted_ratio','tied_pooled_ratio','tied_predicted_ratio')
    assert s.PRIMARY=='tied_predicted_ratio'
    a,t=target();a['scores'][np.flatnonzero(t['y']==6)[:15],3]=0.
    assert s._target(a,t,'clean_DURA',3)['goals']['passed']
    a['scores'][np.flatnonzero(t['y']==6)[15],3]=0.
    assert not s._target(a,t,'clean_DURA',3)['goals']['passed']
    assert s._nonregression(dict(A_K=.8,R_U=.9),dict(A_K=.8,R_U=.9))
    assert not s._nonregression(dict(A_K=.799,R_U=1.),dict(A_K=.8,R_U=.9))
    assert not s._nonregression(dict(A_K=1.,R_U=.899),dict(A_K=.8,R_U=.9))


def test_partial_or_wrong_operator_prediction_never_reads_any_truth(tmp_path,monkeypatch,subtests):
    s=api();reads=[]
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'))
    monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('target'))
    for failure in ('missing star Q','wrong fixed pooled variance'):
        def reject(*args):raise ValueError(failure)
        monkeypatch.setattr(s,'_prediction',reject)
        with subtests.test(failure=failure),pytest.raises(ValueError):
            s.score(tmp_path/'prediction','source','clean','star',tmp_path/'score')
        assert not reads and not (tmp_path/'score').exists()
