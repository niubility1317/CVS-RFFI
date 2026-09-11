import pytest
from scripts.source_probe_v2_development import choose_candidate


def test_candidate_selection_is_frozen_source_fit_only():
    candidates=[dict(name='failed',lag=dict(status='OPTIMIZATION_FAILURE',quality_pass=False,fit_after=.01)),
        dict(name='low',lag=dict(status='RELIABLE_LOW_GAP',quality_pass=True,fit_after=.2,control_ready=True,
             budget=dict(actual_steps=40,requested_steps=40))),
        dict(name='high',lag=dict(status='RELIABLE_HIGH_GAP',quality_pass=True,fit_after=.3,control_ready=True,
             budget=dict(actual_steps=40,requested_steps=40)))]
    assert choose_candidate(candidates)['name']=='low'
    assert choose_candidate(candidates[:1]) is None
