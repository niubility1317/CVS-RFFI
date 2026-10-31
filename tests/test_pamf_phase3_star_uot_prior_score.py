"""Fixed primary, truth ordering and inherited scientific boundaries."""
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_uot_prior_score')


def test_fixed_primary_clean95_star90_and_complete_scene_coverage():
    from test_pamf_phase3_star_class_density_score import target
    s=api();assert s.METHODS==('pooled_ratio','predicted_ratio','UOT_pooled_ratio','UOT_predicted_ratio') and s.PRIMARY==s.METHODS[3]
    a,t=target();a['scores'][t['y']==6,3]=1.
    assert not s._target(a,t,'clean_DURA',3)['goals']['passed']
    a,t=target();assert s._target(a,t,'clean_DURA',3)['goals']['passed']
    t['scenario']=np.asarray(['practical_low_urban']*len(t['y']))
    row=s._target(a,t,'star_DURA',3)
    assert row['goals']['overall_passed'] and not row['goals']['required_scene_coverage_complete'] and not row['goals']['passed']


def test_no_truth_or_output_on_incomplete_predictions(tmp_path,monkeypatch):
    s=api();reads=[]
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'))
    monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('partial Q')))
    with pytest.raises(ValueError,match='partial Q'):s.score('pred','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()
