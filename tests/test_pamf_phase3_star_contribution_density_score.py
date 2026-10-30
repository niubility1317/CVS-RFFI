"""Truth-last source-only scoring and exact frozen-control reuse."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_pamf_phase3_star_contribution_density_run import fixture
from test_pamf_phase3_star_density_score import truth_file


def test_real_toy_core_through_producer_and_independent_scorer_without_fitting(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_score')
    with fixture(tmp_path,monkeypatch) as (m,c,roles,b,counts,e,parent):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        monkeypatch.setattr(m._core(),'fit_known',lambda *a,**k:pytest.fail('Scorer cannot fit'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('Scorer cannot fit OE'))
        got=s.score(c['output_root'],truth_file(tmp_path,s,roles),tmp_path/'score')
        assert got['primary_method']=='contribution640_density_ratio' and got['source_gate_passed']
        assert got['primary_summary']['A_K']==got['primary_summary']['R_U']==1.
        assert got['control_reused_exactly'] and got['source_only'] and got['target_reads']==0
        assert got['gate_scope']=='source_overall_falsification_only' and got['A_N'] is got['H_OW'] is None
        assert len(got['methods'][m.METHODS[1]]['groups'])==42 and not got['final_phase3_complete']


def test_all_models_calibration_and_both_predictions_validate_before_private_truth(tmp_path,monkeypatch,subtests):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_score')
    for defect in ('control_tau','control_score','candidate_raw','candidate_feature','missing_novel','private'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,roles,b,counts,e,parent):
                m.run(c,'cpu');mp.setattr(s,'COUNTS',counts);out=Path(c['output_root'])
                if defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/('thresholds.npz' if defect=='control_tau' else 'novel_predictions.npz')
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='control_tau':d['conditional'][0,0]+=.1
                    elif defect=='control_score':d['scores'][0,0]+=.1
                    elif defect=='candidate_raw':d['raw_scores'][0,1]+=64.
                    elif defect=='candidate_feature':d['contribution_features'][0]=np.roll(d['contribution_features'][0],1)
                    else:d['tx']=np.asarray(['private']*len(d['ids']))
                    np.savez(p,**d)
                mp.setattr(s,'_truth',lambda *a:pytest.fail('Truth read before all public checks'))
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,root/'forbidden_truth',root/'score')


def test_control_cannot_replace_primary_even_with_perfect_control(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_score')
    with fixture(tmp_path,monkeypatch) as (m,c,roles,b,counts,e,parent):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        payload=s._prediction(c['output_root']);payload[-1]['scores'][:,1]=1.
        monkeypatch.setattr(s,'_prediction',lambda *a:payload)
        got=s.score(c['output_root'],truth_file(tmp_path,s,roles),tmp_path/'score')
        assert got['methods'][m.METHODS[0]]['goals']['passed'] and not got['source_gate_passed']
        assert not got['eligible_primary'] and got['next_step']=='NO_PROMOTION'
        assert got['paired_primary_vs_control']['counts']['unknown_rejected_net']==-90


def test_candidate_reconstruction_accepts_FP64_batch_roundoff_control_stays_exact(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_contribution_density_score')
    with fixture(tmp_path,monkeypatch) as (m,c,roles,b,counts,e,parent):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);core=m._core();original=core.score
        def rounded(state,x,method):
            result=original(state,x,method)
            return np.nextafter(result,np.inf) if state['whitener'].shape==(640,640) else result
        monkeypatch.setattr(core,'score',rounded)
        assert s._prediction(c['output_root'])[0]['control_reused_exactly']
