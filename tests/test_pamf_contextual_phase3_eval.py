import numpy as np
import pytest
import torch

from experiments.pamf_phase3_open_world import contextual_restore_eval as experiment


def test_same_ids_per_scene_macro_gaps_and_missing_classes():
    y = np.array(list(range(6))*2)
    scenes = np.array(['a']*6 + ['b']*6)
    clean = y.copy()
    star = y.copy()
    star[0] = 1
    truth = dict(y=y, scenario=scenes)
    gaps = experiment.paired_clean_gaps(clean, star, truth, ('a','b','missing'))
    assert gaps['a'] == pytest.approx(100/6)
    assert gaps['b'] == 0
    assert gaps['missing'] is None
    accepted = np.ones(12, dtype=bool)
    candidate_accepted = accepted.copy()
    candidate_accepted[6] = False
    open_gaps = experiment.paired_clean_gaps(clean, clean, truth, ('b',), accepted, candidate_accepted)
    assert open_gaps['b'] == pytest.approx(100/6)


def test_candidate_iq_reaches_f0_and_both_open_models():
    class G(torch.nn.Module):
        def forward(self, x):
            return 2*x
    class F(torch.nn.Module):
        def features(self, x):
            return x.mean((1,2))[:,None].expand(-1,160)
        def classify_features(self, f):
            return torch.stack([f[:,0]*(i+1) for i in range(6)], dim=1)
    class Open(torch.nn.Module):
        def forward(self, x):
            v = torch.ones(len(x),160,device=x.device)
            v[:,0] = x.mean((1,2))
            return torch.nn.functional.normalize(v, dim=1)
    x = np.ones((2,2,256), dtype=np.float32)
    out = experiment._current({'reader':F()}, {'E0':Open(),'PCD':Open()}, G(), x, 'cpu')
    assert out['default_logits'].shape == (2,6)
    assert np.array_equal(out['closed'], np.array([5,5]))
    assert np.all(out['raw_signal_features'] == 2)
    assert out['signal_features'].shape == (2,2,160)
    assert np.allclose(np.linalg.norm(out['signal_features'],axis=2),1)


def test_score_refuses_incomplete_predictions_before_reading_truth(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    calls = []
    def fail_before_truth(root):
        raise ValueError('Incomplete predictions')
    monkeypatch.setattr(experiment, '_prediction', fail_before_truth)
    monkeypatch.setattr(experiment.reference_score, '_source_truth', lambda *args:calls.append('source'))
    monkeypatch.setattr(experiment.reference_score, '_future_truth', lambda *args:calls.append('target'))
    with pytest.raises(ValueError, match='Incomplete'):
        experiment.score(tmp_path/'missing', tmp_path/'score')
    assert not calls
    assert not (tmp_path/'score').exists()


def test_unverified_dura_reference_stops_before_any_truth(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    done = dict(status=experiment.STATUS, run_id=experiment.RUN,
        counts=dict(known=27000,novel=3150,clean=10238,star=10238),
        all_predictions_before_truth=True, all_networks_and_thresholds_frozen=True,
        private_truth_reads=0, target_U_reads=0, U_ID_reads=0, new_updates=0,
        density_fit_updates=0, threshold_fit_updates=0, config={})
    monkeypatch.setattr(experiment, '_read', lambda path:done)
    monkeypatch.setattr(experiment, 'validate_config', lambda config:({}, {'output_root':'bad-reference'}))
    def reject_reference(path):
        raise ValueError('Unverified DURA reference')
    monkeypatch.setattr(experiment.reference_score, '_prediction', reject_reference)
    truth_calls = []
    monkeypatch.setattr(experiment.reference_score, '_source_truth', lambda *args:truth_calls.append('source'))
    with pytest.raises(ValueError, match='Unverified DURA'):
        experiment.score(tmp_path/'prediction', tmp_path/'score')
    assert truth_calls == []


def test_swapped_clean_and_star_truth_rejected_before_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    rows = {'source':{'novel':{}}, 'clean':{'ids':np.array(['x'])}, 'star':{'ids':np.array(['x'])}}
    runtime = dict(source_novel_truth='source', clean_truth='clean', star_truth='star')
    monkeypatch.setattr(experiment, '_prediction', lambda root:({},runtime,{},rows,{}))
    monkeypatch.setattr(experiment.reference_score, '_source_truth', lambda *args:{})
    def truth(path, ids):
        scenario = 'practical_high' if path == 'clean' else 'clean'
        return dict(ids=ids,y=np.array([0]),tx=np.array(['x']),receiver=np.array([34]),
                    day=np.array(['2021_03_15']),scenario=np.array([scenario]))
    monkeypatch.setattr(experiment.reference_score, '_future_truth', truth)
    with pytest.raises(ValueError, match='scene roster'):
        experiment.score(tmp_path/'prediction', tmp_path/'score')
