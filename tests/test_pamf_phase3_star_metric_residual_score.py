"""Independent source truth isolation, learned-density reconstruction and primary-only promotion."""
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
from test_pamf_phase3_star_metric_residual_eval import fixture,write


def truth_file(root,s,novel):
    rows=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=novel['ids'],rx_i=novel['rx_i'],day_i=novel['day_i'],tx=np.asarray([v[0] for v in rows]),scenario=np.asarray([v[2] for v in rows]))
    p=root/'private_synthetic_truth.npz';np.savez(p,**{k:v[::-1] for k,v in data.items()});return p


def test_actual_toy_checkpoint_evaluation_to_CPUscore_has_no_NN_reload_or_source_refit(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_residual_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,novel):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('Independent CPU scorer cannot load NN/checkpoint'))
        monkeypatch.setattr(m._core(),'fit_known',lambda *a,**k:pytest.fail('No fitting'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('No background fitting'))
        result=s.score(c['output_root'],truth_file(tmp_path,s,novel),tmp_path/'score')
        assert result['primary_arm']=='M_LOE' and result['primary_method']=='learned_metric_density_ratio'
        assert result['truth_last'] and result['source_only'] and result['control_reused_exactly']
        assert result['target_reads']==0 and result['A_N'] is result['H_OW'] is None
        assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        assert len(result['methods'][s.PRIMARY]['groups'])==42
        with pytest.raises(FileExistsError):s.score(c['output_root'],tmp_path/'forbidden',tmp_path/'score')


def test_all_predictions_and_frozen_states_checked_before_private_truth(tmp_path,monkeypatch,subtests):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_residual_score')
    for defect in ('identity','control_tau','control_score','learned_shape','new_margin','missing_novel','private'):
        with subtests.test(defect=defect),monkeypatch.context() as mp:
            root=tmp_path/defect;root.mkdir()
            with fixture(root,mp) as (m,c,counts,events,novel):
                m.evaluate(c,'cpu');mp.setattr(s,'COUNTS',counts);out=Path(c['output_root'])
                if defect=='identity':
                    d=m._read(out/'complete.json');d['model_identity']['native_updates']=2220;write(out/'complete.json',d)
                elif defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/('thresholds.npz' if defect=='control_tau' else 'novel_predictions.npz')
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='control_tau':d['conditional'][0,0]+=.1
                    elif defect=='control_score':d['scores'][0,0]+=.1
                    elif defect=='learned_shape':d['learned_features']=d['learned_features'][:,:159]
                    elif defect=='new_margin':d['scores'][0,1]+=.1
                    else:d['tx']=np.asarray(['private']*len(d['ids']))
                    np.savez(p,**d)
                mp.setattr(s,'_truth',lambda *a:pytest.fail('Private truth opened before all frozen predictions'))
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,root/'forbidden',root/'score')


def test_only_M_LOE_learned_method_can_be_eligible_not_M_L_or_better_F0_control(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_metric_residual_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,novel):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);payload=s._prediction(c['output_root'])
        marker,cal,known,n=deepcopy(payload);known['scores'][:]=1.;n['scores'][:]=-1.;marker['arm']='M_L'
        monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,known,n))
        result=s.score(c['output_root'],truth_file(tmp_path,s,novel),tmp_path/'ML_score')
        assert result['source_gate_passed'] and not result['eligible_for_target_prereg']
        assert result['next_step']=='NO_PROMOTION'
        marker['arm']='M_LOE';n['scores'][:,1]=1.
        result=s.score(c['output_root'],truth_file(tmp_path,s,novel),tmp_path/'main_score')
        assert result['methods'][s.METHODS[0]]['goals']['passed'] and not result['source_gate_passed']
        assert not result['eligible_for_target_prereg'] and result['next_step']=='NO_PROMOTION'
        assert result['paired_primary_vs_control']['counts']['unknown_rejected_net']==-90
