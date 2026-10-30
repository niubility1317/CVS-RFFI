"""Independent four-column reconstruction, truth-last and fixed signal primary."""
import importlib
import numpy as np
from copy import deepcopy
from pathlib import Path
import pytest
from test_pamf_phase3_star_signal_open_eval import fixture,write


def test_four_column_paired_uses_frozen_known_and_signal_ratio_not_legacy_indices():
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_score')
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.asarray([[1,-1,-1,1],[-1,1,1,-1],[-1,1,1,-1],[1,1,-1,-1]],np.float64))
    n=dict(scores=np.asarray([[-1,1,1,-1],[1,-1,-1,1],[-1,-1,1,1]],np.float64))
    a=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=1,reference_index=0)
    b=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=1,reference_index=3)
    assert a['known_correct_accepted_net']==0 and a['unknown_rejected_net']==0
    assert b['known_correct_accepted_net']==1 and b['unknown_rejected_net']==1


def truth_file(root,s,n):
    grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=n['ids'],rx_i=n['rx_i'],day_i=n['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
    p=root/'private_synthetic_truth.npz';np.savez(p,**{k:v[::-1] for k,v in data.items()});return p


def test_actual_checkpoint_IQ_producer_to_independent_CPUscore_no_fit_or_NN_reload(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,roles):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        monkeypatch.setattr(m._train(),'load_trained_model',lambda *a:pytest.fail('No NN checkpoint in independent scorer'))
        monkeypatch.setattr(m._core(),'fit_known',lambda *a,**k:pytest.fail('No source fit in scorer'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('No source background fit in scorer'))
        result=s.score(c['output_root'],truth_file(tmp_path,s,n),tmp_path/'score')
        assert result['primary_method']=='signal_known_support' and result['truth_last'] and result['controls_reused_exactly']
        assert len(result['methods'])==4 and all(len(v['groups'])==42 for v in result['methods'].values())
        assert result['paired_primary_vs_frozen_known']['reference_index']==0 and result['paired_primary_vs_signal_ratio']['reference_index']==3
        assert result['target_reads']==0 and result['native_updates']==0 and result['open_signal_updates']==2220
        assert result['A_N'] is result['H_OW'] is None and result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(c['output_root'],tmp_path/'forbidden',tmp_path/'score')


def test_four_columns_states_roles_and_frozen_controls_all_checked_before_truth(tmp_path,monkeypatch,subtests):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,roles):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);out=Path(c['output_root'])
        originals={p:p.read_bytes() for p in (out/'complete.json',out/'thresholds.npz',out/'novel_predictions.npz',out/'models/signal_density_ratio.npz')}
        monkeypatch.setattr(s,'_truth',lambda *a:pytest.fail('Private truth before all four readouts are verified'))
        for defect in ('native_update','control_tau','signal_shape','main_margin','last_margin','frozen_z','column_swap','shared_known','missing_novel','private_TX'):
            with subtests.test(defect=defect):
                for p,value in originals.items():p.write_bytes(value)
                if defect=='native_update':
                    marker=m._read(out/'complete.json');marker['model_identity']['native_updates']=1;write(out/'complete.json',marker)
                elif defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/('thresholds.npz' if defect=='control_tau' else 'models/signal_density_ratio.npz' if defect=='shared_known' else 'novel_predictions.npz')
                    with np.load(p) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='control_tau':a['conditional'][2,0]+=.1
                    elif defect=='signal_shape':a['signal_features']=a['signal_features'][:,:159]
                    elif defect=='main_margin':a['scores'][0,1]+=.1
                    elif defect=='last_margin':a['scores'][0,3]+=.1
                    elif defect=='frozen_z':a['learned_features']=np.roll(a['learned_features'],1,axis=0)
                    elif defect=='column_swap':
                        a['scores'][:,[1,3]]=a['scores'][:,[3,1]];a['raw_scores'][:,[1,3]]=a['raw_scores'][:,[3,1]]
                    elif defect=='shared_known':a['known_means'][0,0]+=.1
                    else:a['tx']=np.asarray(['private']*counts['novel'])
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,tmp_path/'forbidden',tmp_path/'score')
                assert not (tmp_path/'score/complete.json').exists()


def test_primary_not_replaced_by_frozen_or_signal_ratio_and_missing_coverage_is_NA(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_signal_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,roles):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);marker,cal,k,n=deepcopy(s._prediction(c['output_root']))
        k['scores'][:]=1.;n['scores'][:]=-1.;n['scores'][:,1]=1.
        for v in cal['methods'].values():v['constraints_passed']=True
        monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,k,n))
        truth=truth_file(tmp_path,s,n);result=s.score(c['output_root'],truth,tmp_path/'failed_main')
        assert all(result['methods'][s.METHODS[j]]['goals']['passed'] for j in (0,2,3)) and not result['source_gate_passed']
        assert result['next_step']=='NO_PROMOTION' and not result['eligible_for_target_prereg']
        n['scores'][:,1]=-1.;k['y'][:]=0
        result=s.score(c['output_root'],truth,tmp_path/'missing_known')
        assert result['primary_summary']['A_K'] is None and not result['source_gate_passed']
        assert result['primary_summary']['coverage']['known_complete'] is False
