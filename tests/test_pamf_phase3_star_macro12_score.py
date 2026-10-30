"""Four-column truth-last scoring and explicit hard12/cross30 distinction."""
import importlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest


def test_primary_last_column_pairing_uses_signal42_and_frozen12_controls():
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_score')
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.asarray([[1,1,-1,1],[1,0,1,-1],[-1,-1,-1,1],[1,1,1,-1]],np.float64))
    n=dict(scores=np.asarray([[0,-1,1,-1],[0,1,-1,1],[0,1,1,-1]],np.float64))
    a=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=3,reference_index=2)
    b=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=3,reference_index=1)
    assert a['known_accepted_net']==0 and a['known_correct_accepted_net']==1 and a['unknown_rejected_net']==1
    assert b['known_accepted_net']==-1 and b['known_correct_accepted_net']==0 and b['unknown_rejected_net']==1


def fixture(root,mp):
    from test_pamf_phase3_star_macro12_run import fixture as run_fixture
    return run_fixture(root,mp)


def truth_file(root,s,novel):
    grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=novel['ids'],rx_i=novel['rx_i'],day_i=novel['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
    path=root/'synthetic_truth.npz';np.savez(path,**{k:v[::-1] for k,v in data.items()});return path


def test_actual_synthetic_checkpoint_calibration_to_CPUscore_without_NN_or_fit(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_score')
    with fixture(tmp_path,monkeypatch) as (m,c,parent,rows):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent scorer cannot load neural checkpoint'))
        monkeypatch.setattr(m._core(),'fit_known',lambda *a:pytest.fail('No density fit'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('No background fit'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No recalibration in scorer'))
        result=s.score(c['output_root'],truth_file(tmp_path,s,rows['novel']),tmp_path/'score')
        assert result['primary_method']=='signal_known12' and result['truth_last'] and result['target_reads']==0
        assert result['paired_primary_vs_signal42']['reference_index']==2 and result['paired_primary_vs_frozen12']['reference_index']==1
        for j,name in enumerate(s.METHODS):
            method=result['methods'][name];assert len(method['groups'])==42 and len(method['hard_names'])==(42 if j in (0,2) else 12)
            assert len(method['confidence_intervals']['A_K_per_class'])==6 and len(method['confidence_intervals']['R_U_per_class'])==3
        assert result['native_updates']==result['new_neural_updates']==result['real_signal_forward_rows']==result['G_forward_rows']==0
        assert result['synthetic_signal_forward_rows']==2 and result['A_N'] is result['H_OW'] is None
        assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(c['output_root'],tmp_path/'forbidden',tmp_path/'score')


def test_models_four_columns_controls_and_scope_must_all_verify_before_truth(tmp_path,monkeypatch,subtests):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_score')
    with fixture(tmp_path,monkeypatch) as (m,c,parent,rows):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);out=Path(c['output_root'])
        paths=[out/'complete.json',out/'calibration/complete.json',out/'thresholds.npz',out/'models/signal_known.npz',out/'novel_predictions.npz']
        originals={p:p.read_bytes() for p in paths};monkeypatch.setattr(s,'_truth',lambda *a:pytest.fail('Truth before all four frozen states and predictions'))
        for defect in ('marker','model','control_tau','decreasing_tau','scope','raw_column','margin_column','frozen_features','private_novel','missing_novel'):
            with subtests.test(defect=defect):
                for p,value in originals.items():p.write_bytes(value)
                if defect in ('marker','scope'):
                    p=out/('complete.json' if defect=='marker' else 'calibration/complete.json');d=m._read(p)
                    if defect=='marker':d['new_neural_updates']=2220
                    else:d['method_hard_groups']['signal_known12']=list(s.common.GROUP_NAMES)
                    p.unlink();m._write(p,d)
                elif defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/('models/signal_known.npz' if defect=='model' else 'thresholds.npz' if defect in ('control_tau','decreasing_tau') else 'novel_predictions.npz')
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='model':d['known_means'][0,0]+=.1
                    elif defect=='control_tau':d['conditional'][2,0]+=.1
                    elif defect=='decreasing_tau':d['conditional'][3,0]=d['conditional'][2,0]-1.
                    elif defect=='raw_column':d['raw_scores'][0,3]+=.1
                    elif defect=='margin_column':d['scores'][0,1]+=.1
                    elif defect=='frozen_features':d['learned_features']=np.roll(d['learned_features'],1,axis=0)
                    else:d['tx']=np.asarray(['private']*len(d['ids']))
                    np.savez(p,**d)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,tmp_path/'forbidden',tmp_path/'score')
                assert not (tmp_path/'score/complete.json').exists()


def test_cross_excess_and_missing_are_diagnostic_but_hard12_and_primary_remain_required(tmp_path,monkeypatch):
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_score')
    with fixture(tmp_path,monkeypatch) as (m,c,parent,rows):
        m.run(c,'cpu');monkeypatch.setattr(s,'COUNTS',m.COUNTS);marker,cal,k,n=deepcopy(s._prediction(c['output_root']))
        for method in cal['methods'].values():method['constraints_passed']=True
        # Three rows per TX/cross cell make one lost correct query exceed cross
        # 2pp while remaining below the RX and scene macro budgets.
        k={key:np.repeat(value,3,axis=0) for key,value in k.items()}
        k['ids']=np.asarray([str(v)+'-'+str(i%3) for i,v in enumerate(k['ids'])]);original_y=k['y'].copy()
        k['scores'][:]=1.;n['scores'][:]=-1.;target=(k['rx_i']==1)&(k['scenario']==s.common.SCENES[0])&(k['y']==0)
        assert int(target.sum())==3;k['scores'][np.flatnonzero(target)[0],3]=-1.
        monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,k,n));truth=truth_file(tmp_path,s,n)
        result=s.score(c['output_root'],truth,tmp_path/'cross_excess')
        goals=result['methods'][s.PRIMARY]['goals'];assert goals['known_12_groups_passed'] and not goals['known_42_groups_passed'] and result['source_gate_passed']
        entry=result['methods'][s.PRIMARY]['cross_guard_violations_or_missing']['rxscene/1/'+s.common.SCENES[0]]
        assert 'original_F0_loss_above_2pp' in entry['reasons'] and entry['original_closed_minus_new_open']>.02
        k['scores'][:]=1.;k['y'][target]=1
        result=s.score(c['output_root'],truth,tmp_path/'cross_missing')
        assert result['source_gate_passed'] and not result['methods'][s.PRIMARY]['goals']['known_42_groups_passed']
        entry=result['methods'][s.PRIMARY]['cross_guard_violations_or_missing']['rxscene/1/'+s.common.SCENES[0]]
        assert 'missing_known_class' in entry['reasons'] and entry['A_K'] is None
        k['y']=original_y.copy();n['scores'][:,3]=1.
        result=s.score(c['output_root'],truth,tmp_path/'main_failed')
        assert all(result['methods'][s.METHODS[j]]['goals']['passed'] for j in (0,1,2))
        assert result['methods']['frozen_known12']['goals']['passed'] and not result['source_gate_passed'] and result['next_step']=='NO_PROMOTION'
        n['scores'][:,3]=-1.;k['y'][:]=0
        result=s.score(c['output_root'],truth,tmp_path/'hard_missing')
        assert result['primary_summary']['A_K'] is None and not result['source_gate_passed'] and not result['eligible_for_target_prereg']
