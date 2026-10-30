"""Four-density reconstruction, original-P1 protection and source truth isolation."""
import importlib
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from copy import deepcopy


def truth_file(root,s,novel):
    grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    data=dict(ids=novel['ids'],rx_i=novel['rx_i'],day_i=novel['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
    p=root/'synthetic_truth.npz';np.savez(p,**{k:v[::-1] for k,v in data.items()});return p


def test_actual_checkpoint_four_column_predictions_to_truth_last_CPU_score(tmp_path,monkeypatch):
    from test_pamf_phase3_star_raw_open_eval import fixture
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,values):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts)
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scorer cannot load signal checkpoint'))
        monkeypatch.setattr(m._core(),'fit_known',lambda *a:pytest.fail('No refit'))
        monkeypatch.setattr(m._core(),'fit_background',lambda *a:pytest.fail('No refit'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No recalibration'))
        result=s.score(c['output_root'],truth_file(tmp_path,s,n),tmp_path/'score')
        assert result['primary_method']=='P0_known12' and result['truth_last'] and result['target_reads']==0
        assert result['paired_primary_vs_P1_known']['reference_index']==0 and result['paired_primary_vs_P0_ratio']['reference_index']==3
        assert result['classification_path']==result['identity_baseline_path']=='P1' and result['new_rejection_input_path']==result['teacher_input_path']=='P0'
        for method in result['methods'].values():
            assert len(method['groups'])==42 and len(method['hard_names'])==12 and len(method['diagnostic_names'])==30
            assert len(method['by_TX'])==9 and len(method['by_day'])==3
            assert len(method['confidence_intervals']['A_K_per_class'])==6 and len(method['confidence_intervals']['R_U_per_class'])==3
        assert result['A_N'] is result['H_OW'] is None and result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(c['output_root'],tmp_path/'forbidden',tmp_path/'score')


def test_four_models_original_identity_and_P0_P1_columns_verify_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_raw_open_eval import fixture
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,values):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);out=Path(c['output_root'])
        paths=[out/'complete.json',out/'thresholds.npz',out/'models/P0_known12.npz',out/'novel_predictions.npz']
        originals={p:p.read_bytes() for p in paths};monkeypatch.setattr(s,'_truth',lambda *a:pytest.fail('Truth before complete four-column verification'))
        for defect in ('P1_teacher','control_tau','shared_model','raw_route','margin','P1_features','private_novel','missing_novel'):
            with subtests.test(defect=defect):
                for p,value in originals.items():p.write_bytes(value)
                if defect=='P1_teacher':
                    p=out/'complete.json';d=m._read(p);d['teacher_input_path']='P1';p.unlink();m._write(p,d)
                elif defect=='missing_novel':(out/'novel_predictions.npz').unlink()
                else:
                    p=out/('thresholds.npz' if defect=='control_tau' else 'models/P0_known12.npz' if defect=='shared_model' else 'novel_predictions.npz')
                    with np.load(p) as z:d={k:z[k].copy() for k in z.files}
                    if defect=='control_tau':d['conditional'][0,0]+=.1
                    elif defect=='shared_model':d['known_means'][0,0]+=.1
                    elif defect=='raw_route':d['raw_scores'][0,2]+=.1
                    elif defect=='margin':d['scores'][0,1]+=.1
                    elif defect=='P1_features':d['signal_features']=np.roll(d['signal_features'],1,axis=0)
                    else:d['tx']=np.asarray(['private']*len(d['ids']))
                    np.savez(p,**d)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,tmp_path/'forbidden',tmp_path/'score')
                assert not (tmp_path/'score/complete.json').exists()


def test_cross30_is_diagnostic_and_higher_controls_cannot_replace_failed_main(tmp_path,monkeypatch):
    from test_pamf_phase3_star_raw_open_eval import fixture
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,n,values):
        m.evaluate(c,'cpu');monkeypatch.setattr(s,'COUNTS',counts);marker,cal,k,n=deepcopy(s._prediction(c['output_root']))
        for report in cal['methods'].values():report['constraints_passed']=True
        k={key:np.repeat(value,3,axis=0) for key,value in k.items()};k['ids']=np.asarray([str(v)+'-'+str(i%3) for i,v in enumerate(k['ids'])])
        k['scores'][:]=1.;n['scores'][:]=-1.;ix=np.flatnonzero((k['rx_i']==1)&(k['scenario']==s.common.SCENES[0])&(k['y']==0))[0];k['scores'][ix,2]=-1.
        monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,k,n));truth=truth_file(tmp_path,s,n)
        result=s.score(c['output_root'],truth,tmp_path/'cross_diagnostic');goals=result['methods'][s.PRIMARY]['goals']
        assert goals['known_12_groups_passed'] and not goals['known_42_groups_passed'] and result['source_gate_passed']
        assert result['methods'][s.PRIMARY]['cross_guard_violations_or_missing']
        k['scores'][:]=1.;n['scores'][:,2]=1.
        result=s.score(c['output_root'],truth,tmp_path/'main_fail')
        assert all(result['methods'][s.METHODS[j]]['goals']['passed'] for j in (0,1,3))
        assert not result['source_gate_passed'] and not result['eligible_for_target_prereg'] and result['next_step']=='NO_PROMOTION'
        n['scores'][:,2]=-1.;k['y'][:]=0
        result=s.score(c['output_root'],truth,tmp_path/'missing_coverage')
        assert result['primary_summary']['A_K'] is None and not result['source_gate_passed']


def test_primary_P0_known_index2_has_explicit_P1_and_P0_ratio_pairs():
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_raw_open_score')
    k=dict(y=np.asarray([0,1,2,3]),closed=np.asarray([0,0,2,3]),scores=np.asarray([[1,-1,1,1],[-1,1,1,-1],[-1,1,-1,1],[1,1,-1,-1]],np.float64))
    n=dict(scores=np.asarray([[-1,1,-1,1],[1,-1,-1,-1],[-1,-1,1,-1]],np.float64))
    a=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=2,reference_index=0)
    b=s.paired_counts(k,n,np.ones(4,bool),np.ones(3,bool),primary_index=2,reference_index=3)
    assert a['known_accepted_net']==0 and a['known_correct_accepted_net']==-1 and a['unknown_rejected_net']==0
    assert b['known_accepted_net']==0 and b['known_correct_accepted_net']==-1 and b['unknown_rejected_net']==0
