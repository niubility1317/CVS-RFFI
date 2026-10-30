"""Local-covariance scoring uses fixed identity and three truth-last joins."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_local_cov_score')


def target_rows(scene='clean',count=6):
    y=np.repeat(np.arange(9,dtype=np.int64),count);n=len(y);closed=np.minimum(y,5)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=closed,baseline_closed=closed.copy(),scores=np.ones((n,4),np.float64))
    a['scores'][y>=6]=-1.
    return a,dict(y=y,receiver=np.full(n,26,np.int64),scenario=np.asarray([scene]*n))


def test_primary_target_constraints_and_predicted_control_nonregression():
    s=api()
    assert s.METHODS==('pooled_ratio','predicted_ratio','local_pooled_ratio','local_predicted_ratio')
    assert s.PRIMARY==s.METHODS[3]
    a,t=target_rows()
    good=s._target(a,t,'clean_DURA',3)
    assert good['goals']['passed']
    a['scores'][np.flatnonzero(t['y']==0)[0],3]=-1.
    bad=s._target(a,t,'clean_DURA',3)
    assert not bad['goals']['passed']  # A_K damage exceeds 2 pp despite C_K > 90%.
    a,t=target_rows('practical_low_urban')
    missing=s._target(a,t,'star_DURA',3)
    assert missing['goals']['overall_passed'] and not missing['goals']['passed']
    assert missing['by_scene']['practical_high']['R_U'] is None
    assert s._nonregression(dict(A_K=.8,R_U=.9),dict(A_K=.8,R_U=.9))
    assert not s._nonregression(dict(A_K=.79,R_U=.95),dict(A_K=.8,R_U=.9))


def test_all_predictions_validated_before_any_truth_and_no_partial_output(tmp_path,monkeypatch):
    s=api();reads=[]
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source'))
    monkeypatch.setattr(s,'_target_truth',lambda *a:reads.append('Q'))
    monkeypatch.setattr(s,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('incomplete local covariance predictions')))
    with pytest.raises(ValueError,match='incomplete'):
        s.score('pred','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_source_calibration_mapping_and_main_cannot_borrow_better_control(tmp_path,monkeypatch):
    s=api();kg=[(y,r,scene) for y in range(6) for r in s.common.RECEIVERS for scene in s.common.SCENES]
    ng=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([v[0] for v in kg],np.int64),
        rx_i=np.asarray([v[1] for v in kg],np.int64),day_i=np.arange(len(kg),dtype=np.int64)%3+1,
        scenario=np.asarray([v[2] for v in kg]),scores=np.ones((len(kg),4),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([v[1] for v in ng],np.int64),
        day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),4),np.float64))
    nt=dict(tx=np.asarray([v[0] for v in ng]),scenario=np.asarray([v[2] for v in ng]))
    clean,ct=target_rows(count=60);star,st=target_rows(count=60)
    st['scenario']=np.tile(np.asarray(s.common.SCENES),len(st['y'])//6)
    rows=dict(known=k,novel=n,clean_DURA=clean,star_DURA=star)
    marker=dict(code_commit='a'*40,local_covariance_fit_modes=90,real_signal_forward_rows=6300,real_F_forward_rows=0,G_forward_rows=0,
        synthetic_signal_forward_rows=2,synthetic_F_forward_rows=2,G_synthetic_forward_rows=2)
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS});reads=[]
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,rows))
    monkeypatch.setattr(s,'_source_truth',lambda *a:reads.append('source') or nt)
    monkeypatch.setattr(s,'_target_truth',lambda path,*a:reads.append(path) or (ct if path=='clean' else st))
    good=s.score('pred','source','clean','star',tmp_path/'good')
    assert reads==['source','clean','star'] and good['development_gate_passed']
    assert good['source_known_12_passed'] and not good['source_unknown_90_diagnostic_passed']
    assert set(good['paired'])=={'primary_vs_predicted_ratio','primary_vs_pooled_ratio','primary_vs_local_pooled_ratio'}
    assert len(good['methods'][s.PRIMARY]['source']['groups'])==42
    star['scores'][0,3]=-1.
    bad=s.score('pred','source','clean','star',tmp_path/'bad')
    assert bad['methods'][s.PRIMARY]['star_DURA']['goals']['passed']
    assert not bad['star_not_worse_than_predicted_control'] and not bad['development_gate_passed']
    assert bad['next_step']=='NO_PROMOTION' and bad['full_goal_status']=='ACTIVE'
    assert not bad['final_phase3_complete'] and bad['A_N'] is bad['H_OW'] is None
    assert bad['paired']['primary_vs_predicted_ratio']['counts']['star_DURA']['known_correct_accepted_net']==-1


def test_actual_producer_to_truth_last_CPU_score_and_corruption_boundaries(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_local_cov_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,public,backend):
        m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['L','fit','known','novel','clean_DURA','star_DURA']
        monkeypatch.setattr(s,'COUNTS',counts);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source_truth=s._source_truth;target_truth=s._target_truth
        def source(*a):reads.append('source');return source_truth(*a)
        def target(path,*a):reads.append(Path(path).name);return target_truth(path,*a)
        monkeypatch.setattr(s,'_source_truth',source);monkeypatch.setattr(s,'_target_truth',target)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('CPU scoring never loads a neural checkpoint'))
        monkeypatch.setattr(m.core,'fit',lambda *a:pytest.fail('No local covariance refit in scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No V recalibration or novel-driven threshold choice'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz']
        assert result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['costs']['real_signal_forward_rows']==counts['L'] and result['costs']['real_F_forward_rows']==result['costs']['G_forward_rows']==0
        assert result['source_V_threshold_sets_fitted']==2 and result['local_covariance_fit_modes']==90
        assert result['primary_method']=='local_predicted_ratio' and not result['development_gate_passed']
        assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        for method in s.METHODS:
            assert len(result['methods'][method]['source']['groups'])==42
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'thresholds.npz',out/'calibration/complete.json',out/'models/density.npz',out/'models/local.npz',
            out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('missing_Q','not_frozen','old_tau','old_calibration','density','local_matrix','old_control','local_raw','invalid_tau','private_Q'):
            with subtests.test(defect=defect):
                for p,data in saved.items():p.write_bytes(data)
                reads.clear()
                if defect=='missing_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('not_frozen','old_calibration'):
                    p=out/('complete.json' if defect=='not_frozen' else 'calibration/complete.json');a=m._read(p)
                    if defect=='not_frozen':a['all_thresholds_frozen_before_novel_and_Q']=False
                    else:a['methods'][s.METHODS[0]]['constraints_passed']=not a['methods'][s.METHODS[0]]['constraints_passed']
                    p.unlink();m._write(p,a)
                else:
                    p=(out/'thresholds.npz' if defect in ('old_tau','invalid_tau') else out/'models/density.npz' if defect=='density'
                        else out/'models/local.npz' if defect=='local_matrix' else out/'novel_predictions.npz' if defect=='old_control'
                        else out/'targets/clean_DURA/predictions.npz')
                    with np.load(p,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='invalid_tau':a['conditional'][3,0]=np.nan
                    elif defect=='density':a['known_means'][0,0]+=.1
                    elif defect=='local_matrix':a['transforms'][0,0,0]*=1.1
                    elif defect=='old_control':a['scores'][0,0]+=.1
                    elif defect=='local_raw':a['raw_scores'][0,3]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):
                    s.score(out,*paths,tmp_path/'invalid_score')
                assert not reads and not (tmp_path/'invalid_score').exists()
