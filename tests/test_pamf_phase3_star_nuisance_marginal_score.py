"""One real toy producer and independent CPU scoring; all truths remain last."""
from pathlib import Path
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_marginal_score')


def test_primary_cannot_borrow_predicted_control_or_hide_source_scope(tmp_path,monkeypatch):
    from test_pamf_phase3_star_local_cov_score import target_rows
    s=api();assert s.METHODS==('pooled_ratio','predicted_ratio','marginal_pooled_ratio','marginal_predicted_ratio')
    assert s.PRIMARY==s.METHODS[3]
    kg=[(c,r,sc) for c in range(6) for r in s.common.RECEIVERS for sc in s.common.SCENES]
    ng=[(c,r,sc) for c in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for sc in s.common.SCENES]
    k=dict(ids=np.asarray(['k'+str(i) for i in range(len(kg))]),y=np.asarray([x[0] for x in kg],np.int64),rx_i=np.asarray([x[1] for x in kg],np.int64),
        day_i=np.arange(len(kg),dtype=np.int64)%3+1,scenario=np.asarray([x[2] for x in kg]),scores=np.ones((len(kg),4),np.float64))
    k['closed']=k['y'].copy();k['baseline_closed']=k['y'].copy()
    n=dict(ids=np.asarray(['n'+str(i) for i in range(len(ng))]),rx_i=np.asarray([x[1] for x in ng],np.int64),day_i=np.arange(len(ng),dtype=np.int64)%3+1,scores=np.ones((len(ng),4),np.float64))
    nt=dict(tx=np.asarray([x[0] for x in ng]),scenario=np.asarray([x[2] for x in ng]))
    clean,ct=target_rows(count=60);star,st=target_rows(count=60);st['scenario']=np.tile(np.asarray(s.common.SCENES),len(st['y'])//6)
    rows=dict(known=k,novel=n,clean_DURA=clean,star_DURA=star)
    marker=dict(code_commit='e'*40,prior_rank=0,**{key:0 for key in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows','synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows')})
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    monkeypatch.setattr(s,'_prediction',lambda *a:(marker,cal,rows));monkeypatch.setattr(s,'_source_truth',lambda *a:nt)
    monkeypatch.setattr(s,'_target_truth',lambda path,*a:ct if path=='clean' else st)
    result=s.score('pred','source','clean','star',tmp_path/'pass')
    assert result['development_gate_passed'] and not result['source_unknown_90_diagnostic_passed']
    star['scores'][0,3]=-1.
    result=s.score('pred','source','clean','star',tmp_path/'fail')
    assert result['methods'][s.PRIMARY]['star_DURA']['goals']['passed']
    assert not result['star_not_worse_than_predicted_control'] and result['next_step']=='NO_PROMOTION'
    assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete'] and result['A_N'] is result['H_OW'] is None


def test_actual_aggregate_only_producer_then_CPU_score_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_nuisance_marginal_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,backend):
        m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','prior','known','novel','clean_DURA','star_DURA']
        monkeypatch.setattr(s,'COUNTS',counts);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        def source_truth(*a):reads.append('source');return source(*a)
        def target_truth(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',source_truth);monkeypatch.setattr(s,'_target_truth',target_truth)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No model loading in CPU scorer'))
        monkeypatch.setattr(m.core,'build_prior',lambda *a:pytest.fail('No prior rebuild/refit in CPU scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No calibration in CPU scorer'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz']
        assert result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['nuisance_prior_fitted'] and 0<=result['prior_rank']<=14
        assert result['source_V_threshold_sets_fitted']==2 and result['neural_updates']==result['density_fit_updates']==0
        assert all(result['costs'][key]==0 for key in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows'))
        assert all(result['costs'][key]==2 for key in ('synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows'))
        assert result['next_step']=='NO_PROMOTION' and not result['final_phase3_complete']
        assert all(len(method['source']['groups'])==42 for method in result['methods'].values())
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'models/density.npz',out/'models/prior.npz',out/'thresholds.npz',out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('missing_source','missing_clean','missing_star','old_control','identity','prior','old_tau','new_tau','raw','not_frozen','private_Q'):
            with subtests.test(defect=defect):
                for p,value in saved.items():p.write_bytes(value)
                reads.clear()
                if defect.startswith('missing_'):
                    p=out/'novel_predictions.npz' if defect=='missing_source' else out/'targets'/('clean_DURA' if defect=='missing_clean' else 'star_DURA')/'predictions.npz';p.unlink()
                elif defect in ('identity','not_frozen'):
                    p=out/'complete.json';a=m._read(p)
                    if defect=='identity':a['model_identity']['updates']+=1
                    else:a['all_thresholds_frozen_before_novel_and_Q']=False
                    p.unlink();m._write(p,a)
                else:
                    p=out/'models/prior.npz' if defect=='prior' else out/'thresholds.npz' if defect in ('old_tau','new_tau') else out/'novel_predictions.npz' if defect=='old_control' else out/'targets/star_DURA/predictions.npz'
                    with np.load(p,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect=='prior':
                        if len(a['variances']):a['variances'][0]+=.1
                        else:a['basis']=np.eye(160,dtype=np.float64)[:1];a['variances']=np.ones(1,np.float64)
                    elif defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='new_tau':a['conditional'][3,0]=np.nan
                    elif defect=='old_control':a['scores'][0,0]+=.1
                    elif defect=='raw':a['raw_scores'][0,3]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):
                    s.score(out,*paths,tmp_path/'invalid')
                assert not reads and not (tmp_path/'invalid').exists()
