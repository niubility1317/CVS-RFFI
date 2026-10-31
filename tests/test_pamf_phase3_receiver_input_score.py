"""Candidate identity changes remain separate from same-DURA F0 protection."""
import importlib
import numpy as np
from pathlib import Path
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.receiver_input_score')


def test_candidate_closed_cannot_replace_independent_F0_protection():
    s=api();y=np.arange(10,dtype=np.int64);n=len(y)
    a=dict(closed=(np.minimum(y,5)+1)%6,baseline_closed=np.minimum(y,5),scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,2)))
    t=dict(y=y,receiver=np.full(n,34),day=np.asarray(['2021_03_15']*n),scenario=np.asarray(['clean']*n))
    r=s.fresh._target(a,t,'clean_DURA',1)
    assert r['overall']['original_F0_macro_closed_accuracy']==1. and r['overall']['candidate_macro_closed_accuracy']==0.
    assert r['overall']['original_closed_minus_new_open']==1. and not r['goals']['passed']
    assert r['by_day']['2021_03_23']['A_K'] is None


def test_primary_cannot_borrow_SOURCE_improvement_or_missing_day_coverage(tmp_path,monkeypatch):
    from test_pamf_phase3_prototype_identity_score import fixture
    s=api();_,_,oldcal,oldrows,paths=fixture(tmp_path)
    rows={k:{a:dict(v['PROTO']) for a in s.ARMS} for k,v in oldrows.items()}
    for k in rows:rows[k]['OLD_PCD']=oldrows[k]['OLD_PCD']
    done=dict(code_commit='b'*40,inherited_adapter_optimizer_updates=400)
    cal=dict(methods={name:dict(constraints_passed=True) for name in s.METHODS})
    monkeypatch.setattr(s,'_prediction',lambda *a:(done,cal,rows))
    result=s.score(tmp_path/'pred',*paths,tmp_path/'score')
    assert not result['adaptation_success'] and not result['star_A_K_gain_over_both_controls'] and result['next_step']=='NO_PROMOTION'
    assert result['A_N'] is result['H_OW'] is None and not result['final_phase3_complete']
    assert result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']['A_K'] is None


def test_actual_trained_theta_producer_gate_three_truths_and_launcher(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_receiver_input_run import fixture
    s=api()
    with fixture(tmp_path,monkeypatch) as (p,c,pm,b,pcd,flow,trained):
        for change in (dict(truth='forbidden'),dict(batch_size=True),dict(output_root=c['reference_config']['output_root']),
                dict(train_config=dict(c['train_config'],reference_prediction_root='wrong'))):
            with subtests.test(config=next(iter(change))),pytest.raises(ValueError):p.validate_config(dict(c,**change))
        original_smoke=p._smoke
        with monkeypatch.context() as failure:
            failure.setattr(p,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke failed')))
            with pytest.raises(ValueError,match='smoke failed'):p.run(c,'cpu')
        assert flow==[] and not Path(c['output_root']).exists()
        done=p.run(c,'cpu');out=Path(c['output_root'])
        assert flow==['smoke','source_V','source_NV','Q_clean_DURA','IQ_clean_DURA','Q_star_DURA','IQ_star_DURA']
        assert done['real_F_forward_rows']==done['real_signal_forward_rows']==4*p.Q_COUNT and done['G_forward_rows']==2*p.Q_COUNT
        assert done['new_density_fits']==done['new_threshold_sets_fitted']==done['source_IQ_reads']==0
        assert not done['classification_unchanged'] and done['training_identity']==trained
        for arm in p.ARMS:
            a=p._target_rows(out,'star_DURA',arm);np.testing.assert_array_equal(a['closed'],a['default_logits'].argmax(1))
            old=p._target_rows(out,'star_DURA','OLD_PCD');np.testing.assert_array_equal(a['baseline_logits'],old['default_logits'])
        # Scoring may consume only completed artifacts, not checkpoints, IQ or fitting.
        monkeypatch.setattr(p,'_backend',lambda *a:pytest.fail('No score NN load'));monkeypatch.setattr(p._train(),'load_adapters',lambda *a:pytest.fail('No score checkpoint load'))
        monkeypatch.setattr(p,'_public',lambda *a:pytest.fail('No score Q IQ'));monkeypatch.setattr(p.parent.fresh.risk,'calibrate_known',lambda *a:pytest.fail('No recalibration'))
        _,_,_,_,rows=p._prediction_gate(out)
        nv=rows['source_NV']['OLD_PCD'];grid=[(tx,r,scene) for tx in s.common.UNKNOWN_TX for r in s.common.RECEIVERS for scene in s.common.SCENES]
        source=tmp_path/'source_truth.npz';np.savez(source,ids=nv['ids'],rx_i=nv['rx_i'],day_i=nv['day_i'],tx=np.asarray([v[0] for v in grid]),scenario=np.asarray([v[2] for v in grid]))
        paths=[source];y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);days=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in s.CONDITIONS[1:]:
            a=rows['Q_'+condition]['OLD_PCD'];path=tmp_path/(condition+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=days,
                scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];st=s._source_truth;ft=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),st(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),ft(*a))[1])
        result=s.score(out,*paths,tmp_path/'receiver/score')
        assert reads==['source','future','future'] and result['truth_last'] and result['all_three_predictions_before_truth']
        assert type(result['adaptation_success']) is bool and set(result['methods'])==set(s.METHODS)
        from experiments.pamf_phase3_open_world import receiver_input_launch as launch
        r=dict(run_root=str(tmp_path/'receiver'),code_commit=c['code_commit'])
        assert launch._terminal(r,'predict')==done and launch._terminal(r,'score')['status']==s.STATUS
        main=out/'targets/star_DURA/TARGET_INPUT/predictions.npz';oldpath=out/'targets/star_DURA/OLD_PCD/predictions.npz'
        taupath=out/'thresholds.npz';density=out/'models/density.npz';trainpath=Path(c['train_config']['output_root'])/'complete.json';complete=out/'complete.json'
        files={path:path.read_bytes() for path in (main,oldpath,taupath,density,trainpath,complete)}
        for defect in ('missing_Q','wrong_candidate_route','changed_baseline','changed_control','retau','changed_GMM','wrong_NN_parent','partial_freeze','private_Q'):
            with subtests.test(defect=defect):
                reads.clear()
                if defect=='missing_Q':main.unlink()
                elif defect in ('wrong_NN_parent','partial_freeze'):
                    path=trainpath if defect=='wrong_NN_parent' else complete;v=p._read(path)
                    if defect=='wrong_NN_parent':v['identities']['TARGET_INPUT']['parent_PCD_identity']={'wrong':True}
                    else:v['all_adapters_frozen_before_Q']=False
                    path.unlink();p._write(path,v)
                else:
                    path=oldpath if defect=='changed_control' else taupath if defect=='retau' else density if defect=='changed_GMM' else main
                    a=p._npz(path)
                    if defect=='wrong_candidate_route':a['closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='changed_baseline':a['baseline_closed'][0]=(a['baseline_closed'][0]+1)%6
                    elif defect=='changed_control':a['scores'][0,1]+=.1
                    elif defect=='retau':a['conditional'][5,0]+=.1
                    elif defect=='changed_GMM':a['known_means'][0,0]+=.1
                    else:a['y']=np.zeros(p.Q_COUNT,np.int64)
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/defect)
                assert reads==[]
                for path,value in files.items():path.write_bytes(value)
        with pytest.raises(FileExistsError):p.run(c,'cpu')
