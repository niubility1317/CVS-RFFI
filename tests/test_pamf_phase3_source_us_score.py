"""The source-Us primary must beat both controls without target truth feedback."""
import importlib
import pytest
import numpy as np
from pathlib import Path


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_us_score')


def test_primary_needs_both_accuracy_gains_and_both_rejection_protections():
    m=api();controls=[dict(A_K=.6,R_U=.8),dict(A_K=.7,R_U=.9)]
    assert m._control_gate(dict(A_K=.71,R_U=.9),controls)==(True,True)
    assert m._control_gate(dict(A_K=.65,R_U=.95),controls)==(False,True)
    assert m._control_gate(dict(A_K=.8,R_U=.85),controls)==(True,False)


def test_partial_predictions_never_open_any_truth(tmp_path,monkeypatch):
    m=api();called=[]
    monkeypatch.setattr(m,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('missing star predictions')))
    monkeypatch.setattr(m,'_source_truth',lambda *a:called.append('source'))
    monkeypatch.setattr(m,'_future_truth',lambda *a:called.append('future'))
    with pytest.raises(ValueError,match='missing star'):m.score(tmp_path/'predict','NV','clean','star',tmp_path/'score')
    assert not called and not (tmp_path/'score').exists()


def test_actual_new_student_only_pipeline_then_cpu_three_truths_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_source_us_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,pm,calls,forward_rows):
        done=m.run(c,'cpu');root=Path(c['output_root'])
        assert calls==['L','OE','V','novel','clean_DURA','star_DURA']
        assert forward_rows==[2,m.Q_COUNT,m.Q_COUNT]
        assert done['real_signal_forward_rows']==sum(m.COUNTS.values())+2*m.Q_COUNT
        assert done['real_F_forward_rows']==done['G_forward_rows']==2*m.Q_COUNT
        assert done['new_threshold_sets_fitted']==2 and done['source_density_models_fitted']==1
        for delta in (dict(truth='private'),dict(output_root=c['train_config']['output_root'])):
            with subtests.test(config=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        with pytest.raises(FileExistsError):m.run(c,'cpu')
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda r,role:m._source_rows(r,'source_DURA',role)),_target_rows=staticmethod(m._target_rows)))
        sp=type('Scoring',(),dict(common=s.common,CONDITIONS=()))
        paths=[truth_files(tmp_path,sp,proxy,root)[0]]
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64)
        for name in m.CONDITIONS[1:]:
            a=m._target_rows(root,name);path=tmp_path/(name+'_truth.npz');paths.append(path)
            np.savez(path,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),
                day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean' if name=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];source=s._source_truth;target=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_future_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scorer must not load networks'))
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('CPU scorer must not load checkpoint'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('CPU scorer must not recalibrate'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('CPU scorer must not fit'))
        result=s.score(root,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['truth_last']
        assert result['adaptation_success'] is None and not result['eligible_primary'] and result['full_goal_status']=='ACTIVE'
        late=result['methods'][s.PRIMARY]['paths']['star_DURA']['by_day']['2021_03_23']
        assert late['A_K'] is None and late['R_U'] is not None
        path=root/'targets/star_DURA/predictions.npz';saved=path.read_bytes()
        for defect in ('missing','private','old_control','new_raw','baseline'):
            with subtests.test(defect=defect):
                path.write_bytes(saved);reads.clear()
                if defect=='missing':path.unlink()
                else:
                    a=m._npz(path)
                    if defect=='private':a['y']=y
                    elif defect=='old_control':a['scores'][0,0]=np.nextafter(a['scores'][0,0],np.inf)
                    elif defect=='new_raw':a['raw_scores'][0,5]+=.1
                    else:a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(root,*paths,tmp_path/('bad_'+defect))
                assert reads==[]
