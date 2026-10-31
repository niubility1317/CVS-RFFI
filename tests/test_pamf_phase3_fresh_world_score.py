"""Four-unknown macro scoring keeps missing known dates explicitly undefined."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_world_score')


def test_four_unknowns_RX34_two_dates_missing_known_is_NA():
    s=api();y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);n=len(y)
    a=dict(ids=np.asarray(['q'+str(i) for i in range(n)]),closed=np.minimum(y,5),baseline_closed=np.minimum(y,5),scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,6)))
    t=dict(y=y,tx=np.asarray([s.TX_BY_LABEL[i] for i in y]),receiver=np.full(n,34,np.int64),day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean']*n))
    result=s._target(a,t,'clean_DURA',5);late=result['by_day']['2021_03_23']
    assert result['overall']['R_U']==1. and result['overall']['A_K']==1.
    assert late['A_K'] is late['C_K'] is late['H_score'] is None and late['R_U']==1.
    assert result['late_only_unknown']['R_U']==1. and result['by_RX']['34']['R_U']==1.
    a['scores'][y==9]=-1.;a['scores'][y==6]=1.
    assert s._target(a,t,'clean_DURA',5)['overall']['R_U']==.75


def test_private_future_truth_rejects_old_RX_wrong_day_or_label_map(tmp_path,subtests):
    s=api();ids=np.asarray(['opaque-a','opaque-b','opaque-c']);y=np.asarray([0,6,9],np.int64)
    truth=dict(ids=ids,y=y,tx=np.asarray([s.TX_BY_LABEL[i] for i in y]),receiver=np.full(3,34,np.int64),
        day=np.asarray(['2021_03_15','2021_03_23','2021_03_23']),scenario=np.asarray(['clean']*3))
    path=tmp_path/'truth.npz';np.savez(path,**truth)
    np.testing.assert_array_equal(s._future_truth(path,ids[::-1])['y'],y[::-1])
    for defect in ('old_RX','early_day','TX_map','extra_field'):
        with subtests.test(defect=defect):
            a={k:v.copy() for k,v in truth.items()}
            if defect=='old_RX':a['receiver'][:]=26
            elif defect=='early_day':a['day'][0]='2021_03_01'
            elif defect=='TX_map':a['tx'][2]='5-5'
            else:a['unknown_K']=np.asarray(4)
            np.savez(path,**a)
            with pytest.raises(ValueError):s._future_truth(path,ids)


def test_actual_complete_predictions_then_three_truths_and_negatives(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_fresh_world_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        m.run(c,'cpu');out=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        # Reuse the established source-only truth fixture, then create fresh private future truth.
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'source_DURA',role)),_target_rows=staticmethod(m._target_rows)))
        sp=type('Scoring',(),dict(common=s.common,CONDITIONS=()))
        source=truth_files(tmp_path,sp,proxy,out)[0];paths=[source]
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);day=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for name in m.CONDITIONS[1:]:
            a=m._target_rows(out,name);p=tmp_path/(name+'_truth.npz');paths.append(p)
            data=dict(ids=a['ids'],y=y,tx=np.asarray([s.TX_BY_LABEL[i] for i in y]),receiver=np.full(18,34,np.int64),day=day,scenario=np.asarray(['clean' if name=='clean_DURA' else 'practical_low_urban']*18))
            np.savez(p,**{k:v[::-1] for k,v in data.items()})
        reads=[];sr=s._source_truth;tr=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),sr(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda path,*a:(reads.append(Path(path).name),tr(path,*a))[1])
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('Independent score must not load new model'))
        monkeypatch.setattr(m,'_load_SOURCE',lambda *a:pytest.fail('Independent score must not load SOURCE model'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('Independent score must not load F0/G'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['truth_last']
        assert not result['eligible_primary'] and not result['final_phase3_complete'] and result['full_goal_status']=='ACTIVE'
        from experiments.pamf_phase3_open_world import fresh_world_launch as launch
        actual_read=launch._read
        with monkeypatch.context() as terminal_patch:
            terminal_patch.setattr(launch,'_read',lambda path:m._read(out/'complete.json') if Path(path).parent.name=='predict' else actual_read(tmp_path/'score/complete.json'))
            runtime=dict(run_root=str(tmp_path/'terminal'),code_commit=c['code_commit'])
            assert launch._terminal(runtime,'predict')['status']=='FRESH_WORLD_PREDICTIONS_COMPLETE'
            assert launch._terminal(runtime,'score')['status']=='FRESH_WORLD_SCORES_COMPLETE'
        p=out/'targets/star_DURA/predictions.npz';saved=p.read_bytes()
        for defect in ('partial','private','new_score','old_SOURCE_axis','baseline'):
            with subtests.test(defect=defect):
                p.write_bytes(saved);reads.clear()
                if defect=='partial':p.unlink()
                else:
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='private':a['y']=y
                    elif defect=='new_score':a['raw_scores'][0,5]+=.1
                    elif defect=='old_SOURCE_axis':a['signal_features'][0,1,0]+=.1
                    else:a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/('bad_'+defect))
                assert reads==[]
