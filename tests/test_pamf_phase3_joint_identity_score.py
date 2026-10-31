"""All model-specific predictions reconstruct before any private truth read."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.joint_identity_score')


def test_missing_date_known_roster_stays_NA():
    s=api();y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);n=len(y)
    a=dict(closed=np.minimum(y,5),baseline_closed=np.minimum(y,5),scores=np.where(y[:,None]<6,1.,-1.)*np.ones((n,2)))
    truth=dict(y=y,receiver=np.full(n,34,np.int64),day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean']*n))
    row=s.fresh._target(a,truth,'clean_DURA',1)['by_day']['2021_03_23']
    assert row['A_K'] is row['C_K'] is row['H_score'] is None and row['R_U']==1.
    a['closed']=(a['closed']+1)%6
    main=s.fresh._target(a,truth,'clean_DURA',1)['overall']
    assert main['original_F0_macro_closed_accuracy']==1. and main['candidate_macro_closed_accuracy']==0.
    assert main['original_closed_minus_new_open']==1. and main['candidate_closed_minus_open']==0.


def test_complete_own_class_routing_before_three_truths(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_joint_identity_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        m.run(c,'cpu');out=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'E0',role)),_target_rows=staticmethod(lambda root,name:m._target_rows(root,name,'E0'))))
        source=truth_files(tmp_path,type('Score',(),dict(common=s.common,CONDITIONS=())),proxy,out)[0];paths=[source]
        y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64);day=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition,'E0');p=tmp_path/(condition+'_truth.npz');paths.append(p)
            np.savez(p,ids=a['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),day=day,scenario=np.asarray(['clean' if condition=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];source_read=s._source_truth;future_read=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source_read(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda *a:(reads.append('future'),future_read(*a))[1])
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('Scorer cannot load neural state'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','future','future'] and result['truth_last'] and result['full_goal_status']=='ACTIVE'
        assert result['primary_method']=='TARGET_JOINT_predicted_ratio' and len(result['methods'])==8
        from experiments.pamf_phase3_open_world import joint_identity_launch as launch
        read=launch._read
        with monkeypatch.context() as handoff:
            handoff.setattr(launch,'_read',lambda path:m._read(out/'complete.json') if Path(path).parent.name=='predict' else read(tmp_path/'score/complete.json'))
            runtime=dict(run_root=str(tmp_path/'terminal'),code_commit=c['code_commit'])
            assert launch._terminal(runtime,'predict')['status']=='JOINT_IDENTITY_PREDICTIONS_COMPLETE'
            assert launch._terminal(runtime,'score')['status']=='JOINT_IDENTITY_SCORES_COMPLETE'
        p=out/'targets/star_DURA/TARGET_JOINT/predictions.npz';saved=p.read_bytes()
        for defect in ('partial','F0_routing','own_head','baseline','old_control'):
            with subtests.test(defect=defect):
                p.write_bytes(saved);reads.clear();changed=p
                if defect=='partial':p.unlink()
                else:
                    if defect=='old_control':changed=out/'targets/star_DURA/OLD_PCD/predictions.npz'
                    original=changed.read_bytes()
                    with np.load(changed,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='F0_routing':
                        tau=np.load(out/'thresholds.npz')['conditional'][6:8];a['scores']=a['raw_scores']-tau[:,a['baseline_closed']].T
                        # Ensure corruption even if a fixture class has an identical threshold.
                        a['scores'][0,1]+=.1
                    elif defect=='own_head':a['closed'][0]=(a['closed'][0]+1)%6
                    elif defect=='baseline':a['baseline_closed'][0]=(a['baseline_closed'][0]+1)%6
                    else:a['scores'][0,1]+=.1
                    np.savez(changed,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/('bad_'+defect))
                assert reads==[]
                if defect=='old_control':changed.write_bytes(original)
        p.write_bytes(saved)
