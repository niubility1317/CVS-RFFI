"""All six decisions precede three truths; MD cannot replace the RMD primary."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_relative_metric_score')


def test_primary_requires_gain_and_rejection_against_the_fixed_old_control():
    m=api()
    assert m._control_gate(dict(A_K=.7,R_U=.8),dict(A_K=.6,R_U=.8))==(True,True)
    assert m._control_gate(dict(A_K=.6,R_U=.9),dict(A_K=.6,R_U=.8))==(False,True)
    assert m._control_gate(dict(A_K=.8,R_U=.7),dict(A_K=.6,R_U=.8))==(True,False)


def test_incomplete_artifacts_stop_before_any_truth(tmp_path,monkeypatch):
    m=api();reads=[]
    monkeypatch.setattr(m,'_prediction',lambda *a:(_ for _ in ()).throw(ValueError('not all predictions')))
    monkeypatch.setattr(m,'_source_truth',lambda *a:reads.append(1));monkeypatch.setattr(m,'_future_truth',lambda *a:reads.append(1))
    with pytest.raises(ValueError):m.score('pred','source','clean','star',tmp_path/'score')
    assert reads==[] and not (tmp_path/'score').exists()


def test_actual_metric_artifacts_then_three_truths_and_corruption_gate(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_pcd_relative_metric_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,events):
        done=m.run(c,'cpu');root=Path(c['output_root'])
        assert events==['smoke','L','source_V','source_NV','Q_clean_DURA','Q_star_DURA']
        assert done['source_metric_fits']==1 and done['new_threshold_sets_fitted']==4 and done['models_frozen_before_V']
        assert all(done[k]==0 for k in ('real_F_forward_rows','real_signal_forward_rows','G_forward_rows','optimizer_updates','target_U_reads'))
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._rows(root,role)),_target_rows=staticmethod(m._target_rows)))
        sp=type('Score',(),dict(common=s.common,CONDITIONS=()))
        paths=[truth_files(tmp_path,sp,proxy,root)[0]];y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64)
        for cond in m.CONDITIONS[1:]:
            row=m._target_rows(root,cond);path=tmp_path/(cond+'_truth.npz');paths.append(path)
            np.savez(path,ids=row['ids'],y=y,tx=np.asarray(s.fresh.TX_BY_LABEL)[y],receiver=np.full(len(y),34,np.int64),
                day=np.asarray(['2021_03_15']*10+['2021_03_23']*8),scenario=np.asarray(['clean' if cond=='clean_DURA' else 'practical_low_urban']*len(y)))
        reads=[];source=s._source_truth;future=s._future_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1]);monkeypatch.setattr(s,'_future_truth',lambda path,*a:(reads.append(Path(path).name),future(path,*a))[1])
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('No NN in CPU score'));monkeypatch.setattr(m.core,'fit_metric',lambda *a:pytest.fail('No refit in score'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No recalibration in score'))
        result=s.score(root,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['adaptation_success'] is None
        assert result['primary_method']==m.METHODS[-1] and result['full_goal_status']=='ACTIVE' and not result['eligible_primary']
        from experiments.pamf_phase3_open_world import pcd_relative_metric_launch as launcher
        with monkeypatch.context() as terminal:
            terminal.setattr(launcher,'SOURCE_COUNT',m.SOURCE_COUNT)
            terminal.setattr(launcher,'_read',lambda path:m._read(root/'complete.json') if Path(path).parent.name=='predict' else m._read(tmp_path/'score/complete.json'))
            runtime=dict(run_root=str(tmp_path/'terminal'),code_commit=c['code_commit'])
            assert launcher._terminal(runtime,'predict')==done
            assert launcher._terminal(runtime,'score')['status']==s.STATUS
        path=root/'predictions/Q_star_DURA.npz';saved=path.read_bytes()
        for defect in ('missing','old_control','new_metric_raw','default_head','private'):
            with subtests.test(defect=defect):
                path.write_bytes(saved);reads.clear()
                if defect=='missing':path.unlink()
                else:
                    row=m._npz(path)
                    if defect=='old_control':row['scores'][0,1]=np.nextafter(row['scores'][0,1],np.inf)
                    elif defect=='new_metric_raw':row['raw_scores'][0,5]+=.01
                    elif defect=='default_head':row['closed'][0]=(row['closed'][0]+1)%6
                    else:row['y']=y
                    np.savez(path,**row)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(root,*paths,tmp_path/('bad_'+defect))
                assert reads==[]
