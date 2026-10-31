"""Eight readouts must validate completely before any private truth."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_common_transport_score')


def test_fixed_primary_must_beat_PCD_and_both_older_controls():
    s=api();assert s.PRIMARY=='CCD_predicted_ratio' and s.PRIMARY_INDEX==7
    def metric(ak,ru):return dict(A_K=ak,R_U=ru)
    main=metric(.91,.95);controls=[metric(.90,.94),metric(.90,.95),metric(.91,.95)]
    assert not s._control_gate(main,controls)[0]
    controls[-1]=metric(.90,.951);assert not all(s._control_gate(main,controls))
    controls[-1]=metric(.90,.95);assert all(s._control_gate(main,controls))


def test_actual_predictions_reconstruction_truth_last_and_negative_boundaries(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_common_transport_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        m.run(c,'cpu');out=Path(c['output_root']);monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        proxy=type('Rows',(),dict(_rows=staticmethod(lambda root,role:m._source_rows(root,'source_DURA',role)),_target_rows=staticmethod(m._target_rows)))
        sp=type('Scoring',(),dict(common=s.common,CONDITIONS=s.CONDITIONS[1:]));paths=truth_files(tmp_path,sp,proxy,out)
        reads=[];source=s._source_truth;target=s._target_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1]);monkeypatch.setattr(s,'_target_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m,'_load_model',lambda *a:pytest.fail('CPU scorer must not load new checkpoint'))
        monkeypatch.setattr(m,'_backend',lambda *a:pytest.fail('CPU scorer must not load F0/G/r03'))
        monkeypatch.setattr(m.density,'fit_known',lambda *a:pytest.fail('CPU scorer must not fit stats'))
        monkeypatch.setattr(m.density,'fit_background',lambda *a:pytest.fail('CPU scorer must not fit stats'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and result['truth_last']
        assert result['primary_method']=='CCD_predicted_ratio' and len(result['methods'])==8 and not result['eligible_primary']
        assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        files=[out/'complete.json',out/'thresholds.npz',out/'targets/star_DURA/predictions.npz',out/'models/TARGET_CCD.npz'];saved={p:p.read_bytes() for p in files}
        for defect in ('partial_Q','identity','old_control','new_readout','old_feature','baseline','tau','density','private_Q'):
            with subtests.test(defect=defect):
                for p,data in saved.items():p.write_bytes(data)
                reads.clear()
                if defect=='partial_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect=='identity':
                    p=out/'complete.json';a=m._read(p);a['model_identities']['TARGET_CCD']['updates']=443;p.unlink();m._write(p,a)
                else:
                    p=out/'thresholds.npz' if defect=='tau' else out/'models/TARGET_CCD.npz' if defect=='density' else out/'targets/star_DURA/predictions.npz'
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='tau':a['conditional'][7,0]+=.1
                    elif defect=='density':a['whitener']*=1.2
                    elif defect=='new_readout':a['raw_scores'][0,7]+=.1
                    elif defect=='old_control':a['raw_scores'][0,5]+=.1
                    elif defect=='old_feature':a['signal_features'][0,2,0]+=.1
                    elif defect=='baseline':a['baseline_closed'][0]=(a['closed'][0]+1)%6
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                bad=tmp_path/('bad_'+defect)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,bad)
                assert reads==[] and not bad.exists()
