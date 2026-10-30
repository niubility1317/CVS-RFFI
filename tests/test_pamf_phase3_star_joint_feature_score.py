"""Joint u/z evidence preserves the old controls and three truth-last joins."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_feature_score')


def rows():
    n=3;u=np.eye(160,dtype=np.float32)[:n];z=np.roll(u,1,axis=1)
    raw=np.arange(n*4,dtype=np.float64).reshape(n,4)
    return dict(ids=np.asarray(['a','b','c']),closed=np.asarray([0,1,2],np.int64),signal_features=z,
        raw_scores=raw,scores=raw.copy()),u


def test_target_only_adds_original_unit_features_and_never_changes_old_controls():
    s=api();parent,u=rows();a={k:v.copy() for k,v in parent.items()}
    a['unit_features']=u;a['raw_scores'][:,:2]=parent['raw_scores'][:,[1,3]];a['scores'][:,:2]=parent['scores'][:,[1,3]]
    s._exact_controls(a,parent,allow_unit_features=True)
    with pytest.raises(ValueError):s._exact_controls(a,parent)
    a['private_y']=np.zeros(3,np.int64)
    with pytest.raises(ValueError):s._exact_controls(a,parent,allow_unit_features=True)
    del a['private_y'];a['scores'][0,0]+=.01
    with pytest.raises(ValueError):s._exact_controls(a,parent,allow_unit_features=True)


def test_joint_reconstruction_uses_both_feature_axes_and_actual_class_routed_margin():
    s=api();a,u=rows();a['unit_features']=u;tau=np.arange(24,dtype=np.float64).reshape(4,6)/10
    a['scores']=a['raw_scores']-tau[:,a['closed']].T;seen=[]
    class Core:
        @staticmethod
        def raw_scores(original,joint,original_u,signal_z,closed):
            assert original=='original160' and joint=='joint320'
            assert original_u is a['unit_features'] and signal_z is a['signal_features']
            seen.append(True);return np.arange(12,dtype=np.float64).reshape(3,4)
    s._reconstruct(a,'original160','joint320',tau,Core)
    assert seen==[True]
    a['raw_scores'][0,3]+=.1
    with pytest.raises(ValueError):s._reconstruct(a,'original160','joint320',tau,Core)
    a['raw_scores'][0,3]-=.1;a['scores'][0,3]+=.1
    with pytest.raises(ValueError):s._reconstruct(a,'original160','joint320',tau,Core)
    a['scores']=a['raw_scores']-tau[:,a['closed']].T;a['unit_features'][0]=0
    with pytest.raises(ValueError):s._reconstruct(a,'original160','joint320',tau,Core)


def test_actual_joint_producer_to_CPU_score_and_truth_last_corruption_boundaries(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_joint_feature_run import fixture
    from test_pamf_phase3_star_class_density_score import truth_files
    s=api()
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,public,backend,seen):
        m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['L','OE','fit','known','novel','clean_DURA','star_DURA']
        monkeypatch.setattr(s,'COUNTS',counts);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source_truth=s._source_truth;target_truth=s._target_truth
        def source(*a):reads.append('source');return source_truth(*a)
        def target(path,*a):reads.append(Path(path).name);return target_truth(path,*a)
        monkeypatch.setattr(s,'_source_truth',source);monkeypatch.setattr(s,'_target_truth',target)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No NN checkpoint loading in CPU scorer'))
        monkeypatch.setattr(m.core,'fit',lambda *a:pytest.fail('No joint density refit in scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No V recalibration or novel threshold selection'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz']
        assert result['truth_last'] and result['all_three_predictions_before_truth']
        assert result['costs']['real_signal_forward_rows']==counts['L']+counts['OE']
        assert result['costs']['real_F_forward_rows']==result['costs']['G_forward_rows']==2*m.Q_COUNT
        assert result['costs']['Q_signal_new_forward_rows']==result['costs']['source_F_new_forward_rows']==0
        assert result['source_V_threshold_sets_fitted']==2 and result['density_fit_updates']==result['joint_density_fit_updates']==1
        assert result['primary_method']=='joint_predicted_ratio' and not result['development_gate_passed']
        assert result['full_goal_status']=='ACTIVE' and not result['final_phase3_complete']
        assert set(result['paired'])=={'primary_vs_predicted_ratio','primary_vs_pooled_ratio','primary_vs_joint_pooled_ratio'}
        for method in s.METHODS:assert len(result['methods'][method]['source']['groups'])==42
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'thresholds.npz',out/'calibration/complete.json',out/'models/density.npz',out/'models/joint.npz',
            out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('missing_Q','not_frozen','old_tau','old_calibration','original_density','joint_density','old_control','joint_raw','invalid_tau','new_u','private_Q'):
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
                    p=(out/'thresholds.npz' if defect in ('old_tau','invalid_tau') else out/'models/density.npz' if defect=='original_density'
                        else out/'models/joint.npz' if defect=='joint_density' else out/'novel_predictions.npz' if defect=='old_control'
                        else out/'targets/clean_DURA/predictions.npz')
                    with np.load(p,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect=='old_tau':a['conditional'][0,0]+=.1
                    elif defect=='invalid_tau':a['conditional'][3,0]=np.nan
                    elif defect=='original_density':a['known_means'][0,0]+=.1
                    elif defect=='joint_density':a['origin'][0]+=.1
                    elif defect=='old_control':a['scores'][0,0]+=.1
                    elif defect=='joint_raw':a['raw_scores'][0,3]+=.1
                    elif defect=='new_u':a['unit_features'][0]=0.
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):
                    s.score(out,*paths,tmp_path/'invalid_score')
                assert not reads and not (tmp_path/'invalid_score').exists()
