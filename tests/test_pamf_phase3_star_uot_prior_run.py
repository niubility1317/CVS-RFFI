"""Synthetic cached-U adaptation and independent three-truth integration."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_uot_prior_run')


@contextmanager
def fixture(root,mp):
    import test_pamf_phase3_star_diagonal_uda_run as inherited
    m=api();mp.setattr(inherited,'api',lambda:m)
    if not hasattr(m.core,'density'):mp.setattr(m.core,'density',m.parent.core.density,raising=False)
    with inherited.fixture(root,mp) as bundle:
        mp.setattr(m.diagonal,'COUNTS',m.COUNTS)
        yield bundle


def test_exact_config_and_actual_checkpoint_smoke_precede_cache(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'class'),reference_prediction_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,reference_adaptation_root=str(tmp_path/'mixture'),U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for update in (dict(U_truth='forbidden'),dict(true_K=3),dict(Q_features='forbidden'),dict(U_capsule_id='clean'),dict(output_root=c['reference_adaptation_root']),dict(batch_size=True)):
        with subtests.test(update=update),pytest.raises(ValueError):m.validate_config(dict(c,**update))
    monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('Existing output must not load models'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('checkpoint smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No source/U reads before smoke'))
    with pytest.raises(ValueError,match='smoke failure'):m.run(c,'cpu')
    assert not Path(c['output_root']).exists()


def test_one_real_core_fit_frozen_source_and_all_predictions_before_truth(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_score import truth_files
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_uot_prior_score')
    with fixture(tmp_path,monkeypatch) as (m,c,pm,header,U,backend,events):
        out=Path(c['output_root']);cache=Path(c['reference_adaptation_root']);original=m.core.fit_adapter;inputs=[]
        def fit(state,z):inputs.append(z.copy());return original(state,z)
        monkeypatch.setattr(m.core,'fit_adapter',fit)
        # A scientific source-calibration failure still completes both future conditions.
        ref=m._reference
        def failed_source(config):
            p,state,tau,cal=ref(config);cal=deepcopy(cal)
            for report in cal['methods'].values():report['constraints_passed']=False
            return p,state,tau,cal
        monkeypatch.setattr(m,'_reference',failed_source)
        result=m.run(c,'cpu')
        assert events==['checkpoint','U','known','novel','clean_DURA','star_DURA'] and len(inputs)==1
        np.testing.assert_array_equal(inputs[0],U['signal_features'])
        assert result['target_U_reads']==384 and result['new_target_statistical_iterations']==200
        assert result['new_target_statistical_parameters']==47 and result['new_target_statistical_degrees_of_freedom']==45
        assert result['source_only'] is False and result['target_DA'] and not result['target_REG']
        assert result['real_F_forward_rows']==result['real_signal_forward_rows']==result['G_forward_rows']==0
        assert result['new_neural_updates']==result['density_fit_updates']==result['threshold_fit_updates']==0
        assert m._read(out/'calibration/complete.json')==failed_source(c)[3]
        for role in ('known','novel'):
            a=m._rows(out,role)
            for key in ('raw_scores','scores'):np.testing.assert_array_equal(a[key][:,:2],a[key][:,2:])
        with np.load(out/'adaptation/UOT.npz',allow_pickle=False) as a:assert set(a.files)=={'schema','domain_weights','background_weights'}
        monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_target_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('CPU score cannot load checkpoints'))
        monkeypatch.setattr(m.core,'fit_adapter',lambda *a:pytest.fail('CPU score must not refit 200 Sinkhorn iterations'))
        scored=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and scored['truth_last']
        assert not scored['source_known_12_passed'] and not scored['eligible_primary']
        assert scored['full_goal_status']=='ACTIVE' and not scored['final_phase3_complete']
        assert scored['primary_method']=='UOT_predicted_ratio' and len(scored['methods'])==4
        files=[out/'complete.json',out/'adaptation/UOT.npz',out/'adaptation/U_features.npz',out/'adaptation/fit_diagnostics.json',out/'thresholds.npz',out/'targets/star_DURA/predictions.npz',cache/'complete.json']
        saved={p:p.read_bytes() for p in files}
        for defect in ('partial_Q','old_U_code','identity','diag_budget','negative_weight','control','private_Q','tau','changed_U','private_U'):
            with subtests.test(defect=defect):
                for p,value in saved.items():p.write_bytes(value)
                reads.clear()
                if defect=='partial_Q':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('old_U_code','identity','diag_budget'):
                    p=cache/'complete.json' if defect=='old_U_code' else out/'complete.json' if defect=='identity' else out/'adaptation/fit_diagnostics.json';a=m._read(p)
                    if defect=='old_U_code':a['code_commit']='d'*40
                    elif defect=='identity':a['model_identity']['updates']+=1
                    else:a['iterations']=199
                    p.unlink();m._write(p,a)
                else:
                    p=out/'adaptation/UOT.npz' if defect=='negative_weight' else out/'thresholds.npz' if defect=='tau' else out/'adaptation/U_features.npz' if defect in ('changed_U','private_U') else out/'targets/star_DURA/predictions.npz'
                    with np.load(p,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='negative_weight':a['domain_weights'][0]=-1
                    elif defect=='tau':a['conditional'][3,0]+=.1
                    elif defect=='changed_U':a['signal_features'][0,0]+=.1
                    elif defect=='private_U':a['true_y']=np.zeros(384,np.int64)
                    elif defect=='control':a['raw_scores'][0,1]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(p,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'bad_score')
                assert reads==[] and not (tmp_path/'bad_score').exists()
        for p,value in saved.items():p.write_bytes(value)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_failed_U_fit_cannot_open_source_or_future_rows(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,*_):
        monkeypatch.setattr(m.core,'fit_adapter',lambda *a:(_ for _ in ()).throw(ValueError('UOT failure')))
        monkeypatch.setattr(m,'_parent_rows',lambda *a:pytest.fail('No source rows after failed U fit'))
        monkeypatch.setattr(m,'_parent_target_rows',lambda *a:pytest.fail('No Q after failed U fit'))
        with pytest.raises(ValueError,match='UOT failure'):m.run(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists()
