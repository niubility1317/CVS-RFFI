"""One independently initialized known-only adapter with exact shared controls."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_selective_uda_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_diagonal_uda_run import fixture as old_fixture,write
    with old_fixture(root,mp) as (old,oc,pm,_,U,backend,events):
        old_done=old.run(oc,'cpu');m=api();oc['code_commit']=m.ADAPTATION_CODE;old_done['code_commit']=m.ADAPTATION_CODE;old_done['config']=deepcopy(oc)
        cache=Path(oc['output_root']);write(cache/'complete.json',old_done);write(cache/'resolved_config.json',oc)
        for directory in ('models','adaptation'):
            value=old._read(cache/directory/'complete.json');value['code_commit']=m.ADAPTATION_CODE;write(cache/directory/'complete.json',value)
        mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',old.Q_COUNT);events.clear()
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=oc['reference_config'],reference_adaptation_root=str(cache),U_capsule_id=m.U_CAPSULE_ID,
            U_split_id=m.U_SPLIT_ID,output_root=str(root/'selective/predict'),batch_size=128)
        readU=m._U_features;rows=m._parent_rows;query=m._parent_target_rows;shared_rows=m._shared_target_rows
        def features(config,header):
            if not Path(c['output_root'],'complete.json').exists():
                assert events==['checkpoint'] and Path(c['output_root'],'models/complete.json').is_file();events.append('U')
            return readU(config,header)
        def source(path,role):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'adaptation/complete.json').is_file();events.append(role)
            return rows(path,role)
        def target(path,condition):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'novel_predictions.npz').is_file();events.append(condition)
            return query(path,condition)
        def control(path,condition):
            assert Path(c['output_root'],'adaptation/complete.json').is_file();return shared_rows(path,condition)
        mp.setattr(m,'_U_features',features);mp.setattr(m,'_parent_rows',source);mp.setattr(m,'_parent_target_rows',target);mp.setattr(m,'_shared_target_rows',control)
        mp.setattr(m.shared_core,'fit_adapters',lambda *a:pytest.fail('Never refit or initialize from SHARED'))
        yield m,c,pm,old_done,U,backend,events


def test_private_inputs_overlap_and_checkpoint_failure_precede_U(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'class'),reference_prediction_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,reference_adaptation_root=str(tmp_path/'diagonal'),U_capsule_id=m.U_CAPSULE_ID,
        U_split_id=m.U_SPLIT_ID,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for delta in (dict(U_truth='private'),dict(true_K=3),dict(U_capsule_id='clean'),dict(output_root=c['reference_adaptation_root']),dict(batch_size=True)):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
    monkeypatch.setattr(m,'validate_config',lambda c:c);monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No checkpoint load for an existing output'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('checkpoint smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No source reference before checkpoint smoke'))
    monkeypatch.setattr(m,'_shared',lambda *a:pytest.fail('No shared cache before checkpoint smoke'))
    with pytest.raises(ValueError,match='smoke failure'):m.run(dict(c,output_root=str(tmp_path/'failed')),'cpu')
    assert not (tmp_path/'failed').exists()


def test_actual_producer_exact_shared_then_CPU_three_truths_and_protocol_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_score import truth_files
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_selective_uda_score')
    with fixture(tmp_path,monkeypatch) as (m,c,pm,old,U,backend,events):
        out=Path(c['output_root']);cache=Path(c['reference_adaptation_root']);fit=m.core.fit_adapter;inputs=[]
        def adapter(state,z):inputs.append(z.copy());return fit(state,z)
        monkeypatch.setattr(m.core,'fit_adapter',adapter)
        result=m.run(c,'cpu');assert events==['checkpoint','U','known','novel','clean_DURA','star_DURA']
        assert len(inputs)==1;np.testing.assert_array_equal(inputs[0],U['signal_features'])
        diag=m._read(out/'adaptation/fit_diagnostics.json');assert diag['trace'][0]['scale_min']==diag['trace'][0]['scale_max']==1 and diag['trace'][0]['offset_norm']==0
        assert result['known_only_transform'] and result['selective_background_input_untransformed']
        assert result['new_target_statistical_iterations']==result['inherited_shared_iterations']==20
        assert result['real_signal_forward_rows']==result['real_F_forward_rows']==result['G_forward_rows']==0
        assert result['source_only'] is False and result['original_neural_models_source_only']
        with np.load(out/'adaptation/SHARED.npz') as a,np.load(cache/'adaptation/DIAG.npz') as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        for condition in m.CONDITIONS:
            a=m._target_rows(out,condition);oldq=m.shared_evaluation._target_rows(cache,condition)
            for key,value in oldq.items():np.testing.assert_array_equal(a[key][:,:4] if key in ('scores','raw_scores') else a[key],value[:,[0,1,4,5]] if key in ('scores','raw_scores') else value)
        monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1]);monkeypatch.setattr(s,'_target_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('CPU scorer must not load checkpoints'));monkeypatch.setattr(m.core,'fit_adapter',lambda *a:pytest.fail('CPU scorer must not fit'))
        scored=s.score(out,*paths,tmp_path/'score');assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz']
        assert scored['truth_last'] and scored['target_DA'] and not scored['final_phase3_complete']
        files=[out/'complete.json',out/'adaptation/SHARED.npz',out/'adaptation/SELECTIVE.npz',out/'adaptation/fit_diagnostics.json',out/'thresholds.npz',out/'targets/star_DURA/predictions.npz',cache/'complete.json',cache/'adaptation/U_features.npz']
        saved={path:path.read_bytes() for path in files}
        for defect in ('partial','oldcode','shared','selective','trace','control','private_Q','tau','hidden_U'):
            with subtests.test(defect=defect):
                for path,value in saved.items():path.write_bytes(value)
                reads.clear()
                if defect=='partial':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('oldcode','trace'):
                    path=cache/'complete.json' if defect=='oldcode' else out/'adaptation/fit_diagnostics.json';a=m._read(path)
                    if defect=='oldcode':a['code_commit']='d'*40
                    else:a['trace'].pop()
                    path.unlink();m._write(path,a)
                else:
                    path=out/'adaptation/SHARED.npz' if defect=='shared' else out/'adaptation/SELECTIVE.npz' if defect=='selective' else out/'thresholds.npz' if defect=='tau' else cache/'adaptation/U_features.npz' if defect=='hidden_U' else out/'targets/star_DURA/predictions.npz'
                    with np.load(path,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect in ('shared','selective'):a['inverse_scale'][0]=-1
                    elif defect=='tau':a['conditional'][5,0]+=.1
                    elif defect=='control':a['raw_scores'][0,2]+=.1
                    else:a['y']=np.zeros(384 if defect=='hidden_U' else m.Q_COUNT,np.int64)
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'bad_score')
                assert not reads and not (tmp_path/'bad_score').exists()
        for path,value in saved.items():path.write_bytes(value)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:backend);monkeypatch.setattr(m.parent,'_smoke',lambda *a:dict(status='PASS',actual_F0_signal_DURA_checkpoints=True))
        monkeypatch.setattr(m.core,'fit_adapter',lambda *a:(_ for _ in ()).throw(ValueError('selective fit failure')));monkeypatch.setattr(m,'_U_features',lambda *a:U)
        monkeypatch.setattr(m,'_parent_rows',lambda *a:pytest.fail('No source rows after failed U fit'));monkeypatch.setattr(m,'_parent_target_rows',lambda *a:pytest.fail('No Q rows after failed U fit'))
        with pytest.raises(ValueError,match='fit failure'):m.run(dict(c,output_root=str(tmp_path/'failed_fit')),'cpu')
        assert not (tmp_path/'failed_fit/complete.json').exists()
