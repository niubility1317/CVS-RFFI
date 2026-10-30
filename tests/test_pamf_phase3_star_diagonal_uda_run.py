"""One actual synthetic checkpoint -> cached U EM -> three-truth CPU score."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_diagonal_uda_run')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (parent,pc,_,backend,_,_,_):
        m=api();pc['code_commit']=m.PARENT_CODE;pm=parent.run(pc,'cpu');mp.setattr(m,'COUNTS',dict(parent.COUNTS));mp.setattr(m,'Q_COUNT',parent.Q_COUNT)
        cache=root/'mixture/predict';old=m.mix
        U_manifest=dict(capsule_id=m.U_CAPSULE_ID,split_id=m.U_SPLIT_ID,count=384,registered_classes=parent.signal_train.native.CLASSES)
        mc=dict(run_id=old.RUN_ID,code_commit=m.ADAPTATION_CODE,reference_prediction_root=str(root/'best/predict'),output_root=str(cache),
            U_public_root=str(root/'G10/discovery/public'),U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID,batch_size=128)
        best=dict(run_id=old.best.RUN_ID,code_commit=old.REFERENCE_CODE,config=dict(train_config=pc['train_config']),model_identity=pm['model_identity'],
            parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_only=True,target_U_reads=0,private_truth_reads=0)
        header=dict(schema=old.SCHEMA,status=old.STATUS,run_id=old.RUN_ID,code_commit=m.ADAPTATION_CODE,config=mc,method_names=list(old.METHODS),primary_method=old.PRIMARY_METHOD,
            known_count=m.COUNTS['V'],novel_count=m.COUNTS['novel'],U_count=384,U_manifest=U_manifest,reference_prediction_identity=best,model_identity=pm['model_identity'],
            parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],source_models_frozen_before_U=True,adapters_frozen_before_source_and_Q=True,
            all_three_DURA_conditions_complete=True,source_states_unchanged=True,frozen_target_adapters_unchanged=True,frozen_parameters_buffers_unchanged=True,
            classification_unchanged=True,new_neural_updates=0,native_updates=0,density_fit_updates=0,threshold_fit_updates=0,private_truth_reads=0,U_truth_reads=0,
            source_novel_truth_reads=0,target_truth_reads=0,formal_target_accessed=False,final_phase3_complete=False)
        write(cache/'complete.json',header)
        rng=np.random.default_rng(110);z=rng.normal(size=(384,160));z/=np.linalg.norm(z,axis=1,keepdims=True)
        logits=rng.normal(size=(384,6)).astype(np.float32)
        U=dict(ids=np.asarray(['U-'+str(i).zfill(4) for i in range(384)]),default_logits=logits,closed=logits.argmax(1).astype(np.int64),signal_features=z.astype(np.float32))
        (cache/'adaptation').mkdir();np.savez(cache/'adaptation/U_features.npz',**U)
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=pc,reference_adaptation_root=str(cache),U_capsule_id=m.U_CAPSULE_ID,
            U_split_id=m.U_SPLIT_ID,output_root=str(root/'diagonal/predict'),batch_size=128)
        events=[];load=m.parent._backend;readU=m._U_features;rows=m._parent_rows;target=m._parent_target_rows
        def checkpoint(*args):events.append('checkpoint');return load(*args)
        def features(config,header):
            if not Path(c['output_root'],'complete.json').exists():
                assert events==['checkpoint'] and Path(c['output_root'],'models/complete.json').is_file()
                assert Path(c['output_root'],'thresholds.npz').is_file();events.append('U')
            return readU(config,header)
        def source_rows(path,role):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'adaptation/complete.json').is_file();events.append(role)
            return rows(path,role)
        def query(path,name):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'novel_predictions.npz').is_file();events.append(name)
            return target(path,name)
        mp.setattr(m.parent,'_backend',checkpoint);mp.setattr(m,'_U_features',features);mp.setattr(m,'_parent_rows',source_rows);mp.setattr(m,'_parent_target_rows',query)
        mp.setattr(m.parent,'_public',lambda *a:pytest.fail('No source/U/Q IQ or new real neural forward'))
        mp.setattr(m.core.density,'fit_known',lambda *a:pytest.fail('No source Gaussian refit'));mp.setattr(m.core.density,'fit_background',lambda *a:pytest.fail('No OE refit'))
        mp.setattr(m.parent.risk,'calibrate_known',lambda *a:pytest.fail('No new known-V threshold fit'))
        yield m,c,pm,header,U,backend,events


def test_exact_config_output_and_smoke_precede_U_or_any_rows(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'class'),reference_prediction_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,reference_adaptation_root=str(tmp_path/'mixture'),U_capsule_id=m.U_CAPSULE_ID,
        U_split_id=m.U_SPLIT_ID,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for update in (dict(U_truth='forbidden'),dict(true_K=3),dict(U_capsule_id='clean'),dict(output_root=c['reference_adaptation_root']),dict(batch_size=True)):
        with subtests.test(update=update),pytest.raises(ValueError):m.validate_config(dict(c,**update))
    monkeypatch.setattr(m,'validate_config',lambda c:c);monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('Existing output must not reload'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('synthetic checkpoint smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No source reference before actual checkpoint smoke'))
    monkeypatch.setattr(m,'_cached_U_header',lambda *a:pytest.fail('No old U header before actual checkpoint smoke'))
    with pytest.raises(ValueError,match='smoke failure'):m.run(dict(c,output_root=str(tmp_path/'failed')),'cpu')
    assert not (tmp_path/'failed').exists()


def test_actual_cached_U_two_EMs_then_three_truth_score_and_boundary_failures(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_score import truth_files
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_diagonal_uda_score')
    with fixture(tmp_path,monkeypatch) as (m,c,pm,header,U,backend,events):
        out=Path(c['output_root']);cache=Path(c['reference_adaptation_root']);original=m.core.fit_adapters;fit_inputs=[]
        def fit(state,z):fit_inputs.append(z.copy());return original(state,z)
        monkeypatch.setattr(m.core,'fit_adapters',fit)
        result=m.run(c,'cpu')
        assert events==['checkpoint','U','known','novel','clean_DURA','star_DURA'] and len(fit_inputs)==1
        np.testing.assert_array_equal(fit_inputs[0],U['signal_features'])
        assert result['target_U_reads']==384 and result['U_features_reused'] and result['source_only'] is False
        assert result['real_signal_forward_rows']==result['real_F_forward_rows']==result['G_forward_rows']==0
        assert result['new_neural_updates']==result['density_fit_updates']==result['threshold_fit_updates']==0
        assert m._read(out/'calibration/complete.json')==m._reference(c)[3]
        monkeypatch.setattr(s,'COUNTS',m.COUNTS);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        monkeypatch.setattr(s,'_source_truth',lambda *a:(reads.append('source'),source(*a))[1])
        monkeypatch.setattr(s,'_target_truth',lambda path,*a:(reads.append(Path(path).name),target(path,*a))[1])
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('CPU scorer must not load checkpoints'))
        monkeypatch.setattr(m.core,'fit_adapters',lambda *a:pytest.fail('CPU scorer must not refit'))
        scored=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and scored['truth_last']
        assert scored['target_DA'] and not scored['source_only'] and not scored['final_phase3_complete']
        files=[out/'complete.json',out/'adaptation/DIAG.npz',out/'adaptation/U_features.npz',out/'adaptation/fit_diagnostics.json',out/'thresholds.npz',out/'targets/star_DURA/predictions.npz',cache/'complete.json']
        saved={path:path.read_bytes() for path in files}
        for defect in ('partial','oldcode','identity','trace','negative_scale','control','private_Q','tau','changed_U'):
            with subtests.test(defect=defect):
                for path,value in saved.items():path.write_bytes(value)
                reads.clear()
                if defect=='partial':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('oldcode','identity','trace'):
                    path=cache/'complete.json' if defect=='oldcode' else out/'complete.json' if defect=='identity' else out/'adaptation/fit_diagnostics.json';a=m._read(path)
                    if defect=='oldcode':a['code_commit']='d'*40
                    elif defect=='identity':a['model_identity']['updates']+=1
                    else:a['arms']['DIAG']['trace'].pop()
                    path.unlink();m._write(path,a)
                else:
                    path=out/'adaptation/DIAG.npz' if defect=='negative_scale' else out/'thresholds.npz' if defect=='tau' else out/'adaptation/U_features.npz' if defect=='changed_U' else out/'targets/star_DURA/predictions.npz'
                    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
                    if defect=='negative_scale':a['inverse_scale'][0]=-1
                    elif defect=='tau':a['conditional'][5,0]+=.1
                    elif defect=='changed_U':a['signal_features'][0,0]+=.1
                    elif defect=='control':a['raw_scores'][0,1]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'bad_score')
                assert not reads and not (tmp_path/'bad_score').exists()
        for path,value in saved.items():path.write_bytes(value)
        # A failed U fit cannot proceed to any source or future feature read.
        monkeypatch.setattr(m.parent,'_backend',lambda *a:backend)
        monkeypatch.setattr(m.core,'fit_adapters',lambda *a:(_ for _ in ()).throw(ValueError('EM failure')))
        monkeypatch.setattr(m,'_U_features',lambda *a:U)
        monkeypatch.setattr(m,'_parent_rows',lambda *a:pytest.fail('No source rows after failed U fit'))
        monkeypatch.setattr(m,'_parent_target_rows',lambda *a:pytest.fail('No Q rows after failed U fit'))
        with pytest.raises(ValueError,match='EM failure'):m.run(dict(c,output_root=str(tmp_path/'failed_fit')),'cpu')
        assert not (tmp_path/'failed_fit/complete.json').exists()
