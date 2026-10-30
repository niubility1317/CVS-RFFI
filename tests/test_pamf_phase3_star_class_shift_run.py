"""Cached star U only, preserved shared controls, deferred future features."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_shift_run')
def write(p,v):p.write_text(json.dumps(v),encoding='utf-8')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_mixture_uda_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (old,mc,bpm,backend,u,events,seen,public):
        m=api();mp.setattr(m,'COUNTS',dict(old.COUNTS));mp.setattr(m,'Q_COUNT',8);mp.setattr(m,'U_COUNT',16);mp.setattr(m._core(),'U_COUNT',16)
        pm=old.run(mc,'cpu');mc['code_commit']=m.ADAPTATION_CODE;pm['code_commit']=m.ADAPTATION_CODE;pm['config']=deepcopy(mc)
        parent=Path(mc['output_root']);write(parent/'complete.json',pm);write(parent/'resolved_config.json',mc)
        for name in ('models','adaptation'):
            done=old._read(parent/name/'complete.json');done['code_commit']=m.ADAPTATION_CODE;write(parent/name/'complete.json',done)
        events.clear();seen['F0'].clear();seen['signal'].clear()
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_prediction_root=mc['reference_prediction_root'],reference_adaptation_root=str(parent),
            U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID,output_root=str(root/'class_shift'),batch_size=128)
        source=m._parent_rows;target=m._parent_target_rows;shared=m._shared_target_rows;features=m._U_features
        def read_U(config,done):
            assert events==['checkpoint'] and len(backend['dura'][0].outputs)==1
            assert all((Path(c['output_root'])/p).is_file() for p in ('models/complete.json','adaptation/source_prior.npz','thresholds.npz'))
            events.append('U_features');return features(config,done)
        def rows(path,role):
            assert (Path(c['output_root'])/'adaptation/complete.json').is_file();events.append('parent_'+role);return source(path,role)
        def query(path,condition):
            assert (Path(c['output_root'])/'adaptation/complete.json').is_file();events.append(condition);return target(path,condition)
        def control(path,condition):
            assert (Path(c['output_root'])/'adaptation/complete.json').is_file();return shared(path,condition)
        mp.setattr(m,'_U_features',read_U);mp.setattr(m,'_parent_rows',rows);mp.setattr(m,'_parent_target_rows',query);mp.setattr(m,'_shared_target_rows',control)
        yield m,c,bpm,pm,backend,u,events,seen


def test_cached_U_CLASS_fit_after_actual_smoke_and_two_controls_preserved(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,bpm,pm,b,u,events,seen):
        monkeypatch.setattr(m.mix._core(),'fit_adapters',lambda *a:pytest.fail('Do not refit SHARED'))
        for name in ('fit_known','fit_background'):monkeypatch.setattr(m.best.density,name,lambda *a:pytest.fail('No source density fit'))
        monkeypatch.setattr(m.best.reference.risk,'calibrate_known',lambda *a:pytest.fail('No new thresholds'))
        original=m._core().fit_adapter;inputs=[]
        def fit(state,z,prior):inputs.append(z.copy());return original(state,z,prior)
        monkeypatch.setattr(m._core(),'fit_adapter',fit)
        result=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','U_features','parent_known','parent_novel','clean_DURA','star_DURA']
        assert result['real_F_forward_rows']==result['real_signal_forward_rows']==result['G_forward_rows']==0
        assert result['synthetic_F_forward_rows']==result['synthetic_signal_forward_rows']==result['G_synthetic_forward_rows']==2
        assert result['U_feature_rows_read']==result['target_U_reads']==16
        assert result['source_IQ_reads']==result['target_Q_IQ_reads']==result['private_truth_reads']==0
        assert result['shared_controls_reused_exactly'] and result['all_three_DURA_conditions_complete']
        assert len(inputs)==1 and inputs[0].shape==(16,160)
        assert len(seen['F0'])==len(seen['signal'])==len(b['dura'][0].outputs)==1
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);parent=m.best._source_rows(c['reference_prediction_root'],'source_DURA',role)
            for k,v in parent.items():np.testing.assert_array_equal(a[k],np.tile(v,(1,3)) if k in ('raw_scores','scores') else v)
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition);parent=m.best._target_rows(c['reference_prediction_root'],condition);shared=m.mix._target_rows(c['reference_adaptation_root'],condition)
            for k,v in parent.items():np.testing.assert_array_equal(a[k][:,:2] if k in ('raw_scores','scores') else a[k],v)
            for k in ('raw_scores','scores'):np.testing.assert_array_equal(a[k][:,2:4],shared[k][:,4:6])
            np.testing.assert_array_equal(a['scores'],a['raw_scores']-np.tile(tau,(3,1))[:,a['closed']].T)
        for filename,oldname in (('SHARED.npz','DA_M.npz'),('shared_prior.npz','source_prior.npz'),('U_features.npz','U_features.npz')):
            with np.load(out/'adaptation'/filename) as a,np.load(Path(c['reference_adaptation_root'])/'adaptation'/oldname) as p:
                assert set(a.files)==set(p.files)
                for k in a.files:np.testing.assert_array_equal(a[k],p[k])
        assert m._read(out/'adaptation/shared_diagnostics.json')==m._read(Path(c['reference_adaptation_root'])/'adaptation/fit_diagnostics.json')
        with np.load(out/'adaptation/CLASS.npz') as z:assert set(z.files)=={'schema','delta','background_delta','coefficients'} and z['delta'].shape==(6,160)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_role_config_and_wrong_shared_parent_fail_before_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,bpm,pm,b,u,events,seen):
        for delta in (dict(U_truth='private'),dict(U_public_root='new-input'),dict(U_capsule_id='clean'),dict(output_root=c['reference_adaptation_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        path=Path(c['reference_adaptation_root'])/'complete.json';old=path.read_bytes()
        try:
            wrong=deepcopy(pm);wrong['model_identity']=dict(wrong['model_identity'],input_path='cached_P0_IQ_only');write(path,wrong)
            monkeypatch.setattr(m,'_U_features',lambda *a:pytest.fail('No U before legal inherited SHARED'))
            with pytest.raises(ValueError):m.run(c,'cpu')
            assert not Path(c['output_root'],'complete.json').is_file()
        finally:path.write_bytes(old)


def test_failed_CLASS_fit_never_opens_source_or_query_features(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,bpm,pm,b,u,events,seen):
        monkeypatch.setattr(m._core(),'fit_adapter',lambda *a:(_ for _ in ()).throw(ValueError('synthetic CLASS fit failure')))
        with pytest.raises(ValueError,match='CLASS fit failure'):m.run(c,'cpu')
        assert events==['checkpoint','U_features']
        assert not Path(c['output_root'],'adaptation/complete.json').is_file() and not Path(c['output_root'],'complete.json').is_file()
