"""U-only mixture adaptation, immutable controls and deferred public features."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_mixture_uda_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_dura_best_run import fixture as parent_fixture,write
    with parent_fixture(root,mp) as (best,bc,_,backend,public,events,seen,values):
        m=api();mp.setattr(m,'COUNTS',dict(best.COUNTS));mp.setattr(m,'Q_COUNT',8);mp.setattr(m,'U_COUNT',16);mp.setattr(m._core(),'U_COUNT',16)
        pm=best.run(bc,'cpu');bc['code_commit']=m.REFERENCE_CODE;pm['code_commit']=m.REFERENCE_CODE;pm['config']=deepcopy(bc)
        parent=Path(bc['output_root']);write(parent/'complete.json',pm);write(parent/'resolved_config.json',bc)
        fit=best._read(parent/'models/complete.json');fit['code_commit']=m.REFERENCE_CODE;write(parent/'models/complete.json',fit)
        events.clear();seen['F0'].clear();seen['signal'].clear()
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_prediction_root=str(parent),U_public_root=str(root/'star_g10/discovery/public'),
            U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID,output_root=str(root/'mixture'),batch_size=128)
        rng=np.random.default_rng(41);u=(dict(capsule_id=m.U_CAPSULE_ID,split_id=m.U_SPLIT_ID,count=16,registered_classes=best.signal_train.native.CLASSES),
            np.asarray(['opaque-U-'+str(i) for i in range(16)]),rng.normal(size=(16,2,256)).astype(np.float32))
        def opened(config):
            assert events==['checkpoint'] and len(backend['dura'][0].outputs)==1
            assert all((Path(c['output_root'])/p).is_file() for p in ('models/complete.json','models/density.npz','thresholds.npz'))
            events.append('U');return u
        mp.setattr(m,'_public',opened)
        source=m._parent_rows;target=m._parent_target_rows
        def rows(path,role):
            assert (Path(c['output_root'])/'adaptation/complete.json').is_file();events.append('parent_'+role);return source(path,role)
        def query(path,condition):
            assert (Path(c['output_root'])/'adaptation/complete.json').is_file();events.append(condition);return target(path,condition)
        mp.setattr(m,'_parent_rows',rows);mp.setattr(m,'_parent_target_rows',query)
        yield m,c,pm,backend,u,events,seen,public


def test_actual_U_fit_before_parent_features_all_three_paths_and_exact_DA0(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,seen,public):
        for name in ('fit_known','fit_background'):monkeypatch.setattr(m.best.density,name,lambda *a:pytest.fail('No source density fit'))
        monkeypatch.setattr(m.best.reference.risk,'calibrate_known',lambda *a:pytest.fail('No threshold fit'))
        original=m._core().fit_adapters;fit_inputs=[]
        def fit(state,z,prior):fit_inputs.append(z.copy());return original(state,z,prior)
        monkeypatch.setattr(m._core(),'fit_adapters',fit)
        result=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','U','parent_known','parent_novel','clean_DURA','star_DURA']
        assert result['real_F_forward_rows']==result['real_signal_forward_rows']==result['G_forward_rows']==16
        assert result['synthetic_F_forward_rows']==result['synthetic_signal_forward_rows']==result['G_synthetic_forward_rows']==2
        assert result['source_IQ_reads']==result['target_Q_IQ_reads']==result['private_truth_reads']==0
        assert result['adapters_frozen_before_source_and_Q'] and result['all_three_DURA_conditions_complete']
        assert len(fit_inputs)==1 and fit_inputs[0].shape==(16,160)
        assert len(seen['F0'])==len(seen['signal'])==len(b['dura'][0].outputs)==2
        for x,y,z in zip(seen['F0'],seen['signal'],b['dura'][0].outputs):
            np.testing.assert_array_equal(x.tolist(),z.tolist());np.testing.assert_array_equal(y.tolist(),z.tolist())
        with np.load(out/'thresholds.npz') as z:tau=z['conditional']
        for role in ('known','novel'):
            a=m._source_rows(out,'source_DURA',role);p=m.best._source_rows(c['reference_prediction_root'],'source_DURA',role)
            for key,v in p.items():
                if key in ('raw_scores','scores'):np.testing.assert_array_equal(a[key],np.tile(v,(1,3)))
                else:np.testing.assert_array_equal(a[key],v)
        for condition in m.CONDITIONS[1:]:
            a=m._target_rows(out,condition);p=m.best._target_rows(c['reference_prediction_root'],condition)
            for key,v in p.items():np.testing.assert_array_equal(a[key][:,:2] if key in ('raw_scores','scores') else a[key],v)
            np.testing.assert_array_equal(a['scores'],a['raw_scores']-np.tile(tau,(3,1))[:,a['closed']].T)
        with np.load(out/'adaptation/U_features.npz') as z:
            assert set(z.files)=={'ids','default_logits','closed','signal_features'};np.testing.assert_array_equal(z['signal_features'],fit_inputs[0])
        for arm in ('DA_K','DA_M'):
            with np.load(out/'adaptation'/(arm+'.npz')) as z:assert set(z.files)=={'schema','delta','coefficients'} and z['delta'].shape==(160,)
        assert m._read(out/'calibration/complete.json')==m.best._read(Path(c['reference_prediction_root'])/'calibration/complete.json')
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_config_and_parent_identity_fail_before_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,seen,public):
        for delta in (dict(U_truth='private'),dict(U_public_root=str(tmp_path/'evaluation/public')),dict(U_capsule_id='clean'),dict(output_root=c['reference_prediction_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        parent=Path(c['reference_prediction_root'])/'complete.json';old=parent.read_bytes()
        try:
            bad=deepcopy(pm);bad['code_commit']='c'*40;parent.write_text(json.dumps(bad),encoding='utf-8')
            monkeypatch.setattr(m,'_public',lambda *a:pytest.fail('No U for wrong parent'))
            with pytest.raises(ValueError):m.run(c,'cpu')
            assert not Path(c['output_root'],'complete.json').is_file()
        finally:parent.write_bytes(old)


def test_failed_U_fit_never_opens_source_or_query_features(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,seen,public):
        monkeypatch.setattr(m._core(),'fit_adapters',lambda *a:(_ for _ in ()).throw(ValueError('synthetic U fit failure')))
        with pytest.raises(ValueError,match='U fit failure'):m.run(c,'cpu')
        assert events==['checkpoint','U'] and not Path(c['output_root'],'adaptation/complete.json').is_file()
        assert not Path(c['output_root'],'complete.json').is_file()
