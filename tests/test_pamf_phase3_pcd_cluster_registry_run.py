"""Frozen PCD current features, registry-before-cached-future and no truth."""
from contextlib import contextmanager
import importlib
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.pcd_cluster_registry_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_world_run import fixture as parent
    with parent(root,mp) as (fresh,ref,_,b,events,identity):
        m=api();ref['code_commit']=m.REFERENCE_CODE;ref['train_config']['code_commit']=m.REFERENCE_CODE;identity['code_commit']=m.REFERENCE_CODE;pm=fresh.run(ref,'cpu')
        mp.setattr(m,'COUNTS',dict(fresh.COUNTS));mp.setattr(m,'Q_COUNT',fresh.Q_COUNT);events.clear()
        model,_=fresh._load_model(ref,'cpu');events.clear();mp.setattr(m,'_load_model',lambda *a:(events.append('checkpoint_PCD') or model,identity))
        train=Path(ref['train_config']['output_root']);train.mkdir(exist_ok=True);ids=np.asarray([f'fresh-U{i:04}' for i in range(576)]);rng=np.random.default_rng(77)
        np.savez(train/'U_features.npz',ids=ids,views=rng.normal(size=(3,576,2,256)).astype(np.float32),parent_z=np.tile(np.eye(160)[0],(576,1)).astype(np.float32),neighbors=(np.arange(576)+1)%576)
        manifest=dict(count=576,capsule_id=ref['train_config']['U_capsule_id'],split_id=ref['train_config']['U_split_id'])
        m._write(train/'complete.json',dict(status='FRESH_PCD_TRAINING_COMPLETE',identity=identity,U_manifest=manifest,source_only=False,target_U_reads=576,original_models_unchanged=True))
        c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=ref,output_root=str(root/'registry'),batch_size=128)
        source=fresh._source_rows;target=fresh._target_rows
        def source_rows(*a):
            assert Path(c['output_root'],'registry.json').exists();events.append('source_'+a[-1]);return source(*a)
        def target_rows(*a):
            assert Path(c['output_root'],'registry.json').exists();events.append(a[-1]);return target(*a)
        mp.setattr(fresh,'_source_rows',source_rows);mp.setattr(fresh,'_target_rows',target_rows)
        yield m,c,pm,b,model,events


def test_producer_freezes_current_all_U_registry_before_any_V_NV_Q_and_rebuilds(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,model,events):
        done=m.run(c,'cpu');out=Path(c['output_root'])
        assert done['registry_frozen_before_V_NV_Q'] and done['all_three_conditions_complete'] and not done['classification_unchanged']
        assert done['real_signal_forward_rows']==m.COUNTS['L']+576 and done['real_F_forward_rows']==576 and done['G_forward_rows']==0
        assert events==['checkpoint_PCD','L','source_known','source_novel','clean_DURA','star_DURA']
        gated,state,registry,U,rows=m._prediction_gate(out);assert gated==done and set(rows)=={'source_V','source_NV','Q_clean_DURA','Q_star_DURA'}
        for key in ('Q_clean_DURA','Q_star_DURA'):assert 'y' not in rows[key] and rows[key]['raw_prediction'].shape==(m.Q_COUNT,4)
        with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_private_config_or_failed_registry_never_reads_future_or_completes(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,model,events):
        for delta in (dict(truth='private'),dict(unknown_count=3),dict(batch_size=True)):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        monkeypatch.setattr(m.core,'discover',lambda *a:(_ for _ in ()).throw(ValueError('registry failure')))
        with pytest.raises(ValueError,match='registry failure'):m.run(c,'cpu')
        assert events==['checkpoint_PCD','L'] and not Path(c['output_root'],'complete.json').exists()
