"""Synthetic derived-cache preparation, matched samplers and new-network IO."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pamf_phase3_star_joint_geometry_export import fixture as old_fixture
from test_pamf_phase3_star_protected_open_cache import features


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_metric_residual_train')


@contextmanager
def fixture(root,monkeypatch):
    torch.set_num_threads(1)
    with old_fixture(root,monkeypatch) as (old,oc,L,novel,cp,td,prediction):
        t=api();monkeypatch.setattr(t,'COUNTS',dict(L=360,OE=90))
        feature_root=Path(oc['parent_eval_config']['feature_root']);OE=features('OE')
        np.savez(feature_root/'OE.npz',ids=OE['ids'],**OE['metadata'],**{k:OE[k] for k in t.source_cache.FEATURE_SHAPES})
        source=deepcopy(oc['parent_eval_config']['train_config']['native_config']);out=root/'prepared'
        source.update(run_id=t.RUN_ID,code_commit='b'*40,row_id='star-metric-residual-prepare-s2026092701',output_root=str(out))
        prepare=dict(run_id=t.RUN_ID,code_commit='b'*40,source_config=source,feature_root=str(feature_root),output_root=str(out),batch_size=128)
        b=dict(reader=torch.nn.Linear(1,1).eval().requires_grad_(False),dura=None,reader_identity=cp['feature_manifest']['parent_F0_identity'])
        monkeypatch.setattr(t,'_backend',lambda *a:(b,{}));monkeypatch.setattr(t,'_smoke',lambda *a:dict(status='PASS',actual_F0_checkpoint=True,F0_forward_rows=2))
        original=t._old_features
        def load(path,role):
            assert (out/'smoke.json').is_file();assert role in ('L','OE');return original(path,role)
        monkeypatch.setattr(t,'_old_features',load)
        manifest=t.prepare(prepare,'cpu')
        configs={arm:dict(run_id=t.RUN_ID,code_commit='b'*40,arm=arm,cache_root=str(out),output_root=str(root/'new_train'/arm),
            steps=t.STEPS,lr=.0001,L_seed=2026103017,OE_seed=2026103018) for arm in t.ARMS}
        yield t,prepare,configs,manifest


def frozen_checkpoint(c,manifest):
    """Actual synthetic network state plus nominal-complete metadata, no training."""
    t=api();model=t._model(c,'cpu');identity=t._identity(c,manifest);out=Path(c['output_root']);out.mkdir(parents=True)
    zero={k:0 for k in t.ZEROS}
    torch.save(dict(schema=t.SCHEMA,config=c,identity=identity,network=model.state_dict(),cache_manifest=manifest,source_only=True,**zero),out/'final.pt')
    done=dict(schema=t.SCHEMA,status=t.STATUS,run_id=t.RUN_ID,code_commit=c['code_commit'],arm=c['arm'],config=c,updates=t.STEPS,
        identity=identity,cache_manifest=manifest,source_only=True,final_phase3_complete=False,test_only_nominal_completion=True,**zero)
    t._write(out/'complete.json',done);return model,identity


def test_prepare_exact_schema_parent_and_readonly_unit_feature_pairs(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,configs,manifest):
        assert manifest['status']==t.CACHE_STATUS and manifest['real_F_forward_rows']==0 and manifest['synthetic_F_forward_rows']==2
        assert manifest['source_feature_manifest']['code_commit']==t.SOURCE_CODE
        L=t.load_role(c['output_root'],'L');OE=t.load_role(c['output_root'],'OE')
        assert set(L)=={'ids','metadata','u','c','manifest'} and 'y' not in OE['metadata']
        for role in (L,OE):
            assert role['u'].dtype==role['c'].dtype==np.float32 and not role['u'].flags.writeable
            np.testing.assert_allclose(np.linalg.norm(role['c'].astype(np.float64),axis=1),1.,atol=2e-6)
        for delta in (dict(truth='private'),dict(Q_public_root='Q'),dict(batch_size=64)):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_prepare_config(dict(c,**delta))
        with pytest.raises(FileExistsError):t.prepare(c,'cpu')
        Path(c['output_root'],'OE.npz').unlink();assert len(t.load_role(c['output_root'],'L')['ids'])==360
        with pytest.raises(ValueError):t.load_role(c['output_root'],'V')


def test_sampler_same_L_stream_and_exact_rotating_domains_without_OE_labels(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,configs,m):
        L=t.load_role(c['output_root'],'L');OE=t.load_role(c['output_root'],'OE');a=t.Sampler(L);b=t.Sampler(L,OE)
        assert not hasattr(a,'OE_rng')
        for step in range(1,5):
            left=a.sample(step);right=b.sample(step);np.testing.assert_array_equal(left['L'],right['L'])
            assert len(set(left['L']))==192 and left['OE'].size==0 and len(set(right['OE']))==64
            for label in range(6):
                ix=left['L'][L['metadata']['y'][left['L']]==label];counts=np.bincount(L['metadata']['domain'][ix],minlength=15)
                expected=np.full(15,2);first=(2*(step-1)+label)%15;expected[[first,(first+1)%15]]+=1
                np.testing.assert_array_equal(counts,expected)
            counts=np.bincount(OE['metadata']['domain'][right['OE']],minlength=15);expected=np.full(15,4)
            expected[[(4*(step-1)+j)%15 for j in range(4)]]+=1;np.testing.assert_array_equal(counts,expected)


@pytest.mark.parametrize('arm',['M_L','M_LOE'])
def test_two_actual_updates_smoke_before_cache_read_frozen_source_and_strict_network_reload(tmp_path,monkeypatch,arm):
    with fixture(tmp_path,monkeypatch) as (t,p,configs,manifest):
        monkeypatch.setattr(t,'STEPS',2);c=dict(configs[arm],steps=2);events=[];load=t.load_role
        monkeypatch.setattr(t,'_backend',lambda *a:pytest.fail('Training/loader must not instantiate F0'))
        def read(root,role):
            assert (Path(c['output_root'])/'smoke.json').is_file()
            if arm=='M_L':assert role=='L'
            events.append(role);return load(root,role)
        monkeypatch.setattr(t,'load_role',read)
        done=t.train(c,'cpu');assert events==(['L'] if arm=='M_L' else ['L','OE'])
        assert done['updates']==2 and done['native_updates']==0 and done['new_parameters']==102688
        assert done['OE_training_rows_loaded']==(0 if arm=='M_L' else 90) and done['statistical_background_uses_OE']
        assert done['first_step']['gradient_norms']['last']>0 and done['first_step']['gradient_norms']['hidden']==0
        assert done['second_step']['gradient_norms']['hidden']>0
        model,identity=t.load_trained_model(c,Path(c['output_root'])/'final.pt','cpu')
        assert identity==done['identity'] and all(not p.requires_grad for p in model.parameters())
        assert not hasattr(model,'id_backbone') and not hasattr(model,'relation_scorer')
        payload=torch.load(Path(c['output_root'])/'final.pt',weights_only=False)
        assert set(payload['network'])=={'network.0.weight','network.0.bias','network.2.weight','network.2.bias'}
        del payload['network']['network.2.bias'];torch.save(payload,Path(c['output_root'])/'final.pt')
        with pytest.raises(ValueError):t.load_trained(c,'cpu')
        with pytest.raises(FileExistsError):t.train(c,'cpu')


def test_forbidden_train_config_cache_labels_and_smoke_failure_do_not_read_training_values(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,p,configs,manifest):
        c=configs['M_L']
        for delta in (dict(steps=222),dict(L_seed=2026092701),dict(truth='private'),dict(OE_seed=1),dict(arm='R1')):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke fail')))
        monkeypatch.setattr(t,'load_role',lambda *a:pytest.fail('Source values read before smoke'))
        with pytest.raises(ValueError,match='smoke fail'):t.train(c,'cpu')
        assert not Path(c['output_root']).exists()
