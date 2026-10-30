"""Synthetic source-only adapter export, with no native F/G or real files."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_pamf_phase3_star_protected_open_cache import native_config, identities, features


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_joint_geometry_export')


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@contextmanager
def fixture(root,monkeypatch):
    torch.set_num_threads(1);s=api();cache=s.cache;ow=s.ow;n=native_config(root,'R1');n['code_commit']=s.PARENT_COMMIT
    L=features('L');nl=len(L['ids']);nv=45
    monkeypatch.setattr(s,'COUNTS',dict(L=nl,novel=nv,union=nl+nv));monkeypatch.setattr(cache,'COUNTS',dict(L=nl,OE=90))
    cn=dict(n,row_id='cache-s2026092701');parent,original,dura=identities(n)
    cc=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,native_config=cn,L_root=str(root/'cache/L'),
        OE_root=str(root/'cache/OE'),output_root=str(root/'features'),batch_size=128)
    source={r:dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',
        role=r,count=count,source_only=True,all_new_IQ_satellite=True,restored_input_path='P1',parent_F0_identity=parent,dura_identity=dura)
        for r,count in (('L',nl),('OE',90))}
    fm=dict(schema=cache.SCHEMA,status=cache.STATUS,run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,config=cc,counts=dict(L=nl,OE=90),
        parent_F0_identity=parent,dura_identity=dura,original_parent_identity=original,source_manifests=source,source_only=True,
        input_path='cached_P1_only',model_native_state_unchanged=True,model_states_unchanged=True,optimizer_updates=0,model_updates=0,
        F_forward_rows=nl+90,synthetic_F_forward_rows=2,**{k:0 for k in cache.ZEROS})
    write(root/'features/complete.json',fm);np.savez(root/'features/L.npz',ids=L['ids'],**L['metadata'],
        **{k:L[k] for k in cache.FEATURE_SHAPES})
    n['output_root']=str(root/'train/R1')
    train=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,arm='R1',native_config=n,feature_root=str(root/'features'),
        output_root=n['output_root'],steps=2220,lr=.0001,sampling_seed=2026103017,slot_seed=2026103018)
    ec=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,arm='R1',train_config=train,feature_root=train['feature_root'],
        cache_root=str(root/'cache'),output_root=str(root/'predict/R1'),batch_size=128)
    identity=dict(run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,arm='R1',checkpoint=str(root/'train/R1/final.pt'),
        adapter_updates=2220,original_parent_updates=11100,native_updates=0,original_parent_identity=original,
        parent_F0_identity=parent,dura_identity=dura,identifier_parameters=299170,open_world_parameters=66177,
        training_container_parameters=966899,trainable_parameters=66177,native_identity_frozen=True,identity_head_replaced=False,
        input_path='cached_P1_only',residual_enabled=True,OE_member_supervision=True)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(27);parts=ow.ProtectedParts(True)
    with torch.no_grad():parts.residual_gain.fill_(.2)
    handle=dict(classes=cache.native.CLASSES,source_rxs=cache.native.RX,source_days=[1,2,3],split_seed=392005,
        counts=dict(L_s=6300,U_s=56700,V=27000),source_contract=n['source_contract'])
    cp=dict(schema='pamf_phase3_star_protected_open_train_v1',config=train,identity=identity,feature_manifest=fm,
        source_data_handle=handle,open_world=parts.state_dict(),source_only=True,model_native_state_unchanged=True,**{k:0 for k in cache.ZEROS})
    td=dict(schema=cp['schema'],status='STAR_PROTECTED_SOURCE_FROZEN',run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,config=train,
        identity=identity,updates=2220,adapter_updates=2220,original_parent_updates=11100,feature_manifest=fm,
        source_only=True,model_native_state_unchanged=True,**{k:0 for k in cache.ZEROS})
    write(root/'train/R1/complete.json',td);torch.save(cp,root/'train/R1/final.pt')
    prediction=dict(schema=s.evaluation.SCHEMA,status=s.evaluation.STATUS,run_id=s.PARENT_RUN,code_commit=s.PARENT_COMMIT,
        arm='R1',config=ec,model_identity=identity,parent_F0_identity=parent,dura_identity=dura,source_only=True,
        known_count=27000,novel_count=nv,source_L_count=nl,source_truth_reads=0,target_reads=0,source_OE_reads=0,original_U_s_reads=0,
        native_identity_frozen=True,native_updates=0,eval_neural_updates=0,frozen_models_verified=True,identity_predictions_unchanged=True,
        novel_features_same_forward=True,aggregates_frozen_before_V=True,thresholds_frozen_before_novel=True,source_aggregates_unchanged=True,
        source_roles_disjoint=True,G_forward_rows=0,received_cache_values_read=0,formal_target_accessed=False)
    write(root/'predict/R1/complete.json',prediction)
    rng=np.random.default_rng(21);nvmeta=dict(ids=np.asarray(['NV-'+str(i) for i in range(nv)]),
        rx_i=np.tile(np.repeat(np.asarray(cache.native.RX,np.int64),3),3),day_i=np.tile(np.arange(1,4,dtype=np.int64),15))
    novel=dict(nvmeta,**{k:np.asarray(F.normalize(torch.tensor(rng.standard_normal((nv,160)).tolist(),dtype=torch.float32),dim=1).tolist(),np.float32)
        for k in ('original_fused160','protected160')})
    novel['original_fused160'][0]=0
    np.savez(root/'predict/R1/novel_features.npz',**novel)
    nm=dict(source['L'],role='novel',count=nv)
    write(root/'cache/novel/manifest.json',nm);np.save(root/'cache/novel/ids.npy',nvmeta['ids'],allow_pickle=False)
    np.savez(root/'cache/novel/metadata.npz',rx_i=nvmeta['rx_i'],day_i=nvmeta['day_i'],domain=np.searchsorted(cache.native.RX,nvmeta['rx_i'])*3+nvmeta['day_i']-1)
    c=dict(run_id=s.RUN_ID,code_commit='b'*40,parent_eval_config=ec,output_root=str(root/'joint'),batch_size=128)
    yield s,c,L,novel,cp,td,prediction


def test_config_rejects_extra_private_fields_wrong_parent_and_output_overlap(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (s,c,*rest):
        for delta in (dict(truth='private'),dict(U_root='U'),dict(Q_root='Q'),dict(K=3),dict(batch_size=64),dict(output_root=c['parent_eval_config']['feature_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):s.validate_config(dict(c,**delta))
        bad=deepcopy(c);bad['parent_eval_config']['code_commit']='c'*40
        with pytest.raises(ValueError):s.validate_config(bad)
        Path(c['output_root']).mkdir()
        monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Checkpoint opened on collision')))
        with pytest.raises(FileExistsError):s.export(c,'cpu')


def test_complete_adapter_only_flow_smoke_first_L_only_and_novel_exact_reuse(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (s,c,L,novel,cp,td,prediction):
        from experiments.pamf_phase3_open_world import star_protected_open_train as train
        def forbidden(*a,**k):raise AssertionError('Native/F/G/IQ access is forbidden')
        monkeypatch.setattr(train,'_model',forbidden);monkeypatch.setattr(train,'load_trained_model',forbidden)
        monkeypatch.setattr(s.cache.native,'make_trainable_model',forbidden);monkeypatch.setattr(s.ow,'forward',forbidden)
        calls=[];load=s._load_L
        def source(ec):
            assert (Path(c['output_root'])/'smoke.json').is_file();calls.append('L');return load(ec)
        monkeypatch.setattr(s,'_load_L',source)
        marker=s.export(c,'cpu');assert calls==['L']
        assert s.validate_metadata(marker) is marker and marker['actual_adapter_checkpoint_loaded']
        assert not marker['native_F0_checkpoint_loaded'] and not marker['G_checkpoint_loaded']
        assert marker['real_F_forward_rows']==marker['synthetic_F_forward_rows']==marker['G_forward_rows']==0
        assert marker['adapter_L_feature_rows']==len(L['ids']) and marker['synthetic_adapter_rows']==2
        assert marker['source_data_handle']==cp['source_data_handle'] and marker['reused_novel_prediction_identity']==prediction
        with np.load(Path(c['output_root'])/'union.npz',allow_pickle=False) as z:
            assert set(z.files)=={'ids','rx_i','day_i','pool','original_fused160','protected160'}
            assert z['original_fused160'].shape==(len(L['ids'])+len(novel['ids']),160)
            np.testing.assert_array_equal(z['pool'],[0]*len(L['ids'])+[1]*len(novel['ids']))
            for k in ('ids','rx_i','day_i','original_fused160','protected160'):np.testing.assert_array_equal(z[k][len(L['ids']):],novel[k])
            expected=np.asarray(F.normalize(torch.tensor(L['raw_original_fused160'].tolist()),dim=1).tolist(),np.float32)
            np.testing.assert_array_equal(z['original_fused160'][:len(L['ids'])],expected)
        assert marker['eligible_primary'] is False and marker['final_phase3_complete'] is False


def test_unverified_parts_or_source_handle_abort_before_L_values(tmp_path,monkeypatch,subtests):
    for defect in ('handle','identity','parts','target','cache_manifest'):
        with subtests.test(defect=defect),fixture(tmp_path/defect,monkeypatch) as (s,c,L,novel,cp,td,prediction):
            if defect=='handle':cp['source_data_handle']['split_seed']=0
            elif defect=='identity':cp['identity']=dict(cp['identity'],native_updates=1)
            elif defect=='parts':del cp['open_world']['direction_projection.bias']
            elif defect=='target':cp['target_reads']=1
            else:cp['feature_manifest']=dict(cp['feature_manifest'],counts=dict(L=1,OE=90))
            torch.save(cp,Path(c['parent_eval_config']['train_config']['output_root'])/'final.pt')
            monkeypatch.setattr(s,'_load_L',lambda *a:(_ for _ in ()).throw(AssertionError('L opened before source identity')))
            with pytest.raises(ValueError):s.export(c,'cpu')
            assert not (Path(c['output_root'])/'complete.json').exists()


def test_novel_private_keys_pool_wrong_metadata_nonfinite_or_overlap_never_complete(tmp_path,monkeypatch,subtests):
    for defect in ('y','pool','rx','norm','nonfinite','overlap'):
        with subtests.test(defect=defect),fixture(tmp_path/defect,monkeypatch) as (s,c,L,novel,cp,td,prediction):
            if defect=='y':novel['y']=np.zeros(len(novel['ids']),np.int64)
            elif defect=='pool':novel['pool']=np.ones(len(novel['ids']),np.int64)
            elif defect=='rx':novel['rx_i'][0]=3
            elif defect=='norm':novel['protected160'][0]*=2
            elif defect=='nonfinite':novel['protected160'][0,0]=np.nan
            else:novel['ids'][0]=L['ids'][0]
            np.savez(Path(c['parent_eval_config']['output_root'])/'novel_features.npz',**novel)
            with pytest.raises(ValueError):s.export(c,'cpu')
            assert not (Path(c['output_root'])/'complete.json').exists()


def test_adapter_state_mutation_or_smoke_failure_cannot_complete(tmp_path,monkeypatch,subtests):
    for defect in ('smoke','mutation'):
        with subtests.test(defect=defect),monkeypatch.context() as local,fixture(tmp_path/defect,local) as (s,c,*rest):
            if defect=='smoke':local.setattr(s,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('smoke failed')))
            else:
                encode=s._encode_L
                def changed(model,*a):
                    result=encode(model,*a)
                    with torch.no_grad():model.open_world.residual_gain.add_(1.)
                    return result
                local.setattr(s,'_encode_L',changed)
            with pytest.raises(ValueError):s.export(c,'cpu')
            assert not (Path(c['output_root'])/'complete.json').exists()
