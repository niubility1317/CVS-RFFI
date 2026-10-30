"""Synthetic source-feature cache boundaries; no real IQ, weights or truth."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pamf_phase3_star_reference_open_model import fixture


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_protected_open_cache')


def native_config(root,arm='R0'):
    from experiments.pamf_phase3_open_world import source_open_embedding_train as n
    return dict(run_id='20261030-phase3-star-protected-open-s2026092701-r01',row_id=arm+'-s2026092701',
        code_commit='a'*40,model_seed=2026092701,batch_size=128,registered_classes=n.CLASSES,
        base_release=str(root/'base'),frozen_sources={k:str(root/k) for k in ('F','AF','BASE','FOLD','G')},
        source_output=str(root/'original'),source_contract=str(root/'contract.json'),source_cache=str(root/'native_cache'),
        output_root=str(root/'cache_native_R0'))


def identities(n):
    p=dict(model='original_F0_E50',parameters=232993,epoch=50,updates=11100,model_seed=2026092701,
        **{k:n[k] for k in ('source_output','source_contract','base_release')})
    original=dict(parent_epoch=50,parent_updates=11100,model_seed=2026092701,
        **{k:n[k] for k in ('source_output','source_contract','base_release','source_cache','frozen_sources')})
    return p,original,dict(deployment_id='dura_v1',parameters=266951,epoch=20,updates=4940)


def features(role):
    grid=[(r,d) for r in (1,3,4,6,8) for d in (1,2,3)]
    records=[(y,r,d) for y in range(6) for r,d in grid for _ in range(4)] if role=='L' else [(0,r,d) for r,d in grid for _ in range(6)]
    y=np.asarray([a[0] for a in records],np.int64);rx=np.asarray([a[1] for a in records],np.int64);day=np.asarray([a[2] for a in records],np.int64)
    rng=np.random.default_rng(17 if role=='L' else 18);f=rng.standard_normal((len(y),160)).astype(np.float32)
    if role=='L':f[:,:6]+=np.eye(6,dtype=np.float32)[y]*4
    meta=dict(rx_i=rx,day_i=day,domain=np.searchsorted((1,3,4,6,8),rx)*3+day-1)
    if role=='L':meta['y']=y
    return dict(ids=np.asarray([role+'-'+str(i) for i in range(len(y))]),metadata=meta,raw_original_fused160=f,
        contributions=np.stack((.4*f,.3*f,.2*f,.1*f),1),default_logits=rng.standard_normal((len(y),6)).astype(np.float32))


class Unread:
    def __getitem__(self,key):raise AssertionError('Forbidden P0/cache-teacher values accessed')


@contextmanager
def cache_fixture(root,monkeypatch):
    torch.set_num_threads(1);s=api();n=native_config(root);p,original,dura=identities(n)
    c=dict(run_id=s.RUN_ID,code_commit='a'*40,native_config=n,L_root=str(root/'cache/L'),OE_root=str(root/'cache/OE'),
        output_root=str(root/'features'),batch_size=128)
    roles={};created=[];events=[]
    for role in ('L','OE'):
        f=features(role);count=len(f['ids']);rng=np.random.default_rng(2)
        manifest=dict(schema='pamf_phase3_star_representation_cache_role_v1',status='STAR_REPRESENTATION_ROLE_COMPLETE',role=role,
            count=count,parent_F0_identity=p,dura_identity=dura,source_only=True,all_new_IQ_satellite=True,restored_input_path='P1')
        roles[role]=dict(manifest=manifest,ids=f['ids'],metadata=f['metadata'],received=Unread(),teacher_logits=Unread(),
            restored=rng.standard_normal((count,2,256)).astype(np.float32))
    def construct(native,device,residual_enabled):
        from experiments.pamf_phase3_open_world import star_protected_open_model as ow
        model,_=fixture();model._phase3_parent_identity=deepcopy(original);model._phase3_source_handle={'source_only':True}
        ow.install(model,residual_enabled);created.append(model);return model
    def load(path,role):
        assert (Path(c['output_root'])/'smoke.json').is_file();events.append(role);return roles[role]
    monkeypatch.setattr(s,'COUNTS',dict(L=360,OE=90));monkeypatch.setattr(s,'_construct',construct)
    monkeypatch.setattr(s,'_load_role',load);monkeypatch.setattr(s,'_close_role',lambda role:None)
    monkeypatch.setattr(s,'_synthetic',lambda device:torch.randn(2,2,256))
    yield s,c,roles,created,events


def test_config_rejects_private_roles_wrong_run_budget_and_overwrite_before_model(tmp_path,monkeypatch,subtests):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        assert s.validate_config(c)==c
        for field in ('truth','Q_public_root','novel_root','U_root'):
            with subtests.test(field=field),pytest.raises(ValueError):s.build(dict(c,**{field:'forbidden'}),'cpu')
        for change in (dict(batch_size=64),dict(run_id='old-run'),dict(output_root=c['L_root'])):
            with subtests.test(change=change),pytest.raises(ValueError):s.build(dict(c,**change),'cpu')
        Path(c['output_root']).mkdir()
        with pytest.raises(FileExistsError):s.build(c,'cpu')
        assert created==[] and events==[]


def test_build_smoke_precedes_P1_only_reads_and_exports_same_pass_frozen_features(tmp_path,monkeypatch):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        marker=s.build(c,'cpu');assert events==['L','OE']
        assert marker['status']==s.STATUS and marker['F_forward_rows']==450 and marker['synthetic_F_forward_rows']==2
        assert marker['model_native_state_unchanged'] and marker['target_reads']==marker['received_cache_values_read']==0
        for role in ('L','OE'):
            bundle=s.load_features(c['output_root'],role)
            assert set(bundle)=={'ids','metadata','raw_original_fused160','contributions','default_logits','manifest'}
            assert set(bundle['metadata'])==({'y','rx_i','day_i','domain'} if role=='L' else {'rx_i','day_i','domain'})
            np.testing.assert_array_equal(bundle['ids'],roles[role]['ids'])
            assert not bundle['raw_original_fused160'].flags.writeable
            r=created[0].id_backbone.encoder
            with torch.no_grad():expected=r.classify_features(torch.tensor(bundle['raw_original_fused160'].tolist()))
            np.testing.assert_array_equal(bundle['default_logits'],np.asarray(expected.tolist(),np.float32))
        assert all(not p.requires_grad for p in created[0].parameters())


def test_incomplete_and_corrupt_role_features_are_rejected_without_opening_other_role(tmp_path,monkeypatch,subtests):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        s.build(c,'cpu');root=Path(c['output_root']);(root/'OE.npz').unlink()
        assert s.load_features(root,'L')['manifest']['counts']==dict(L=360,OE=90)
        with np.load(root/'L.npz',allow_pickle=False) as z:clean={k:z[k].copy() for k in z.files}
        for defect in ('label','shape','nonfinite','domain','duplicate'):
            a={k:v.copy() for k,v in clean.items()}
            if defect=='label':a['tx']=np.asarray(['forbidden']*360)
            elif defect=='shape':a['contributions']=a['contributions'][:,:3]
            elif defect=='nonfinite':a['default_logits'][0,0]=np.nan
            elif defect=='domain':a['domain'][0]=14
            else:a['ids'][0]=a['ids'][1]
            np.savez(root/'L.npz',**a)
            with subtests.test(defect=defect),pytest.raises(ValueError):s.load_features(root,'L')
        with pytest.raises(ValueError):s.load_features(root,'V')
        (root/'complete.json').unlink()
        with pytest.raises(FileNotFoundError):s.load_features(root,'L')


def test_private_OE_metadata_aborts_completion(tmp_path,monkeypatch):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        roles['OE']['metadata']['y']=np.zeros(90,np.int64)
        with pytest.raises(ValueError):s.build(c,'cpu')
        assert not (Path(c['output_root'])/'complete.json').exists()


def test_changed_original_buffer_aborts_completion(tmp_path,monkeypatch):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        original=s._encode
        def changed(model,x,device):
            result=original(model,x,device)
            with torch.no_grad():model.id_backbone.encoder.source_buffer.add_(1.)
            return result
        monkeypatch.setattr(s,'_encode',changed)
        with pytest.raises(ValueError,match='buffers changed'):s.build(c,'cpu')
        assert not (Path(c['output_root'])/'complete.json').exists()


def test_feature_manifest_cannot_relabel_original_constructor_or_DURA(tmp_path,monkeypatch,subtests):
    with cache_fixture(tmp_path,monkeypatch) as (s,c,roles,created,events):
        marker=s.build(c,'cpu');path=Path(c['output_root'])/'complete.json'
        for defect in ('constructor','dura','source_status'):
            bad=deepcopy(marker)
            if defect=='constructor':bad['original_parent_identity']['source_cache']='different-original-cache'
            elif defect=='dura':bad['dura_identity']['updates']=1
            else:bad['source_manifests']['L']['status']='INCOMPLETE'
            path.write_text(json.dumps(bad),encoding='utf-8')
            with subtests.test(defect=defect),pytest.raises(ValueError):s.load_features(c['output_root'],'L')
