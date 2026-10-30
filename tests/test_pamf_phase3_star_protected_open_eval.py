"""Synthetic protected160 frozen source evaluation; no real IQ/checkpoints."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_protected_open_eval')


def config(root, arm='R1'):
    run = '20261030-phase3-star-protected-open-s2026092701-r01'
    native = dict(base_release='original-release', source_output='original-F0/source', source_contract='contract.json', model_seed=2026092701,
        run_id=run,code_commit='a'*40,row_id=arm+'-s2026092701',output_root=str(root/'train'/arm),batch_size=128,
        frozen_sources={k:k for k in ('F','AF','BASE','FOLD','G')},source_cache=str(root/'native-cache'),
        registered_classes=['14-10','14-7','20-15','20-19','6-15','8-20'])
    train = dict(run_id=run, code_commit='a'*40, arm=arm, native_config=native, feature_root=str(root/'features'),
                 output_root=str(root/'train'/arm), steps=2220, lr=.0001, sampling_seed=2026103017, slot_seed=2026103018)
    return dict(run_id=run, code_commit='a'*40, arm=arm, train_config=train, feature_root=train['feature_root'],
                cache_root=str(root/'cache'), output_root=str(root/'predict'/arm), batch_size=128)


class NeverP0:
    def __getitem__(self, key):
        raise AssertionError('Evaluation must not consume P0')


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(160, 6)
        with torch.no_grad():
            self.head.weight.zero_(); self.head.weight[:, :6] = 4*torch.eye(6); self.head.bias.fill_(-2.)

    def classify_features(self, x):
        return self.head(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.id_backbone = nn.Module(); self.id_backbone.encoder = Reader()
        self.eval().requires_grad_(False)


@contextmanager
def fixture(root, arm='R1', defect=None, novel_accept=False):
    e = module(); c = config(root, arm); events = []
    parent = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701,
                  **{k:c['train_config']['native_config'][k] for k in ('source_output','source_contract','base_release')})
    dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
    identity = dict(run_id=e.RUN_ID, code_commit=c['code_commit'], arm=arm, checkpoint=str(Path(c['train_config']['output_root'])/'final.pt'),
        adapter_updates=2220, original_parent_updates=11100, native_updates=0, parent_F0_identity=parent, dura_identity=dura,
        original_parent_identity=dict(parent_epoch=50, parent_updates=11100, model_seed=2026092701,
            **{k:c['train_config']['native_config'][k] for k in ('source_output','source_contract','base_release')}),
        identifier_parameters=299170, open_world_parameters=66177, training_container_parameters=966899,
        native_identity_frozen=True, identity_head_replaced=False, input_path='cached_P1_only', OE_member_supervision=arm!='N0',residual_enabled=arm!='R0')
    backend = dict(model=Model(), identity=identity, parent=parent, dura_identity=dura)
    cells = [(y,r,d) for y in range(6) for r in e.RECEIVERS for d in (1,2,3)]
    ly = np.asarray([x[0] for x in cells], np.int64)
    lr = np.asarray([x[1] for x in cells], np.int64); ld = np.asarray([x[2] for x in cells], np.int64)
    raw = np.eye(160, dtype=np.float32)[ly]
    global_manifest = dict(schema='pamf_phase3_protected_source_features_v1', status='STAR_PROTECTED_SOURCE_FEATURES_COMPLETE',
        parent_F0_identity=parent, dura_identity=dura, counts=dict(L=len(ly), OE=64), source_manifests={})
    features = dict(ids=np.asarray(['L-'+str(i) for i in range(len(ly))]), metadata=dict(y=ly, rx_i=lr, day_i=ld,
        domain=np.searchsorted(e.RECEIVERS,lr)*3+ld-1), raw_original_fused160=raw,
        contributions=np.repeat(raw[:,None,:]/4,4,axis=1), default_logits=raw[:,:6]*4-2, manifest=global_manifest)
    if defect == 'L_grid': features['metadata']['day_i'][:] = 1
    vrows = [(y,r,s) for y in range(6) for r in e.RECEIVERS for s in e.SCENES]
    nrows = [(y,r,s) for y in range(6,9) for r in e.RECEIVERS for s in e.SCENES]
    bundles = {}; truth = None
    for role, rows in (('V',vrows),('novel',nrows)):
        y = np.asarray([x[0] for x in rows], np.int64); rx = np.asarray([x[1] for x in rows], np.int64)
        day = np.arange(len(y),dtype=np.int64)%3+1; scenario = np.asarray([x[2] for x in rows])
        values = np.zeros((len(y),2,256),np.float32); values[:,0,0] = y
        meta = dict(rx_i=rx,day_i=day,domain=np.searchsorted(e.RECEIVERS,rx)*3+day-1)
        if role == 'V': meta.update(y=y,scenario=scenario)
        bundles[role] = dict(ids=np.asarray([role+'-'+str(i) for i in range(len(y))]),metadata=meta,restored=values,
            received=NeverP0(),manifest=dict(role=role,count=len(y),parent_F0_identity=parent,dura_identity=dura,
                source_only=True,all_new_IQ_satellite=True,restored_input_path='P1'))
        if role == 'V': bundles[role]['baseline_closed'] = np.column_stack((y,y)).astype(np.int64)
        else: truth = dict(ids=bundles[role]['ids'],tx=np.asarray([e.common.UNKNOWN_TX[x-6] for x in y]),rx_i=rx,day_i=day,scenario=scenario)
    if defect == 'baseline': bundles['V']['baseline_closed'][0,1] = 1
    if defect == 'novel_y': bundles['novel']['metadata']['y'] = np.zeros(len(nrows),np.int64)
    if defect == 'overlap': bundles['V']['ids'][0] = features['ids'][0]

    def load_features(path, role):
        assert events == ['smoke']; assert role == 'L'; events.append('L'); return features

    def load_role(cache, role):
        assert role in ('V','novel')
        out = Path(c['output_root'])
        assert (out/'source_aggregates.npz').is_file()
        if role == 'novel':
            assert all((out/f).is_file() for f in ('thresholds.npy','calibration/complete.json','known_predictions.npz'))
        events.append(role); return bundles[role]

    def encode(model, f, contributions):
        return dict(protected160=F.normalize(f,dim=1))

    def forward(model, x):
        y = x[:,0,0].long(); f = F.one_hot(y,num_classes=160).float()
        if defect == 'mutation' and events[-1:] == ['V']:
            with torch.no_grad(): model.id_backbone.encoder.head.bias.add_(.1)
        return dict(raw_original_fused160=f,protected160=F.normalize(f,dim=1),logits=model.id_backbone.encoder.classify_features(f))

    def relation(model,z,p):
        scores = (z[:,None,:]*p).sum(-1)*4-2
        return scores+4 if novel_accept and events[-1:] == ['novel'] else scores

    probe = e._probe
    def smoke(b,device):
        events.append('smoke')
        if defect == 'smoke': raise ValueError('synthetic smoke failure')
        return probe(b,device)

    with ExitStack() as stack:
        for name, value in (('_backend',lambda c,d:backend),('_load_features',load_features),
                            ('_load_role',load_role),('_close_role',lambda r:None),('_probe',smoke),
                            ('COUNTS',dict(L=len(ly),V=len(vrows),novel=len(nrows)))):
            stack.enter_context(patch.object(e,name,value))
        stack.enter_context(patch.object(e.model_api,'encode',encode))
        stack.enter_context(patch.object(e.model_api,'forward',forward))
        stack.enter_context(patch.object(e.model_api,'relation',relation))
        stack.enter_context(patch.object(e.model_api,'clear',lambda m:None))
        stack.enter_context(patch.object(e,'_synthetic',lambda d:torch.zeros((2,2,256))))
        yield e,c,backend,features,bundles,events,truth


def test_exact_source_only_config_rejects_extra_fields_and_input_output_overlap(tmp_path):
    with fixture(tmp_path) as (e,c,*_):
        assert e.validate_config(c) is c
        for key in ('truth','Q_public_root','U_root','unknown_count','G_checkpoint'):
            with pytest.raises(ValueError): e.validate_config(dict(c,**{key:'forbidden'}))
        for key,value in (('arm','best'),('batch_size',64),('feature_root','other'),('output_root',c['cache_root']+'/out')):
            with pytest.raises(ValueError): e.validate_config(dict(c,**{key:value}))


def test_aggregate_query_mask_matches_direct_source_support_and_requires_all_cells(tmp_path):
    from experiments.pamf_phase3_open_world.star_protected_open_loss import query_prototypes
    with fixture(tmp_path) as (e,c,b,f,*_):
        m=f['metadata']; z=f['raw_original_fused160']; a=e.aggregate_source(z,m['y'],m['rx_i'],m['day_i'])
        rx=np.asarray([1,3],np.int64);day=np.asarray([1,2],np.int64)
        p=e.query_prototypes(a,rx,day)
        expected=query_prototypes(torch.tensor(z),torch.tensor(m['y']),torch.tensor(m['rx_i']),torch.tensor(m['day_i']),torch.tensor(rx),torch.tensor(day))
        assert np.array_equal(p,expected.tolist())
        with pytest.raises(ValueError): e.aggregate_source(z,m['y'],m['rx_i'],np.ones(len(z),np.int64))


def test_frozen_checkpoint_smoke_L_aggregate_V_tau_then_novel_no_OE_or_P0(tmp_path):
    with fixture(tmp_path) as (e,c,b,f,roles,events,truth):
        done=e.evaluate(c,'cpu'); out=Path(c['output_root'])
        assert events == ['smoke','L','V','novel']
        assert done['native_updates']==0 and done['real_F_forward_rows']==270 and done['adapter_L_feature_rows']==90
        assert done['G_forward_rows']==0 and done['source_truth_reads']==done['target_reads']==0
        assert done['identity_predictions_unchanged'] and done['thresholds_frozen_before_novel']
        with np.load(out/'source_aggregates.npz') as z: assert set(z.files)==e.AGGREGATE_FIELDS
        with np.load(out/'known_predictions.npz') as z:
            assert set(z.files)==e.KNOWN_FIELDS and np.array_equal(z['closed'],roles['V']['baseline_closed'][:,1])
        with np.load(out/'novel_predictions.npz') as z:
            assert set(z.files)==e.NOVEL_FIELDS and not {'y','tx','scenario'}&set(z.files)
            with np.load(out/'novel_features.npz') as features:
                assert set(features.files)==e.FEATURE_FIELDS
                assert all(np.array_equal(features[k],z[k]) for k in ('ids','rx_i','day_i'))
                assert features['original_fused160'].shape==features['protected160'].shape==(90,160)
        with pytest.raises(FileExistsError): e.evaluate(c,'cpu')


def test_invalid_grid_baseline_role_overlap_smoke_or_state_never_complete(tmp_path):
    for defect in ('L_grid','baseline','novel_y','overlap','smoke','mutation'):
        with fixture(tmp_path/defect,defect=defect) as (e,c,b,f,roles,events,truth):
            with pytest.raises(ValueError): e.evaluate(c,'cpu')
            assert not (Path(c['output_root'])/'complete.json').exists()
            if defect != 'novel_y': assert 'novel' not in events


def test_all_arm_identity_and_native_frozen_fields_are_explicit(tmp_path):
    for arm in ('N0','R0','R1'):
        with fixture(tmp_path/arm,arm=arm) as (e,c,b,*_):
            assert e._identity(b['identity'],c)==b['parent']
            for key,value in (('native_updates',1),('adapter_updates',1),('native_identity_frozen',False),
                              ('identifier_parameters',315106),('OE_member_supervision',arm=='N0')):
                with pytest.raises(ValueError): e._identity(dict(b['identity'],**{key:value}),c)
