"""Relocated source21 export routing; mocked F, synthetic IQ only."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pamf_phase3_open_world import source_open_export as source
from experiments.pamf_phase3_open_world import source_open_run as parent


def exporter():
    return importlib.import_module('experiments.pamf_phase3_open_world.source_diverse_export')


def write(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding='utf-8')


@pytest.fixture
def case(tmp_path, monkeypatch):
    module = exporter()
    monkeypatch.setattr(module, 'COUNT', 15)
    parent_root = tmp_path / '20261029-phase3-source-open-readout-s2026092701-r01'
    classes = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
    contract_path = tmp_path / 'contract.json'
    contract = dict(classes=classes, counts=dict(L_s=6300, V=27000), source_rxs=[1, 3, 4, 6, 8],
                    source_days=[1, 2, 3], role_ids=dict(L_s=['legal-L0', 'legal-L1']))
    write(contract_path, contract)
    pc = dict(run_id=parent_root.name, row_id='source-open-readout-s2026092701', model_seed=2026092701,
        base_release=str(tmp_path / 'base'), frozen_sources={key: str(tmp_path / key) for key in ['F', 'AF', 'BASE', 'FOLD', 'G']},
        source_output=str(tmp_path / 'source-model'), source_contract=str(contract_path), registered_classes=classes,
        code_commit='a' * 40, batch_size=128, output_root=str(parent_root / 'export'), source_cache=str(tmp_path / 'cache'),
        retention=.99, oe_data_root=str(tmp_path / 'old-source-OE'))
    pm = dict(status='SOURCE_OPEN_FEATURES_EXPORTED', source_only=True, target_reads=0, optimizer_updates=0,
        original_U_s_reads=0, OE_truth_reads=0, head_changed=False, model_classes=6, parameters=232993,
        counts=parent.COUNTS.copy(), feature_refs=source.FEATURE_REFS.copy(), config=pc,
        source_OE_handles={role:dict(count=parent.COUNTS[role],capsule_id='parent-'+role,split_id='parent-'+role)
                           for role in ['exposure_train', 'novel_validation']})
    write(parent_root / 'export/complete.json', pm)
    data = tmp_path / 'relocated-source21'
    pub = data / 'exposure_train/public'; pub.mkdir(parents=True)
    np.save(pub / 'iq.npy', np.full((15, 2, 256), 7., np.float32))
    np.save(pub / 'ids.npy', np.asarray([f'oe{i}' for i in range(15)]))
    rx = np.repeat([1, 3, 4, 6, 8], 3); days = np.tile([1, 2, 3], 5)
    np.savez(pub / 'metadata.npz', rx_i=rx, day_i=days, domain=np.arange(15))
    handle = dict(public_root='E:/retired-builder/source21/public', truth_path='E:/retired-builder/private/truth.npz',
                  count=15, capsule_id=module.CAPSULE_ID, split_id=module.SPLIT_ID,
                  data_status='VALIDATED_ONCE', iq_shape=[15, 2, 256], iq_dtype='float32')
    cm = dict(status='SOURCE_OE_DIVERSE_DATA_VALIDATED', data_status='VALIDATED_ONCE', count=15,
        counts=dict(exposure_train=15), exposure_train_count=15, exposure_train=handle,
        source_domains=dict(receiver_names=source.RX_NAMES,receiver_indices=source.MAIN_RX,
            dates=source.DATES,day_indices=[1,2,3],domain_count=15), source_only=True,
        **{key:0 for key in ['target_reads','original_L_s_reads','original_U_s_reads','original_V_reads',
          'novel_validation_IQ_reads','model_reads','model_predictions','optimizer_updates','channel_observations_generated']})
    write(data / 'complete.json', cm)
    config = dict(run_id='source-diverse-test', row_id='source-diverse-test-row', code_commit='b' * 40,
                  parent_run_root=str(parent_root), diverse_data_root=str(data), output_root=str(tmp_path / 'export'))
    return dict(config=config, parent=pm, parent_root=parent_root, data=data, public=pub, complete=cm, output=tmp_path/'export')


def fake_model(monkeypatch, case):
    import torch
    module = exporter(); events=[]; reader=SimpleNamespace(tag='actual-frozen-complete-six-class')
    def load(config, contract, device):
        assert config['code_commit']=='a'*40  # Old cache identity is not relabelled as new execution code.
        assert config['source_cache']==case['parent']['config']['source_cache']
        events.append(('load',device)); return reader
    def dataset(config,role,contract):
        assert role=='L_s', 'No V, U_s or old OE dataset may be opened'
        events.append(('source-role',role)); return SimpleNamespace(x=torch.full((2,2,256),3.,dtype=torch.float32))
    def meta(dataset,role,contract):
        assert role=='L_s';return dict(ids=np.asarray(['legal-L0','legal-L1']))
    def capture(model,x):
        assert model is reader and torch.is_tensor(x) and tuple(x.shape[1:])==(2,256)
        assert bool(torch.isfinite(x).all())
        events.append(('forward',len(x),float(x[0,0,0])))
        values={name:torch.full((len(x),d),float(x[0,0,0]),dtype=torch.float32) for name,d in source.DIMENSIONS.items()}
        values['reference29'][:,27]=0
        return values,torch.arange(6,dtype=torch.float32)[None,:].repeat(len(x),1)+100.
    monkeypatch.setattr(module.runner,'_load_model',load)
    monkeypatch.setattr(module.runner,'_source_dataset',dataset)
    monkeypatch.setattr(module.source,'_source_meta',meta)
    monkeypatch.setattr(module.source,'_validate_model',lambda model:events.append(('validate',model.tag)))
    monkeypatch.setattr(module.source,'extract_branches',capture)
    original=np.load
    def guard(path,*args,**kwargs):
        text=str(path)
        assert 'truth' not in text and 'retired-builder' not in text
        if Path(path).parent==case['public']:
            assert any(v[0]=='forward' and v[1]==2 and v[2]==3. for v in events), 'OE opened before real L IQ smoke'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(module.np,'load',guard)
    return events


def test_relocated_public_path_real_l_smoke_then_only_new_oe(case,monkeypatch):
    events=fake_model(monkeypatch,case)
    result=exporter().export(case['config'],'cpu')
    assert [v for v in events if v[0]=='forward']==[('forward',2,3.),('forward',15,7.)]
    assert [v for v in events if v[0]=='source-role']==[('source-role','L_s')]
    assert result['parent_export_identity']==parent._identity(case['parent'])
    assert result['code_commit']=='b'*40 and result['parent_export_identity']['model']['code_commit']=='a'*40
    assert result['status']=='SOURCE_DIVERSE_OE_FEATURES_EXPORTED' and result['count']==15
    assert result['model_forward_samples']==17 and result['source_L_smoke_samples']==2
    assert result['source_OE_handle']=={k:case['complete']['exposure_train'][k] for k in ['capsule_id','split_id','count']}
    with np.load(case['output']/result['feature_ref'],allow_pickle=False) as stored:
        assert set(stored.files)==parent.COMMON
        assert stored['logits'].shape==(15,6)
        np.testing.assert_array_equal(stored['logits'][0],np.arange(6)+100.)
        for name,d in source.DIMENSIONS.items():assert stored['features__'+name].shape==(15,d)
        assert np.all(stored['features__reference29'][:,27]==0)
    smoke=json.loads((case['output']/'smoke.json').read_text())
    assert smoke['actual_checkpoint'] is True and smoke['source_samples']==2 and smoke['OE_reads']==0
    assert result['OE_truth_reads']==result['original_U_s_reads']==result['target_reads']==0


@pytest.mark.parametrize('fault',['old_status','count','capsule','split','domain','public_tx'])
def test_bad_new_data_never_reaches_oe_model_forward(case,monkeypatch,fault):
    events=fake_model(monkeypatch,case)
    cm=case['complete']
    if fault=='old_status':cm['status']='SOURCE_OE_DATA_VALIDATED'
    elif fault=='count':cm['count']=16
    elif fault=='capsule':cm['exposure_train']['capsule_id']='other'
    elif fault=='split':cm['exposure_train']['split_id']='other'
    elif fault=='domain':
        np.savez(case['public']/'metadata.npz',rx_i=np.repeat([1,3,4,6,8],3),day_i=np.tile([1,2,3],5),domain=np.ones(15,dtype=np.int64))
    else:
        np.savez(case['public']/'metadata.npz',rx_i=np.repeat([1,3,4,6,8],3),day_i=np.tile([1,2,3],5),domain=np.arange(15),tx=np.asarray(['forbidden']*15))
    write(case['data']/'complete.json',cm)
    with pytest.raises(ValueError):exporter().export(case['config'],'cpu')
    assert [v for v in events if v[0]=='forward']==[('forward',2,3.)]
    assert not (case['output']/'complete.json').exists()


def test_output_cannot_be_overwritten_before_any_load(case,monkeypatch):
    case['output'].mkdir();(case['output']/'existing').write_text('keep')
    monkeypatch.setattr(exporter().parent,'_export_marker',lambda root:pytest.fail('existing output must fail first'))
    with pytest.raises(FileExistsError):exporter().export(case['config'],'cpu')
    assert (case['output']/'existing').read_text()=='keep'


def test_only_declared_new_config_and_original_parent_are_allowed(case,monkeypatch):
    wrong=dict(case['config'],target_root='forbidden')
    with pytest.raises(ValueError):exporter().validate_config(wrong)
    cm=case['parent'];cm['config']['run_id']='different-parent'
    write(case['parent_root']/'export/complete.json',cm)
    monkeypatch.setattr(exporter().runner,'_load_model',lambda *a:pytest.fail('wrong parent must fail before weight load'))
    with pytest.raises(ValueError):exporter().export(case['config'],'cpu')
