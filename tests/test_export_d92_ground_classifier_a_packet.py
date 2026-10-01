"""Synthetic native head/checkpoint/metadata only; no real source or target reads."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import export_d92_ground_classifier_a_packet as exporter
from test_d92_ground_classifier_a import original_head_class


def fixture(tmp_path,prefix='',scale=17.25,extra_state=None):
    generator=torch.Generator().manual_seed(394)
    weight=torch.randn(6,160,generator=generator,dtype=torch.float32)
    weight[0].zero_();weight[1]*=1e-8
    state={prefix+exporter.WEIGHT_KEY:weight,prefix+'encoder.unused':torch.ones(2)}
    if extra_state:state.update(extra_state)
    checkpoint=tmp_path/'synthetic.pth';torch.save(dict(model=state,args={'seed':4,'from_scratch':True}),checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    classes=['tx-z','tx-a','tx-q','tx-b','tx-m','tx-c']
    binding=dict(checkpoint_sha256=digest,checkpoint_weight_key=exporter.WEIGHT_KEY,ordered_classes=classes,
        scale=scale,norm_eps=1e-4,feature_contract=dict(checkpoint_sha256=digest,cache_key='z_id',
        source_tensor='feat_joint',representation='raw',dtype='float32',feature_dim=160),logit_corrections='none',
        class_row_order_source='Synthetic original loader explicit class-to-index contract',
        factory_scale_source='Synthetic factory original CosFaceHead.s declaration',
        corrections_source='Synthetic original factory: no correction branch')
    provenance=dict(checkpoint_sha256=digest,verdict='MATCHED_SOURCE_ONLY_SCRATCH',target_access_before_freeze=False,
        checkpoint_inheritance=[],classes=sorted(classes),model_seed=4,checkpoint_epoch=200,
        source_role_comparison='EXACT_MATCH')
    return checkpoint,weight,binding,provenance


def native_fixture(tmp_path):
    """Toy producer modules, deliberately unrelated to real production source."""
    native=tmp_path/'native-release'/'code';(native/'cvsrffi').mkdir(parents=True)
    (native/'baseline_origin_sat_view.py').write_text(
        'class SatViewStage:\n    def __init__(self):\n        self.synthetic = 1\n',encoding='utf-8')
    (native/'cvsrffi'/'muse_ssdg.py').write_text(
        'from baseline_origin_sat_view import SatViewStage\n'
        'class RC4Calibration:\n    def __init__(self):\n        self.synthetic = SatViewStage()\n',encoding='utf-8')
    return native


def native_payload(checkpoint,weight,binding,provenance,native):
    with exporter.native_checkpoint_types(native):
        first=importlib.import_module('baseline_origin_sat_view').SatViewStage()
        second=importlib.import_module('cvsrffi.muse_ssdg').RC4Calibration()
        torch.save(dict(model={exporter.WEIGHT_KEY:weight},native_stage=first,calibration=second),checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding['checkpoint_sha256']=binding['feature_contract']['checkpoint_sha256']=provenance['checkpoint_sha256']=digest


def test_explicit_native_types_checkpoint_preserves_bits_and_restores_namespace(tmp_path):
    checkpoint,weight,binding,provenance=fixture(tmp_path);native=native_fixture(tmp_path)
    package=importlib.import_module('cvsrffi');package_path=package.__path__;search=sys.path[:]
    native_payload(checkpoint,weight,binding,provenance,native)
    assert all(name not in sys.modules for name in exporter.NATIVE_TYPES)
    before=checkpoint.read_bytes();out=tmp_path/'packet'
    exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out,native_code=native)
    assert (out/exporter.WEIGHT_FILE).read_bytes()==bytes(weight.view(torch.uint8).reshape(-1).tolist())
    assert checkpoint.read_bytes()==before and sys.path==search and package.__path__ is package_path
    assert all(name not in sys.modules for name in exporter.NATIVE_TYPES)
    assert not hasattr(package,'muse_ssdg') and not list(native.rglob('*.pyc'))
    saved=json.loads((out/exporter.METADATA_FILE).read_text(encoding='utf-8'))
    resolution=saved['export_measurements']['native_type_resolution']
    assert resolution['resolved_native_code']==str(native.resolve())
    assert set(resolution['module_files'])==set(exporter.NATIVE_TYPES)
    assert exporter.load_packet(out).classes==tuple(binding['ordered_classes'])


def test_existing_correct_native_modules_preserved_across_nested_context(tmp_path):
    native=native_fixture(tmp_path);package=importlib.import_module('cvsrffi')
    with exporter.native_checkpoint_types(native):
        baseline=sys.modules['baseline_origin_sat_view'];calibration=sys.modules['cvsrffi.muse_ssdg']
        package_path=package.__path__;search=sys.path[:]
        with exporter.native_checkpoint_types(native):
            assert sys.modules['baseline_origin_sat_view'] is baseline
            assert sys.modules['cvsrffi.muse_ssdg'] is calibration
        assert sys.path==search and package.__path__ is package_path and package.muse_ssdg is calibration
    assert all(name not in sys.modules for name in exporter.NATIVE_TYPES)


@pytest.mark.parametrize('case',['missing_directory','missing_module','wrong_cached_origin','wrong_import_origin','missing_class','load_failure'])
def test_native_resolution_failure_preserved_and_namespace_restored(tmp_path,case):
    checkpoint,weight,binding,provenance=fixture(tmp_path);native=native_fixture(tmp_path)
    package=importlib.import_module('cvsrffi');package_path=package.__path__;search=sys.path[:]
    if case=='missing_directory':native=native/'absent'
    elif case=='missing_module':(native/'cvsrffi/muse_ssdg.py').unlink()
    elif case=='missing_class':(native/'cvsrffi/muse_ssdg.py').write_text('OTHER = 1\n',encoding='utf-8')
    wrong=ModuleType('cvsrffi.muse_ssdg');wrong.__file__=str(tmp_path/'outside.py')
    from contextlib import ExitStack
    out=tmp_path/'packet'
    with ExitStack() as stack:
        if case=='wrong_cached_origin':stack.enter_context(patch.dict(sys.modules,{'cvsrffi.muse_ssdg':wrong}))
        if case=='wrong_import_origin':
            from types import SimpleNamespace
            stack.enter_context(patch.object(exporter.PathFinder,'find_spec',return_value=SimpleNamespace(origin=wrong.__file__)))
        loader=stack.enter_context(patch.object(exporter.torch,'load',side_effect=RuntimeError('synthetic load failure')))
        with pytest.raises((ValueError,RuntimeError)):
            exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out,native_code=native)
        assert loader.call_count==(1 if case=='load_failure' else 0)
        if case=='wrong_cached_origin':assert sys.modules['cvsrffi.muse_ssdg'] is wrong
    assert sys.path==search and package.__path__ is package_path
    assert all(name not in sys.modules for name in exporter.NATIVE_TYPES)
    failure=json.loads((out/'export_failed.json').read_text(encoding='utf-8'))
    assert failure['phase']==('checkpoint_load' if case=='load_failure' else 'native_checkpoint_types')
    assert not (out/exporter.WEIGHT_FILE).exists()


@pytest.mark.parametrize('prefix,scale',[('',17.25),('module.',3.75)])
def test_export_and_load_native_prediction_oracle_and_measured_bytes(tmp_path,original_head_class,prefix,scale):
    checkpoint,weight,binding,provenance=fixture(tmp_path,prefix,scale)
    before=checkpoint.read_bytes();out=tmp_path/'packet'
    with patch.object(exporter.torch,'load',wraps=torch.load) as loader:
        marker=exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)
    assert loader.call_count==1 and loader.call_args.kwargs==dict(map_location='cpu',weights_only=False)
    assert checkpoint.read_bytes()==before
    head=exporter.load_packet(out);features=torch.randn(9,160,generator=torch.Generator().manual_seed(57))
    features[0].zero_();features[1]*=1e-8;features[2]*=100
    original=original_head_class(160,6,s=scale,m=.95).eval()
    with torch.no_grad():original.weight.copy_(weight);expected=original(features)
    torch.testing.assert_close(head.score(z_id=features,feature_contract=head.metadata.feature_contract),expected,rtol=0,atol=0)
    assert head.classes==tuple(binding['ordered_classes'])
    assert head.predict(z_id=features[:1],feature_contract=head.metadata.feature_contract)==('tx-z',)
    assert head.predict(z_id=features,feature_contract=head.metadata.feature_contract)==tuple(head.classes[i] for i in expected.argmax(1))
    assert marker['weight_numeric_bytes']==3840
    assert marker['packet_total_file_bytes']==sum(p.stat().st_size for p in out.iterdir())>3840
    assert marker['completion_file_bytes']==(out/'complete.json').stat().st_size
    assert marker==json.loads((out/'complete.json').read_text(encoding='utf-8'))
    saved=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    assert saved['weight']['actual_checkpoint_key']==prefix+exporter.WEIGHT_KEY
    assert saved['class_row_mapping']==[dict(row=i,class_id=name) for i,name in enumerate(binding['ordered_classes'])]
    assert saved['head_metadata']['scale']==scale and saved['head_metadata']['ordered_classes']!=provenance['classes']
    assert saved['export_measurements']['actual_A_evaluated'] is marker['actual_A_evaluated'] is False
    assert saved['export_measurements']['encoder_executed'] is False
    assert set(p.name for p in out.iterdir())=={'metadata.json','complete.json','head_weight.float32.bin'}
    with patch.object(exporter.torch,'load') as loader,pytest.raises(FileExistsError):
        exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)
    loader.assert_not_called()


@pytest.mark.parametrize('change',['missing_scale','scale_none','scale_nan','scale_default_unknown','missing_order','duplicate_classes',
    'wrong_class_set','unknown_order_source','wrong_key','eps','unknown_corrections','enabled_corrections','unknown_source',
    'target_contact','inheritance','provenance_identity','feature_identity','feature_normalized'])
def test_fail_closed_invalid_or_unknown_binding_before_checkpoint_load(tmp_path,change):
    checkpoint,weight,binding,provenance=fixture(tmp_path)
    if change=='missing_scale':binding.pop('scale')
    elif change=='scale_none':binding['scale']=None
    elif change=='scale_nan':binding['scale']=float('nan')
    elif change=='scale_default_unknown':binding['scale']=30.;binding['factory_scale_source']='UNKNOWN'
    elif change=='missing_order':binding.pop('ordered_classes')
    elif change=='duplicate_classes':binding['ordered_classes']=['a']*6
    elif change=='wrong_class_set':binding['ordered_classes'][0]='not-a-source-class'
    elif change=='unknown_order_source':binding['class_row_order_source']=''
    elif change=='wrong_key':binding['checkpoint_weight_key']='prototype'
    elif change=='eps':binding['norm_eps']=1e-12
    elif change=='unknown_corrections':binding['logit_corrections']='UNKNOWN'
    elif change=='enabled_corrections':binding['logit_corrections']='Dual'
    elif change=='unknown_source':provenance['verdict']='CHECKPOINT_PROVENANCE_UNVERIFIED'
    elif change=='target_contact':provenance['target_access_before_freeze']=True
    elif change=='inheritance':provenance['checkpoint_inheritance']=['previous']
    elif change=='provenance_identity':provenance['checkpoint_sha256']='b'*64
    elif change=='feature_identity':binding['feature_contract']['checkpoint_sha256']='b'*64
    elif change=='feature_normalized':binding['feature_contract']['representation']='unit'
    with patch.object(exporter.torch,'load') as loader,pytest.raises((ValueError,TypeError)):
        exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=tmp_path/'packet')
    loader.assert_not_called()
    assert not (tmp_path/'packet').exists()


@pytest.mark.parametrize('kind',['head_collision','other_collision','double_prefix','wrong_shape','wrong_dtype','nonfinite','alternate_key','alternate_container'])
def test_exact_key_and_prefix_ambiguity_no_fallback_and_failure_preserved(tmp_path,kind):
    checkpoint,weight,binding,provenance=fixture(tmp_path)
    state={exporter.WEIGHT_KEY:weight}
    if kind=='head_collision':state['module.'+exporter.WEIGHT_KEY]=weight.clone()
    elif kind=='other_collision':state.update({'unrelated':torch.ones(1),'module.unrelated':torch.zeros(1)})
    elif kind=='double_prefix':state={'module.module.'+exporter.WEIGHT_KEY:weight}
    elif kind=='wrong_shape':state[exporter.WEIGHT_KEY]=weight[:5]
    elif kind=='wrong_dtype':state[exporter.WEIGHT_KEY]=weight.double()
    elif kind=='nonfinite':state[exporter.WEIGHT_KEY]=torch.full((6,160),float('nan'))
    elif kind=='alternate_key':state={'cls_head.head.weight':weight,'ground_prototype':weight}
    payload={'state_dict' if kind=='alternate_container' else 'model':state}
    torch.save(payload,checkpoint);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding['checkpoint_sha256']=binding['feature_contract']['checkpoint_sha256']=provenance['checkpoint_sha256']=digest
    out=tmp_path/'packet'
    with pytest.raises(ValueError):exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)
    failure=json.loads((out/'export_failed.json').read_text(encoding='utf-8'))
    assert failure['phase']=='exact_head_extraction' and failure['status']=='GROUND_CLASSIFIER_A_PACKET_EXPORT_FAILED'
    assert not (out/'complete.json').exists()
    with pytest.raises(ValueError,match='Failed packet'):exporter.load_packet(out)
    with pytest.raises(FileExistsError):exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)


def test_actual_checkpoint_identity_checked_before_deserialization(tmp_path):
    checkpoint,weight,binding,provenance=fixture(tmp_path)
    checkpoint.write_bytes(checkpoint.read_bytes()+b'synthetic identity corruption')
    with patch.object(exporter.torch,'load') as loader,pytest.raises(ValueError,match='SHA256'):
        exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=tmp_path/'packet')
    loader.assert_not_called()
    assert json.loads((tmp_path/'packet/export_failed.json').read_text(encoding='utf-8'))['phase']=='checkpoint_identity'


def test_readback_rejects_damage_and_no_array_shared_abi(tmp_path):
    checkpoint,weight,binding,provenance=fixture(tmp_path)
    out=tmp_path/'packet';exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)
    raw=(out/exporter.WEIGHT_FILE).read_bytes()
    assert raw==bytes(weight.contiguous().view(torch.uint8).reshape(-1).tolist())
    (out/exporter.WEIGHT_FILE).write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='size mismatch'):exporter.load_packet(out)
    source=Path(exporter.__file__).read_text(encoding='utf-8')
    assert 'from_numpy' not in source and '.numpy(' not in source and 'build_baseline_model' not in source


def test_readback_failure_retains_completed_bytes_and_failure_status(tmp_path):
    checkpoint,weight,binding,provenance=fixture(tmp_path);out=tmp_path/'packet'
    with patch.object(exporter,'load_packet',side_effect=ValueError('synthetic readback failure')),pytest.raises(ValueError):
        exporter.export_packet(checkpoint=checkpoint,binding=binding,provenance=provenance,output=out)
    assert (out/exporter.WEIGHT_FILE).stat().st_size==3840
    assert (out/'metadata.json').exists() and (out/'complete.json').exists()
    failure=json.loads((out/'export_failed.json').read_text(encoding='utf-8'))
    assert failure['phase']=='packet_readback'
    with pytest.raises(ValueError,match='Failed packet'):exporter.load_packet(out)
