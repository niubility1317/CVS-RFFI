"""Isolated exact source bundle import and mocked publication plumbing only."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock,patch

import pytest

from test_run_d92_ground_a_packet_export import fixture,ROOT
import publish_d92_ground_a_packet_export as publisher
import publish_d92_branch_support_probe as existing


def write_spec(root,spec):
    path=root/spec['spec_path'];path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(spec),encoding='utf-8')


def test_minimal_explicit_whitelist_and_isolated_real_import():
    assert set(publisher.REMOTE_CODE_PATHS)=={'code/cvsrffi/__init__.py','code/cvsrffi/d92_ground_classifier_a.py',
        'tools/export_d92_ground_classifier_a_packet.py','tools/run_d92_ground_a_packet_export.py'}
    assert 'code' not in publisher.PATHS and not any(p.endswith(('.pth','.npz','.npy')) for p in publisher.PATHS)
    assert not any('conditional' in p or 'affine' in p or 'score_d92' in p for p in publisher.PATHS)
    result=publisher.verify_bundle_imports(ROOT)
    assert result['status']=='VERIFIED' and set(result['module_files'])==set(publisher.REMOTE_CODE_PATHS)
    assert 'NOT_REMOTE' in result['scope']


def test_existing_transport_preserved_except_cpu_supervisor(tmp_path):
    spec=fixture(tmp_path);transport=SimpleNamespace(REMOTE=existing.REMOTE)
    publisher.configure_transport(transport,spec,ROOT)
    assert 'nvidia-smi' not in transport.REMOTE and 'GPU0 occupied' not in transport.REMOTE
    assert 'tools/run_d92_ground_a_packet_export.py' in transport.REMOTE
    assert "if release.exists() or root.exists()" in transport.REMOTE
    assert "start_new_session=True" in transport.REMOTE and "CUDA_VISIBLE_DEVICES=''" in transport.REMOTE
    assert transport.PATHS==publisher.PATHS and transport.validate_spec is publisher.validate_spec
    for changed in (existing.REMOTE.replace('used=int','changed=int'),existing.REMOTE.replace('tools/run_d92_branch_support_probe.py','other.py')):
        with pytest.raises(RuntimeError):publisher.configure_transport(SimpleNamespace(REMOTE=changed),spec,ROOT)


def test_missing_dependency_blocks_before_import_or_any_publication(tmp_path):
    spec=fixture(tmp_path);write_spec(tmp_path,spec);check=Mock()
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader:
        with pytest.raises(RuntimeError,match='Incomplete'):publisher.dispatch(spec['spec_path'],tmp_path,import_check=check)
    check.assert_not_called();loader.assert_not_called()
    state=publisher.readiness(spec,tmp_path)
    assert state['missing'] and state['launched'] is False
    dependency=state['external_native_dependency']
    assert Path(dependency['code_ref'])==Path(spec['rows'][0]['native_code_ref'])
    assert dependency['copied_into_bundle'] is dependency['runtime_module_origins_verified'] is False
    assert not any('baseline_origin' in path or 'muse_ssdg' in path for path in state['paths'])


def test_dispatch_checks_exact_isolated_bundle_then_reuses_sole_transport(tmp_path):
    spec=fixture(tmp_path);write_spec(tmp_path,spec)
    for relative in publisher.PATHS:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    transport=SimpleNamespace(REMOTE=existing.REMOTE,main=Mock());loader=SimpleNamespace(exec_module=Mock())
    location=SimpleNamespace(loader=loader);check=Mock(return_value=dict(status='VERIFIED'))
    with patch.object(publisher.importlib.util,'spec_from_file_location',return_value=location), \
        patch.object(publisher.importlib.util,'module_from_spec',return_value=transport):
        publisher.dispatch(spec['spec_path'],tmp_path,import_check=check)
    check.assert_called_once_with(tmp_path);loader.exec_module.assert_called_once_with(transport);transport.main.assert_called_once_with()
    assert transport.ROOT==tmp_path and 'tools/run_d92_ground_a_packet_export.py' in transport.REMOTE
    check=Mock(return_value=dict(status='UNKNOWN'))
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader,pytest.raises(ValueError):
        publisher.dispatch(spec['spec_path'],tmp_path,import_check=check)
    loader.assert_not_called()


def test_publication_spec_path_binding_rejected_before_transport(tmp_path):
    spec=fixture(tmp_path);(tmp_path/'wrong.json').write_text(json.dumps(spec),encoding='utf-8')
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader,pytest.raises(ValueError,match='declared spec'):
        publisher.dispatch('wrong.json',tmp_path)
    loader.assert_not_called()
