"""Exact isolated runtime bundle and mocked existing CPU publication plumbing."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock,patch

import pytest

from test_run_d92_ground_a_support import fixture,ROOT
import publish_d92_ground_a_support as publisher
import publish_d92_branch_support_probe as existing


def write_spec(root,spec):
    path=root/spec['spec_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(spec),encoding='utf-8')


def test_exact_seven_runtime_files_import_in_isolated_process():
    assert set(publisher.REMOTE_CODE_PATHS)=={'code/cvsrffi/__init__.py','code/cvsrffi/d92_ground_classifier_a.py',
        'tools/export_d92_ground_classifier_a_packet.py','tools/score_d92_ground_a_support.py',
        'tools/export_d92_branch_support_features.py','tools/cvs_native_artifacts.py','tools/run_d92_ground_a_support.py'}
    assert not any('baseline_origin' in path or 'muse_ssdg' in path or path.endswith(('.pth','.npz','.bin')) for path in publisher.PATHS)
    assert not any('conditional' in path or 'affine' in path or 'summarize_' in path for path in publisher.PATHS)
    result=publisher.verify_bundle_imports(ROOT)
    assert result['status']=='VERIFIED' and set(result['module_files'])==set(publisher.REMOTE_CODE_PATHS)
    assert 'NOT_REMOTE' in result['scope']


def test_cpu_transport_only_changes_gpu_guard_and_supervisor(tmp_path):
    spec=fixture(tmp_path);transport=SimpleNamespace(REMOTE=existing.REMOTE)
    publisher.configure_transport(transport,spec,ROOT)
    assert 'nvidia-smi' not in transport.REMOTE and 'GPU0 occupied' not in transport.REMOTE
    assert 'tools/run_d92_ground_a_support.py' in transport.REMOTE
    assert 'if release.exists() or root.exists()' in transport.REMOTE
    assert 'start_new_session=True' in transport.REMOTE and "CUDA_VISIBLE_DEVICES=''" in transport.REMOTE
    assert transport.PATHS==publisher.PATHS and transport.validate_spec is publisher.validate_spec
    for changed in (existing.REMOTE.replace('used=int','changed=int'),existing.REMOTE.replace('tools/run_d92_branch_support_probe.py','other.py')):
        with pytest.raises(RuntimeError):publisher.configure_transport(SimpleNamespace(REMOTE=changed),spec,ROOT)


def test_missing_dependency_stops_before_import_or_transport(tmp_path):
    spec=fixture(tmp_path);write_spec(tmp_path,spec);check=Mock()
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader:
        with pytest.raises(RuntimeError,match='Incomplete'):publisher.dispatch(spec['spec_path'],tmp_path,import_check=check)
    check.assert_not_called();loader.assert_not_called()
    state=publisher.readiness(spec,tmp_path)
    assert state['missing'] and state['launched'] is state['native_training_release_required'] is False


def test_dispatch_checks_isolated_bundle_then_sole_existing_transport(tmp_path):
    spec=fixture(tmp_path);write_spec(tmp_path,spec)
    for relative in publisher.PATHS:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    transport=SimpleNamespace(REMOTE=existing.REMOTE,main=Mock());loader=SimpleNamespace(exec_module=Mock())
    location=SimpleNamespace(loader=loader);check=Mock(return_value=dict(status='VERIFIED'))
    with patch.object(publisher.importlib.util,'spec_from_file_location',return_value=location), \
        patch.object(publisher.importlib.util,'module_from_spec',return_value=transport):
        publisher.dispatch(spec['spec_path'],tmp_path,import_check=check)
    check.assert_called_once_with(tmp_path);loader.exec_module.assert_called_once_with(transport);transport.main.assert_called_once_with()
    assert transport.ROOT==tmp_path and 'tools/run_d92_ground_a_support.py' in transport.REMOTE
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader,pytest.raises(ValueError):
        publisher.dispatch(spec['spec_path'],tmp_path,import_check=Mock(return_value=dict(status='UNKNOWN')))
    loader.assert_not_called()


def test_declared_spec_path_is_bound_before_transport(tmp_path):
    spec=fixture(tmp_path);(tmp_path/'wrong.json').write_text(json.dumps(spec),encoding='utf-8')
    with patch.object(publisher.importlib.util,'spec_from_file_location') as loader,pytest.raises(ValueError,match='declared spec'):
        publisher.dispatch('wrong.json',tmp_path)
    loader.assert_not_called()
