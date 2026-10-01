"""Release capability/whitelist only; no Git, SSH, archive or launch executed."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest
from test_prepare_d92_margin_joint_probe import spec
import publish_d92_margin_joint_probe as publish


def test_whitelist_contains_margin_head_and_exact_projected_helper_dependencies():
    paths=set(publish.release_paths(spec()))
    assert {'code/cvsrffi/d92_margin_qp_head.py','code/cvsrffi/d92_margin_joint_local_ridge.py',
        'code/cvsrffi/d92_affine_joint_local_ridge.py',publish.SUMMARY}<=paths
    assert {path for path in paths if 'conditional' in path.lower()}=={
        'code/cvsrffi/d92_conditional_joint_local_ridge.py','code/cvsrffi/d92_conditional_affine_kernel.py'}
    assert 'code' not in paths and not any(p.endswith(('.npz','.pt','.pth')) for p in paths)
    assert not any('state_arrays' in p or 'handoff' in p for p in paths)


def test_missing_summary_cannot_be_published_or_launched(tmp_path):
    s=spec();path=tmp_path/s['spec_path'];path.parent.mkdir(parents=True);path.write_text(json.dumps(s),encoding='utf-8')
    state=publish.readiness(s,tmp_path)
    assert state['status']=='INCOMPLETE_RELEASE_CAPABILITY' and not state['launched'] and not state['independent_summary_available']
    with patch.object(publish.importlib.util,'spec_from_file_location') as loader:
        with pytest.raises(RuntimeError,match='incomplete'):publish.dispatch(s['spec_path'],tmp_path)
    loader.assert_not_called()


def test_declared_spec_path_must_match_publication_input(tmp_path):
    s=spec();path=tmp_path/'different.json';path.write_text(json.dumps(s),encoding='utf-8')
    with patch.object(publish.importlib.util,'spec_from_file_location') as loader:
        with pytest.raises(ValueError,match='declared spec_path'):publish.dispatch('different.json',tmp_path)
    loader.assert_not_called()


def test_declared_entry_bundle_imports_without_workspace_or_implicit_helper_fallback(tmp_path):
    """Only explicit source files, isolated interpreter; no real data or launch."""
    bundle=tmp_path/'isolated_bundle';bundle.mkdir()
    for name in publish.PATHS:
        source=publish.ROOT/name
        assert source.is_file(),name
        target=bundle/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    smoke=bundle/'import_smoke.py'
    smoke.write_text(
        'import importlib,json,pathlib,sys\n'
        'root=pathlib.Path(__file__).resolve().parent\n'
        'sys.path[:0]=[str(root/"code"),str(root/"tools")]\n'
        'names=["prepare_d92_margin_joint_probe","preflight_d92_margin_joint_probe",'
        '"run_d92_margin_joint_probe","publish_d92_margin_joint_probe","evaluate_d92_margin_joint_probe"]\n'
        'for name in names:\n'
        ' module=importlib.import_module(name)\n'
        ' assert pathlib.Path(module.__file__).resolve().is_relative_to(root),name\n'
        'for name,module in list(sys.modules.items()):\n'
        ' if name=="cvsrffi" or name.startswith("cvsrffi."):\n'
        '  assert pathlib.Path(module.__file__).resolve().is_relative_to(root),name\n'
        'from cvsrffi import d92_margin_joint_local_ridge as margin,d92_conditional_joint_local_ridge as projected\n'
        'assert margin.project_coordinates is projected.project_coordinates\n'
        'print(json.dumps({"imports":names,"network_or_launch":False}))\n',encoding='utf-8')
    result=subprocess.run([sys.executable,'-I','-X','utf8',str(smoke)],cwd=bundle,
        text=True,encoding='utf-8',capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    observed=json.loads(result.stdout)
    assert observed['network_or_launch'] is False and len(observed['imports'])==5


def test_complete_entry_files_do_not_claim_analysis_capability(tmp_path):
    s=spec()
    for relative in publish.release_paths(s):
        if relative in publish.ANALYSIS_PATHS:continue
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}',encoding='utf-8')
    state=publish.readiness(s,tmp_path)
    assert state['entry_bundle_available'] and not state['analysis_chain_available']
    assert set(state['missing'])==set(publish.ANALYSIS_PATHS)
    assert not state['launched']
