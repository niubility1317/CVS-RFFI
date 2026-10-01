"""Release capability/whitelist only; no Git, SSH, archive or launch executed."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from test_prepare_d92_conditional_joint_probe import spec
import publish_d92_conditional_joint_probe as publish


def test_whitelist_contains_pure_conditional_core_helpers_and_independent_certificate():
    paths=set(publish.release_paths(spec()))
    assert {'code/cvsrffi/d92_conditional_affine_kernel.py','code/cvsrffi/d92_conditional_joint_local_ridge.py',
        'code/cvsrffi/d92_affine_joint_local_ridge.py','tools/d92_conditional_analysis_math.py',publish.SUMMARY}<=paths
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
