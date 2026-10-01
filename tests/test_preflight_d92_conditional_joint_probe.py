"""Preflight transport and metadata-only boundaries; subprocess always mocked."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_prepare_d92_conditional_joint_probe import spec
import preflight_d92_conditional_joint_probe as preflight


def test_metadata_only_script_and_explicit_transport_and_exclusive_output(tmp_path):
    s=spec(3);script=preflight.remote_script(s)
    compile(script,'synthetic-preflight','exec')
    assert 'support_branch_features.npz' in script and 'np.load' not in script and 'torch.load' not in script
    assert 'query_truth' not in script and 'fit_trace.jsonl' not in script
    response=dict(status='VERIFIED',run_id=s['run_id'],query_access=False,source_sample_access=False,feature_values_read=False)
    call=Mock(return_value=SimpleNamespace(returncode=0,stdout=json.dumps(response),stderr=''))
    output=tmp_path/'preflight.json'
    args=dict(ssh_host='synthetic-host',ssh_config='explicit-config',remote_python='/python with space',output=output,run_fn=call)
    assert preflight.preflight(s,**args)==response
    assert call.call_args.args[0][0]=='ssh' and call.call_args.kwargs['input']==script
    assert preflight.shlex.split(call.call_args.args[0][-1])==['/python with space','-']
    with pytest.raises(FileExistsError):preflight.preflight(s,**args)
    assert call.call_count==1


def test_invalid_binding_rejected_before_transport(tmp_path):
    s=spec();s['probe']['query_access']=True;call=Mock()
    with pytest.raises(ValueError):preflight.preflight(s,ssh_host='synthetic',ssh_config='config',remote_python='python',output=tmp_path/'out',run_fn=call)
    call.assert_not_called()
