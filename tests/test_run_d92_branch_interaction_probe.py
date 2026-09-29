"""Cached support diagnostics cannot export, score queries, or silently retry."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from test_run_d92_branch_support_probe import spec as old_spec, fake_launcher
import run_d92_branch_interaction_probe as runner


def spec(tmp_path):
    s = old_spec(tmp_path)
    s['execution'].update(export_device=None, export_batch_size=None)
    s['probe']['reuse_support_cache'] = True
    for row in s['rows']:
        row['support_features'] = str(tmp_path/'cache'/row['row_id'])
    return s


def test_all_eight_cached_rows_complete_without_export(tmp_path):
    s=spec(tmp_path);fn,calls=fake_launcher(s)
    runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='SUPPORT_PROBE_COMPLETE' and complete['episodes']==4800
    assert complete['completed_rows']==8 and len(calls)==8 and all(v[1]=='probe' for v in calls)
    assert complete['checkpoint_loaded'] is complete['gpu_use'] is False
    with pytest.raises(FileExistsError):runner.run(s,'commit',launch_fn=fn)


def test_failed_lane_preserved_and_no_retry(tmp_path):
    s=spec(tmp_path);rid=s['rows'][0]['row_id'];fn,calls=fake_launcher(s,rid)
    with pytest.raises(RuntimeError,match='lane failed'):runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='FAILED' and complete['completed_rows']==7 and calls.count((rid,'probe'))==1


@pytest.mark.parametrize('change',[
    lambda s:s['probe'].update(query_access=True), lambda s:s['probe'].update(reuse_support_cache=False),
    lambda s:s['execution'].update(export_device='cuda:0'),lambda s:s['rows'].pop(),
    lambda s:s['rows'][0].update(support_features=s['rows'][0]['output_root']),
    lambda s:s['rows'][0].update(output_root='/outside')])
def test_bad_scope_before_mutation(tmp_path,change):
    s=spec(tmp_path);change(s)
    with pytest.raises(ValueError):runner.run(s,'x',launch_fn=lambda *a:None)
    assert not Path(s['execution']['remote_run_root']).exists()


def test_cpu_command_and_readback(tmp_path):
    import read_d92_run
    s=spec(tmp_path);argv=runner.command(s,s['rows'][0])
    assert argv[argv.index('--support-features')+1]==s['rows'][0]['support_features']
    assert not any(v in ' '.join(argv) for v in ('--truth','--source','--device','--ground','export_d92'))
    text=read_d92_run.readback_script(s);compile(text,'readback','exec')
    assert "candidate_folder='probe'" in text


def test_cpu_publisher_transport_compiles_without_gpu_gate():
    # Separate process avoids mutating the imported publisher in other tests.
    import subprocess
    p=subprocess.run([sys.executable,'-c',
        "import sys;sys.path.insert(0,'tools');import publish_d92_branch_interaction_probe as p;"
        "s=p.transport.REMOTE.replace('CONFIG','{}');compile(s,'remote','exec');"
        "assert 'nvidia-smi' not in s;assert 'run_d92_branch_interaction_probe.py' in s"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
