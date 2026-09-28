"""Stage barriers and full-matrix completion, without model/data access."""
import copy
import json
from pathlib import Path
import sys
import threading
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_d92_branch_support_probe as runner
import read_d92_run as reader
import publish_d92_branch_support_probe as publisher


def spec(tmp_path):
    out=tmp_path/'run';release=tmp_path/'release'
    value=dict(permissions=dict(query_use='none; query IQ/labels/truth/scores never read'),
        execution=dict(remote_run_root=str(out),cpu_lanes=4,blas_threads_per_lane=2,export_device='cuda:0',export_batch_size=32),
        code=dict(cwd=str(release)),probe=dict(query_access=False,native_code='native',source_contract='contract',source_receivers=['r0'],cohorts={}),rows=[])
    for cohort,count in [('rx3',900),('rx1',300)]:
        value['probe']['cohorts'][cohort]=dict(capsule='capsule-'+cohort,capsule_id='id-'+cohort,expected_split_count=count,evaluation_config='config-'+cohort)
        for seed in range(2026092701,2026092705):
            rid=cohort+'-'+str(seed)
            value['rows'].append(dict(row_id=rid,cohort=cohort,seeds=dict(model=seed),output_root=str(out/rid),
                source_root='source-'+str(seed),expected_checkpoint_sha256='a'*64,
                data_overrides=dict(capsule='capsule-'+cohort,capsule_id='id-'+cohort,expected_split_count=count)))
    return value


def fake_launcher(value,fail_row=None,bad_export=False):
    calls=[];lock=threading.Lock()
    def fake(argv,log,cwd,stage):
        out=Path(argv[argv.index('--output')+1]);rid=out.parent.name
        with lock:calls.append((rid,stage))
        if rid==fail_row and not bad_export:
            raise RuntimeError('synthetic technical failure')
        out.mkdir()
        row=next(r for r in value['rows'] if r['row_id']==rid);co=value['probe']['cohorts'][row['cohort']]
        marker=dict(status='BRANCH_SUPPORT_FEATURES_COMPLETE' if stage=='export' else 'SUPPORT_PROBE_COMPLETE',
            capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'])
        if stage=='export':
            marker.update(count=20,query_rows_read=1 if rid==fail_row and bad_export else 0)
        else:marker.update(episodes=co['expected_split_count'],query_rows_used=0,source_rows_used=0)
        runner.write(out/('features_complete.json' if stage=='export' else 'probe_complete.json'),marker)
    return fake,calls


def test_complete_requires_all_eight_bound_artifacts(tmp_path):
    s=spec(tmp_path);fn,calls=fake_launcher(s);runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='SUPPORT_PROBE_COMPLETE' and complete['episodes']==4800
    assert complete['completed_rows']==8 and len(calls)==16
    assert complete['query_access'] is False
    for row in s['rows']:
        assert calls.index((row['row_id'],'export'))<calls.index((row['row_id'],'probe'))
    with pytest.raises(FileExistsError):runner.run(s,'commit',launch_fn=fn)


@pytest.mark.parametrize('bad_marker',[False,True])
def test_failed_export_never_probes_or_retries_and_healthy_rows_finish(tmp_path,bad_marker):
    s=spec(tmp_path);rid=s['rows'][0]['row_id'];fn,calls=fake_launcher(s,rid,bad_marker)
    with pytest.raises(RuntimeError,match='One or more'):runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='FAILED' and complete['completed_rows']==7
    assert calls.count((rid,'export'))==1 and (rid,'probe') not in calls
    assert len([x for x in calls if x[1]=='probe'])==7


def test_commands_and_readback_never_add_query_or_ground_arguments(tmp_path):
    s=spec(tmp_path);export,probe=runner.commands(s,s['rows'][0]);payload=' '.join(export+probe)
    assert '--support-features' in probe and '--capsule' in export
    assert not any(x in payload for x in ('--truth','--ground','score_d92','predictions.jsonl'))
    assert '--source-contract' in export and '--expected-checkpoint-sha256' in export
    script=reader.readback_script(s);compile(script,'remote-readback','exec')
    assert "candidate_folder='probe'" in script and 'probe_complete.json' in script


def test_remote_archive_is_posix_on_windows():
    assert publisher.remote_archive_path('/home/user/releases/probe','probe.tar')=='/home/user/releases/probe.tar'
    with pytest.raises(ValueError):publisher.remote_archive_path('\\home\\user\\releases\\probe','probe.tar')


@pytest.mark.parametrize('mutation',[lambda s:s['rows'].pop(),lambda s:s['probe'].update(query_access=True),
    lambda s:s['rows'][0].update(output_root='/outside'),lambda s:s['rows'][0].update(expected_checkpoint_sha256='bad')])
def test_invalid_scope_rejected_before_output_creation(tmp_path,mutation):
    s=spec(tmp_path);mutation(s)
    with pytest.raises(ValueError):runner.run(s,'commit',launch_fn=lambda *x:None)
    assert not Path(s['execution']['remote_run_root']).exists()
