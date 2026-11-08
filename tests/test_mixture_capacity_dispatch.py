import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.pamf_phase3_restoration import mixture_capacity_dispatch as d


def test_only_exact_unstarted_retired_owner_can_be_adopted(tmp_path):
    release=tmp_path/'release'; release.mkdir(); (release/'release_commit.txt').write_text(d.SCIENCE_COMMIT)
    out=tmp_path/'run';out.mkdir();(out/'rows'/d.ARM/'prediction').mkdir(parents=True)
    (out/'launch.json').write_text(json.dumps({'pid':42,'CWD':str(release),'code_commit':d.SCIENCE_COMMIT}))
    proc=tmp_path/'proc';(proc/'42').mkdir(parents=True);(proc/'42'/'cmdline').write_bytes(b'live')
    with pytest.raises(ValueError):d.verify_unstarted_retired(out,release,proc)
    (proc/'42'/'cmdline').unlink();d.verify_unstarted_retired(out,release,proc)
    (out/'submitted-test.json').write_text('{}')
    with pytest.raises(ValueError):d.verify_unstarted_retired(out,release,proc)


def test_stage_command_and_gpu_cap():
    args=SimpleNamespace(root=Path('/root'),legacy_release=Path('/release'),cache=Path('/cache'))
    for stage in ('train','cross_rx','cross_both','score'):
        cmd=d.stage_command(args,stage)
        assert d.SCIENCE_MODULE in cmd and d.SCIENCE_COMMIT in cmd
        assert ('--task' in cmd)==(stage in ('cross_rx','cross_both'))
    assert not d.cap_exceeded(5,{1,2,5})
    assert d.cap_exceeded(5,{1,2,3,5})
    assert not d.cap_exceeded(5,{1,2,3,4})


def test_feature_wrapper_uses_its_original_science(monkeypatch):
    from experiments.pamf_phase3_restoration import feature_response_capacity_dispatch as f
    old = (d.RUN, d.ARM, d.SCIENCE_MODULE, d.SCIENCE_COMMIT)
    try:
        f.install()
        assert (d.RUN, d.ARM, d.SCIENCE_MODULE, d.SCIENCE_COMMIT) == (f.RUN, f.ARM, f.SCIENCE_MODULE, f.SCIENCE_COMMIT)
        args = SimpleNamespace(root=Path('/root'), legacy_release=Path('/release'), cache=Path('/cache'))
        for stage in d.STAGES:
            cmd = d.stage_command(args, stage)
            assert f.SCIENCE_MODULE in cmd and f.SCIENCE_COMMIT in cmd and f.ARM in cmd
    finally:
        d.RUN, d.ARM, d.SCIENCE_MODULE, d.SCIENCE_COMMIT = old
