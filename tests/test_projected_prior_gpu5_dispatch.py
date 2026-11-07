import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.pamf_phase3_restoration import projected_prior_gpu5_dispatch as m


def test_exact_original_release_and_cuda_empty_required(tmp_path,monkeypatch):
    args=SimpleNamespace(root=tmp_path,legacy_release=tmp_path/'wrong',cache=tmp_path/'cache')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError):m.controller(args)
    args.legacy_release=tmp_path/'releases'/(m.RUN+'-release')
    args.legacy_release.mkdir(parents=True)
    (args.legacy_release/'release_commit.txt').write_text('wrong')
    with pytest.raises(ValueError):m.controller(args)


def test_old_owner_or_missing_retirement_rejected(tmp_path,monkeypatch):
    args=SimpleNamespace(root=tmp_path,legacy_release=tmp_path/'releases'/(m.RUN+'-release'),cache=tmp_path/'cache')
    args.legacy_release.mkdir(parents=True)
    (args.legacy_release/'release_commit.txt').write_text(m.SCIENCE_COMMIT)
    logs=tmp_path/'logs'/m.RUN;logs.mkdir(parents=True)
    (logs/'resource_handover_GPU5.json').write_text(json.dumps({'old_PID':123,'status':'VERIFIED_UNSTARTED_OWNER_RETIRED','scientific_commit':m.SCIENCE_COMMIT}))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    real_read=Path.read_bytes
    def fake_read(path):
        if path.as_posix().endswith('/proc/123/cmdline'):return (m.SCIENCE_MODULE+' run '+m.SCIENCE_COMMIT).encode()
        return real_read(path)
    monkeypatch.setattr(Path,'read_bytes',fake_read)
    with pytest.raises(ValueError,match='Old queue owner'):m.controller(args)
