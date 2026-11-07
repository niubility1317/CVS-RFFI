import json
import pytest
from experiments.pamf_phase3_restoration import low_rank_source_probe_shared as m


def test_resource_route_uses_only_existing_gpu6_exception(monkeypatch):
    called=[]
    def choice(**kwargs):called.append(kwargs);return []
    monkeypatch.setattr(m.method,'recovery_gpu_choice',choice)
    assert m.resource_choice()==[]
    assert called==[dict(allowed_gpus=(6,),maximum_processes=5)]


def test_started_or_live_previous_probe_prevents_duplicate(tmp_path):
    old=tmp_path/'runs'/m.PREVIOUS;old.mkdir(parents=True)
    (old/'launch.json').write_text(json.dumps(dict(pid=123)),encoding='utf-8')
    proc=tmp_path/'proc/123';proc.mkdir(parents=True)
    (proc/'cmdline').write_bytes((m.OLD_MODULE+' run --inputs '+m.PREVIOUS+'/inputs.json').encode())
    with pytest.raises(ValueError):m.previous_unstarted(tmp_path,tmp_path/'proc')
    (proc/'cmdline').write_bytes(b'an unrelated process')
    m.previous_unstarted(tmp_path,tmp_path/'proc')
    (old/'submitted.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError):m.previous_unstarted(tmp_path,tmp_path/'proc')
