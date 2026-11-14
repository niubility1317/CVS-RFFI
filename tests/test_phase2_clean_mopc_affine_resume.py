import json
import pytest

from experiments.pamf_phase2_practical.clean_mopc_affine_resume import partition


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return str(path)


def test_partition_reuses_completed_other_methods_and_replaces_proto_version(tmp_path, monkeypatch):
    root = tmp_path/'old'; (root/'scheduler/logs').mkdir(parents=True)
    entries = []
    for name, group, profile in (('p','hybrid','protonet_cda'), ('d','hybrid','dadda_sda'),
                                 ('n','native_cil','mopc_hr')):
        out = root/'tasks'/name
        job = put(tmp_path/(name+'.json'), dict(profile=profile, methods=['mopc_hr'] if group=='hybrid' else [], output=str(out)))
        entry = dict(id=name, group=group, job=job, runtime='unused', completion=str(out/'complete.json'))
        entries.append(entry)
    for entry in entries[:2]:
        put(__import__('pathlib').Path(entry['completion']), {})
    monkeypatch.setattr('experiments.pamf_phase2_practical.clean_mopc_affine_resume.completion_matches', lambda entry: True)
    result = partition(root, entries, set())
    assert result['completed_other'] == [entries[1]]
    assert result['replace_proto'] == [entries[0]]
    assert result['pending_other'] == [entries[2]]


def test_unclassified_partial_cannot_be_silently_relaunched(tmp_path):
    root = tmp_path/'old'; (root/'scheduler/logs').mkdir(parents=True)
    (root/'scheduler/logs/d.log').touch()
    job = put(tmp_path/'d.json', dict(profile='dadda_sda', methods=['mopc_hr'], output=str(root/'tasks/d')))
    entry = dict(id='d', group='hybrid', job=job, runtime='unused', completion=str(root/'tasks/d/complete.json'))
    with pytest.raises(ValueError, match='partial'):
        partition(root, [entry], set())
