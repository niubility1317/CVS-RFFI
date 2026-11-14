import json
import pytest
from experiments.pamf_phase2_practical import clean_scale_takeover as takeover


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_partition_reuses_complete_and_refuses_partial(tmp_path, monkeypatch):
    root=tmp_path/'parent';control=root/'scheduler';(control/'logs').mkdir(parents=True)
    done=root/'tasks/done';done.mkdir(parents=True);(done/'complete.json').write_text('{}')
    entries=[dict(id='done',completion=str(done/'complete.json')),
             dict(id='new',completion=str(root/'tasks/new/complete.json'))]
    monkeypatch.setattr(takeover,'completion_matches',lambda e: e['id']=='done')
    completed,pending=takeover.partition(root, entries)
    assert [e['id'] for e in completed]==['done']
    assert [e['id'] for e in pending]==['new']
    (control/'logs/new.log').write_text('started')
    with pytest.raises(ValueError,match='partial'):
        takeover.partition(root, entries)
