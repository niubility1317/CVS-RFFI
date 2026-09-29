import contextlib
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import collect_d92_adaptation_registration_identities as mod


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path):
    specs = []
    for cohort in ('rx3', 'rx1'):
        base = tmp_path/cohort
        cap, run = base/'capsule', base/'run'
        old = [f'old{i}' for i in range(6)]
        classes = old+['new0','new1']
        ids = [f'{cohort}-s{i}' for i in range(8)]+[f'{cohort}-q{i}' for i in range(8)]
        cap.mkdir(parents=True)
        np.savez(cap/'received.npz', ids=np.array(ids), iq=np.zeros((16,2,256)))
        save(cap/'manifest.json', dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',
             capsule_id=cohort,split_count=2,received_count=16))
        for sid, c in (('before',6),('after',8)):
            save(cap/'splits'/f'{sid}.json', dict(split_id=sid,capsule_id=cohort,receiver=cohort,
                 scenario='scene',k=1,support_seed=7,registered_classes=classes[:c],
                 support_indices=list(range(c)),support_labels=list(range(c)),query_indices=list(range(8,8+c))))
        rows = []
        for seed in range(4):
            origin = base/f'original{seed}'
            sha, source = str(seed)*64, f'/source/{seed}'
            rows.append(dict(row_id=f'm{seed}',seeds=dict(model=seed),expected_checkpoint_sha256=sha,
                             source_root=source,reuse_row_root=str(origin)))
            save(origin/'d92_startup.json',dict(checkpoint_sha256=sha,seed=seed,capsule=str(cap),
                 query_fit_access=False,truth_read=False))
            save(origin/'ground_provenance.json',dict(checkpoint_sha256=sha,source_root=source))
            save(origin/'predictions_complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id=cohort,
                 split_count=2,truth_read=False,predictions=3))
            records = [dict(mode='d92_registration',split_id=sid) for sid in ('before','after')]
            records.append(dict(mode='frozen_dg',split_id='before',capsule_id=cohort,receiver=cohort,
                 scenario='scene',classes=old,query_ids=ids[8:14],k=1,support_seed=7,
                 scores='NUMERIC_PAYLOAD_NOT_USED',predicted_indices='NOT_EXPORTED'))
            (origin/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        spec = dict(run_id=cohort,execution=dict(remote_run_root=str(run)),rows=rows,
             data=dict(capsule_id=cohort,target_receivers=[cohort],scenarios=['scene']),
             confirmation=dict(capsule=str(cap),expected_split_count=2,reuse_validated_capsule_id=cohort,
                               candidate_method='D92-BranchLocalRidge-v1',old_classes=old))
        save(run/'startup.json',dict(spec=spec,commit='fixed'))
        save(run/'complete.json',dict(status='SCORED',selection_feedback_forbidden=True,commit='fixed',
                                     model_rows=4,records=20))
        specs.append(spec)
    return specs


def change_json(path, **updates):
    value = mod.read(path)
    value.update(updates)
    save(path,value)


def test_projection_only_accesses_ids_member_and_exports_identity(tmp_path, monkeypatch):
    specs = fixture(tmp_path)
    original = np.load
    members = []
    class IdOnly:
        def __init__(self,path,**kwargs): self.value=original(path,**kwargs)
        def __enter__(self): return self
        def __exit__(self,*args): self.value.close()
        def __getitem__(self,key):
            assert key == 'ids'
            members.append(key)
            return self.value[key]
    monkeypatch.setattr(np,'load',IdOnly)
    result = mod.collect_all(specs)
    assert members == ['ids','ids']
    assert len(result) == 2 and all(len(v['models']) == 4 for v in result)
    assert result[0]['models'][0]['splits'][0]['support_ids']
    raw = json.dumps(result)
    assert all(v not in raw for v in ('NUMERIC_PAYLOAD_NOT_USED','NOT_EXPORTED','predicted_indices','scores','truth'))
    assert result[0]['models'][0]['frozen_dg'][0]['query_ids'] == [f'rx3-q{i}' for i in range(6)]


def test_both_terminals_checked_before_any_projection(tmp_path, monkeypatch):
    specs = fixture(tmp_path)
    change_json(Path(specs[1]['execution']['remote_run_root'])/'complete.json',status='RUNNING')
    monkeypatch.setattr(mod,'collect_identity',lambda *_: pytest.fail('Projection before full completion'))
    with pytest.raises(ValueError,match='completed'): mod.collect_all(specs)


@pytest.mark.parametrize('mutation', ['checkpoint','source','seed','capsule','count','index','dg_query','dg_duplicate','main_duplicate'])
def test_wrong_identity_or_coverage_rejected(tmp_path, mutation):
    specs = fixture(tmp_path)
    spec = specs[0]
    origin = Path(spec['rows'][0]['reuse_row_root'])
    cap = Path(spec['confirmation']['capsule'])
    if mutation=='checkpoint': change_json(origin/'d92_startup.json',checkpoint_sha256='x'*64)
    elif mutation=='source': change_json(origin/'ground_provenance.json',source_root='/other')
    elif mutation=='seed': change_json(origin/'d92_startup.json',seed=99)
    elif mutation=='capsule': change_json(cap/'manifest.json',capsule_id='other')
    elif mutation=='count': change_json(origin/'predictions_complete.json',predictions=1)
    elif mutation=='index': change_json(cap/'splits/after.json',query_indices=[-1])
    else:
        path=origin/'predictions.jsonl'
        records=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        if mutation=='dg_query': records[-1]['query_ids']=['other']
        elif mutation=='dg_duplicate': records.append(copy.deepcopy(records[-1]))
        else: records[1]=copy.deepcopy(records[0])
        path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    with pytest.raises(ValueError): mod.collect_all(specs)


def test_generated_remote_payload_is_self_contained(tmp_path):
    specs=fixture(tmp_path)
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(mod.remote_script(specs),'remote_projection','exec'),{})
    values=json.loads(stream.getvalue())
    assert [v['run_id'] for v in values]==['rx3','rx1']
    assert values==mod.collect_all(specs)
