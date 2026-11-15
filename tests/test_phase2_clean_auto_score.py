import json
import pytest

from experiments.pamf_phase2_practical.clean_auto_score import expected_units, coverage, merge_rows


def test_global_barrier_requires_every_method_and_every_kn():
    task = dict(capsule='capsules/P01/MIX01/1-1', seed=2026092701,
                points=[[1, 2], [5, 2]])
    expected = expected_units([task])
    assert len(expected) == 17
    proto = next(u for u in expected if u[:2] == ('hybrid', 'protonet_cda'))
    assert ('protonet_cda__mopc_hr_affine_v1', 'C', 5, 2, 'P0') in expected[proto]
    assert ('protonet_cda__mopc_hr', 'C', 5, 2, 'P0') not in expected[proto]
    available = {unit: set(rows) for unit, rows in expected.items()}
    assert coverage(expected, available)['ready']
    hybrid = next(unit for unit in expected if unit[0] == 'hybrid')
    available[hybrid].remove(next(iter(available[hybrid])))
    assert not coverage(expected, available)['ready']
    del available[hybrid]
    assert not coverage(expected, available)['ready']


def test_stabilized_scoring_never_mixes_original_and_repaired_mopc(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_auto_score as auto
    task=dict(capsule='capsules/P01/MIX01/1-1',seed=2026092701,points=[[5,20]],
              profile='dadda_sda',methods=['mopc_hr'])
    plans=tmp_path/'plans';entries=[]
    for folder,version in (('20261115-phase2-clean-original','none'),('mopc-stable-r01','clip1000_v1')):
        root=plans/folder;root.mkdir(parents=True)
        job=root/'job.json';job.write_text(json.dumps(dict(task,mopc_stabilization=version)))
        done=root/'complete.json';done.write_text('{}')
        entry=dict(id=folder,group='hybrid',job=str(job),runtime='unused',completion=str(done))
        (root/'scheduler.json').write_text(json.dumps(dict(input_condition='clean_ground_pre_satellite_normalized',jobs=[entry])))
        entries.append(entry)
    monkeypatch.setattr(auto,'completion_matches',lambda _:True)
    u=auto.unit('hybrid',task)
    old=auto.expected_units([task]);stable=auto.expected_units([task],'clip1000_v1')
    old_rows,old_entries=auto.collect(plans,old)
    stable_rows,stable_entries=auto.collect(plans,stable,mopc_stabilization='clip1000_v1')
    assert [x['id'] for x in old_entries[u]]==['20261115-phase2-clean-original']
    assert [x['id'] for x in stable_entries[u]]==['mopc-stable-r01']
    assert ('dadda_sda__mopc_hr','C',5,20,'P0') in old_rows[u]
    assert ('dadda_sda__mopc_hr_clip1000_v1','C',5,20,'P0') in stable_rows[u]


def test_conflicting_duplicate_predictions_do_not_silently_score(tmp_path):
    import numpy as np
    a = tmp_path / 'a.npz'; b = tmp_path / 'b.npz'
    np.savez(a, query_indices=[1], predicted_indices=[0])
    np.savez(b, query_indices=[1], predicted_indices=[1])
    row = dict(arm='mopc_hr', stage='C', k=5, new_count=2, input_path='P0',
               split_id='s', registered_classes=[0], prediction=str(a))
    other = dict(row, prediction=str(b))
    with pytest.raises(ValueError, match='Conflicting'):
        merge_rows([row, other])


def test_all_units_are_validated_before_any_truth_is_opened(monkeypatch):
    from experiments.pamf_phase2_practical import clean_auto_score as auto
    seen = []
    def fake_assemble(payload):
        seen.append(payload[0])
        if payload[0] == 'bad':
            raise ValueError('invalid later prediction')
    monkeypatch.setattr(auto, 'assemble_unit', fake_assemble)
    monkeypatch.setattr(auto, 'validate_prediction_rows', lambda item: None)
    with pytest.raises(ValueError, match='later prediction'):
        auto.validate_before_truth({'good': set(), 'bad': set()}, {'good': [], 'bad': []})
    assert seen == ['good', 'bad']


def test_prediction_preflight_rejects_wrong_query_without_truth(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.pamf_phase2_practical.clean_auto_score import validate_prediction_rows
    cap = tmp_path / 'data/capsules/P01/MIX01/1-1'
    (cap / 'splits').mkdir(parents=True)
    (cap / 'stage_rows').mkdir()
    spec = dict(split_id='abc', k=1, new_count=2, registered_classes=list(range(8)),
                support_indices=[0, 2, 4, 6, 8, 10, 12, 14], query_count=8)
    (cap / 'splits/abc.json').write_text(json.dumps(spec))
    np.save(cap / 'stage_rows/N02.npy', np.arange(16))
    prediction = tmp_path / 'pred.npz'
    np.savez(prediction, query_indices=np.arange(8), predicted_indices=np.zeros(8))
    row = dict(split_id='abc', k=1, new_count=2, registered_classes=list(range(8)),
               stage='C', prediction=str(prediction))
    original = np.load
    def no_truth(path, *args, **kwargs):
        assert 'score_only' not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', no_truth)
    item = dict(job=dict(capsule='capsules/P01/MIX01/1-1'), rt=dict(data_root=str(tmp_path/'data')),
                payload=dict(rows=[row]))
    with pytest.raises(ValueError, match='query'):
        validate_prediction_rows(item)
    np.savez(prediction, query_indices=np.arange(1, 16, 2, dtype=np.float64),
             predicted_indices=np.zeros(8, dtype=np.int32))
    with pytest.raises(ValueError, match='query'):
        validate_prediction_rows(item)


def test_transient_incomplete_completion_marker_is_not_counted(tmp_path, monkeypatch):
    import json
    from experiments.pamf_phase2_practical import clean_auto_score as auto
    plan = tmp_path/'plans/20261115-phase2-clean-example/scheduler.json'
    plan.parent.mkdir(parents=True)
    job = tmp_path/'job.json'
    job.write_text(json.dumps(dict(capsule='capsules/P01/MIX01/1-1', seed=2026092701,
                                  points=[[5, 20]], profile='protonet_cda')))
    marker = tmp_path/'complete.json'; marker.write_text('{')
    plan.write_text(json.dumps(dict(input_condition='clean_ground_pre_satellite_normalized',
        jobs=[dict(id='a', group='native_da', job=str(job), runtime='unused', completion=str(marker))])))
    monkeypatch.setattr(auto, 'completion_matches', lambda entry: json.loads(marker.read_text()))
    task = dict(capsule='capsules/P01/MIX01/1-1', seed=2026092701, points=[[5, 20]])
    expected = auto.expected_units([task]); unstable = {}
    available, _ = auto.collect(tmp_path/'plans', expected, unstable)
    assert not available
    available, _ = auto.collect(tmp_path/'plans', expected, unstable)
    assert not available
    with pytest.raises(ValueError, match='Malformed'):
        auto.collect(tmp_path/'plans', expected, unstable)


def test_transient_scheduler_plan_is_not_counted(tmp_path):
    from experiments.pamf_phase2_practical import clean_auto_score as auto
    plan = tmp_path/'plans/20261115-phase2-clean-new/scheduler.json'
    plan.parent.mkdir(parents=True)
    plan.write_text('{')
    unstable = {}
    assert auto.collect(tmp_path/'plans', {}, unstable) == ({}, {})
    assert auto.collect(tmp_path/'plans', {}, unstable) == ({}, {})
    with pytest.raises(ValueError, match='Malformed'):
        auto.collect(tmp_path/'plans', {}, unstable)


def test_primary_matrix_excludes_legacy_proto_mopc_but_keeps_affine(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_auto_score as auto
    plan = tmp_path/'plans/20261115-phase2-clean-successor/scheduler.json'
    plan.parent.mkdir(parents=True)
    task = dict(capsule='capsules/P01/MIX01/1-1', seed=2026092701,
                points=[[5, 20]], profile='protonet_cda', methods=['mopc_hr'])
    entries = []
    for name, bridge in (('old', 'original'), ('new', 'proto_affine_v1')):
        job = tmp_path/(name+'-job.json'); job.write_text(json.dumps(dict(task, mopc_bridge=bridge)))
        done = tmp_path/(name+'-done.json'); done.write_text('{}')
        entries.append(dict(id=name, group='hybrid', job=str(job), runtime='unused', completion=str(done)))
    plan.write_text(json.dumps(dict(input_condition='clean_ground_pre_satellite_normalized', jobs=entries)))
    monkeypatch.setattr(auto, 'completion_matches', lambda entry: True)
    expected = auto.expected_units([task])
    available, candidates = auto.collect(tmp_path/'plans', expected)
    u = auto.unit('hybrid', task)
    assert [entry['id'] for entry in candidates[u]] == ['new']
    assert ('protonet_cda__mopc_hr_affine_v1', 'C', 5, 20, 'P0') in available[u]
    assert ('protonet_cda__mopc_hr', 'C', 5, 20, 'P0') not in available[u]

