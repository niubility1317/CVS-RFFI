"""Synthetic IO boundaries for frozen source-novel physical geometry export."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_phase3_open_world import star_physical_geometry_export as e
from experiments.pamf_phase3_open_world import star_physical_geometry_launch as launch
from experiments.pamf_phase3_open_world import star_representation_launch as process_api
from experiments.pamf_phase3_open_world import star_channel
from tests.test_pamf_phase3_star_physical_open_train import config as parent_config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ForbiddenP0:
    def __getitem__(self, key):
        raise AssertionError('Geometry export must not consume received P0')


class Frozen(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer('offset', torch.zeros(1))
        self.eval().requires_grad_(False)


def fixture(root, count=5):
    configs = {}
    for arm in e.ARMS:
        c = parent_config(root / 'parents', arm)
        c['code_commit'] = c['native_config']['code_commit'] = e.PARENT_COMMIT
        configs[arm] = c
    config = dict(run_id=e.RUN_ID, code_commit='a' * 40, train_configs=configs,
        cache_root=str(root / 'parents/cache'), baseline_feature_root=str(root / 'old_features'),
        output_root=str(root / 'run/features'), batch_size=128)
    parent = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701)
    dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
    models = {arm: Frozen(arm) for arm in e.ARMS}
    identities = {}
    for arm, c in configs.items():
        identity = dict(run_id=e.PARENT_RUN, code_commit=e.PARENT_COMMIT, arm=arm,
                        updates=2220, cumulative_updates=13320, parent_F0_identity=deepcopy(parent))
        identities[arm] = identity
        write_json(Path(c['output_root']) / 'complete.json', dict(status=e.parent.STATUS,
            identity=identity, code_commit=e.PARENT_COMMIT, source_only=True, updates=2220,
            model_buffers_unchanged=True, dura_identity=dura, target_reads=0, validation_reads=0,
            novel_validation_reads=0, original_U_s_reads=0, OE_truth_reads=0))
    rx = np.asarray([1, 3, 4, 6, 8][:count], np.int64)
    day = np.arange(count, dtype=np.int64) % 3 + 1
    role = dict(ids=np.asarray(['novel-opaque-' + str(i) for i in range(count)]),
        metadata=dict(rx_i=rx, day_i=day, domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1),
        manifest=dict(role='novel', count=count, source_only=True, all_new_IQ_satellite=True,
            restored_input_path='P1', parent_F0_identity=deepcopy(parent), dura_identity=deepcopy(dura)),
        received=ForbiddenP0(), restored=np.zeros((count, 2, 256), np.float32))
    role['restored'][:, 0, 0] = np.arange(count) + 1
    unit = np.zeros((count, 160), np.float32)
    unit[np.arange(count), np.arange(count) + 70] = 1.
    baseline = dict(ids=role['ids'].copy(), rx_i=rx.copy(), day_i=day.copy(), fused160=unit)
    marker = dict(schema='pamf_phase3_open_feature_export_v1', status='STAR_OPEN_FEATURES_COMPLETE',
        run_id='20261030-phase3-star-open-feature-diagnostic-s2026092701-r01', source_count=count,
        source_role='novel', model_states_unchanged=True, model_updates=0, optimizer_updates=0,
        threshold_updates=0, target_reads=0, private_truth_reads=0, input_path='cached_P1_only',
        received_cache_values_read=0, parent_identities={'F0': deepcopy(parent)}, source_manifest=deepcopy(role['manifest']))
    path = Path(config['baseline_feature_root'])
    write_json(path / 'complete.json', marker)
    np.savez(path / 'F0.npz', **baseline)
    return config, role, baseline, parent, dura, models, identities


def execute(monkeypatch, values, failure=None):
    config, role, baseline, parent, dura, models, identities = values
    events = []
    snapshots = {arm: {k: v.clone() for k, v in model.state_dict().items()} for arm, model in models.items()}

    def load_model(c, checkpoint, device):
        assert c['arm'] in e.ARMS and c['code_commit'] == e.PARENT_COMMIT
        assert Path(checkpoint) == Path(c['output_root']) / 'final.pt'
        events.append(('checkpoint', c['arm']))
        return models[c['arm']], identities[c['arm']]

    def forward(model, x):
        assert not torch.is_grad_enabled()
        smoke = bool((x[:, 0, 0] < 0).all())
        events.append(('smoke' if smoke else 'P1', model.arm, len(x)))
        if smoke and failure == 'smoke':
            raise RuntimeError('synthetic checkpoint smoke failed')
        if not smoke and failure == 'mutation':
            model.weight.add_(1.)
        raw = torch.zeros((len(x), 160), dtype=torch.float32)
        raw[:, 0] = model.weight
        raw[:, 1] = x[:, 0, 0] + model.offset
        return dict(raw_fused160=raw, open64=raw[:, :64])

    def load_role(root, name):
        assert events[:4] == [('checkpoint', 'P_L'), ('checkpoint', 'P_LOE'), ('smoke', 'P_L', 2), ('smoke', 'P_LOE', 2)]
        assert (Path(config['output_root']) / 'smoke.json').is_file()
        assert name == 'novel' and Path(root) == Path(config['cache_root']) / 'novel'
        events.append(('load', name))
        return role

    smoke = torch.zeros((2, 2, 256), dtype=torch.float32)
    smoke[:, 0, 0] = -1.
    with monkeypatch.context() as patch:
        patch.setattr(e, 'COUNT', len(role['ids']))
        patch.setattr(star_channel, '_channel_api', lambda: None)
        patch.setattr(e.parent, 'load_trained_model', load_model)
        patch.setattr(e.old.original, '_backend', lambda *a, **k: pytest.fail('F0 must never be loaded'))
        patch.setattr(e.old, '_synthetic', lambda device: smoke)
        patch.setattr(e.model_api, 'forward', forward)
        patch.setattr(e.model_api, 'clear', lambda model: None)
        patch.setattr(e, 'load_role', load_role)
        patch.setattr(e, 'close_role', lambda role: None)
        done = e.export(config, 'cpu')
    assert all(torch.equal(snapshots[arm][k], value) for arm, model in models.items() for k, value in model.state_dict().items())
    return done, events


def test_config_rejects_private_target_wrong_parent_and_output_collisions(monkeypatch, tmp_path):
    config, *_ = fixture(tmp_path)
    assert e.validate_config(config) is config
    for case in ('truth', 'target', 'parent', 'overlap', 'existing'):
        c = deepcopy(config)
        if case == 'truth':
            c['novel_truth'] = 'forbidden.npz'
        elif case == 'target':
            c['target_root'] = 'forbidden-target'
        elif case == 'parent':
            c['train_configs']['P_LOE']['code_commit'] = 'b' * 40
            c['train_configs']['P_LOE']['native_config']['code_commit'] = 'b' * 40
        elif case == 'overlap':
            c['output_root'] = str(Path(c['baseline_feature_root']) / 'new')
        else:
            Path(c['output_root']).mkdir(parents=True)
        with monkeypatch.context() as patch:
            patch.setattr(e, '_backend', lambda *a: pytest.fail('invalid input reached model loading'))
            with pytest.raises(FileExistsError if case == 'existing' else ValueError):
                e.export(c, 'cpu')


def test_baseline_is_exact_and_rejects_parent_manifest_labels_or_metadata(monkeypatch, tmp_path):
    for case in ('valid', 'parent', 'manifest', 'labels', 'ids', 'rx', 'day'):
        config, role, data, parent, dura, *_ = fixture(tmp_path / case)
        root = Path(config['baseline_feature_root'])
        marker = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
        if case == 'parent':
            marker['parent_identities']['F0']['model'] = 'different-F0'
        elif case == 'manifest':
            marker['source_manifest']['role'] = 'L'
        elif case == 'labels':
            data['tx'] = np.full(len(role['ids']), 'forbidden')
        elif case == 'ids':
            data['ids'] = data['ids'][::-1]
        elif case in ('rx', 'day'):
            data['rx_i' if case == 'rx' else 'day_i'][0] += 1
        write_json(root / 'complete.json', marker)
        np.savez(root / 'F0.npz', **data)
        with monkeypatch.context() as patch:
            patch.setattr(e, 'COUNT', len(role['ids']))
            if case == 'valid':
                reused, _ = e._baseline(root, role, parent, dura)
                assert all(np.array_equal(reused[k], data[k]) for k in data)
            else:
                with pytest.raises(ValueError):
                    e._baseline(root, role, parent, dura)


def test_export_smokes_both_parents_then_only_novel_P1_and_reuses_F0_exactly(monkeypatch, tmp_path):
    values = fixture(tmp_path)
    config, _, baseline, *_ = values
    done, events = execute(monkeypatch, values)
    assert events[-3:] == [('load', 'novel'), ('P1', 'P_L', 5), ('P1', 'P_LOE', 5)]
    assert done['actual_frozen_checkpoints'] == list(e.ARMS) and not done['F0_checkpoint_loaded']
    assert done['real_F_forward_rows'] == 10 and done['synthetic_F_forward_rows'] == 4
    assert done['F0_feature_rows_reused'] == 5 and done['G_forward_rows'] == 0 and done['model_updates'] == 0
    assert len(done['representations']) == 5 and done['received_cache_values_read'] == 0
    for role in ('F0', 'P_L', 'P_LOE'):
        with np.load(Path(config['output_root']) / (role + '.npz')) as saved:
            if role == 'F0':
                assert all(np.array_equal(saved[k], baseline[k]) for k in baseline)
            else:
                assert saved['fused160'].shape == (5, 160) and saved['open64'].shape == (5, 64)
                assert np.allclose(np.linalg.norm(saved['open64'], axis=1), 1.)
    assert done['status'] == e.STATUS


def test_smoke_baseline_failure_or_model_mutation_never_writes_complete(monkeypatch, tmp_path):
    for failure in ('smoke', 'baseline', 'mutation'):
        values = fixture(tmp_path / failure)
        if failure == 'baseline':
            path = Path(values[0]['baseline_feature_root']) / 'complete.json'
            marker = json.loads(path.read_text(encoding='utf-8'))
            marker['status'] = 'RUNNING'
            write_json(path, marker)
        with pytest.raises(RuntimeError if failure == 'smoke' else ValueError):
            execute(monkeypatch, values, failure)
        assert not (Path(values[0]['output_root']) / 'complete.json').exists()


def test_launcher_failed_export_stops_and_success_passes_truth_only_to_CPU_score(monkeypatch, tmp_path):
    for fail in (True, False):
        config, *_ = fixture(tmp_path / ('failed' if fail else 'success'))
        root = Path(config['output_root']).parent
        (root / 'logs').mkdir(parents=True)
        runtime = dict(run_id=e.RUN_ID, code_commit=config['code_commit'], run_root=str(root),
            release_root=str(root.parent / 'release'), python='synthetic-python', launch_owner='test-owner',
            allowed_gpus=list(range(8)), minimum_free_mb=4096, max_added_workers_per_gpu=2,
            export_config=config, novel_truth='scorer-only-synthetic-private.npz')
        path = root.parent / 'runtime.json'
        write_json(path, runtime)
        calls = []

        def popen(command, **kwargs):
            scoring = '--novel-truth' in command
            calls.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))

            def wait():
                if not scoring and fail:
                    return 1
                marker = dict(run_id=e.RUN_ID, code_commit=config['code_commit'], source_count=3150,
                              target_reads=0, optimizer_updates=0)
                if scoring:
                    marker.update(status='STAR_PHYSICAL_GEOMETRY_SCORES_COMPLETE', truth_last=True,
                                  all_neighbors_fixed_before_truth=True, final_phase3_complete=False)
                else:
                    marker.update(status=e.STATUS, private_truth_reads=0, model_states_unchanged=True, source_role='novel',
                        baseline_F0_features_reused=True, actual_frozen_checkpoints=['P_L', 'P_LOE'],
                        real_F_forward_rows=6300, synthetic_F_forward_rows=4, F0_feature_rows_reused=3150, F0_checkpoint_loaded=False)
                write_json(root / ('score' if scoring else 'features') / 'complete.json', marker)
                return 0

            return SimpleNamespace(pid=100 + len(calls), wait=wait)

        with monkeypatch.context() as patch:
            patch.setattr(launch, '_gpu_free', lambda r: [4])
            patch.setattr(process_api.subprocess, 'Popen', popen)
            launch.driver(path)
        state = json.loads((root / 'pipeline_state.json').read_text(encoding='utf-8'))
        assert calls[0][1] == '4' and runtime['novel_truth'] not in calls[0][0]
        if fail:
            assert len(calls) == 1 and state['status'] == 'FAILED_PRESERVED'
            assert not (root / 'score').exists()
        else:
            assert len(calls) == 2 and calls[1][1] == ''
            assert calls[1][0][calls[1][0].index('--novel-truth') + 1] == runtime['novel_truth']
            assert state['status'] == 'ANALYZED_SOURCE_DIAGNOSTIC' and state['all_stages_complete']
