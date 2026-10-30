"""Small differentiable graphs for physical-open source training boundaries."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.pamf_phase3_open_world import star_physical_open_train as t


def config(root, arm='P_LOE'):
    out = str(root / arm / 'train')
    native = dict(run_id=t.RUN_ID, row_id=arm + '-s2026092701', code_commit='a' * 40,
        output_root=out, model_seed=2026092701, base_release='synthetic-original-release',
        frozen_sources={k: k for k in ('F', 'AF', 'BASE', 'FOLD', 'G')},
        source_output=str(root / 'original-F0/source'), source_contract='synthetic-contract.json',
        source_cache=str(root / 'legacy-cache-NOT-OPEN'), registered_classes=t.native.CLASSES.copy(), batch_size=128)
    return dict(run_id=t.RUN_ID, code_commit='a' * 40, arm=arm, native_config=native,
        L_root=str(root / 'cache/L'), OE_root=str(root / 'cache/OE'), output_root=out,
        steps=2220, lr=.0001, sampling_seed=2026103017, slot_seed=2026103018)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.encoder = nn.Linear(4, 64)
        self.head = nn.Linear(64, 6)
        self.open_world = nn.Module()
        self.open_world.conditioner = nn.Sequential(nn.Linear(29, 4), nn.GELU(), nn.Linear(4, 64))
        self.open_world.relation_scorer = nn.Linear(128, 1)
        self._phase3_parent_args = {}


def roles(c):
    n = 192
    y = np.repeat(np.arange(6, dtype=np.int64), 32)
    rx = np.tile(np.asarray([1, 3, 4, 6] * 8, np.int64), 6)
    day = np.tile(np.asarray([1, 2, 3, 1] * 8, np.int64), 6)
    domain = np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1
    parent = dict(model='original_F0_E50', model_seed=2026092701, epoch=50, updates=11100,
                  parameters=232993, source_output=c['native_config']['source_output'])
    dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
    result = {}
    for role, count, ids_offset in (('L', n, 0), ('OE', 80, 1000)):
        restored = np.zeros((count, 2, 256), np.float32)
        restored[:, 0, 0] = np.arange(count) + ids_offset
        restored[:, 0, 1] = 1.  # Same record; only P0/P1 view differs.
        received = restored.copy()
        received[:, 0, 1] = 0.
        meta = dict(rx_i=rx[:count].copy(), day_i=day[:count].copy(), domain=domain[:count].copy())
        if role == 'L':
            meta['y'] = y.copy()
        manifest = dict(role=role, count=count, parent_F0_identity=deepcopy(parent), dura_identity=deepcopy(dura),
                        received_input_path='P0', restored_input_path='P1', teacher_input_path='P1' if role == 'L' else None)
        bundle = dict(ids=np.asarray([role + '-opaque-' + str(i) for i in range(count)]),
                      received=received, restored=restored, metadata=meta, manifest=manifest)
        if role == 'L':
            bundle['teacher_logits'] = np.zeros((count, 6), np.float32)
        result[role] = bundle
    return result['L'], result['OE']


def run_step(monkeypatch, root, arm):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    c = config(root, arm)
    c['steps'] = 1
    L, OE = roles(c)
    rng = np.random.default_rng(37)
    sample = dict(native_L=rng.permutation(192)[:128], support=np.arange(192).reshape(6, 32)[:, :16],
        query=np.arange(192).reshape(6, 32)[:, 16:], OE=rng.permutation(80)[:64],
        held_class=2, held_rx=3, held_day=2, slot_order=np.asarray([2, 0, 5, 1, 4, 3], np.int64))
    model = Tiny()
    calls, pairs = [], []

    class Sampler:
        def __init__(self, y, rx, day, ids, domains, *, seed, slot_seed):
            assert np.array_equal(y, L['metadata']['y'])
            assert np.array_equal(domains, OE['metadata']['domain'])
            assert seed == c['sampling_seed'] and slot_seed == c['slot_seed']

        def sample(self, step):
            assert step == 1
            return sample

    def forward(m, x):
        record, view = x[:, 0, 0], x[:, 0, 1]
        inputs = torch.stack((torch.sin(.13 * record), torch.cos(.17 * record), view, record / 1200.), dim=1)
        raw = m.id_backbone.encoder(inputs) + m.open_world.conditioner(torch.zeros((len(x), 29)))
        calls.append(dict(ids=np.asarray(record.detach().tolist(), np.int64),
                          view=np.asarray(view.detach().tolist(), np.int64), raw=raw))
        return dict(open_raw64=raw, open64=F.normalize(raw, dim=1), logits=m.head(raw))

    def relation(m, z, p):
        return m.open_world.relation_scorer(torch.cat(((z[:, None] - p[None]).abs(), z[:, None] * p[None]), -1)).squeeze(-1)

    def native_loss(m, x, y, domains, metadata, augmentor, no_pa, step):
        assert m.training and not m.id_backbone.encoder.training and len(x) == 128
        assert set(metadata) == {'rx_i', 'day_i', 'domain'} and step == 1
        # Exercise the full 6x128 count with the same differentiable toy graph.
        logits = forward(m, x.repeat(6, 1, 1))['logits']
        return F.cross_entropy(logits, y.repeat(6)), dict(T1_L_packets=768, CRIC_supcon_weight=.1,
            T1_PL_seen=0, T1_sat_seen=0, T1_As_KL_weight=0.)

    physical = t.physical.physical

    def paired(z0, z1):
        pairs.append((z0.detach().clone(), z1.detach().clone()))
        return physical(z0, z1)

    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    with monkeypatch.context() as patch:
        patch.setattr(t.episode, 'EpisodeSampler', Sampler)
        patch.setattr(t.native, '_augmentor', lambda args: (None, True, {}))
        patch.setattr(t.native, 'supervised_loss', native_loss)
        patch.setattr(t.ow, 'forward', forward)
        patch.setattr(t.ow, 'relation', relation)
        patch.setattr(t.ow, 'clear', lambda m: None)
        patch.setattr(t.physical, 'physical', paired)
        patch.setattr(t, '_clip', lambda optimizer, params: nn.utils.clip_grad_norm_(params, 5.))
        rows, state = t._loop(model, L, OE, c, 'cpu', torch.optim.AdamW(model.parameters(), lr=c['lr']))
    return rows[0], state, calls, pairs, sample, model, before


def test_source_config_roles_private_fields_and_output_isolation(tmp_path):
    valid = [config(tmp_path, arm) for arm in t.ARMS]
    for c in valid:
        assert t.validate_config(c) is c
        assert c['native_config']['row_id'] == c['arm'] + '-s2026092701'
    assert valid[0]['output_root'] != valid[1]['output_root']
    assert valid[0]['L_root'] == valid[1]['L_root'] and valid[0]['OE_root'] == valid[1]['OE_root']
    for change in ('truth', 'role', 'overlap', 'row', 'budget'):
        c = deepcopy(valid[0])
        if change == 'truth':
            c['OE_truth'] = 'forbidden.npz'
        elif change == 'role':
            c['OE_root'] = c['L_root']
        elif change == 'overlap':
            c['output_root'] = str(Path(c['L_root']) / 'train')
            c['native_config']['output_root'] = c['output_root']
        elif change == 'row':
            c['native_config']['row_id'] = 'P_LOE-s2026092701'
        else:
            c['steps'] = 1
        with pytest.raises(ValueError):
            t.validate_config(c)


def test_one_step_pairs_budget_weight_equation_and_OE_raw_gradient_control(monkeypatch, tmp_path):
    results = {}
    for arm in t.ARMS:
        row, state, calls, pairs, sample, model, before = run_step(monkeypatch, tmp_path, arm)
        results[arm] = row
        ix, oi = sample['native_L'], sample['OE']
        assert [len(call['ids']) for call in calls] == [768, 128, 256, 128]
        assert sum(len(call['ids']) for call in calls) == row['total_F_packet_views'] == 1280
        assert np.array_equal(calls[-1]['ids'][:64], ix[:64])
        assert np.array_equal(calls[-1]['ids'][64:], oi + 1000)
        assert np.array_equal(calls[1]['ids'][:64], calls[-1]['ids'][:64])
        assert np.array_equal(calls[2]['ids'][192:], calls[-1]['ids'][64:])
        assert all(np.all(call['view'] == 1) for call in calls[:3]) and np.all(calls[-1]['view'] == 0)
        assert torch.equal(pairs[0][0], calls[-1]['raw'].detach()[:64])
        assert torch.equal(pairs[0][1], calls[1]['raw'].detach()[:64])
        assert torch.equal(pairs[1][0], calls[-1]['raw'].detach()[64:])
        assert torch.equal(pairs[1][1], calls[2]['raw'].detach()[192:])
        weight = .1 if arm == 'P_LOE' else 0.
        expected = .1 * (row['member_loss'] + row['crossRX_contrast'] + row['physical_L']['total']) + weight * row['physical_OE']['total']
        assert row['source_objective_weighted'] == pytest.approx(expected, rel=1e-6)
        assert row['anchor_loss'] == pytest.approx(row['full_L_loss'] + row['KD'], rel=1e-6)
        assert row['physical_L_weight'] == row['crossRX_weight'] == row['member_weight'] == .1
        assert row['physical_OE_weight'] == weight and row['physical_OE']['total'] > 0.
        assert row['OE_physical_openraw_grad_norm'] == 0. if arm == 'P_L' else row['OE_physical_openraw_grad_norm'] > 0.
        assert row['crossRX_positive_pairs'] > 0 and row['crossRX_anchors'] == 192
        assert row['member_counts']['prototype_count'] == 5 and row['reference_mode'] == 'constant'
        assert row['physical_P0_L_views'] == row['physical_P0_OE_views'] == 64
        assert row['As_cached_P1_replay_samples'] == 64 and row['As_new_channel_generated_samples'] == 0
        assert torch.count_nonzero(model.open_world.conditioner[0].weight.grad) == 0
        assert not torch.equal(before['head.weight'], model.head.weight)
        assert not torch.equal(before['id_backbone.encoder.weight'], model.id_backbone.encoder.weight)
        assert state['actual_unique_source_records']['L_P0'] == 64
        assert state['actual_unique_source_records']['OE_P0'] == state['actual_unique_source_records']['OE_P1'] == 64
    assert results['P_L']['physical_OE']['total'] == pytest.approx(results['P_LOE']['physical_OE']['total'])


def test_train_rejects_source_parent_or_OE_label_leak_before_any_update(monkeypatch, tmp_path):
    for invalid in ('parent', 'OE_y', 'overlapping_ids'):
        c = config(tmp_path / invalid)
        L, OE = roles(c)
        if invalid == 'parent':
            OE['manifest']['parent_F0_identity']['source_output'] = 'different-original-F0'
        elif invalid == 'OE_y':
            OE['metadata']['y'] = np.zeros(len(OE['ids']), np.int64)
        else:
            OE['ids'][0] = L['ids'][0]
        calls = []
        with monkeypatch.context() as patch:
            patch.setattr(t, '_model', lambda *args: Tiny())
            patch.setattr(t, '_smoke', lambda *args: dict(status='PASS'))
            patch.setattr(t, 'load_L', lambda *args: L)
            patch.setattr(t, 'load_OE', lambda *args: OE)
            patch.setattr(t, 'close_role', lambda bundle: None)
            patch.setattr(t.ow, 'clear', lambda model: None)
            patch.setattr(t.previous, 'COUNTS', dict(L=len(L['ids']), OE=len(OE['ids'])))
            patch.setattr(t, '_loop', lambda *args: calls.append('unexpected update'))
            with pytest.raises(ValueError):
                t.train(c, 'cpu')
        assert calls == []
        assert not (Path(c['output_root']) / 'complete.json').exists()
        assert not (Path(c['output_root']) / 'final.pt').exists()
