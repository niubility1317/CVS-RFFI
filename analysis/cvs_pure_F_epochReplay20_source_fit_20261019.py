"""Frozen source-only readout: cached As fitting versus new channel and new IDs.

Existing B0/B1 predictions are reused.  Only M's fixed L panel and B1 on M's
cached IQ are forwarded; no target, optimizer, EMA, or head update is accessed.
Named-stream reconstruction is not a byte comparison with historical saved IQ.
"""
import argparse
import gzip
import importlib.util
from itertools import zip_longest
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUN = '20261019-diagnostic-F-epochReplay-sourceFit-e20-s2026092701-m2-r01'
ER = '20261019-phase1-F-epochReplay-e20-s2026092701-m1-r01'
DR = '20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01'
OLD_RELEASE = 'cvs_F_observedChannel_E20_20261019_r01'
M = 'EPOCH_REPLAY'
SEED = 2026092701
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
NAMESPACES = ('F/observedChannel/20261019/new/draw0',
              'F/observedChannel/20261019/new/draw1')
KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf8')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_selected(selected, base, legal, opaque):
    """Check the existing panel's physical roles before any Dataset IQ fetch."""
    if len(selected) != 180 or len({r['physical_id'] for r in selected}) != 180:
        raise ValueError('Reuse complete fixed180 L panel without reselection')
    for r in selected:
        pid = tuple(r['pid'])
        if len(pid) != 5 or not 0 <= pid[4] < len(base.index):
            raise ValueError('Original physical base_index required before IQ fetch')
        it = base.index[pid[4]]
        if (pid != (it.eq_i, it.rx_i, it.day_i, it.sig_i, pid[4])
                or opaque(it) != r['physical_id'] or r['physical_id'] not in legal
                or pid[0] != 1 or pid[1] not in (1, 3, 4, 6, 8) or pid[2] not in (1, 2, 3)
                or (r['TX'], r['RX'], r['day']) != (it.tx_i, it.rx_i, it.day_i)
                or r['TX'] not in range(6) or r['epoch'] != 20
                or r['step'] not in range(4219, 4441) or r['slot'] not in range(128)):
            raise ValueError('Saved source L identity/role/coordinates differ before IQ fetch')


def lines(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        yield from map(json.loads, stream)


def recover_origins(mechanism, replay, batches, selected):
    """Close all4440 logs; independently derive E20 origins from actual L order.

    This pure metadata function never consumes IQ, labels, or online predictions.
    In particular, online correctness is not a frozen-fit measurement.
    """
    by_step = {r['step']: r for r in batches}
    if set(by_step) != set(range(4219, 4441)):
        raise ValueError('Complete actual final-epoch L sequence required')
    observed, coordinates, count = {}, {}, 0
    list_keys = ('source_steps', 'source_slots', 'cacheflags', 'used_namespaces',
                 'used_scenes', 'used_states', 'used_locks', 'fresh_scenes')
    for count, (a, b) in enumerate(zip_longest(mechanism, replay), 1):
        epoch = (count - 1) // 222 + 1
        if (a is None or b is None or count > 4440
                or a['step'] != count or b['step'] != count
                or a['epoch'] != epoch or b['epoch'] != epoch
                or len(a['packets']) != 128
                or any(len(b[k]) != 128 for k in list_keys)
                or a['As_packets'] != 128 or a['fresh_As_packets'] != 128
                or a['used_As_packets'] != 128 or a['L_forwards'] != 2
                or not a['ordinary_first_forward_verified']
                or a['hook_raw_MSE'] != 0. or a['encoder_used_As_MSE'] != 0.
                or b['fresh_generated_packets'] != 128 or b['used_packets'] != 128
                or b['cumulative_generated'] != count * 128
                or b['cumulative_used'] != count * 128
                or b['miss_packets'] + b['hit_packets'] != 128
                or b['epoch_cache_reset'] != ((count - 1) % 222 == 0)
                or b['reset_count'] != epoch - 1):
            raise ValueError('Closed native mechanism/replay budgets or order differ')
        if epoch != 20:
            continue
        namespace = f'FcrossRX/train/e20/step{count}/L_S'
        ids = by_step[count]['ids']
        hits = 0
        for slot, p in enumerate(a['packets']):
            pid = tuple(ids[slot])
            if (p['slot'] != slot or tuple(p['sample_id']) != pid
                    or p['fresh_namespace'] != namespace
                    or p['fresh_scene'] != b['fresh_scenes'][slot]):
                raise ValueError('Actual E20 mechanism physical slot differs from L sequence')
            expected = observed.get(pid)
            hit = expected is not None
            if expected is None:
                expected = dict(source_step=count, source_slot=slot,
                    namespace=namespace, scene=p['fresh_scene'],
                    state=b['used_states'][slot], lock=b['used_locks'][slot])
                observed[pid] = expected
            origin = (expected['source_step'], expected['source_slot'])
            if (p['cache_hit'] != hit or b['cacheflags'][slot] != hit
                    or (p['source_step'], p['source_slot']) != origin
                    or (b['source_steps'][slot], b['source_slots'][slot]) != origin
                    or p['used_namespace'] != expected['namespace']
                    or b['used_namespaces'][slot] != expected['namespace']
                    or p['used_scene'] != expected['scene']
                    or b['used_scenes'][slot] != expected['scene']
                    or b['used_states'][slot] != expected['state']
                    or b['used_locks'][slot] != expected['lock']
                    or tuple(by_step[origin[0]]['ids'][origin[1]]) != pid):
                raise ValueError('Cache origin/scene/state does not equal actual epoch first occurrence')
            coordinates[count, slot] = dict(expected, pid=list(pid), cache_hit=hit)
            hits += hit
        if b['hit_packets'] != hits or b['miss_packets'] != 128 - hits:
            raise ValueError('Actual E20 cache hit/miss packet accounting differs')
    if count != 4440:
        raise ValueError('All4440 mechanism and replay steps must be complete first')
    result = []
    for r in selected:
        entry = coordinates[r['step'], r['slot']]
        if entry['pid'] != r['pid']:
            raise ValueError('Existing selected physical L does not occupy its actual M step/slot')
        result.append(dict(entry, selected_step=r['step'], selected_slot=r['slot'],
                           physical_id=r['physical_id']))
    return result


def with_correct(value):
    if isinstance(value, dict):
        value = {k: with_correct(v) for k, v in value.items()}
        if 'accuracy' in value and 'n' in value:
            value['correct'] = int(round(value['accuracy'] * value['n']))
    return value


def matching_origins(old_coordinates, origins, observed_records):
    if not len(old_coordinates) == len(origins) == len(observed_records):
        raise ValueError('Matched origin coordinate panels required')
    return np.asarray([
        r['source_step'] == old_coordinates[i]['source_step']
        and r['source_slot'] == old_coordinates[i]['source_slot']
        and r['namespace'] == observed_records[i]['namespace']
        and r['scene'] == observed_records[i]['scene']
        and r['state'] == observed_records[i]['state_end']
        and r['lock'] == observed_records[i]['receiver_lock_status']
        for i, r in enumerate(origins)], dtype=bool)


def analyze(old, logits, vdata, y, rx, old_scene, cache_scene):
    """All method differences use common IQ and identical physical/scene groups."""
    def summary(z, labels, receivers, scene=None):
        return with_correct(old.summary(z, labels, receivers, scene))
    def paired(a, b, labels, receivers, scene=None):
        result = old.paired(a, b, labels, receivers, scene)
        sa, sb = summary(a, labels, receivers, scene), summary(b, labels, receivers, scene)
        result['CE_delta'] = dict(
            hard_label_NLL=sb['overall']['hard_label_NLL']-sa['overall']['hard_label_NLL'],
            label_smoothed_CE_001=sb['overall']['label_smoothed_CE_001']-sa['overall']['label_smoothed_CE_001'])
        return result
    L, V, seen_to_new, retention, v_retention = {}, {}, {}, {}, {}
    for arm in ('B0', 'B1', M):
        z = logits[arm]
        L[arm] = dict(raw180=summary(z['raw'], y, rx),
            old_B1_observed=summary(z['old_observed'], y, rx, old_scene),
            new_all180_by_scene={s: summary(z['new'][:, j].reshape(-1, 6),
                np.tile(y, 2), np.tile(rx, 2)) for j, s in enumerate(SCENES)},
            new_per_draw={str(k): {s: summary(z['new'][k, j], y, rx)
                for j, s in enumerate(SCENES)} for k in range(2)},
            new_paired_to_observed_scene={})
        retention[arm] = dict(old_B1_observed=paired(z['raw'], z['old_observed'], y, rx, old_scene),
            new_all180_by_scene={s: paired(np.tile(z['raw'], (2, 1)),
                z['new'][:, j].reshape(-1, 6), np.tile(y, 2), np.tile(rx, 2))
                for j, s in enumerate(SCENES)})
        for label, scene in (('old_B1_observed', old_scene), ('M_cached_used', cache_scene)):
            new = np.stack([z['new'][k, scene, np.arange(len(y))] for k in range(2)])
            z['new_paired_'+label] = new
            L[arm]['new_paired_to_observed_scene'][label] = summary(
                new.reshape(-1, 6), np.tile(y, 2), np.tile(rx, 2), np.tile(scene, 2))
        v = vdata[arm]
        V[arm] = dict(clean1800=summary(v['clean'], v['TX'], v['RX']),
            new1800_by_scene={s: summary(v['satellite'][:, j].reshape(-1, 6),
                np.tile(v['TX'], 2), np.tile(v['RX'], 2)) for j, s in enumerate(SCENES)},
            new_per_draw={str(k): {s: summary(v['satellite'][k, j], v['TX'], v['RX'])
                for j, s in enumerate(SCENES)} for k in range(2)})
        v_retention[arm] = {s: paired(np.tile(v['clean'], (2, 1)),
            v['satellite'][:, j].reshape(-1, 6), np.tile(v['TX'], 2), np.tile(v['RX'], 2))
            for j, s in enumerate(SCENES)}
    for arm in ('B1', M):
        z = logits[arm]
        L[arm]['M_cached_used'] = summary(z['cached_used'], y, rx, cache_scene)
        retention[arm]['M_cached_used'] = paired(z['raw'], z['cached_used'], y, rx, cache_scene)
        seen_to_new[arm] = {}
        for label, scene, seen in (('old_B1_observed', old_scene, z['old_observed']),
                                   ('M_cached_used', cache_scene, z['cached_used'])):
            seen_to_new[arm][label] = paired(np.tile(seen, (2, 1)),
                z['new_paired_'+label].reshape(-1, 6), np.tile(y, 2),
                np.tile(rx, 2), np.tile(scene, 2))
    pairs = {}
    for arm in ('B0', 'B1'):
        a, b = logits[arm], logits[M]
        entry = dict(raw180=paired(a['raw'], b['raw'], y, rx),
            old_B1_observed=paired(a['old_observed'], b['old_observed'], y, rx, old_scene),
            new_all180_by_scene={s: paired(a['new'][:, j].reshape(-1, 6),
                b['new'][:, j].reshape(-1, 6), np.tile(y, 2), np.tile(rx, 2))
                for j, s in enumerate(SCENES)}, new_paired_to_observed_scene={}, V={})
        for label, scene in (('old_B1_observed', old_scene), ('M_cached_used', cache_scene)):
            entry['new_paired_to_observed_scene'][label] = paired(
                a['new_paired_'+label].reshape(-1, 6), b['new_paired_'+label].reshape(-1, 6),
                np.tile(y, 2), np.tile(rx, 2), np.tile(scene, 2))
        if arm == 'B1':
            entry['M_cached_used'] = paired(a['cached_used'], b['cached_used'], y, rx, cache_scene)
        va, vb = vdata[arm], vdata[M]
        entry['V'] = dict(clean1800=paired(va['clean'], vb['clean'], vb['TX'], vb['RX']),
            new1800_by_scene={s: paired(va['satellite'][:, j].reshape(-1, 6),
                vb['satellite'][:, j].reshape(-1, 6), np.tile(vb['TX'], 2), np.tile(vb['RX'], 2))
                for j, s in enumerate(SCENES)})
        pairs[M+'_minus_'+arm] = entry
    return dict(L=L, V=V, common_IQ_method_pairs=pairs,
                observed_to_new_sameL_pairs=seen_to_new, raw180_correct_retention=retention,
                V1800_clean_correct_retention=v_retention)


def run(project, output):
    import time
    import torch
    start = time.perf_counter()
    module_at('_epochReplay_source_fit_runtime',
              Path.cwd()/'experiments/cvs_F_epochReplay20/runner.py').bootstrap()
    from experiments.cvs_F_epochReplay20 import design as d
    from experiments.cvs_F_epochReplay20.model import own_model
    from experiments.cvs_F_nontrue20 import design as ref
    from experiments.cvs_F_nontrue20.model import own_model as own_ref
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_fullsource_concat20.channel import Practical, serial
    from experiments.cvs_fullsource_concat20 import design as channel_design
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        raise ValueError('Ordinary account required')
    if d.RUN != ER or Path(d.PROJECT).resolve() != Path(project).resolve():
        raise ValueError('Existing epochReplay immutable runtime/project differs')
    root, oldroot = Path(project)/'runs'/ER, Path(project)/'runs'/DR
    path = root/(M+'-s'+str(SEED))
    decision = read(root/'source_matrix_decision.json')
    done, qual = read(path/'source/completion.json'), read(path/'source_qualification/complete.json')
    if (decision['status'] != 'SOURCE_DECISION_FIXED' or decision['formal_scores_read'] is not False
            or done['epoch'] != 20 or done['optimizer_steps'] != 4440
            or qual['status'] != 'SOURCE_READOUT_COMPLETE' or qual['target_access']):
        raise ValueError('Fixed source decision and own FINAL20/readout must exist first')
    old = module_at('_epochReplay_existing_helpers', Path(project)/'releases'/OLD_RELEASE/
                    'analysis/cvs_pure_F_observed_fit_20261019.py')
    oldmeta, reconstruction = read(oldroot/'metadata.json'), read(oldroot/'reconstruction.json')
    if (read(oldroot/'complete.json')['status'] != 'SOURCE_ONLY_FROZEN_READOUT_COMPLETE'
            or oldmeta['run_id'] != DR or oldmeta['model_seed'] != SEED
            or oldmeta['new_namespaces'] != list(NAMESPACES) or oldmeta['scenes'] != list(SCENES)
            or oldmeta['target_access'] or not reconstruction['all_replays_exact_equal']):
        raise ValueError('Existing frozen source180 diagnostic handles differ')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    roles = ref.read(ref.SOURCE)['role_ids']
    if set(roles) != {'L_s', 'U_s', 'V'} or any(len(roles[k]) != n for k, n in
            (('L_s', 6300), ('U_s', 56700), ('V', 27000))):
        raise ValueError('Original L6300/U56700/V27000 source roles required')
    b1path = Path(project)/'runs'/ref.RUN/('B1-s'+str(SEED))
    config = ref.config(next(r for r in ref.rows() if r['arm'] == 'B1'))
    args = SimpleNamespace(**read(b1path/'source/actual_native_args.json'))
    if (not args.a1_source_screen_only or args.baseline_ckpt or args.teacher_ckpt
            or args.use_concat_sat_channel_aug or args.wisig_out_len != 256):
        raise ValueError('Original source-only Dataset builder required')
    with installed(config) as native:
        ctx = native._build_ssdg_wisig_data(args, torch.device('cpu'))
    if ctx['named_test_loaders'] or role_ids_from_native(ctx) != roles:
        raise ValueError('Source-only Dataset physical roles differ')
    base = ctx['train_loader'].dataset.base
    if list(base.tx_list) != ref.CLASSES or base.transform is not None or not base.normalize or base.center:
        raise ValueError('Original deterministic float32 raw Dataset required')
    selected, legal = oldmeta['selected'], set(roles['L_s'])
    check_selected(selected, base, legal, opaque_id)
    with np.load(oldroot/'predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k].copy() for k in f.files if k in
                 ('ids', 'TX', 'RX', 'day', 'step', 'slot', 'observed_scene_index',
                  'V_ids', 'V_TX', 'V_RX', 'V_day') or k.startswith(('B0_', 'B1_'))}
    for field, key in (('ids', 'physical_id'), ('TX', 'TX'), ('RX', 'RX'), ('day', 'day'),
                       ('step', 'step'), ('slot', 'slot')):
        if not np.array_equal(saved[field], [r[key] for r in selected]):
            raise ValueError('Existing saved L panel IDs/labels/coordinates differ')
    y, rx, old_scene = saved['TX'], saved['RX'], saved['observed_scene_index']
    if old_scene.shape != (180,) or set(old_scene.tolist()) - set(range(3)):
        raise ValueError('Original observed scene panel differs')
    old_batches = old.epoch20(b1path/'source/training_sequence.jsonl.gz')
    batches = old.epoch20(path/'source/training_sequence.jsonl.gz')
    origins = recover_origins(lines(path/'source/mechanism_steps.jsonl.gz'),
                              lines(path/'source/replay_steps.jsonl.gz'), batches, selected)
    cache_scene = np.asarray([SCENES.index(r['scene']) for r in origins], np.int64)
    raw_reads = 0
    def raw_at(pid):
        nonlocal raw_reads
        if len(pid) != 5 or not 0 <= pid[4] < len(base.index):
            raise ValueError('Invalid original L index before IQ access')
        it = base.index[pid[4]]
        if (tuple(pid) != (it.eq_i, it.rx_i, it.day_i, it.sig_i, pid[4])
                or opaque_id(it) not in legal or it.eq_i != 1
                or it.rx_i not in ref.FIT or it.day_i not in (1, 2, 3)):
            raise ValueError('Invalid original L physical role before IQ access')
        x, _, _, meta = base[pid[4]]; raw_reads += 1
        if (tuple(int(meta[k]) for k in KEYS) != tuple(pid) or x.shape != (2, 256)
                or x.dtype != torch.float32 or not bool(torch.isfinite(x).all())):
            raise ValueError('Original native raw hook reconstruction differs')
        return x
    raw = torch.stack([raw_at(r['pid']) for r in selected])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        raise ValueError('Registered source-fit diagnostic uses physical GPU6')
    with lock(Path(project)/'runs/receiver_residual_capacity4.lock'):
        state = capacity()[6]
        if len(state['pids'] | {os.getpid()}) > 2 or state['free_mb'] < 12000:
            raise ValueError('GPU6 total-owner cap2/free12GB insufficient')
        torch.cuda.init()
        if len(capacity()[6]['pids'] | {os.getpid()}) > 2:
            raise ValueError('GPU6 capacity changed at CUDA initialization')
    with numerical_context(d.FULL_FP32_POLICY):
        model = own_model(d.config(d.rows()[0]), 'cuda:0')
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            smoke = model.forward_identity_only(raw[:1].to('cuda:0'))['tx_logits']
        if smoke.shape != (1, 6) or not bool(torch.isfinite(smoke).all()):
            raise ValueError('New FINAL20 one-source-packet smoke failed')
    write(out/'source_smoke.json', dict(status='PASS', checkpoint=M+' own FINAL20',
        source_packets=1, optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False))
    p = Practical('train')
    audit = serial(p.audit())
    actual = read(path/'source/actual_channel.json')['physical_channel']
    if (audit['configurations'] != actual['configurations']
            or audit['configurations'] != reconstruction['channel_audit']['configurations']
            or tuple(channel_design.TRAIN_SCENES) != SCENES):
        raise ValueError('Original actual channel configurations/scene recipe differ')
    generated = 0
    def reconstruct_flow(sequence, coordinates, saved_records=None, only=None):
        nonlocal generated
        lookup = {b['step']: b for b in sequence}
        positions = {}
        for i in (range(180) if only is None else only):
            r = coordinates[i]
            positions.setdefault(r['source_step'], []).append(i)
        result, records_out = torch.empty_like(raw), [None] * 180
        for step, indices in positions.items():
            ids = lookup[step]['ids']
            xb = torch.stack([raw_at(pid) for pid in ids])
            meta = {k: torch.tensor([pid[j] for pid in ids]) for j, k in enumerate(KEYS)}
            full, records = p.apply(xb, meta, epoch=20, step=step); generated += len(xb)
            for i in indices:
                r = coordinates[i]; slot = r['source_slot']; record = records[slot]
                if (list(ids[slot]) != selected[i]['pid']
                        or record['namespace'] != r['namespace'] or record['scene'] != r['scene']
                        or ('state' in r and record['state_end'] != r['state'])
                        or ('lock' in r and record['receiver_lock_status'] != r['lock'])
                        or (saved_records is not None and record != saved_records[i])):
                    raise ValueError('Deterministic origin reconstruction differs from logged source metadata')
                result[i], records_out[i] = full[slot], record
        return result, records_out
    old_coordinates = [dict(source_step=r['step'], source_slot=r['slot'],
        namespace=f'FcrossRX/train/e20/step{r["step"]}/L_S', scene=SCENES[int(old_scene[i])])
        for i, r in enumerate(selected)]
    observed, observed_records = reconstruct_flow(old_batches, old_coordinates,
                                                  reconstruction['observed_records'])
    same_origin = matching_origins(old_coordinates, origins, observed_records)
    different = np.flatnonzero(~same_origin)
    cached, cached_records = observed.clone(), list(observed_records)
    if len(different):
        partial, partial_records = reconstruct_flow(batches, origins, only=different)
        for i in different:
            cached[i], cached_records[i] = partial[i], partial_records[i]
    sessions = read(path/'source/training_result.json')['channel']['sessions']
    if any(sessions.get(k) != v for k, v in p.sessions.items()):
        raise ValueError('Actual M source hardware sessions differ')
    fresh = torch.empty((2, 3, 180, 2, 256), dtype=torch.float32)
    fresh_records = []
    for draw, namespace in enumerate(NAMESPACES):
        for si, scene in enumerate(SCENES):
            cfg = p.configs[scene]
            for i, r in enumerate(selected):
                pid = tuple(r['pid']); session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
                seed = p.ch.stable_seed(channel_design.CHANNEL_SEEDS['train'], namespace, scene, pid)
                hw = p.ch.receiver_for_session(cfg, channel_design.CHANNEL_SEEDS['receiver'], session)
                a = np.asarray(raw[i].tolist(), np.float64)
                normalized, _ = p.ch.normalize_input(a[0]+1j*a[1])
                signal, meta = p.residual.ResidualChannel(cfg, seed, hw).process(normalized)
                record = serial(dict(meta, draw=draw, scene=scene, namespace=namespace,
                                     sample_id=pid, session_id=session))
                if (record != reconstruction['fresh_records'][len(fresh_records)]
                        or not np.isfinite(signal).all() or meta['channel_equalization_applied']):
                    raise ValueError('Existing independent new namespace producer metadata differs')
                fresh_records.append(record)
                a = np.ascontiguousarray(np.stack((signal.real, signal.imag)), dtype=np.float32)
                fresh[draw, si, i] = torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(2, 256)
    count = 0
    def infer(network, x):
        nonlocal count
        values = []
        with torch.no_grad():
            for begin in range(0, len(x), 90):
                z = network.forward_identity_only(x[begin:begin+90].to('cuda:0'))['tx_logits']
                if z.shape != (min(90, len(x)-begin), 6) or not bool(torch.isfinite(z).all()):
                    raise ValueError('Finite frozen default-head source logits required')
                values.append(np.asarray(z.cpu().tolist(), np.float32)); count += len(z)
        return np.concatenate(values)
    def unchanged(network, initial, expected):
        if (network.training or any(v.requires_grad or v.grad is not None for v in network.parameters())
                or network.id_backbone.encoder._full_eval_packets != expected
                or any(not torch.equal(v, network.state_dict()[k].detach().cpu()) for k, v in initial.items())):
            raise ValueError('Frozen weights/buffers/gradients/forward counters differ')
        return dict(weights_buffers_unchanged=True, parameter_grads_none=True,
                    optimizer_updates=0, EMA_updates=0, head_fit_updates=0)
    logits = {arm: dict(raw=saved[arm+'_raw'], old_observed=saved[arm+'_observed_reconstructed'],
                       new=saved[arm+'_new_all_scenes']) for arm in ('B0', 'B1')}
    with numerical_context(d.FULL_FP32_POLICY):
        logits[M] = dict(raw=infer(model, raw), old_observed=infer(model, observed),
                         new=infer(model, fresh.reshape(-1, 2, 256)).reshape(2, 3, 180, 6))
        logits[M]['cached_used'] = logits[M]['old_observed'].copy()
        if len(different):
            logits[M]['cached_used'][different] = infer(model, cached[different])
        state_checks = {M: unchanged(model, before, 1441 + len(different))}
        del model, before
        logits['B1']['cached_used'] = logits['B1']['old_observed'].copy()
        if len(different):
            baseline = own_ref(config, 'cuda:0')
            before = {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
            logits['B1']['cached_used'][different] = infer(baseline, cached[different])
            state_checks['B1'] = unchanged(baseline, before, len(different))
            del baseline, before
        else:
            state_checks['B1'] = dict(checkpoint_loaded=False,
                status='REUSED_FROZEN_PREDICTIONS_NO_MODEL_LOAD',
                source_forward_packets=0, optimizer_updates=0, EMA_updates=0, head_fit_updates=0)
    if count != 1440 + 2 * len(different) or count > 1800:
        raise ValueError('Only distinct-IQ cached subsets may add source forwards; maximum1800')
    forwards = {M:1440 + len(different), 'B1_cached_only':len(different)}
    vdata = {a: dict(clean=saved[a+'_V_clean'], satellite=saved[a+'_V_satellite'],
        ids=saved['V_ids'], TX=saved['V_TX'], RX=saved['V_RX'], day=saved['V_day']) for a in ('B0', 'B1')}
    with np.load(path/'source_qualification/predictions.npz', allow_pickle=False) as f:
        vdata[M] = {k: f[k].copy() for k in ('ids', 'TX', 'RX', 'day', 'clean', 'satellite')}
    for arm, v in vdata.items():
        if (v['clean'].shape != (1800, 6) or v['satellite'].shape != (2, 3, 1800, 6)
                or set(v['ids'].tolist()) - set(roles['V'])
                or any(not np.array_equal(v[k], vdata['B0'][k]) for k in ('ids', 'TX', 'RX', 'day'))):
            raise ValueError('Exact existing source V1800 qualification panel required')
    full_v = {}
    for arm, location in (('B0', d.reference('B0')), ('B1', b1path), (M, path)):
        metrics = read(location/'source/completion.json')['final_source_metrics']
        if metrics['source_val_count'] != 27000:
            raise ValueError('Full V27000 clean completion metric required')
        full_v[arm] = dict(n=27000, accuracy=metrics['source_val_accuracy'],
                           provenance='own FINAL20 completion; not V1800 panel', new_NN=0)
    metadata = dict(run_id=RUN, model_seed=SEED, selected=selected, cached_origins=origins,
        reused_diagnostic=DR, scenes=list(SCENES), new_namespaces=list(NAMESPACES),
        raw_input='original source-only Dataset float32 raw, membership before IQ fetch',
        closed_mechanism_replay_steps=4440, E20_L_sequence_slots=28416,
        reconstruction='deterministic named first-origin producer/config/seed/session; no historical IQ saved',
        historical_IQ_byte_comparison=False, observed_records=observed_records,
        cached_records=cached_records, fresh_records=fresh_records,
        old_and_cached_IQ_equal_packets=int((observed == cached).flatten(1).all(1).sum()),
        same_origin_packets=int(same_origin.sum()), different_origin_packets=len(different),
        same_origin_mask=same_origin.tolist(),
        prediction_reuse=dict(M_cached_from_own_old_observed=int(same_origin.sum()),
                              B1_cached_from_saved_old_observed=int(same_origin.sum())),
        NN_budget='Maximum1801 including smoke; identical origins reuse predictions without NN',
        scene_comparison='Each method pair shares identical IQ and scene groups; two observed streams may have different scene composition; no cross-composition net effect',
        B0_As_exposure='NoAs reference; old observed and new As were not training observations',
        fitting='All reported logits use frozen FINAL20 default heads; online training counts are not frozen fit',
        model_seed_count=1, new_draw_count=2, draws_are_not_model_seeds=True,
        raw180_is_not_fullL6300=True, V1800_is_not_fullV27000=True,
        source_decision_fixed=decision, source_promotion_changed=False,
        source_threshold_added=False, target_access=False,
        channel_generations=dict(reconstructed_full_batch_packets=generated, new_namespace_packets=1080),
        checked_L_raw_fetches=raw_reads, CE_definitions=dict(
            hard_label_NLL='Standard true-label default-logit CE without temperature/smoothing',
            label_smoothed_CE_001='Training CE with label_smoothing0.01'),
        optimizer_updates=0, EMA_updates=0, head_fit_updates=0, new_V_forward_packets=0)
    result = analyze(old, logits, vdata, y, rx, old_scene, cache_scene)
    result.update(status='SOURCE_ONLY_FROZEN_READOUT_COMPLETE', run_id=RUN,
        metadata=metadata, fullV27000_clean=full_v, state_checks=state_checks,
        forwards=forwards, source_smoke_packets=1,
        new_source_NN_packets=count+1, maximum_source_NN_packets=1801, old_B0_forward_packets=0,
        reused_B1_existing_views_new_NN=0, new_V_forward_packets=0,
        optimizer_updates=0, EMA_updates=0, head_fit_updates=0,
        source_promotion_changed=False, target_access=False,
        model_seed_count=1, sample_SD=None, wall_seconds=time.perf_counter()-start)
    predictions = dict(saved, M_cached_scene_index=cache_scene)
    for arm, values in logits.items():
        predictions.update({arm+'_'+key: value for key, value in values.items()})
    predictions.update({M+'_V_clean':vdata[M]['clean'], M+'_V_satellite':vdata[M]['satellite']})
    np.savez_compressed(out/'predictions.npz', **predictions)
    write(out/'metadata.json', metadata); write(out/'analysis.json', result)
    for arm in (M, 'B1'):
        folder = out/(arm+'-s'+str(SEED)); folder.mkdir(exist_ok=False)
        write(folder/'analysis.json', dict(run_id=RUN, arm=arm, model_seed=SEED,
            L=result['L'][arm], V=result['V'][arm], fullV27000_clean=full_v[arm],
            observed_to_new_sameL_pairs=result['observed_to_new_sameL_pairs'][arm],
            raw180_correct_retention=result['raw180_correct_retention'][arm],
            V1800_clean_correct_retention=result['V1800_clean_correct_retention'][arm],
            state_check=state_checks[arm], source_forward_packets=forwards[M if arm == M else 'B1_cached_only'],
            source_smoke_packets=int(arm == M), new_V_forward_packets=0,
            optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False))
    write(out/'complete.json', dict(status=result['status'], source_L_selected=180,
        M_source_forward_packets=forwards[M], B1_cached_forward_packets=len(different), source_smoke_packets=1,
        source_forward_packets=count+1, maximum_source_forward_packets=1801,
        same_origin_packets=int(same_origin.sum()), different_origin_packets=len(different),
        reused_M_cached_predictions=int(same_origin.sum()), reused_B1_cached_predictions=int(same_origin.sum()),
        new_V_forward_packets=0, all_loaded_weights_buffers_unchanged=True,
        optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False,
        historical_IQ_byte_comparison=False, source_promotion_changed=False))
    print(json.dumps(dict(status=result['status'], new_source_NN=count+1,
                         maximum_source_NN=1801, V_new_NN=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.project, args.output)
