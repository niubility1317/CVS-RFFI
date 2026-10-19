"""Frozen pure-F source fitting/channel-generalization readout; never opens target.

Run from the existing sceneRisk release. Historical As IQ was not stored: the
"observed" panel is a deterministic reconstruction of its named training stream,
not an independent byte comparison with a historical IQ artifact.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SEED = 2026092701
REFERENCE = '20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01'
CURRENT = '20261019-phase1-F-sceneRisk-e20-s2026092701-m2-r01'
ARMS = ('B0', 'B1', 'BALANCED', 'RISK')
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
NAMESPACES = ('F/observedChannel/20261019/new/draw0',
              'F/observedChannel/20261019/new/draw1')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf8')


def epoch20(path):
    """The nth consumed L batch is step n; keep only its final-epoch metadata."""
    result, count = [], 0
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['role'] not in ('L', 'U'):
                raise ValueError('Unknown source sequence role')
            if r['role'] == 'L':
                count += 1
                ids = r['ids']
                if len(ids) != 128 or any(len(pid) != 5 for pid in ids):
                    raise ValueError('Actual L128 physical sequence required')
                if count > 19 * 222:
                    result.append(dict(step=count, epoch=(count - 1) // 222 + 1,
                                       ids=ids))
    if count != 4440 or len(result) != 222:
        raise ValueError('Complete cumulative E20 L consumption required')
    return result


def select(sequence, base, legal, opaque):
    """Check identity/role before any Dataset IQ access; never refill a cell."""
    counts, used, selected = Counter(), set(), []
    for batch in sequence:
        for slot, raw_pid in enumerate(batch['ids']):
            pid = tuple(map(int, raw_pid))
            if pid[4] < 0 or pid[4] >= len(base.index):
                raise ValueError('Source base_index outside original index')
            it = base.index[pid[4]]
            actual = (it.eq_i, it.rx_i, it.day_i, it.sig_i, pid[4])
            physical = opaque(it)
            if (pid != actual or physical not in legal or pid[0] != 1
                    or pid[1] not in (1, 3, 4, 6, 8) or pid[2] not in (1, 2, 3)):
                raise ValueError('L role or physical metadata mismatch before IQ read')
            cell = (int(it.tx_i), pid[1], pid[2])
            if cell[0] not in range(6):
                raise ValueError('Legal six-TX source label required')
            if physical not in used and counts[cell] < 2:
                used.add(physical)
                counts[cell] += 1
                selected.append(dict(physical_id=physical, pid=list(pid),
                                     TX=cell[0], RX=cell[1], day=cell[2],
                                     step=batch['step'], epoch=20, slot=slot))
    if not selected or len(selected) > 180:
        raise ValueError('Empty or excessive fixed L diagnostic panel')
    return selected, counts


def metric(z, y):
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if (z.shape != (len(y), 6) or not len(y) or not np.isfinite(z).all()
            or (y < 0).any() or (y > 5).any()):
        raise ValueError('Finite default-head six-class source logits required')
    lp = z - z.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    nll = -lp[np.arange(len(y)), y]
    return dict(n=len(y), accuracy=float((z.argmax(1) == y).mean()),
                hard_label_NLL=float(nll.mean()),
                label_smoothed_CE_001=float((.99 * nll - .01 * lp.mean(1)).mean()))


def summary(z, y, rx, scene=None):
    out = dict(overall=metric(z, y), by_RX={str(r): metric(z[rx == r], y[rx == r])
               for r in sorted(set(rx.tolist()))})
    if scene is not None:
        out['by_scene'] = {SCENES[s]: metric(z[scene == s], y[scene == s])
                           for s in sorted(set(scene.tolist()))}
    return out


def pair(b, m, y):
    bc, mc = b.argmax(1) == y, m.argmax(1) == y
    both, damaged = int((bc & mc).sum()), int((bc & ~mc).sum())
    return dict(n=len(y), corrected=int((~bc & mc).sum()), damaged=damaged,
                both_correct=both, both_wrong=int((~bc & ~mc).sum()),
                net_pp=float(100 * (mc.mean() - bc.mean())),
                baseline_correct_retention=None if both + damaged == 0 else
                both / (both + damaged))


def paired(b, m, y, rx, scene=None):
    out = dict(overall=pair(b, m, y), by_RX={str(r): pair(b[rx == r], m[rx == r], y[rx == r])
               for r in sorted(set(rx.tolist()))})
    if scene is not None:
        out['by_scene'] = {SCENES[s]: pair(b[scene == s], m[scene == s], y[scene == s])
                           for s in sorted(set(scene.tolist()))}
    return out


def bootstrap(project):
    runner = Path.cwd() / 'experiments/cvs_scene_risk20/runner.py'
    spec = importlib.util.spec_from_file_location('_pure_F_observed_bootstrap', runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bootstrap()
    from experiments.cvs_scene_risk20 import design as risk
    from experiments.cvs_F_nontrue20 import design as ref
    if Path(project).resolve() != Path(risk.PROJECT).resolve() or risk.RUN != CURRENT or ref.RUN != REFERENCE:
        raise ValueError('Existing fixed pure-F runtime/project differs')
    return risk, ref


def run(project, output):
    import time
    import torch
    start = time.perf_counter()
    risk, ref = bootstrap(project)
    from experiments.cvs_scene_risk20.model import own_model as own_risk
    from experiments.cvs_F_nontrue20.model import own_model as own_ref
    from experiments.cvs_phase1_stack.runtime import installed
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_fullsource_concat20.channel import Practical, serial
    from experiments.cvs_fullsource_concat20 import design as channel_design
    from experiments.cvs_receiver_residual_capacity4.control import lock
    from experiments.cvs_multi_state_action.dispatch import capacity
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    from cvsrffi.game_tracking.data import opaque_id
    if hasattr(os, 'geteuid') and os.geteuid() == 0:
        raise ValueError('Ordinary account required')
    torch.set_num_threads(2)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    paths = {a: Path(project) / 'runs' / (REFERENCE if a in ('B0', 'B1') else CURRENT)
             / (a + '-s' + str(SEED)) for a in ARMS}
    contract = ref.read(ref.SOURCE)
    roles = contract['role_ids']
    if set(roles) != {'L_s', 'U_s', 'V'} or any(len(roles[k]) != n for k, n in
            (('L_s', 6300), ('U_s', 56700), ('V', 27000))):
        raise ValueError('Original full source role handles required')
    legal = set(roles['L_s'])
    config = ref.config(next(r for r in ref.rows() if r['arm'] == 'B1'))
    actual_args = read(paths['B1'] / 'source/actual_native_args.json')
    args = SimpleNamespace(**actual_args)
    if (not args.a1_source_screen_only or args.baseline_ckpt or args.teacher_ckpt
            or args.use_concat_sat_channel_aug or args.wisig_out_len != 256):
        raise ValueError('Original source-only raw data construction required')
    # Existing training scope masks U and builds no named target datasets.
    # Constructing Dataset indices does not fetch IQ. Only checked L indices below
    # are passed to __getitem__; the U loader is never iterated.
    with installed(config) as native:
        ctx = native._build_ssdg_wisig_data(args, torch.device('cpu'))
    if ctx['named_test_loaders'] or role_ids_from_native(ctx) != roles:
        raise ValueError('Original source-only physical roles differ')
    base = ctx['train_loader'].dataset.base
    if list(base.tx_list) != ref.CLASSES or base.transform is not None or not base.normalize or base.center:
        raise ValueError('Original deterministic float32 raw Dataset required')
    sequences = {a: epoch20(p / 'source/training_sequence.jsonl.gz') for a, p in paths.items()}
    if any(v != sequences['B1'] for v in sequences.values()):
        raise ValueError('Matched final-epoch actual L consumption differs')
    selected, counts = select(sequences['B1'], base, legal, opaque_id)
    metadata = dict(status='SOURCE_ONLY_DIAGNOSTIC_FIXED', model_seed=SEED,
        run_id='20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01',
        checkpoints={a: str(p / 'source/final_ssdg.pth') for a, p in paths.items()},
        source_roles={k: len(v) for k, v in roles.items()}, selected=selected,
        selection='E20 consumed order, first at most two unique L IDs per TX/RX/day cell; no refill',
        cell_counts=[dict(TX=t, RX=r, day=d, n=counts[(t, r, d)]) for t in range(6)
                     for r in ref.FIT for d in (1, 2, 3)],
        new_namespaces=list(NAMESPACES), new_draws=2, scenes=list(SCENES),
        historical_IQ_byte_comparison_available=False,
        reconstruction='Exact named training-stream deterministic reconstruction; historical IQ not retained',
        B0_As_exposure='NoAs reference; reconstructed As were not seen by B0',
        optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False,
        affects_existing_source_promotion=False)
    write(out / 'metadata.json', metadata)

    def raw_at(pid):
        it = base.index[int(pid[4])]
        if opaque_id(it) not in legal:
            raise ValueError('Non-L physical identity before IQ access')
        x, y, domain, meta = base[int(pid[4])]
        if tuple(int(meta[k]) for k in KEYS) != tuple(pid) or x.shape != (2, 256) or not torch.isfinite(x).all():
            raise ValueError('Raw hook input reconstruction differs')
        return x, int(y), int(domain), meta

    raw = torch.stack([raw_at(r['pid'])[0] for r in selected])
    y = np.asarray([r['TX'] for r in selected], np.int64)
    rx = np.asarray([r['RX'] for r in selected], np.int64)
    day = np.asarray([r['day'] for r in selected], np.int64)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        raise ValueError('This fixed diagnostic is assigned physical GPU5')
    with lock(Path(project) / 'runs/receiver_residual_capacity4.lock'):
        state = capacity()[5]
        if len(state['pids'] | {os.getpid()}) > 2 or state['free_mb'] < 12000:
            raise ValueError('GPU5 total-owner capacity/free memory insufficient')
        torch.cuda.init()
        if len(capacity()[5]['pids'] | {os.getpid()}) > 2:
            raise ValueError('GPU5 total-owner capacity changed at CUDA initialization')
    device = 'cuda:0'
    # The first model operation is one finite real-source-packet checkpoint smoke.
    b0c = ref.config(next(r for r in ref.rows() if r['arm'] == 'B0'))
    with numerical_context(ref.FULL_FP32_POLICY):
        smoke_model = own_ref(b0c, device)
        smoke_state = {k: v.detach().cpu().clone() for k, v in smoke_model.state_dict().items()}
        with torch.no_grad():
            smoke = smoke_model.forward_identity_only(raw[:1].to(device))['tx_logits']
        if smoke.shape != (1, 6) or not torch.isfinite(smoke).all() or any(
                not torch.equal(v, smoke_model.state_dict()[k].cpu()) for k, v in smoke_state.items()):
            raise ValueError('Single source-packet frozen smoke failed')
        del smoke_model, smoke_state
    write(out / 'source_smoke.json', dict(status='PASS', source_packets=1,
          checkpoint='B0 FINAL20 native default head', updates=0, target_access=False))

    p = Practical('train')
    audit = serial(p.audit())
    if tuple(channel_design.TRAIN_SCENES) != SCENES:
        raise ValueError('Existing practical training scene set differs')
    for a in ('B1', 'BALANCED', 'RISK'):
        saved = read(paths[a] / 'source/actual_channel.json')
        if saved['configurations'] != audit['configurations']:
            raise ValueError('Actual frozen channel configurations differ')
    observed = torch.empty_like(raw)
    scene_index = np.empty(len(selected), np.int64)
    replay_meta, full_packets = [None] * len(selected), 0
    positions = {}
    for i, r in enumerate(selected):
        positions.setdefault(r['step'], []).append(i)
    saved_scenes = {}
    for a in ('BALANCED', 'RISK'):
        with gzip.open(paths[a] / 'source/mechanism_steps.jsonl.gz', 'rt', encoding='utf8') as f:
            saved_scenes[a] = {r['step']: r['scene_index'] for r in map(json.loads, f)
                               if r['step'] in positions}
        if set(saved_scenes[a]) != set(positions):
            raise ValueError('Saved actual scene slots missing')
    for batch in sequences['B1']:
        step = batch['step']
        if step not in positions:
            continue
        samples = [raw_at(pid) for pid in batch['ids']]
        x = torch.stack([q[0] for q in samples])
        meta = {k: torch.tensor([q[3][k] for q in samples]) for k in KEYS}
        full, records = p.apply(x, meta, epoch=20, step=step)
        full_packets += len(x)
        for i in positions[step]:
            slot = selected[i]['slot']
            single_meta = {k: v[slot:slot+1] for k, v in meta.items()}
            single, one = Practical('train').apply(x[slot:slot+1], single_meta, epoch=20, step=step)
            if not torch.equal(single[0], full[slot]) or one[0] != records[slot]:
                raise ValueError('Independent full-batch/single-packet deterministic replay differs')
            si = SCENES.index(records[slot]['scene'])
            if any(saved_scenes[a][step][slot] != si for a in saved_scenes):
                raise ValueError('Reconstruction differs from saved actual training scene slot')
            observed[i], scene_index[i], replay_meta[i] = full[slot], si, records[slot]
    for a in ('B1', 'BALANCED', 'RISK'):
        sessions = read(paths[a] / 'source/training_result.json')['channel']['sessions']
        if any(sessions.get(k) != v for k, v in p.sessions.items()):
            raise ValueError('Fixed RX/day/eq hardware sessions differ')

    fresh = torch.empty((2, 3, len(selected), 2, 256), dtype=torch.float32)
    fresh_meta = []
    for draw, namespace in enumerate(NAMESPACES):
        for si, scene in enumerate(SCENES):
            cfg = p.configs[scene]
            for i, record in enumerate(selected):
                pid = tuple(record['pid'])
                session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
                seed = p.ch.stable_seed(channel_design.CHANNEL_SEEDS['train'], namespace, scene, pid)
                hw = p.ch.receiver_for_session(cfg, channel_design.CHANNEL_SEEDS['receiver'], session)
                a = np.asarray(raw[i].tolist(), dtype=np.float64)
                normalized, _ = p.ch.normalize_input(a[0] + 1j*a[1])
                signal, m = p.residual.ResidualChannel(cfg, seed, hw).process(normalized)
                if not np.isfinite(signal).all() or m['channel_equalization_applied']:
                    raise ValueError('Invalid new source-only physical observation')
                array = np.ascontiguousarray(np.stack((signal.real, signal.imag)), dtype=np.float32)
                fresh[draw, si, i] = torch.frombuffer(bytearray(array.tobytes()),
                                                     dtype=torch.float32).reshape(2, 256)
                fresh_meta.append(serial(dict(m, draw=draw, scene=scene, namespace=namespace,
                                               sample_id=pid, session_id=session)))
    write(out / 'reconstruction.json', dict(full_batch_replay_packets=full_packets,
          independent_single_replay_packets=len(selected), all_replays_exact_equal=True,
          saved_BALANCED_RISK_scene_slots_equal=True, historical_IQ_byte_comparison=False,
          channel_audit=audit, observed_records=replay_meta, fresh_records=fresh_meta))

    predictions = dict(ids=np.asarray([r['physical_id'] for r in selected]), TX=y, RX=rx,
                       day=day, step=np.asarray([r['step'] for r in selected]),
                       slot=np.asarray([r['slot'] for r in selected]), observed_scene_index=scene_index)
    logits, vdata, frozen, forwards = {}, {}, {}, {}
    with numerical_context(ref.FULL_FP32_POLICY):
        for arm in ARMS:
            design, loader = (ref, own_ref) if arm in ('B0', 'B1') else (risk, own_risk)
            c = design.config(next(r for r in design.rows() if r['arm'] == arm))
            model = loader(c, device)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            seen = 0
            def infer(x):
                nonlocal seen
                parts = []
                with torch.no_grad():
                    for begin in range(0, len(x), 90):
                        z = model.forward_identity_only(x[begin:begin+90].to(device))['tx_logits']
                        if z.shape != (min(90, len(x)-begin), 6) or not torch.isfinite(z).all():
                            raise ValueError('Frozen default-head source prediction invalid')
                        parts.append(np.asarray(z.cpu().tolist(), dtype=np.float32))
                        seen += len(z)
                return np.concatenate(parts)
            zraw, zold = infer(raw), infer(observed)
            znew = infer(fresh.reshape(-1, 2, 256)).reshape(2, 3, len(selected), 6)
            zmatched = np.stack([znew[d, scene_index, np.arange(len(selected))] for d in range(2)])
            if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(
                    not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
                raise ValueError('Frozen source diagnostic changed model state or parameter gradients')
            frozen[arm] = dict(weights_buffers_unchanged=True, parameter_grads_none=True,
                               optimizer_updates=0, EMA_updates=0, head_fit_updates=0)
            forwards[arm] = seen
            logits[arm] = dict(raw=zraw, observed_reconstructed=zold,
                              new_paired=zmatched.reshape(-1, 6),
                              **{'new_draw'+str(d)+'_paired': zmatched[d] for d in range(2)},
                              **{s: znew[:, j].reshape(-1, 6) for j, s in enumerate(SCENES)})
            predictions.update({arm+'_raw': zraw, arm+'_observed_reconstructed': zold,
                                arm+'_new_all_scenes': znew, arm+'_new_paired': zmatched})
            # No new V inference: reuse only existing legal source qualification.
            with np.load(paths[arm] / 'source_qualification/predictions.npz', allow_pickle=False) as f:
                v = {k: f[k].copy() for k in ('ids', 'TX', 'RX', 'day', 'clean', 'satellite')}
            if (v['clean'].shape != (1800, 6) or v['satellite'].shape != (2, 3, 1800, 6)
                    or set(v['ids'].tolist()) - set(roles['V'])):
                raise ValueError('Original frozen source V1800 two-draw readout differs')
            if vdata and any(not np.array_equal(v[k], vdata['B0'][k]) for k in ('ids', 'TX', 'RX', 'day')):
                raise ValueError('Paired V physical IDs or labels differ')
            vdata[arm] = v
            predictions.update({arm+'_V_clean': v['clean'], arm+'_V_satellite': v['satellite']})
            del model, before
    predictions.update({'V_'+k: vdata['B0'][k] for k in ('ids', 'TX', 'RX', 'day')})
    result = dict(status='SOURCE_ONLY_FROZEN_READOUT_COMPLETE', metadata=metadata,
                  L={}, V={}, method_pairs={}, L_observed_to_new_pairs={},
                  forwards=forwards, source_smoke_packets=1, state_checks=frozen,
                  V_new_forward_packets=0, model_seed_count=1, sample_SD=None,
                  target_access=False, target_scores_read=False, existing_source_decision_unchanged=True)
    for arm in ARMS:
        result['L'][arm] = {}
        for view, z in logits[arm].items():
            rep = len(z) // len(y)
            result['L'][arm][view] = summary(z, np.tile(y, rep), np.tile(rx, rep),
                                           np.tile(scene_index, rep) if view == 'observed_reconstructed' or 'paired' in view else None)
        v = vdata[arm]
        result['V'][arm] = dict(clean=summary(v['clean'], v['TX'], v['RX']),
            new_by_scene={s: summary(v['satellite'][:, j].reshape(-1, 6),
                np.tile(v['TX'], 2), np.tile(v['RX'], 2)) for j, s in enumerate(SCENES)},
            new_per_draw={str(d): {s: summary(v['satellite'][d, j], v['TX'], v['RX'])
                for j, s in enumerate(SCENES)} for d in range(2)})
        result['L_observed_to_new_pairs'][arm] = paired(
            np.tile(logits[arm]['observed_reconstructed'], (2, 1)), logits[arm]['new_paired'],
            np.tile(y, 2), np.tile(rx, 2), np.tile(scene_index, 2))
    for a, b in (('RISK', 'B0'), ('RISK', 'B1'), ('RISK', 'BALANCED'), ('BALANCED', 'B1')):
        key = a + '_minus_' + b
        result['method_pairs'][key] = dict(L={}, V={})
        for view, z in logits[a].items():
            rep = len(z) // len(y)
            result['method_pairs'][key]['L'][view] = paired(logits[b][view], z,
                np.tile(y, rep), np.tile(rx, rep),
                np.tile(scene_index, rep) if view == 'observed_reconstructed' or 'paired' in view else None)
        va, vb = vdata[a], vdata[b]
        result['method_pairs'][key]['V'] = dict(clean=paired(vb['clean'], va['clean'], va['TX'], va['RX']),
            new_by_scene={s: paired(vb['satellite'][:, j].reshape(-1, 6), va['satellite'][:, j].reshape(-1, 6),
                np.tile(va['TX'], 2), np.tile(va['RX'], 2)) for j, s in enumerate(SCENES)})
    result['wall_seconds'] = time.perf_counter() - start
    np.savez_compressed(out / 'predictions.npz', **predictions)
    write(out / 'analysis.json', result)
    for arm in ARMS:
        row = out / (arm + '-s' + str(SEED))
        row.mkdir(exist_ok=False)
        write(row / 'analysis.json', dict(run_id=metadata['run_id'], arm=arm, model_seed=SEED,
            L=result['L'][arm], V=result['V'][arm],
            observed_to_new_pairs=result['L_observed_to_new_pairs'][arm],
            state_checks=frozen[arm], source_forwards=forwards[arm],
            standard_CE='hard_label_NLL; default logits, no temperature',
            training_CE='label_smoothed_CE_001; smoothing=0.01',
            target_access=False, optimizer_updates=0, EMA_updates=0, head_fit_updates=0))
    write(out / 'complete.json', dict(status=result['status'], source_L_selected=len(selected),
        source_forward_packets=sum(forwards.values())+1, new_V_forwards=0,
        all_weights_buffers_unchanged=True, updates=0, target_access=False,
        historical_IQ_byte_comparison=False, existing_source_promotion_unchanged=True))
    print(json.dumps(dict(status=result['status'], selected=len(selected), forwards=forwards,
                         source_smoke_packets=1, target_access=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.project, args.output)
