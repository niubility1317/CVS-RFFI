"""Add only two frozen rawAs20 checkpoints to the existing source-fit panel.

No target artifacts are opened. Existing B0/B1 logits are reused without a
model load or forward; historical As IQ was not saved, so reconstruction does
not constitute a byte comparison with historical received IQ.
"""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RUN = '20261019-diagnostic-F-rawAs-sourceFit-e20-s2026092701-m2-r01'
NR = '20261019-phase1-F-rawAs-e20-s2026092701-m2-r01'
DR = '20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01'
OLD_RELEASE = 'cvs_F_observedChannel_E20_20261019_r01'
SEED = 2026092701
NEW_ARMS = ('RAW_NOAS', 'RAW_AS')
ARMS = ('B0', 'B1', *NEW_ARMS)
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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_selected(selected, base, legal, opaque):
    """Validate saved physical metadata/roles before any Dataset IQ access."""
    if len(selected) != 180 or len({r['physical_id'] for r in selected}) != 180:
        raise ValueError('Reuse the complete existing180 physical L panel; no reselection')
    for r in selected:
        pid = tuple(r['pid'])
        if len(pid) != 5 or pid[4] < 0 or pid[4] >= len(base.index):
            raise ValueError('Original L base_index required before IQ access')
        it = base.index[pid[4]]
        if (pid != (it.eq_i, it.rx_i, it.day_i, it.sig_i, pid[4])
                or opaque(it) != r['physical_id'] or r['physical_id'] not in legal
                or pid[0] != 1 or pid[1] not in (1, 3, 4, 6, 8) or pid[2] not in (1, 2, 3)
                or (r['TX'], r['RX'], r['day']) != (it.tx_i, it.rx_i, it.day_i)
                or r['TX'] not in range(6) or r['epoch'] != 20
                or r['step'] not in range(4219, 4441) or r['slot'] not in range(128)):
            raise ValueError('Saved L role/physical identity/step/slot differs before IQ access')


def run(project, output):
    import time
    import torch
    start = time.perf_counter()
    boot = module_at('_rawAs_source_fit_runtime',
                     Path.cwd()/'experiments/cvs_F_rawAs20/runner.py')
    boot.bootstrap()
    from experiments.cvs_F_rawAs20 import design as d
    from experiments.cvs_F_rawAs20.model import own_model
    from experiments.cvs_F_nontrue20 import design as ref
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
    if d.RUN != NR or Path(d.PROJECT).resolve() != Path(project).resolve():
        raise ValueError('Existing rawAs20 immutable runtime/project differs')
    root, oldroot = Path(project)/'runs'/NR, Path(project)/'runs'/DR
    decision = read(root/'source_matrix_decision.json')
    if decision['status'] != 'SOURCE_DECISION_FIXED' or decision['formal_scores_read'] is not False:
        raise ValueError('Registered source decision must already be fixed without formal scores')
    paths = {a: root/(a+'-s'+str(SEED)) for a in NEW_ARMS}
    for a, path in paths.items():
        complete = read(path/'source/completion.json')
        qualified = read(path/'source_qualification/complete.json')
        if (complete['epoch'] != 20 or complete['optimizer_steps'] != 4440
                or qualified['status'] != 'SOURCE_READOUT_COMPLETE' or qualified['target_access']):
            raise ValueError('Both own FINAL20 source completions/readouts must exist first')
    old = module_at('_existing_source_fit_helpers', Path(project)/'releases'/OLD_RELEASE/
                    'analysis/cvs_pure_F_observed_fit_20261019.py')
    old_done = read(oldroot/'complete.json')
    oldmeta = read(oldroot/'metadata.json')
    oldanalysis = read(oldroot/'analysis.json')
    reconstruction = read(oldroot/'reconstruction.json')
    if (old_done['status'] != 'SOURCE_ONLY_FROZEN_READOUT_COMPLETE'
            or oldmeta['run_id'] != DR or oldmeta['model_seed'] != SEED
            or oldmeta['new_namespaces'] != list(NAMESPACES)
            or oldmeta['scenes'] != list(SCENES) or oldmeta['target_access']
            or not reconstruction['all_replays_exact_equal']
            or not reconstruction['saved_BALANCED_RISK_scene_slots_equal']):
        raise ValueError('Existing frozen source diagnostic handles differ')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    contract = ref.read(ref.SOURCE)
    roles = contract['role_ids']
    if set(roles) != {'L_s', 'U_s', 'V'} or any(len(roles[k]) != n for k, n in
            (('L_s', 6300), ('U_s', 56700), ('V', 27000))):
        raise ValueError('Original full L6300/U56700/V27000 role handles required')
    config = ref.config(next(r for r in ref.rows() if r['arm'] == 'B1'))
    original_B1 = Path(project)/'runs'/ref.RUN/('B1-s'+str(SEED))
    args = SimpleNamespace(**read(original_B1/'source/actual_native_args.json'))
    if (not args.a1_source_screen_only or args.baseline_ckpt or args.teacher_ckpt
            or args.use_concat_sat_channel_aug or args.wisig_out_len != 256):
        raise ValueError('Existing source-only raw Dataset builder required')
    with installed(config) as native:
        ctx = native._build_ssdg_wisig_data(args, torch.device('cpu'))
    if ctx['named_test_loaders'] or role_ids_from_native(ctx) != roles:
        raise ValueError('Source-only Dataset roles differ')
    base = ctx['train_loader'].dataset.base
    if list(base.tx_list) != ref.CLASSES or base.transform is not None or not base.normalize or base.center:
        raise ValueError('Original deterministic float32 Dataset policy required')
    selected, legal = oldmeta['selected'], set(roles['L_s'])
    check_selected(selected, base, legal, opaque_id)
    with np.load(oldroot/'predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k].copy() for k in f.files}
    ids = np.asarray([r['physical_id'] for r in selected])
    y, rx, day = (np.asarray([r[k] for r in selected], np.int64) for k in ('TX', 'RX', 'day'))
    if (not np.array_equal(saved['ids'], ids) or not np.array_equal(saved['TX'], y)
            or not np.array_equal(saved['RX'], rx) or not np.array_equal(saved['day'], day)
            or not np.array_equal(saved['step'], [r['step'] for r in selected])
            or not np.array_equal(saved['slot'], [r['slot'] for r in selected])):
        raise ValueError('Reuse exact existing L IDs/labels/steps/slots')
    scene_index = saved['observed_scene_index']
    if scene_index.shape != (180,) or set(scene_index.tolist()) - set(range(3)):
        raise ValueError('Existing observed scene slots differ')
    batches = {b['step']: b for b in old.epoch20(paths['RAW_AS']/'source/training_sequence.jsonl.gz')}
    positions = {}
    for i, r in enumerate(selected):
        if batches[r['step']]['ids'][r['slot']] != r['pid']:
            raise ValueError('RAW_AS did not consume the saved physical L at that actual step/slot')
        positions.setdefault(r['step'], []).append(i)
    with gzip.open(paths['RAW_AS']/'source/mechanism_steps.jsonl.gz', 'rt', encoding='utf8') as f:
        consumed = {r['step']: r for r in map(json.loads, f) if r['step'] in positions}
    if set(consumed) != set(positions) or any(r['As_packets'] != 128 or
            not r['raw_first_forward_verified'] or r['hook_raw_MSE'] != 0. for r in consumed.values()):
        raise ValueError('Actual RAW_AS full-As consumption witness differs')

    checked_raw_reads = 0
    def raw_at(pid):
        # Only already-validated L metadata reaches Dataset IQ access.
        nonlocal checked_raw_reads
        if len(pid) != 5 or int(pid[4]) < 0 or int(pid[4]) >= len(base.index):
            raise ValueError('Invalid original L base_index before IQ access')
        it = base.index[int(pid[4])]
        if (tuple(pid) != (it.eq_i, it.rx_i, it.day_i, it.sig_i, int(pid[4]))
                or opaque_id(it) not in legal or it.eq_i != 1
                or it.rx_i not in ref.FIT or it.day_i not in (1, 2, 3)):
            raise ValueError('Invalid L physical metadata or role before IQ access')
        x, _, _, meta = base[int(pid[4])]
        checked_raw_reads += 1
        if tuple(int(meta[k]) for k in KEYS) != tuple(pid) or x.shape != (2, 256) or not torch.isfinite(x).all():
            raise ValueError('Original hook raw reconstruction differs')
        return x
    raw = torch.stack([raw_at(r['pid']) for r in selected])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6':
        raise ValueError('Registered source-fit diagnostic uses physical GPU6')
    with lock(Path(project)/'runs/receiver_residual_capacity4.lock'):
        state = capacity()[6]
        if len(state['pids']|{os.getpid()}) > 2 or state['free_mb'] < 12000:
            raise ValueError('GPU6 total-owner cap2/free12GB insufficient')
        torch.cuda.init()
        if len(capacity()[6]['pids']|{os.getpid()}) > 2:
            raise ValueError('GPU6 capacity changed at CUDA initialization')
    first_config = d.config(next(r for r in d.rows() if r['arm'] == 'RAW_NOAS'))
    with numerical_context(d.FULL_FP32_POLICY):
        first_model = own_model(first_config, 'cuda:0')
        first_state = {k: v.detach().cpu().clone() for k, v in first_model.state_dict().items()}
        with torch.no_grad():
            smoke = first_model.forward_identity_only(raw[:1].to('cuda:0'))['tx_logits']
        if smoke.shape != (1, 6) or not torch.isfinite(smoke).all():
            raise ValueError('New FINAL20 single-source-packet smoke failed')
    write(out/'source_smoke.json', dict(status='PASS', source_packets=1, optimizer_updates=0,
                                        checkpoint='RAW_NOAS FINAL20', target_access=False))
    p = Practical('train')
    audit = serial(p.audit())
    actual = read(paths['RAW_AS']/'source/actual_channel.json')
    if (actual['configurations'] != reconstruction['channel_audit']['configurations']
            or audit['configurations'] != actual['configurations']
            or tuple(channel_design.TRAIN_SCENES) != SCENES):
        raise ValueError('Original actual channel Config/scene recipe differs')
    observed = torch.empty_like(raw)
    replay_packets = 0
    for step, indices in positions.items():
        batch = batches[step]
        xb = torch.stack([raw_at(pid) for pid in batch['ids']])
        meta = {k: torch.tensor([pid[j] for pid in batch['ids']]) for j, k in enumerate(KEYS)}
        full, records = p.apply(xb, meta, epoch=20, step=step)
        replay_packets += len(xb)
        for i in indices:
            slot = selected[i]['slot']
            record = records[slot]
            if (SCENES.index(record['scene']) != int(scene_index[i])
                    or record != reconstruction['observed_records'][i]):
                raise ValueError('Same named observed channel reconstruction differs from prior diagnostic')
            observed[i] = full[slot]
    sessions = read(paths['RAW_AS']/'source/training_result.json')['channel']['sessions']
    if any(sessions.get(k) != v for k, v in p.sessions.items()):
        raise ValueError('RAW_AS actual RX/day/eq hardware sessions differ')
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
                    raise ValueError('Same prior fresh namespace/channel metadata differs')
                fresh_records.append(record)
                arr = np.ascontiguousarray(np.stack((signal.real, signal.imag)), dtype=np.float32)
                fresh[draw, si, i] = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32).reshape(2, 256)
    metadata = dict(run_id=RUN, model_seed=SEED, reused_diagnostic=DR, selected=selected,
        new_namespaces=list(NAMESPACES), scenes=list(SCENES), raw_input='original Dataset float32 raw',
        RAW_AS_consumed_matching_step_slot=True, RAW_AS_scene_evidence='Deterministic producer/config/'
        'seed/session and actual128 As consumption; no persisted per-packet scene log',
        RAW_NOAS_As_exposure='NoAs model; this reconstructed As was not seen during its training',
        observed_reconstruction_metadata_matches_prior=True, fresh_metadata_matches_prior=True,
        historical_IQ_byte_comparison=False, source_promotion_fixed_before_readout=decision,
        source_promotion_changed=False, target_access=False, optimizer_updates=0, EMA_updates=0,
        head_fit_updates=0, new_V_forward_packets=0)
    metadata['channel_generations'] = dict(reconstructed_training_full_batch_packets=replay_packets,
                                           prior_new_namespace_packets=1080)
    metadata['explicit_checked_L_raw_fetches'] = checked_raw_reads
    metadata['CE_definitions'] = dict(hard_label_NLL='Standard true-label CE on default logits, '
        'no temperature or smoothing', label_smoothed_CE_001='Training label smoothing0.01 CE')
    write(out/'metadata.json', metadata)
    logits = {a: dict(raw=saved[a+'_raw'], observed_reconstructed=saved[a+'_observed_reconstructed'],
                         new_paired=saved[a+'_new_paired'].reshape(-1, 6),
                         **{'new_draw'+str(k)+'_paired': saved[a+'_new_paired'][k] for k in range(2)},
                         **{s: saved[a+'_new_all_scenes'][:, j].reshape(-1, 6) for j, s in enumerate(SCENES)})
              for a in ('B0', 'B1')}
    predictions = {k: saved[k] for k in ('ids', 'TX', 'RX', 'day', 'step', 'slot', 'observed_scene_index',
                   'V_ids', 'V_TX', 'V_RX', 'V_day')}
    vdata = {a: dict(clean=saved[a+'_V_clean'], satellite=saved[a+'_V_satellite'],
                    ids=saved['V_ids'], TX=saved['V_TX'], RX=saved['V_RX'], day=saved['V_day'])
             for a in ('B0', 'B1')}
    for a in ('B0', 'B1'):
        predictions.update({k: saved[k] for k in saved if k.startswith(a+'_')})
    state_checks, forwards = {}, {}
    with numerical_context(d.FULL_FP32_POLICY):
        for arm in NEW_ARMS:
            if arm == 'RAW_NOAS':
                model, before = first_model, first_state
            else:
                c = d.config(next(r for r in d.rows() if r['arm'] == arm))
                model = own_model(c, 'cuda:0')
                before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            count = 0
            def infer(x):
                nonlocal count
                values = []
                with torch.no_grad():
                    for begin in range(0, len(x), 90):
                        z = model.forward_identity_only(x[begin:begin+90].to('cuda:0'))['tx_logits']
                        if z.shape != (min(90, len(x)-begin), 6) or not torch.isfinite(z).all():
                            raise ValueError('Finite frozen default-head source logits required')
                        values.append(np.asarray(z.cpu().tolist(), np.float32)); count += len(z)
                return np.concatenate(values)
            zr, zo = infer(raw), infer(observed)
            zn = infer(fresh.reshape(-1, 2, 256)).reshape(2, 3, 180, 6)
            zm = np.stack([zn[k, scene_index, np.arange(180)] for k in range(2)])
            if (count != 1440 or model.id_backbone.encoder._full_eval_packets != count+int(arm == 'RAW_NOAS')
                    or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters())
                    or any(not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items())):
                raise ValueError('Source forward count/frozen model state differs')
            state_checks[arm] = dict(weights_buffers_unchanged=True, parameter_grads_none=True,
                                     optimizer_updates=0, EMA_updates=0, head_fit_updates=0)
            forwards[arm] = count
            logits[arm] = dict(raw=zr, observed_reconstructed=zo, new_paired=zm.reshape(-1, 6),
                **{'new_draw'+str(k)+'_paired': zm[k] for k in range(2)},
                **{s: zn[:, j].reshape(-1, 6) for j, s in enumerate(SCENES)})
            predictions.update({arm+'_raw': zr, arm+'_observed_reconstructed': zo,
                                arm+'_new_all_scenes': zn, arm+'_new_paired': zm})
            with np.load(paths[arm]/'source_qualification/predictions.npz', allow_pickle=False) as f:
                v = {k: f[k].copy() for k in ('ids', 'TX', 'RX', 'day', 'clean', 'satellite')}
            if (v['clean'].shape != (1800, 6) or v['satellite'].shape != (2, 3, 1800, 6)
                    or set(v['ids'].tolist()) - set(roles['V'])
                    or any(not np.array_equal(v[k], vdata['B0'][k]) for k in ('ids', 'TX', 'RX', 'day'))):
                raise ValueError('Exact existing V1800 physical readout required')
            vdata[arm] = v
            predictions.update({arm+'_V_clean': v['clean'], arm+'_V_satellite': v['satellite']})
            if arm == 'RAW_NOAS':
                del first_model, first_state
            del model, before
    result = dict(status='SOURCE_ONLY_FROZEN_READOUT_COMPLETE', run_id=RUN, metadata=metadata,
        L={a: oldanalysis['L'][a] for a in ('B0', 'B1')},
        V={a: oldanalysis['V'][a] for a in ('B0', 'B1')}, method_pairs={},
        L_observed_to_new_pairs={}, state_checks=state_checks, forwards=forwards,
        reused_B0_B1_forward_packets=0, new_V_forward_packets=0, source_smoke_packets=1,
        model_seed_count=1, sample_SD=None, source_promotion_changed=False, target_access=False)
    for arm in NEW_ARMS:
        result['L'][arm] = {view: old.summary(z, np.tile(y, len(z)//180), np.tile(rx, len(z)//180),
            np.tile(scene_index, len(z)//180) if view == 'observed_reconstructed' or 'paired' in view else None)
            for view, z in logits[arm].items()}
        v = vdata[arm]
        result['V'][arm] = dict(clean=old.summary(v['clean'], v['TX'], v['RX']),
            new_by_scene={s: old.summary(v['satellite'][:, j].reshape(-1, 6),
                np.tile(v['TX'], 2), np.tile(v['RX'], 2)) for j, s in enumerate(SCENES)},
            new_per_draw={str(k): {s: old.summary(v['satellite'][k, j], v['TX'], v['RX'])
                for j, s in enumerate(SCENES)} for k in range(2)})
        result['L_observed_to_new_pairs'][arm] = old.paired(np.tile(logits[arm]['observed_reconstructed'], (2, 1)),
            logits[arm]['new_paired'], np.tile(y, 2), np.tile(rx, 2), np.tile(scene_index, 2))
    for a, b in (('RAW_AS', 'B0'), ('RAW_AS', 'B1'), ('RAW_AS', 'RAW_NOAS'), ('RAW_NOAS', 'B0')):
        entry = dict(L={}, V={})
        for view, z in logits[a].items():
            rep = len(z)//180
            entry['L'][view] = old.paired(logits[b][view], z, np.tile(y, rep), np.tile(rx, rep),
                np.tile(scene_index, rep) if view == 'observed_reconstructed' or 'paired' in view else None)
        va, vb = vdata[a], vdata[b]
        entry['V'] = dict(clean=old.paired(vb['clean'], va['clean'], va['TX'], va['RX']),
            new_by_scene={s: old.paired(vb['satellite'][:, j].reshape(-1, 6), va['satellite'][:, j].reshape(-1, 6),
                np.tile(va['TX'], 2), np.tile(va['RX'], 2)) for j, s in enumerate(SCENES)})
        result['method_pairs'][a+'_minus_'+b] = entry
    result['raw_correct_retention'] = {key: dict(
        L_raw=v['L']['raw']['overall']['baseline_correct_retention'],
        V_clean=v['V']['clean']['overall']['baseline_correct_retention'])
        for key, v in result['method_pairs'].items()}
    result['wall_seconds'] = time.perf_counter()-start
    np.savez_compressed(out/'predictions.npz', **predictions)
    write(out/'analysis.json', result)
    for a in NEW_ARMS:
        row = out/(a+'-s'+str(SEED)); row.mkdir(exist_ok=False)
        write(row/'analysis.json', dict(run_id=RUN, model_seed=SEED, arm=a,
            L=result['L'][a], V=result['V'][a], state_checks=state_checks[a],
            observed_to_new_pairs=result['L_observed_to_new_pairs'][a], source_forwards=forwards[a],
            source_smoke_packets=int(a == 'RAW_NOAS'),
            optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False))
    write(out/'complete.json', dict(status=result['status'], source_L_selected=180,
        new_CP_forward_packets=2880, source_smoke_packets=1, source_forward_packets=2881,
        reused_B0_B1_forward_packets=0, new_V_forward_packets=0, weights_buffers_unchanged=True,
        optimizer_updates=0, EMA_updates=0, head_fit_updates=0, target_access=False,
        historical_IQ_byte_comparison=False, source_promotion_changed=False))
    print(json.dumps(dict(status=result['status'], new_source_NN=2881, old_CP_NN=0, V_new_NN=0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True); p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.project, a.output)
