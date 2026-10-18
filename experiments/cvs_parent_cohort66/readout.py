"""Exact As20 *student* cohort; cached source observations, no new channel draws."""
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np

RUN = '20261018-diagnostic-FcrossRX-parent20-cohort-s2026092701-m1-r01'
RELEASE = 'cvs_FcrossRX_parent20_cohort_20261018_r01'
PARENT_RUN = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
PARENT_ROW = 'B01-s2026092701'
SEED = 2026092701
COUNT = 1800
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
RX, DAYS = (1, 3, 4, 6, 8), (1, 2, 3)
STATES, LOCKS = ('LOS', 'PARTIAL', 'BLOCKED'), ('locked', 'degraded', 'unlocked')
SOURCE_NN, SMOKE_NN = 12600, 2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_metadata(value):
    expected = dict(status='SOURCE_READOUT_COMPLETE', run_id=RUN, model_seed=SEED,
        checkpoint_run=PARENT_RUN, checkpoint_row=PARENT_ROW, checkpoint_stage='stage20',
        checkpoint_key='model', checkpoint_epoch=20, checkpoint_optimizer_steps=4440,
        checkpoint_selection='final_only', source_role='V', source_unique_V=COUNT,
        channel_draws=2, scenes=list(SCENES), condition='full', condition_index=0,
        NN=12602, source_NN=SOURCE_NN, smoke_NN=SMOKE_NN, training_updates=0,
        head_fit_updates=0, teacher_training_updates=0, teacher_bank_created=False,
        U_truth_read=False, U_records_used=0, target_access=False, formal_scores_read=False,
        query_access=False, new_channel_calls=0, model_state_unchanged=True,
        all_parameters_frozen=True, model_parameters=900722, model_state_keys=206,
        default_head_shape=[6, 160], default_head_scale=30., eval_counter=12602,
        train_counter=0, source_role_counts=dict(L_s=6300, U_s=56700, V=27000))
    require(isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items()),
            'Exact As20 student/source-only metadata or forward budget differs')
    for key in ('checkpoint', 'source_contract', 'source_panel', 'source_observations'):
        require(isinstance(value.get(key), str) and bool(value[key]), 'Missing concrete source handle: ' + key)
    for key in ('wall_seconds', 'smoke_wall_seconds', 'source_wall_seconds'):
        require(isinstance(value.get(key), (int, float)) and not isinstance(value[key], bool)
                and np.isfinite(value[key]) and value[key] >= 0, 'Invalid measured wall time: ' + key)
    return value


def validate_predictions(value, count=COUNT):
    require(set(value) == {'ids', 'TX', 'RX', 'day', 'clean', 'satellite'}, 'Exact source prediction fields required')
    ids = np.asarray(value['ids'])
    require(ids.shape == (count,) and ids.dtype.kind in 'US' and len(set(ids.tolist())) == count,
            'Unique ordered source physical IDs required')
    for key, allowed in (('TX', range(6)), ('RX', RX), ('day', DAYS)):
        x = np.asarray(value[key])
        require(x.shape == (count,) and x.dtype.kind in 'iu' and set(x.tolist()) == set(allowed),
                'Source role labels differ: ' + key)
    require(count % 90 == 0 and Counter(zip(value['TX'].tolist(), value['RX'].tolist(), value['day'].tolist()))
            == Counter({(t, r, d): count // 90 for t in range(6) for r in RX for d in DAYS}),
            'Balanced original TX/RX/day source panel required')
    for key, shape in (('clean', (count, 6)), ('satellite', (2, 3, count, 6))):
        z = np.asarray(value[key])
        require(z.shape == shape and z.dtype.kind == 'f' and np.isfinite(z).all(),
                'Finite original default-head source logits required: ' + key)
    return value


def cohort_metrics(parent, current, physics=None):
    """Pure saved-array comparison; same As20 clean-correct IDs in every row.

    physics, when supplied, contains condition0 state/lock arrays [2,3,N].
    This function does not read files, access U labels, fit a head or infer IQ.
    """
    n = len(parent['ids'])
    validate_predictions(parent, n)
    validate_predictions(current, n)
    require(all(np.array_equal(parent[k], current[k]) for k in ('ids', 'TX', 'RX', 'day')),
            'Same ordered source physical IDs and labels required; no silent reordering')
    if physics is not None:
        require(set(physics) == {'state', 'lock'}, 'Only saved source state/lock strata accepted')
        for key in ('state', 'lock'):
            v = np.asarray(physics[key])
            require(v.shape == (2, 3, n) and v.dtype.kind in 'iu' and np.all((v >= 0) & (v < 3)),
                    'Condition0 source state/lock indices differ')
    clean_correct = np.asarray(parent['clean']).argmax(1) == parent['TX']
    records = []
    for scene in ('clean', *SCENES):
        si = None if scene == 'clean' else SCENES.index(scene)
        for draw in ((-1,) if si is None else (-1, 0, 1)):
            draws = (0, 1) if draw == -1 else (draw,)
            copies = 1 if si is None else len(draws)
            y = np.tile(parent['TX'], copies)
            cohort = np.tile(clean_correct, copies)
            rx, day = np.tile(parent['RX'], copies), np.tile(parent['day'], copies)
            zp = parent['clean'] if si is None else np.concatenate([parent['satellite'][d, si] for d in draws])
            zc = current['clean'] if si is None else np.concatenate([current['satellite'][d, si] for d in draws])
            masks = [('overall', 'ALL', np.ones(len(y), bool))]
            for dimension, values, allowed in (('RX', rx, RX), ('day', day, DAYS), ('TX', y, range(6))):
                masks.extend((dimension, str(v), values == v) for v in allowed)
            if si is not None and physics is not None:
                st = np.concatenate([physics['state'][d, si] for d in draws])
                lk = np.concatenate([physics['lock'][d, si] for d in draws])
                masks.extend(('state', name, st == i) for i, name in enumerate(STATES))
                masks.extend(('lock', name, lk == i) for i, name in enumerate(LOCKS))
                masks.extend(('state_lock', s + '/' + l, (st == i) & (lk == j))
                             for i, s in enumerate(STATES) for j, l in enumerate(LOCKS))
            pgood, cgood = zp.argmax(1) == y, zc.argmax(1) == y
            other = zc.copy()
            other[np.arange(len(y)), y] = -np.inf
            margin = (zc[np.arange(len(y)), y] - other.max(1)) / 30.
            for dimension, name, mask in masks:
                cc = cohort & mask
                den = int(cc.sum())
                retained = int((cgood & cc).sum())
                baseline = int((pgood & cc).sum())
                repair, damage = int((~pgood & cgood & cc).sum()), int((pgood & ~cgood & cc).sum())
                records.append(dict(scene=scene, draw=draw, dimension=dimension, stratum=name,
                    count=int(mask.sum()), parent_clean_correct_count=den,
                    parent_retained=baseline, current_retained=retained,
                    parent_retention=baseline / den if den else None,
                    current_retention=retained / den if den else None,
                    retention_delta_pp=100. * (retained - baseline) / den if den else None,
                    repairs_within_parent_cohort=repair, new_errors_within_parent_cohort=damage,
                    mean_margin_over30_within_parent_cohort=float(margin[cc].mean()) if den else None,
                    cohort_reference='P43 B01 As20 student clean correct; not NoAs30 or EMA',
                    empty_cohort=not den, small_cohort=den < 30))
    return dict(parent_clean_correct_count=int(clean_correct.sum()), unique_source_V=n,
                channel_draws=2, model_seed_count=1, records=records,
                NN=0, IQ_reads=0, checkpoint_reads=0, updates=0, target_access=False)


def execute(device='cuda:0'):
    import torch
    from experiments.cvs_u_quality66 import design as d, source
    from experiments.cvs_u_quality66.quality import panel
    from experiments.cvs_u_quality66.channel import tensor_iq
    from experiments.cvs_noise_shape_factorial43 import model as original
    from experiments.cvs_equivariant_identity.precision import numerical_context

    require(device in ('cpu', 'cuda:0'), 'Only declared local CPU or assigned CUDA device')
    out = Path(d.PROJECT) / 'runs' / RUN
    require(not out.exists(), 'Immutable parent cohort run exists; reconcile rather than overwrite')
    out.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        config = d.config(d.rows()[0])
        checkpoint = Path(config['parent_source']) / 'final_ssdg.pth'
        with numerical_context(d.FULL_FP32_POLICY):
            cp = source.parent_payload(config)
            require(cp['run_id'] == PARENT_RUN and cp['candidate_id'] == PARENT_ROW
                    and cp['epoch'] == 20 and cp['a1_ema_successful_updates'] == 4440
                    and cp['checkpoint_selection'] == 'final_only', 'Exact source-fixed parent identity differs')
            model = original.build(d.old_config(), device)
            model.load_state_dict(cp['model'], strict=True)  # Deliberately student, never ema_model.
            model.eval().requires_grad_(False)
            head = model.id_backbone.encoder.core.id_backbone.cls_head
            require(len(model.state_dict()) == 206 and sum(p.numel() for p in model.parameters()) == 900722
                    and list(head.weight.shape) == [6, 160] and float(head.scale) == 30., 'Fixed F/default cosine head differs')
            require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
                    'Every parent parameter and module must be frozen')
            before = {k: v.clone() for k, v in model.state_dict().items()}
            encoder = model.id_backbone.encoder
            require(encoder._full_eval_packets == 0 and encoder._full_train_packets == 0, 'Unexpected pre-readout network calls')
            tick = time.perf_counter()
            with torch.no_grad():
                smoke = model.forward_identity_only(torch.zeros(2, 2, 256, device=device))['tx_logits']
            require(smoke.shape == (2, 6) and bool(torch.isfinite(smoke).all())
                    and encoder._full_eval_packets == 2, 'Real parent checkpoint no-query smoke failed')
            smoke_wall = time.perf_counter() - tick
            write(out / 'real_checkpoint_smoke.json', dict(status='PASS', checkpoint=str(checkpoint),
                checkpoint_key='model', NN=2, source_IQ_read_before_smoke=0,
                query_access=False, target_access=False, training_updates=0, head_fit_updates=0,
                wall_seconds=smoke_wall))
            tick = time.perf_counter()
            data, roles = panel(torch)
            observed = d.read(d.PANEL / 'observations/completion.json')
            require(observed['draws'] == 2 and observed['conditions'][0] == 'full'
                    and observed['scenes'] == list(SCENES) and not observed['target_access'], 'Exact cached source observation order differs')
            observations = np.load(d.PANEL / 'observations/iq.npy', mmap_mode='r', allow_pickle=False)
            require(observations.shape == (2, 3, len(observed['conditions']), COUNT, 2, 256),
                    'Cached source observation dimensions differ')
            clean = np.empty((COUNT, 6), np.float32)
            satellite = np.empty((2, 3, COUNT, 6), np.float32)
            with torch.no_grad():
                for i in range(0, COUNT, 90):
                    clean[i:i+90] = model.forward_identity_only(data['x'][i:i+90].to(device))['tx_logits'].cpu().tolist()
                for draw in range(2):
                    for si in range(3):
                        for i in range(0, COUNT, 90):
                            z = model.forward_identity_only(tensor_iq(observations[draw, si, 0, i:i+90], device))['tx_logits']
                            satellite[draw, si, i:i+90] = z.cpu().tolist()
            require(encoder._full_eval_packets == 12602 and encoder._full_train_packets == 0
                    and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
                    and all(p.grad is None for p in model.parameters()), 'Frozen original parent state or forward cost changed')
            arrays = dict(ids=np.asarray(data['ids']), TX=np.asarray(data['y'].tolist(), np.int64),
                RX=np.asarray(data['rx'].tolist(), np.int64), day=np.asarray(data['day'].tolist(), np.int64),
                clean=clean, satellite=satellite)
            validate_predictions(arrays)
            np.savez_compressed(out / 'predictions.npz', **arrays)
            source_wall = time.perf_counter() - tick
            contract = d.read(d.SOURCE)
            metadata = dict(status='SOURCE_READOUT_COMPLETE', run_id=RUN, release=RELEASE, model_seed=SEED,
                checkpoint=str(checkpoint), checkpoint_run=PARENT_RUN, checkpoint_row=PARENT_ROW,
                checkpoint_stage='stage20', checkpoint_key='model', checkpoint_epoch=20,
                checkpoint_optimizer_steps=4440, checkpoint_selection='final_only',
                source_contract=str(d.SOURCE), source_role_counts={k: len(v) for k, v in contract['role_ids'].items()},
                source_role='V', source_unique_V=COUNT, source_panel=str(d.PANEL),
                source_observations=str(d.PANEL / 'observations/iq.npy'), channel_draws=2,
                scenes=list(SCENES), condition='full', condition_index=0, new_channel_calls=0,
                NN=12602, source_NN=SOURCE_NN, smoke_NN=SMOKE_NN,
                training_updates=0, head_fit_updates=0, teacher_training_updates=0,
                teacher_bank_created=False, U_truth_read=False, U_records_used=0,
                target_access=False, formal_scores_read=False, query_access=False,
                model_state_unchanged=True, all_parameters_frozen=True, model_parameters=900722,
                model_state_keys=206, default_head_shape=[6, 160], default_head_scale=30.,
                eval_counter=int(encoder._full_eval_packets), train_counter=int(encoder._full_train_packets),
                parent_clean_correct_count=int((clean.argmax(1) == arrays['TX']).sum()),
                clean_accuracy=float((clean.argmax(1) == arrays['TX']).mean()),
                source_accuracies=(satellite.argmax(-1) == arrays['TX']).mean(-1).tolist(),
                source_IQ_read_after_smoke=True, source_cache_used_only_role='V',
                smoke_wall_seconds=smoke_wall, source_wall_seconds=source_wall,
                wall_seconds=time.perf_counter()-started,
                modules={k: str(v.__file__) for k, v in sys.modules.items() if getattr(v, '__file__', None)
                         and k in ('experiments.cvs_u_quality66.source', 'experiments.cvs_noise_shape_factorial43.model',
                                   'leo_practical.channel', 'leo_practical.residual')})
            validate_metadata(metadata)
            write(out / 'complete.json', metadata)
            return metadata
    except Exception as error:
        write(out / 'failure.json', dict(status='FAILED', run_id=RUN, error=repr(error),
            traceback=traceback.format_exc(), preserve_outputs=True, automatic_retry=False,
            target_access=False, wall_seconds=time.perf_counter()-started))
        raise
