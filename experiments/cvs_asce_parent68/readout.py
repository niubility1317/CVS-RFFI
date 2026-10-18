"""Actual AS_CE20 student: frozen source V logits, with no new channel or training."""
from collections import Counter
import csv
import json
from pathlib import Path
import time
import traceback
import numpy as np

RUN = '20261018-diagnostic-FcrossRX-ASCE20cohort-sourceV-s2026092701-m1-r01'
RELEASE = 'cvs_FcrossRX_ASCE20_cohort_sourceV_20261018_r01'
OWNER = 'codex/root/ASCE20-source-cohort68'
PARENT_RUN = '20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
PARENT_ROW = 'AS_CE-s2026092701'
SEED, COUNT, SOURCE_NN, SMOKE_NN = 2026092701, 1800, 12600, 2
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
RX, DAYS = (1, 3, 4, 6, 8), (1, 2, 3)
STATES, LOCKS = ('LOS', 'PARTIAL', 'BLOCKED'), ('locked', 'degraded', 'unlocked')
COHORT_REFERENCE = 'Actual SAT18 AS_CE20 student clean correct; not P43 As20 or EMA'


def require(value, message):
    if not value:
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
        head_fit_updates=0, teacher_forwards=0, teacher_bank_created=False,
        U_truth_read=False, U_records_used=0, target_access=False, formal_scores_read=False,
        query_access=False, new_channel_calls=0, model_state_unchanged=True,
        all_parameters_frozen=True, model_parameters=900722, model_state_keys=206,
        default_head_shape=[6, 160], default_head_scale=30., eval_counter=12602,
        train_counter=0, source_role_counts=dict(L_s=6300, U_s=56700, V=27000))
    require(isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items()),
        'Fixed AS_CE20 student/source-only metadata or budget differs')
    for key in ('checkpoint', 'source_contract', 'source_panel', 'source_observations'):
        require(isinstance(value.get(key), str) and bool(value[key]), 'Missing source handle: ' + key)
    for key in ('wall_seconds', 'smoke_wall_seconds', 'source_wall_seconds'):
        require(isinstance(value.get(key), (int, float)) and not isinstance(value[key], bool)
            and np.isfinite(value[key]) and value[key] >= 0, 'Invalid measured time: ' + key)
    return value


def validate_predictions(value, count=COUNT):
    require(set(value) == {'ids', 'TX', 'RX', 'day', 'clean', 'satellite'}, 'Fixed source prediction fields required')
    ids = np.asarray(value['ids'])
    require(ids.shape == (count,) and ids.dtype.kind in 'US' and len(set(ids.tolist())) == count,
        'Unique ordered source physical IDs required')
    for key, allowed in (('TX', range(6)), ('RX', RX), ('day', DAYS)):
        v = np.asarray(value[key])
        require(v.shape == (count,) and v.dtype.kind in 'iu' and set(v.tolist()) == set(allowed),
            'Source V labels differ: ' + key)
    require(count % 90 == 0 and Counter(zip(value['TX'].tolist(), value['RX'].tolist(), value['day'].tolist()))
        == Counter({(t, r, d): count // 90 for t in range(6) for r in RX for d in DAYS}),
        'Original balanced source V TX/RX/day panel required')
    for key, shape in (('clean', (count, 6)), ('satellite', (2, 3, count, 6))):
        v = np.asarray(value[key])
        require(v.shape == shape and v.dtype.kind == 'f' and np.isfinite(v).all(),
            'Finite fixed default-head source logits required: ' + key)
    return value


def cohort_metrics(parent, current=None, physics=None):
    """Pure saved-logit comparison on the actual parent's clean-correct IDs."""
    current = parent if current is None else current
    n = len(parent['ids'])
    validate_predictions(parent, n)
    validate_predictions(current, n)
    require(all(np.array_equal(parent[k], current[k]) for k in ('ids', 'TX', 'RX', 'day')),
        'Same ordered legal source V IDs and labels required; no silent reordering')
    if physics is not None:
        require(set(physics) == {'state', 'lock'}, 'Only cached source state/lock arrays accepted')
        for key in physics:
            v = np.asarray(physics[key])
            require(v.shape == (2, 3, n) and v.dtype.kind in 'iu' and np.all((v >= 0) & (v < 3)),
                'Condition0 source state/lock shape differs')
    clean_correct = parent['clean'].argmax(1) == parent['TX']
    records = []
    for scene in ('clean', *SCENES):
        si = None if scene == 'clean' else SCENES.index(scene)
        for draw in ((-1,) if si is None else (-1, 0, 1)):
            draws = (0, 1) if draw == -1 else (draw,)
            copies = 1 if si is None else len(draws)
            y, cohort = np.tile(parent['TX'], copies), np.tile(clean_correct, copies)
            rx, day = np.tile(parent['RX'], copies), np.tile(parent['day'], copies)
            zp = parent['clean'] if si is None else np.concatenate([parent['satellite'][d, si] for d in draws])
            zc = current['clean'] if si is None else np.concatenate([current['satellite'][d, si] for d in draws])
            masks = [('overall', 'ALL', np.ones(len(y), bool))]
            for dim, values, allowed in (('RX', rx, RX), ('day', day, DAYS), ('TX', y, range(6))):
                masks.extend((dim, str(v), values == v) for v in allowed)
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
                baseline, retained = int((pgood & cc).sum()), int((cgood & cc).sum())
                records.append(dict(scene=scene, draw=draw, dimension=dimension, stratum=name,
                    count=int(mask.sum()), parent_clean_correct_count=den,
                    parent_retained=baseline, current_retained=retained,
                    parent_retention=baseline / den if den else None,
                    current_retention=retained / den if den else None,
                    retention_delta_pp=100. * (retained-baseline) / den if den else None,
                    repairs_within_parent_cohort=int((~pgood & cgood & cc).sum()),
                    new_errors_within_parent_cohort=int((pgood & ~cgood & cc).sum()),
                    mean_margin_over30_within_parent_cohort=float(margin[cc].mean()) if den else None,
                    cohort_reference=COHORT_REFERENCE, empty_cohort=not den, small_cohort=den < 30))
    return dict(parent_clean_correct_count=int(clean_correct.sum()), unique_source_V=n,
        channel_draws=2, model_seed_count=1, records=records, NN=0, IQ_reads=0,
        checkpoint_reads=0, updates=0, target_access=False)


def execute(device='cuda:0'):
    import torch
    from experiments.cvs_full_channel_pair68 import design as d, source, model as api
    from experiments.cvs_full_channel_pair68.quality import panel
    from experiments.cvs_full_channel_pair68.channel import tensor_iq
    from experiments.cvs_equivariant_identity.precision import numerical_context

    require(device in ('cpu', 'cuda:0'), 'Only declared CPU or assigned CUDA device')
    out = Path(d.PROJECT) / 'runs' / RUN
    out.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    try:
        torch.set_num_threads(2)
        config = d.config(d.rows()[0])
        parent_checkpoint = Path(config['parent_source']) / 'final_ssdg.pth'
        require(parent_checkpoint.parent.name == 'stage20'
            and parent_checkpoint.parent.parent.name == PARENT_ROW
            and parent_checkpoint.parent.parent.parent.name == PARENT_RUN,
            'Only actual SAT18 AS_CE20 parent allowed')
        with numerical_context(d.FULL_FP32_POLICY):
            cp = source.parent_payload(config)
            require(cp['run_id'] == PARENT_RUN and cp['candidate_id'] == PARENT_ROW
                and cp['epoch'] == 20 and cp['a1_ema_successful_updates'] == 4440
                and cp['checkpoint_selection'] == 'final_only', 'Actual AS_CE20 student identity differs')
            model = api.build(config, device)
            model.load_state_dict(cp['model'], strict=True)
            model.eval().requires_grad_(False)
            head = model.id_backbone.encoder.core.id_backbone.cls_head
            require(len(model.state_dict()) == 206 and sum(p.numel() for p in model.parameters()) == 900722
                and list(head.weight.shape) == [6, 160] and float(head.scale) == 30., 'Fixed F/default cosine head differs')
            require(all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
                'Every parent module and parameter must be frozen')
            before = {k: v.clone() for k, v in model.state_dict().items()}
            encoder = model.id_backbone.encoder
            require(encoder._full_eval_packets == 0 and encoder._full_train_packets == 0,
                'Unexpected forward before actual parent smoke')
            tick = time.perf_counter()
            with torch.no_grad():
                z = model.forward_identity_only(torch.zeros(2, 2, 256, device=device))['tx_logits']
            require(z.shape == (2, 6) and bool(torch.isfinite(z).all()) and encoder._full_eval_packets == 2,
                'Actual parent checkpoint no-query smoke failed')
            smoke_wall = time.perf_counter() - tick
            write(out / 'real_checkpoint_smoke.json', dict(status='PASS', checkpoint=str(parent_checkpoint),
                checkpoint_key='model', NN=2, source_IQ_read_before_smoke=0, query_access=False,
                target_access=False, training_updates=0, head_fit_updates=0, wall_seconds=smoke_wall))
            # Continue immediately after the same-run smoke. No new channel draw.
            tick = time.perf_counter()
            data, roles = panel(torch)
            observed = d.read(d.PANEL / 'observations/completion.json')
            require(observed['draws'] == 2 and observed['conditions'][0] == 'full'
                and observed['scenes'] == list(SCENES) and not observed['target_access'],
                'Existing source observation order differs')
            observations = np.load(d.PANEL / 'observations/iq.npy', mmap_mode='r', allow_pickle=False)
            require(observations.shape == (2, 3, len(observed['conditions']), COUNT, 2, 256),
                'Existing source observation dimensions differ')
            clean, satellite = np.empty((COUNT, 6), np.float32), np.empty((2, 3, COUNT, 6), np.float32)
            with torch.no_grad():
                for i in range(0, COUNT, 90):
                    clean[i:i+90] = model.forward_identity_only(data['x'][i:i+90].to(device))['tx_logits'].cpu().tolist()
                for draw in range(2):
                    for si in range(3):
                        for i in range(0, COUNT, 90):
                            x = tensor_iq(observations[draw, si, 0, i:i+90], device)
                            satellite[draw, si, i:i+90] = model.forward_identity_only(x)['tx_logits'].cpu().tolist()
            require(encoder._full_eval_packets == 12602 and encoder._full_train_packets == 0
                and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
                and all(p.grad is None for p in model.parameters()) and model._full_channel.packets == 0
                and model._pair_generated == 0, 'Frozen parent state, zero channel calls or cost differs')
            arrays = dict(ids=np.asarray(data['ids']), TX=np.asarray(data['y'].tolist(), np.int64),
                RX=np.asarray(data['rx'].tolist(), np.int64), day=np.asarray(data['day'].tolist(), np.int64),
                clean=clean, satellite=satellite)
            validate_predictions(arrays)
            np.savez_compressed(out / 'predictions.npz', **arrays)
            cohort = cohort_metrics(arrays)
            write(out / 'parent_cohort_metrics.json', cohort)
            with (out / 'parent_cohort_metrics.csv').open('x', encoding='utf8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(cohort['records'][0]))
                writer.writeheader()
                writer.writerows(cohort['records'])
            source_wall = time.perf_counter() - tick
            contract = d.read(d.SOURCE)
            metadata = dict(status='SOURCE_READOUT_COMPLETE', run_id=RUN, release=RELEASE, owner=OWNER,
                model_seed=SEED, checkpoint=str(parent_checkpoint), checkpoint_run=PARENT_RUN,
                checkpoint_row=PARENT_ROW, checkpoint_stage='stage20', checkpoint_key='model',
                checkpoint_epoch=20, checkpoint_optimizer_steps=4440, checkpoint_selection='final_only',
                source_contract=str(d.SOURCE), source_role_counts={k: len(v) for k, v in contract['role_ids'].items()},
                source_role='V', source_unique_V=COUNT, source_panel=str(d.PANEL),
                source_observations=str(d.PANEL / 'observations/iq.npy'), channel_draws=2,
                scenes=list(SCENES), condition='full', condition_index=0, new_channel_calls=0,
                NN=12602, source_NN=SOURCE_NN, smoke_NN=SMOKE_NN, training_updates=0, head_fit_updates=0,
                teacher_forwards=0, teacher_bank_created=False, U_truth_read=False, U_records_used=0,
                target_access=False, formal_scores_read=False, query_access=False, model_state_unchanged=True,
                all_parameters_frozen=True, model_parameters=900722, model_state_keys=206,
                default_head_shape=[6, 160], default_head_scale=30., eval_counter=int(encoder._full_eval_packets),
                train_counter=int(encoder._full_train_packets), parent_clean_correct_count=cohort['parent_clean_correct_count'],
                parent_cohort_reference=COHORT_REFERENCE,
                retention_summary=[r for r in cohort['records'] if r['dimension'] == 'overall'],
                clean_accuracy=float((clean.argmax(1) == arrays['TX']).mean()),
                source_accuracies=(satellite.argmax(-1) == arrays['TX']).mean(-1).tolist(),
                smoke_wall_seconds=smoke_wall, source_wall_seconds=source_wall,
                wall_seconds=time.perf_counter()-started,
                formal_evaluation_required=False, reason='Source-only frozen-parent diagnostic; no training',
                P68_training_and_source_decision_unchanged=True)
            validate_metadata(metadata)
            write(out / 'complete.json', metadata)
            return metadata
    except Exception as error:
        write(out / 'failure.json', dict(status='FAILED', run_id=RUN, error=repr(error),
            traceback=traceback.format_exc(), preserve_outputs=True, automatic_retry=False,
            target_access=False, wall_seconds=time.perf_counter()-started))
        raise
