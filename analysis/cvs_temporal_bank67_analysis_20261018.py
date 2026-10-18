"""P67 CPU-only complete-log/four-row readout; never opens IQ or checkpoints."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import importlib.util
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np

RUN = '20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01'
REF = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
SEED, STEPS = 2026092701, 222
ARMS = ('NOAS30', 'AS30', 'NOAS_BANK', 'AS_BANK')
NEW_ARMS = ARMS[2:]
PAIRS = (('AS30', 'NOAS30'), ('NOAS_BANK', 'NOAS30'),
         ('AS_BANK', 'AS30'), ('AS_BANK', 'NOAS_BANK'), ('AS_BANK', 'NOAS30'))


def load_common():
    # A private module instance: no mutation of P66 files or other imports.
    path = Path(__file__).with_name('cvs_u_quality66_analysis_20261018.py')
    spec = importlib.util.spec_from_file_location('_p67_cpu_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN, module.NEW_ARMS, module.PAIRS = RUN, NEW_ARMS, PAIRS
    module.training_complete = training_complete
    module.row_specs = row_specs
    module.write = processed_write
    return module


def row_specs(project):
    runs = Path(project) / 'runs'
    return [dict(key=key, run=REF if i < 2 else RUN,
                 row_id=('B00' if i % 2 == 0 else 'B01') + '-s' + str(SEED),
                 arm='B00' if i % 2 == 0 else 'B01', reused=i < 2,
                 root=runs / (REF if i < 2 else RUN) /
                 (('B00' if i % 2 == 0 else 'B01') + '-s' + str(SEED)))
            for i, key in enumerate(ARMS)]


def training_complete(spec, project):
    root = spec['root']
    parent = (root if spec['reused'] else Path(project) / 'runs' / REF / spec['row_id']) / 'stage20'
    roles = c.read(Path(project) / 'runs' / c.CONTRACT_RUN / 'source_contract.json')['role_ids']
    is_as = spec['arm'] == 'B01'
    for folder, parent_stage, epochs, labels, pseudo, updates in (
            (parent, True, 20, 13, 7, 4440), (root / 'source', False, 10, 0, 10, 2220)):
        done = c.read(folder / 'completion.json'); cfg = done['config']
        frozen = c.read(folder / 'source_frozen.json'); init = c.read(folder / 'initialization.json')
        result = c.read(folder / 'result.json'); args = c.read(folder / 'actual_native_args.json')
        args = args.get('args', args)
        c.require((done.get('epoch'), done.get('optimizer_steps'), cfg.get('epochs'),
                   cfg.get('label_epochs'), cfg.get('pseudo_epochs'), cfg.get('updates')) ==
                  (epochs, updates, epochs, labels, pseudo, updates), 'Exact E20+10 calendar differs')
        c.require(cfg.get('run_id') == (REF if parent_stage else spec['run']) and
                  cfg.get('row_id') == spec['row_id'] and cfg.get('model_seed') == SEED and
                  cfg.get('As') is is_as, 'Actual run/row/seed/As differs')
        expected = [] if parent_stage else [str(parent / 'final_ssdg.pth')]
        c.require(cfg.get('checkpoint_sources') == expected and init.get('checkpoint_sources') == expected and
                  init.get('scratch_only') is parent_stage and init.get('source_roles') == 'EXACT_MATCH' and
                  init.get('target_contact') is False, 'Exact matched own P43 parent provenance differs')
        c.require(frozen.get('status') == 'SOURCE_FROZEN' and frozen.get('config') == cfg and
                  frozen.get('selection') == 'final_only' and frozen.get('target_access') is False,
                  'Fixed final-only model required')
        c.require(done.get('target_access') is False and done.get('head_fit_updates') == 0 and
                  done.get('native_fitted_probes_disabled') is True and result.get('config') == cfg and
                  result.get('U_truth_read') is False and result.get('target_access') is False and
                  result.get('head_fit_updates') == 0, 'Joint default head/hidden U/source boundary differs')
        c.require(c.read(folder / 'source_contract.json')['role_ids'] == roles and
                  result.get('source_RX') == list(c.RX) and result.get('source_RX_holdout') is None and
                  done['final_source_metrics'].get('source_val_count') == 27000,
                  'Original complete L/U/V and all five source RX required')
        c.require(cfg.get('Ag_mode') == 'NONE' and cfg.get('new_Ag') is False and
                  cfg.get('L_NN_per_step') == 768 and cfg.get('total_L_loss_scale') == 1.25,
                  'Unchanged sixL/1.25/no-Ag recipe required')
        expected_mode = 'epoch_bank' if not spec['reused'] and not parent_stage else 'batch_neighbor'
        c.require(args.get('pseudo_temporal_mode') == expected_mode and args.get('pseudo_temporal_gate') is True and
                  args.get('pseudo_temporal_bank_min_streak') == 2 and args.get('pseudo_temporal_min_conf') == .8,
                  'Actual native temporal mode/minimum confidence/streak differs')
        if not parent_stage:
            c.require(cfg.get('parent_source') == str(parent) and cfg.get('epoch_offset') == 20 and
                      cfg.get('step_offset') == 4440, 'Matching own20 and reset tail clock required')
    return c.read(root / 'source/completion.json')


def interaction(rows, keys):
    """B11-B10-B01+B00 in absolute percentage points; never a selection rule."""
    c.require(set(ARMS) <= set(rows), 'All four factorial cells required')
    result = {}
    for key in keys:
        values = [rows[a].get(key) for a in ARMS]
        result[key + '_interaction_pp'] = (100 * (values[3] - values[2] - values[1] + values[0])
                                               if all(v is not None for v in values) else None)
    return result


def enrich_formal(obj):
    rows = obj['rows']; obj['legacy_reference_stage'] = obj.get('stage', {})
    obj['interactions'] = {view: interaction({a: rows[a][view] for a in ARMS},
                                            ('accuracy', 'macro_f1', 'worst_RX', 'satellite_retention'))
                           for view in c.VIEWS}
    obj['stage'] = {}
    for new, control in (('AS30', 'NOAS30'), ('AS_BANK', 'NOAS_BANK')):
        clean = 100 * (rows[new]['clean']['accuracy'] - rows[control]['clean']['accuracy'])
        urban = 100 * np.mean([rows[new][v]['accuracy'] - rows[control][v]['accuracy'] for v in c.VIEWS[-3:]])
        obj['stage'][new] = dict(matched_noAs=control, clean_delta_pp=float(clean),
            low_urban_delta_pp=100 * (rows[new][c.VIEWS[-1]]['accuracy'] - rows[control][c.VIEWS[-1]]['accuracy']),
            mid_urban_delta_pp=100 * (rows[new][c.VIEWS[-2]]['accuracy'] - rows[control][c.VIEWS[-2]]['accuracy']),
            urban_mean_delta_pp=float(urban), clean_within_0p1pp=bool(clean >= -.1 - 1e-10),
            single_seed_observed_dual_gain=bool(clean > 0 and urban > 0), reproducible_dual_gain=False)
    obj['factorial_interaction_is_descriptive'] = True


def processed_write(path, obj):
    if Path(path).name == 'formal_summary.json':
        enrich_formal(obj)
    # Use the unmodified P66 writer, retained separately below.
    _write(path, obj)


c = load_common()
# load_common assigns our write hook, so retain the original implementation via
# its source module's primitive rather than recursively invoking the hook.
def _write(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False, default=c.json_scalar)
        stream.write('\n')


def read_source_decision(project):
    decision = c.read(Path(project) / 'runs' / RUN / 'source_matrix_decision.json')
    c.require(decision.get('status') in c.DECISION_COMPLETE | c.DECISION_PARTIAL and
              decision.get('formal_scores_read') is False and decision.get('target_feedback') is False and
              decision.get('NN') == 0 and decision.get('model_seed_count') == 1,
              'Already fixed source-only producer decision required before any formal read')
    if decision['status'] in c.DECISION_COMPLETE:
        checks = decision.get('checks')
        c.require(isinstance(checks, dict) and checks and all(type(v) is bool for v in checks.values()) and
                  type(decision.get('source_qualified')) is bool and decision['source_qualified'] == all(checks.values()),
                  'Stored source verdict and checks disagree')
    return decision


def replay_temporal(records, expected_steps=2220, expected_packets=567000):
    """Recount the native same-physical-ID bank, including duplicate visits."""
    bank = {}; epochs = defaultdict(Counter); output = []; totals = Counter()
    for step, record in enumerate(records, 1):
        epoch = (step - 1) // STEPS + 1
        c.require(record.get('step') == step and record.get('cumulative_step') == step + 4440 and
                  record.get('epoch') == epoch and record.get('cumulative_epoch') == epoch + 20 and
                  record.get('min_conf') == .8 and record.get('min_streak') == 2,
                  'Contiguous native epoch-bank clock/threshold required')
        entries = record.get('entries'); c.require(isinstance(entries, list) and len(entries) == record.get('packets'), 'All U entries required')
        c.require(record.get('bank_size_before') == len(bank), 'Bank before-size differs')
        counts = Counter(); histogram = Counter(); keys = set()
        for item in entries:
            key = item.get('key'); c.require(isinstance(key, list) and len(key) == 5 and
                all(type(v) is int for v in key) and key[0] in c.RX and key[1] in c.DAYS and key[-1] >= 0,
                'Opaque native RX/day/eq/sig/base_index key required; no TX field')
            key = tuple(key); keys.add(key); previous = bank.get(key)
            c.require(item.get('previous') == (list(previous) if previous else None), 'Saved bank history differs from prior observations')
            pseudo = item.get('pseudo'); confidence = c.finite(item.get('confidence'), 'teacher confidence')
            c.require(type(pseudo) is int and 0 <= pseudo < 6 and 0 <= confidence <= 1, 'Finite default-head pseudo decision required')
            streak = 1
            if previous is not None:
                counts['same_epoch_repeats'] += previous[1] == epoch
                if previous[0] == pseudo and previous[1] == epoch - 1:
                    streak = previous[2] + 1; counts['cross_epoch_same_prediction'] += 1
                elif previous[0] == pseudo and previous[1] == epoch:
                    streak = previous[2]
            eligible = confidence >= .8
            if eligible:
                bank[key] = (pseudo, epoch, streak)
            after = bank.get(key); selected = eligible and streak >= 2
            c.require(item.get('after') == (list(after) if after else None) and
                      type(item.get('temporal_pass')) is bool and item['temporal_pass'] == selected,
                      'Native epoch-bank transition/selection differs')
            counts['packets'] += 1; counts['confidence_eligible'] += eligible
            counts['temporal_selected'] += selected
            histogram[str(after[2] if after is not None else 0)] += 1
            c.require(epoch != 1 or not selected, 'First tail epoch cannot contain cross-epoch confirmation')
        c.require(record.get('unique_batch_keys') == len(keys) and record.get('bank_size_after') == len(bank), 'Physical-key count/bank after-size differs')
        for field in ('confidence_eligible', 'temporal_selected', 'same_epoch_repeats', 'cross_epoch_same_prediction'):
            c.require(record.get(field) == counts[field], 'Saved bank count differs: ' + field)
        c.require(record.get('streak_histogram') == dict(histogram) and record.get('first_epoch_all_false') is True and
                  record.get('no_same_epoch_streak_growth') is True, 'Saved streak/first-epoch invariants differ')
        totals.update(counts); epochs[epoch].update(counts); epochs[epoch]['steps'] += 1
        if step % STEPS == 0 or step == expected_steps:
            # These fields describe the complete retained bank, not the visits.
            epochs[epoch]['bank_size'] = len(bank)
            epochs[epoch]['streak_histogram'] = dict(Counter(str(v[2]) for v in bank.values()))
        output.append(dict(step=step, epoch=epoch, cumulative_epoch=epoch + 20,
                           unique_batch_keys=len(keys), bank_size=len(bank), **dict(counts)))
    c.require(len(output) == expected_steps and totals['packets'] == expected_packets, 'Complete temporal logs required')
    return dict(steps=len(output), packets=totals['packets'], bank_size=len(bank),
                temporal_selected=totals['temporal_selected'], confidence_eligible=totals['confidence_eligible'],
                first_epoch_all_false=True, no_same_epoch_streak_growth=True,
                epoch_summaries=[dict(epoch=e, **dict(v)) for e, v in sorted(epochs.items())],
                entries_replayed=True, U_TX_truth_read=False, extra_NN=0), output


def validate_temporal_summary(saved, actual):
    c.require(isinstance(saved, dict), 'Producer temporal summary required')
    for key in ('steps', 'packets', 'temporal_selected', 'confidence_eligible', 'bank_size',
                'first_epoch_all_false', 'no_same_epoch_streak_growth', 'extra_NN', 'U_TX_truth_read'):
        c.require(saved.get(key) == actual[key], 'Final temporal summary differs: ' + key)
    c.require(saved.get('bank_empty_at_segment_start') is True and saved.get('bank_owned_exclusively') is True,
              'One empty native bank per tail required')
    summaries = saved.get('epoch_summaries'); c.require(isinstance(summaries, list) and len(summaries) == len(actual['epoch_summaries']), 'All temporal epoch summaries required')
    for a, b in zip(summaries, actual['epoch_summaries']):
        for key in ('epoch', 'steps', 'packets', 'temporal_selected', 'confidence_eligible', 'bank_size', 'streak_histogram'):
            c.require(a.get(key) == b.get(key), 'Temporal epoch aggregate differs: ' + key)


def parse_tail(folder, is_as, bank=False):
    folder = Path(folder); result = c.read(folder / 'result.json')
    c.require(result.get('optimizer_steps') == 2220 and result.get('L_NN') == 1704960 and
              result.get('L_As') == (142080 if is_as else 0) and result.get('Ag_packets') == 0 and
              result.get('Ag_control_NN') == 568320 and result.get('extra_U_NN') == 0 and
              result.get('U_As') == 0, 'Actual native sixL/U/no-extra-forward counts differ')
    cost = c.native_cost(folder)
    c.require(cost['native_NN'] == 3162962 and cost['extra_student_U_NN'] == cost['extra_frozen_teacher_U_NN'] == 0,
              'Only original native training forwards permitted')
    streams = [c.lines(folder / name) for name in ('reference_steps.jsonl.gz', 'shape_steps.jsonl.gz',
                                                  'step_metrics.jsonl.gz', 'unit_steps.jsonl.gz')]
    sums = defaultdict(Counter); counts = Counter(); ordinary = []; total_u = selected = 0; max_error = 0.
    for step, values in enumerate(zip_longest(*streams), 1):
        c.require(all(x is not None for x in values), 'Incomplete reference/shape/native/unit logs')
        reference, shape, native, units = values; epoch = (step - 1) // STEPS + 1; metrics = native['metrics']
        c.require(all(x.get('step') == step for x in values) and native.get('epoch') == epoch and
                  all(x.get('cumulative_step') == step + 4440 for x in (reference, shape, units)) and
                  units.get('physical_clip_calls') == step, 'Complete unchanged update clocks required')
        c.require(reference.get('L_forwards') == 6 and reference.get('As_packets') == (64 if is_as else 0) and
                  reference.get('Ag_packets') == 0 and reference.get('Ag_mode') == 'NONE' and
                  reference.get('extra_control_packets') == 128 and reference.get('ordinary_redraws') == 0 and
                  reference.get('same_IQ_each_pair') is True and reference.get('all_satellite_L_supervised') is True,
                  'Original complete L supervision differs')
        c.require(shape.get('mode') == 'NONE' and shape.get('NN') == 256 and shape.get('packets') == 128 and
                  shape.get('applied') == 0 and shape.get('weight') == .25 and shape.get('same_IQ_each_pair') is True,
                  'Original raw control RDrop pair required')
        for key in ('common_CE', 'common_KL', 'extra_CE', 'extra_KL', 'SupCon', 'total_L', 'base_L_before_shape', 'ordinary_raw_MSE'):
            c.finite(reference[key], key)
        for key in ('shape_CE', 'shape_KL', 'shape_RD'): c.finite(shape[key], key)
        expected = .5 * (reference['common_CE'] + reference['common_KL']) + .5 * (reference['extra_CE'] + reference['extra_KL']) + .1 * reference['SupCon'] + .25 * shape['shape_RD']
        error = abs(expected - reference['total_L']); max_error = max(max_error, error)
        c.require(reference['SupCon_weight'] == .1 and error < 3e-5 and
                  abs(shape['shape_RD'] - shape['shape_CE'] - shape['shape_KL']) < 3e-5 and
                  abs(reference['total_L'] - reference['base_L_before_shape'] - .25 * shape['shape_RD']) < 1e-6,
                  'Original CE/RDrop/SupCon normalized decomposition differs')
        details = reference['SupCon_details']
        c.require(reference.get('crossview_mode') == 'ORD128' and not reference.get('new_losses') and
                  not reference.get('extra_U_NN') and len(details) == 4 and
                  all(q.get('mode') == 'ORD128' and q.get('total_anchors') == 128 and
                      q.get('ordinary_anchors') == 128 and q.get('additional_anchors') == 0 and
                      q.get('same_RX_positive_pairs') == 0 and q.get('true_satellite_anchors') == 0 for q in details),
                  'Original ordinary-only crossRX SupCon required')
        c.require(not any(metrics.get(k) or 0 for k in ('train/skipped_nonfinite_loss', 'train/skipped_nonfinite_grad', 'train/pseudo_truth_available')),
                  'Invalid update or hidden U truth exposure')
        n, accepted = int(metrics.get('train/pseudo_total') or 0), int(metrics.get('train/pseudo_selected') or 0)
        c.require(n == (124 if step % STEPS == 0 else 256) and 0 <= accepted <= n, 'Original unpadded U batch differs')
        total_u += n; selected += accepted; counts[epoch] += 1
        for key in ('common_CE', 'common_KL', 'extra_CE', 'extra_KL', 'SupCon', 'total_L'): sums[epoch][key] += reference[key]
        sums[epoch]['native_loss'] += c.finite(metrics['train/loss'], 'native loss')
        sums[epoch]['U_packets'] += n; sums[epoch]['pseudo_selected'] += accepted
        for key in ('train/pseudo_domain_pass', 'train/pseudo_temporal_pass', 'train/pseudo_strong_pass'):
            if metrics.get(key) is not None: sums[epoch][key] += c.finite(metrics[key], key)
        ordinary.append(dict(step=step + 4440, MSE=reference['ordinary_raw_MSE']))
    c.require(sum(counts.values()) == 2220 and set(counts) == set(range(1, 11)) and
              set(counts.values()) == {222} and total_u == 567000, 'Complete ten native U epochs required')
    raw_epochs = list(c.lines(folder / 'metrics_epoch.jsonl')); epochs = list(c.lines(folder / 'epoch_metrics.jsonl'))
    c.require([x['epoch'] for x in raw_epochs] == list(range(1, 11)) and
              [x['epoch'] for x in epochs] == list(range(1, 11)), 'Full raw and compact epoch telemetry required')
    temporal = None; temporal_steps = []
    if bank:
        temporal, temporal_steps = replay_temporal(c.lines(folder / 'temporal_steps.jsonl.gz'))
        validate_temporal_summary(result.get('temporal_summary'), temporal)
        validate_temporal_summary(c.read(folder / 'temporal_progress.json'), temporal)
        for t, native in zip(temporal_steps, c.lines(folder / 'step_metrics.jsonl.gz')):
            c.require(native['metrics']['train/pseudo_selected'] <= t['temporal_selected'],
                      'Final native acceptance exceeds temporal-only gate')
    curves = []
    for row in epochs:
        e = row['epoch']; n = sums[e]['U_packets']
        point = dict(epoch=e + 20, local_epoch=e, **{k: v / STEPS for k, v in sums[e].items()},
                     lr=c.finite(row['lr'], 'lr'), source_val_accuracy_pct=c.finite(row['val_tx_acc'], 'val accuracy'),
                     grad_backbone=c.finite(row['train_grad_backbone'], 'backbone gradient'),
                     epoch_wall_seconds=c.finite(row['epoch_time_s'], 'epoch wall'),
                     native_accepted_count=sums[e]['pseudo_selected'], native_U_count=n,
                     native_accepted_fraction=sums[e]['pseudo_selected'] / n)
        if temporal:
            t = temporal['epoch_summaries'][e - 1]
            point.update(temporal_selected=t['temporal_selected'], confidence_eligible=t['confidence_eligible'],
                         temporal_selected_fraction=t['temporal_selected'] / n,
                         bank_size=temporal_steps[e * STEPS - 1]['bank_size'])
        curves.append(point)
    c.require(abs(curves[0]['lr'] - .0002) < 1e-15 and abs(curves[-1]['lr'] - .000001) < 1e-15 and
              np.all(np.diff([r['lr'] for r in curves]) <= 1e-15), 'Actual reset cosine10 differs')
    return dict(status='FULL_TAIL_LOGS_VERIFIED', updates=2220, epochs=10, L_NN=1704960,
                As_packets=142080 if is_as else 0, U_packets=total_u, native_pseudo_selected=selected,
                native_acceptance_is_not_precision=True, maximum_L_decomposition_error=max_error,
                temporal=temporal, cost=cost, curve=curves), ordinary, temporal_steps


def same_sequence(first, second):
    count = 0; roles = Counter()
    for a, b in zip_longest(c.lines(first), c.lines(second)):
        c.require(a is not None and b is not None and a == b, 'Original complete physical L/U sequence differs')
        c.require(a.get('role') in ('L', 'U') and isinstance(a.get('ids'), list), 'Physical sequence record differs')
        roles[a['role']] += len(a['ids']); count += 1
    c.require(count > 0 and roles['L'] == 2220 * 128 and roles['U'] == 567000, 'Full L/U sequence coverage differs')
    return dict(records=count, packet_occurrences=dict(roles), exact=True)


def training_analysis(project, specs, out):
    tails = {}; curves = []; coverage = []; temporal_steps = []
    by_key = {s['key']: s for s in specs}
    for spec in specs:
        arm = spec['key']; tail, ordinary, steps = parse_tail(spec['root'] / 'source', spec['arm'] == 'B01', not spec['reused'])
        if not spec['reused']:
            ref = by_key['AS30' if spec['arm'] == 'B01' else 'NOAS30']
            tail['physical_sequence'] = same_sequence(spec['root'] / 'source/training_sequence.jsonl.gz', ref['root'] / 'source/training_sequence.jsonl.gz')
            reference_ordinary = tails[ref['key']]['ordinary']
            c.require(ordinary == reference_ordinary, 'Original ordinary augmentation stream differs')
            temporal_steps.extend(dict(row_id=arm, **r) for r in steps)
        tail['ordinary'] = ordinary
        result = c.read(spec['root'] / 'source/result.json')
        if spec['arm'] == 'B01':
            channel = result['L_channel']; cells = channel['strata']
            c.require(channel.get('packets') == 142080 and channel.get('session_count') == 15 and
                      sum(r['count'] for r in cells) == 142080 and
                      {r['RX'] for r in cells} == set(c.RX) and {r['day'] for r in cells} == set(c.DAYS) and
                      {r['scene'] for r in cells} == set(c.SCENES), 'Complete original As RX/day/scene coverage required')
            coverage.extend(dict(row_id=arm, **r) for r in cells)
        native_args = c.read(spec['root'] / 'source/actual_native_args.json')
        native_args = native_args.get('args', native_args)
        tail['native_pseudo_temporal_mode'] = native_args['pseudo_temporal_mode']
        source_curves = c.epoch_curve(spec['root'] / 'stage20', 0) if spec['reused'] else []
        curves.extend(dict(row_id=arm, reused=spec['reused'], model_seed=SEED, **r) for r in source_curves + tail['curve'])
        tails[arm] = tail
    for tail in tails.values(): tail.pop('ordinary')
    c.csv_write(out / 'training_curves.csv', curves); c.csv_write(out / 'temporal_steps.csv', temporal_steps)
    c.csv_write(out / 'training_As_coverage.csv', coverage)
    result = dict(status='FULL_TRAINING_LOGS_VERIFIED', rows=tails, new_epochs=20, new_updates=4440,
                  U_truth_read=False, extra_NN=0, extra_teacher_training=0,
                  temporal_confirmation_coverage_is_not_identity_gain=True)
    c.write(out / 'training_summary.json', result)
    return result, curves


def interaction_strata(records, out, filename):
    grouped = defaultdict(dict)
    dimensions = ('view', 'scene', 'draw', 'dimension', 'stratum')
    for r in records:
        context = tuple((k, r[k]) for k in dimensions if k in r)
        c.require(r['row_id'] not in grouped[context], 'Duplicate factorial stratum')
        grouped[context][r['row_id']] = r
    output = [dict(context, **interaction(rows, ('accuracy', 'macro_f1', 'satellite_retention')))
              for context, rows in grouped.items()]
    c.csv_write(out / filename, output)
    return output


def costs(project, specs, local_nn, smoke_nn):
    rows = {}; new = {}; historical = {}
    for spec in specs:
        root = spec['root']; tail = c.native_cost(root / 'source')
        source = c.read(root / 'source_qualification/complete.json'); prediction = c.read(root / 'prediction/complete.json')
        c.require(source.get('NN') == 12600 and prediction.get('actual_NN') == 1176002, 'Actual readout/prediction cost differs')
        r = dict(tail=tail, source_readout_NN=12600, formal_prediction_NN=1176002,
                 source_wall_seconds=source.get('wall_seconds'), formal_wall_seconds=prediction.get('wall_seconds'))
        if spec['reused']:
            r['stage20'] = c.native_cost(root / 'stage20'); historical[spec['key']] = r
        else:
            c.require(tail['native_NN'] == 3162962 and tail['extra_student_U_NN'] == tail['extra_frozen_teacher_U_NN'] == 0,
                      'New native-only forward budget differs')
            new[spec['key']] = r
        rows[spec['key']] = r
    total = sum(r['tail']['training_total_NN'] + r['source_readout_NN'] + r['formal_prediction_NN'] for r in new.values())
    c.require(total == 8703128, 'Actual two-tail native/readout/formal subtotal differs')
    smoke = c.read(Path(project) / 'runs' / RUN / 'real_checkpoint_smoke.json')
    c.require(smoke.get('status') == 'PASS' and type(smoke.get('NN')) is int and smoke['NN'] >= 0,
              'Actual shared no-query checkpoint smoke required')
    if smoke_nn is not None: c.require(smoke['NN'] == smoke_nn, 'Owner-declared smoke budget differs from actual')
    historical_total = sum(sum(r[s]['training_total_NN'] for s in ('stage20', 'tail')) + r['source_readout_NN'] + r['formal_prediction_NN'] for r in historical.values())
    return dict(rows=rows, new_native_NN=6325924, new_source_NN=25200, new_formal_NN=2352004,
                new_remote_subtotal_NN=total, shared_remote_smoke_NN=smoke['NN'],
                new_total_remote_NN=total + smoke['NN'], local_verification_NN=local_nn,
                new_model_epochs=20, new_updates=4440, matched_parent_epochs_each=20,
                matched_parent_updates_each=4440, cumulative_student_epochs_each=30,
                cumulative_student_updates_each=6660, historical_reference_epochs=60,
                historical_reference_NN_known=historical_total, unique_total_training_epochs=80,
                new_training_wall_seconds_sum=sum(r['tail']['wall_seconds'] for r in new.values()),
                extra_student_NN=0, extra_teacher_NN=0, analysis_NN=0,
                parent_training_reused_not_repeated=True, reference_reprediction=0, reference_rescoring=0,
                budget_exhaustion_is_not_convergence=True)


def plots(out, curves, source_rows, formal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(ARMS, ('#64748b', '#d97706', '#8b5cf6', '#2563eb'))); files = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            path = out / (name + '.' + ext); fig.savefig(path, dpi=180); files.append(str(path))
        plt.close(fig)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for arm in ARMS:
            seq = [r for r in curves if r['row_id'] == arm]
            for ax, key, title in zip(axes, ('native_loss', 'source_val_accuracy_pct', 'grad_backbone'),
                                      ('Native training loss', 'Source V accuracy (%)', 'Backbone gradient norm')):
                ax.plot([r['epoch'] for r in seq], [r[key] for r in seq], color=colors[arm], label=arm)
                ax.set(xlabel='Cumulative epoch', title=title); ax.grid(alpha=.2)
        axes[0].legend(); fig.suptitle('Complete logged epochs; new tails E21-30 only'); save(fig, 'training_curves')
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for arm in ARMS:
            seq = [r for r in curves if r['row_id'] == arm and r.get('local_epoch')]
            axes[0].plot([r['epoch'] for r in seq], [100 * r['native_accepted_fraction'] for r in seq], label=arm, color=colors[arm])
            if arm in NEW_ARMS:
                axes[1].plot([r['epoch'] for r in seq], [100 * r['temporal_selected_fraction'] for r in seq], label=arm, color=colors[arm])
                axes[2].plot([r['epoch'] for r in seq], [r['bank_size'] for r in seq], label=arm, color=colors[arm])
        for ax, title in zip(axes, ('Final native U acceptance (%)', 'Temporal gate only (%)', 'Native bank entries')):
            ax.set(xlabel='Cumulative epoch', title=title); ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.suptitle('Coverage and history only: hidden U truth; acceptance is not precision'); save(fig, 'bank_acceptance')
    if source_rows:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for ax, scene in zip(axes, c.SCENES[1:]):
            seq = [r for r in source_rows if r['scene'] == scene and r['draw'] == -1 and r['dimension'] == 'signal_to_added_noise_db']
            labels = list(dict.fromkeys(r['stratum'] for r in seq)); xx = np.arange(len(labels))
            for i, arm in enumerate(ARMS):
                rr = [next(r for r in seq if r['row_id'] == arm and r['stratum'] == label) for label in labels]
                ax.bar(xx + (i - 1.5) * .2, [100*r['accuracy'] if r['accuracy'] is not None else np.nan for r in rr], .2, label=arm, color=colors[arm])
            ax.set(xticks=xx, xticklabels=labels, title=scene, ylabel='Source accuracy (%)', xlabel='Signal / added noise (dB); two repeated draws'); ax.grid(axis='y', alpha=.2)
        axes[0].legend(); save(fig, 'source_quality')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5)); xx = np.arange(7)
    for new, old, label in (('AS30', 'NOAS30', 'As effect / old temporal'), ('AS_BANK', 'NOAS_BANK', 'As effect / physical-ID temporal'), ('AS_BANK', 'AS30', 'Bank effect / As')):
        axes[0].plot(xx, [100*(formal['rows'][new][v]['accuracy']-formal['rows'][old][v]['accuracy']) for v in c.VIEWS], marker='.', label=label)
    axes[1].bar(xx, [formal['interactions'][v]['accuracy_interaction_pp'] for v in c.VIEWS], color='#2563eb')
    for ax, title in zip(axes, ('Matched formal paired accuracy changes', 'As × temporal interaction: B11-B10-B01+B00')):
        ax.set(xticks=xx, xticklabels=['clean','high sub','mid sub','low sub','high urb','mid urb','low urb'], ylabel='Absolute percentage points', title=title); ax.tick_params(axis='x', labelrotation=30); ax.axhline(0, color='grey', lw=.7); ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8); save(fig, 'formal_paired')
    return files


def report(out, result):
    formal = result['formal']; rows = formal['rows']; cost = result['cost']
    text = ['# P67：同物理样本跨轮时间确认', '',
        '只替换原 U 时间门控：batch 内相邻录音预测一致，改为同一 opaque physical ID 跨相邻 epoch 预测一致。置信阈值、strong 一致、原 U CE/熵、own evolving EMA、六次 L 前向、默认 cosine30 头、完整 practical 和 SupCon 保留。两个新尾段分别严格继承匹配 P43 NoAs20／As20 学生及自身 EMA，重置 AdamW/cos10；每条学生累计 E30。', '',
        '源判定在正式结果读取前固定；本分析只报告，不重新晋级。时间确认覆盖及 native 接受率不是真实伪标签精度，也不等于身份识别收益。', '',
        '| 环境 | NoAs30 % | As30 % | NoAs bank % | As bank % | As×bank 交互 pp |',
        '|---|---:|---:|---:|---:|---:|']
    for v in c.VIEWS:
        text.append('| '+v+' | '+' | '.join(f'{100*rows[a][v]["accuracy"]:.6f}' for a in ARMS)+f' | {formal["interactions"][v]["accuracy_interaction_pp"]:+.6f} |')
    text += ['', '| 比较 | clean变化 pp | mid urban变化 pp | low urban变化 pp | urban均值变化 pp | clean下降≤0.1pp |', '|---|---:|---:|---:|---:|---|']
    for arm, s in formal['stage'].items():
        text.append(f'| {arm}−{s["matched_noAs"]} | {s["clean_delta_pp"]:+.6f} | {s["mid_urban_delta_pp"]:+.6f} | {s["low_urban_delta_pp"]:+.6f} | {s["urban_mean_delta_pp"]:+.6f} | {s["clean_within_0p1pp"]} |')
    text += ['', '阶段判断使用匹配的 AS_BANK−NOAS_BANK；同时保留原 As 效应、bank 在有／无 As 下的配对变化与四格交互。只有一个模型 seed，两个源信道 draw 是重复观测，不能证明可重复双提升。小于 0.1 个百分点的正变化只能作初步观察。', '',
        '全部七环境 accuracy／Macro-F1／最差 RX、RX／day／TX、配对修复与新增错误、margin/30 和固定 NoAs30 clean 正确集合保持率见 CSV。该集合不是父 As20 clean 正确集合。源面板提供 LOS/PARTIAL/BLOCKED、锁定状态、接收质量代理及 signal_to_added_noise_db 分层；新增噪声比不是总 SNR，分层关联不构成物理因果。原训练信道保存 RX/day/scene/state/lock 联合覆盖，没有额外保存 TX 与物理状态的逐步联合表。正式逐包信道状态与完整 V27000 F1 未保存，不作推断。', '',
        '原 P43 两条历史模型共60 model-epochs；本次两个尾段新增20 model-epochs／4440更新。匹配父训练已包含在历史来源，不重复记作新训练。'+f'新增 native NN={cost["new_native_NN"]}，源读出={cost["new_source_NN"]}，正式预测={cost["new_formal_NN"]}，共享 smoke={cost["shared_remote_smoke_NN"]}，新增远端合计={cost["new_total_remote_NN"]}；本地验证 NN={cost["local_verification_NN"]}。额外学生／教师 NN=0。新增训练墙钟之和={cost["new_training_wall_seconds_sum"]:.3f} 秒，这是行耗时之和，不是并行矩阵经过时间。预算耗尽不等于收敛。', '',
        '已存源判定：'+json.dumps(result['stored_source_decision'], ensure_ascii=False), '',
        '未完成／限制：'+json.dumps(result['missing'], ensure_ascii=False), '', '实际图表：']
    text += ['- '+p for p in result['figures']]
    text += ['', '此离线分析新增 NN、IQ／checkpoint 读取、训练更新、头拟合、重新推理与 scorer 启动均为0。']
    (out / 'final_analysis.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


def run(project, out, truth=None, code_commit=None, local_nn=0, smoke_nn=None):
    project = Path(project); decision = read_source_decision(project)  # Before any formal artifact/truth.
    c.require(type(local_nn) is int and local_nn >= 0 and (smoke_nn is None or type(smoke_nn) is int and smoke_nn >= 0), 'Nonnegative external verification costs required')
    out = Path(out); c.require(not out.exists(), 'Immutable new analysis output required'); out.mkdir(parents=False)
    c.write(out / 'stored_source_decision.json', decision); specs = row_specs(project); started = time.perf_counter()
    missing = []; training = None; curves = []; source = None; source_rows = []
    if decision['status'] in c.DECISION_PARTIAL:
        missing.append(dict(component='stored_source_decision', error='Producer source decision is PARTIAL; not recomputed'))
    for name in ('training', 'source'):
        try:
            if name == 'training': training, curves = training_analysis(project, specs, out)
            else:
                source, source_rows = c.source_analysis(project, specs, out, decision)
                interaction_strata(source_rows, out, 'source_interactions.csv')
        except Exception as error:
            failure = dict(component=name, error=repr(error), traceback=traceback.format_exc(), formal_still_required=True)
            missing.append(failure); c.write(out / (name + '_failure.json'), failure)
    truth = Path(truth) if truth else project / 'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json'
    formal = c.formal_analysis(project, specs, out, truth)
    interaction_strata(c.read(out / 'formal_confusions.json'), out, 'formal_interactions.csv')
    cost = costs(project, specs, local_nn, smoke_nn); c.write(out / 'costs.json', cost)
    try:
        figures = plots(out, curves, source_rows, formal)
        if training and source_rows:
            c.require(len(figures) == 8, 'Four actual PNG/PDF figure pairs required')
    except Exception as error:
        figures = []; failure = dict(component='plot', error=repr(error), traceback=traceback.format_exc())
        missing.append(failure); c.write(out / 'plot_failure.json', failure)
    result = dict(status='ANALYZED' if not missing else 'FORMAL_ANALYZED_WITH_PARTIAL_COMPONENTS',
        run_id=RUN, code_commit=code_commit, stored_source_decision=decision, source=source, formal=formal,
        training=training, cost=cost, figures=figures, missing=missing, model_seed_count=1,
        crossseed_confirmed=False, reproducible_dual_gain=False, source_decision_recomputed=False,
        target_feedback=False, goal_complete=False, NN=0, IQ_reads=0, checkpoint_reads=0,
        training_updates=0, wall_seconds=time.perf_counter()-started)
    report(out, result); c.write(out / 'summary.json', {k: v for k, v in result.items() if k != 'training'})
    c.write(out / 'complete.json', result)
    return result


def checks():
    """Small deterministic metadata/prediction tests only; no Torch/CP/IQ/network."""
    import copy
    import tempfile
    import unittest
    from unittest.mock import patch
    def record(step, entries, before, after, selected, eligible, histogram, repeat=0, cross=0):
        epoch=(step-1)//STEPS+1
        return dict(step=step,cumulative_step=4440+step,epoch=epoch,cumulative_epoch=epoch+20,
            packets=len(entries),unique_batch_keys=len({tuple(x['key']) for x in entries}),
            bank_size_before=before,bank_size_after=after,confidence_eligible=eligible,
            temporal_selected=selected,min_conf=.8,min_streak=2,streak_histogram=histogram,
            same_epoch_repeats=repeat,cross_epoch_same_prediction=cross,first_epoch_all_false=True,
            no_same_epoch_streak_growth=True,entries=entries)
    key=[1,1,1,8,80]
    def item(p=2,conf=.9,previous=None,after=None,selected=False):
        return dict(key=key,pseudo=p,confidence=conf,previous=previous,after=after,temporal_pass=selected)
    class Checks(unittest.TestCase):
        def test_factorial_interaction_and_matched_clean(self):
            rows={a:dict(accuracy=v) for a,v in zip(ARMS,(.7,.8,.75,.9))}
            self.assertAlmostEqual(interaction(rows,['accuracy'])['accuracy_interaction_pp'],5.)
            with self.assertRaises(ValueError): interaction({k:v for k,v in rows.items() if k!='AS30'},['accuracy'])
            rows={a:{v:dict(accuracy=q,macro_f1=q,worst_RX=q,satellite_retention=q) for v in c.VIEWS} for a,q in zip(ARMS,(.7,.8,.85,.8495))}
            obj=dict(rows=rows,stage={}); enrich_formal(obj)
            self.assertAlmostEqual(obj['stage']['AS_BANK']['clean_delta_pp'],-.05)
            self.assertTrue(obj['stage']['AS_BANK']['clean_within_0p1pp'])
        def test_native_bank_same_epoch_does_not_confirm(self):
            a=item(after=[2,1,1]); b=item(previous=[2,1,1],after=[2,1,1])
            rows=[record(1,[a,b],0,1,0,2,{'1':2},repeat=1)]
            result,_=replay_temporal(rows,1,2); self.assertEqual(result['temporal_selected'],0)
            bad=copy.deepcopy(rows);bad[0]['entries'][1]['after']=[2,1,2]
            with self.assertRaises(ValueError):replay_temporal(bad,1,2)
        def test_native_bank_cross_epoch_and_low_confidence(self):
            rows=[record(i,[item(after=[2,1,1]) if i==1 else item(previous=[2,1,1],after=[2,1,1])],0 if i==1 else 1,1,0,1,{'1':1},repeat=0 if i==1 else 1) for i in range(1,STEPS+1)]
            rows.append(record(STEPS+1,[item(previous=[2,1,1],after=[2,2,2],selected=True)],1,1,1,1,{'2':1},cross=1))
            result,_=replay_temporal(rows,STEPS+1,STEPS+1);self.assertEqual(result['temporal_selected'],1)
            bad=copy.deepcopy(rows);bad[-1]['entries'][0]['confidence']=.79
            with self.assertRaises(ValueError):replay_temporal(bad,STEPS+1,STEPS+1)
        def test_missing_temporal_log_and_wrong_history_rejected(self):
            r=record(1,[item(after=[2,1,1])],0,1,0,1,{'1':1})
            with self.assertRaises(ValueError):replay_temporal([r],2,1)
            r['entries'][0]['previous']=[2,0,1]
            with self.assertRaises(ValueError):replay_temporal([r],1,1)
        def test_low_confidence_histogram_and_changed_same_epoch_prediction(self):
            a=item(conf=.7); b=item(after=[2,1,1]); d=item(p=3,previous=[2,1,1],after=[3,1,1])
            rows=[record(1,[a,b,d],0,1,0,2,{'0':1,'1':2},repeat=1)]
            value,_=replay_temporal(rows,1,3);self.assertEqual(value['confidence_eligible'],2)
            bad=copy.deepcopy(rows);bad[0]['streak_histogram']={'1':3}
            with self.assertRaises(ValueError):replay_temporal(bad,1,3)
        def test_empty_metrics_and_exact_pair_counts(self):
            self.assertIsNone(c.metric(np.array([],dtype=int),np.array([],dtype=int))['accuracy'])
            y=np.arange(6); old=y.copy();new=y.copy();new[0]=1
            p=c.paired(y,old,new);self.assertEqual(p['repairs'],0);self.assertEqual(p['new_errors'],1)
            self.assertAlmostEqual(c.logit_metric(y,np.eye(6)*30,np.ones(6,bool))['mean_margin_over30'],1.)
        def test_source_decision_precedes_and_forbids_formal_feedback(self):
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'runs'/RUN/'source_matrix_decision.json';p.parent.mkdir(parents=True)
                d=dict(status='SOURCE_READOUT_DECIDED',formal_scores_read=False,target_feedback=False,NN=0,model_seed_count=1,source_qualified=False,checks=dict(a=False,b=True))
                p.write_text(json.dumps(d),encoding='utf8');self.assertEqual(read_source_decision(temp),d)
                d['formal_scores_read']=True;p.write_text(json.dumps(d),encoding='utf8')
                with self.assertRaises(ValueError):read_source_decision(temp)
        def test_four_rows_all_views_before_truth(self):
            with tempfile.TemporaryDirectory() as temp:
                specs=row_specs(temp);ids=np.asarray(['toy'+str(i) for i in range(6)])
                def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj),encoding='utf8')
                for s in specs:
                    root=s['root'];cfg=dict(row_id=s['row_id']);p=root/'prediction'
                    dump(p/'complete.json',dict(status='PREDICTIONS_COMPLETE',row_id=root.name,count=6,views=list(c.VIEWS),config=cfg,truth_read=False,query_fit=False,all_parameters_frozen=True,head_fit_updates=0,actual_NN=44))
                    dump(p/'resolved_config.json',dict(config=cfg,views=list(c.VIEWS),model_frozen=True,separate_head_fit=False,truth_read=False,query_fit=False,checkpoint=str(root/'source/final_ssdg.pth')))
                    dump(root/'scoring/complete.json',dict(status='SCORED_COMPLETE',row_id=root.name,views=list(c.VIEWS),truth_last=True))
                    dump(root/'complete.json',dict(status='TRAINED_AND_SCORED',row_id=root.name))
                    dump(root/'scoring/scores.json',dict(truth_last=True,target_feedback_forbidden=True,results=[dict(view=v,dimension='overall',stratum='ALL',query_count=6) for v in c.VIEWS]))
                    np.savez(p/'predictions.npz',ids=ids,**{v:np.arange(6) for v in c.VIEWS})
                    for view in c.VIEWS:np.savez(p/(view+'_scores.npz'),ids=ids,logits=np.eye(6)*30)
                truth=Path(temp)/'truth.json';dump(truth,{v:dict(label=i,receiver='toyRX',day='toyday') for i,v in enumerate(ids)})
                observed=[];read=c.read
                def spy(path):observed.append('truth' if Path(path)==truth else str(path));return read(path)
                checker=lambda s,p:dict(config=dict(row_id=s['row_id']))
                with patch.object(c,'read',side_effect=spy):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertEqual(observed.count('truth'),1);self.assertEqual(observed[-1],'truth')
                (specs[-1]['root']/'prediction/practical_low_urban_scores.npz').unlink();observed.clear()
                with patch.object(c,'read',side_effect=spy),self.assertRaises(FileNotFoundError):c.truth_last_arrays(specs,ids,truth,temp,6,checker)
                self.assertNotIn('truth',observed)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--truth',type=Path);parser.add_argument('--code-commit')
    parser.add_argument('--local-nn',type=int,default=0);parser.add_argument('--smoke-nn',type=int)
    parser.add_argument('--checks',action='store_true');args=parser.parse_args()
    if args.checks:checks();return
    if args.project is None or args.out is None:parser.error('--project and --out are required unless --checks')
    result=run(args.project,args.out,args.truth,args.code_commit,args.local_nn,args.smoke_nn)
    print(json.dumps(dict(status=result['status'],out=str(args.out),missing=[r['component'] for r in result['missing']],NN=0,IQ_reads=0,checkpoint_reads=0)),flush=True)


if __name__=='__main__':main()
