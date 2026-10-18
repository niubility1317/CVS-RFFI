"""Complete source training-log recount; never opens IQ, checkpoints or target truth.

run(project) requires completed own20+10 logs for AS_CE/AS_DET and reused
P43 B00/B01. Missing/truncated artifacts raise IncompleteTrainingLogs, not a
training-failure verdict. All return values are JSON serializable and omit IDs,
private absolute paths and stdout text. No Torch or experiment imports.
"""
import argparse
import csv
import gzip
import json
import math
import re
import time
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

RUN = '20261018-phase1-FcrossRX-satelliteDropout-e30-s2026092701-m2-r01'
REFERENCE = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
ARMS = ('B00', 'B01', 'AS_CE', 'AS_DET')
RX = (1, 3, 4, 6, 8)
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
SEGMENTS = (('stage20', 20, 13, 0), ('source', 10, 0, 20))
COST_KEYS = ('student_train_NN', 'student_eval_NN', 'teacher_train_NN', 'teacher_eval_NN')


class IncompleteTrainingLogs(RuntimeError):
    """Observable source artifact is missing/incomplete; do not infer process state."""


def read(path):
    path = Path(path)
    if not path.is_file():
        raise IncompleteTrainingLogs('Missing source artifact: ' + path.name)
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError as exc:
        raise IncompleteTrainingLogs('Incomplete JSON artifact: ' + path.name) from exc


def records(path):
    path = Path(path)
    if not path.is_file():
        raise IncompleteTrainingLogs('Missing structured source log: ' + path.name)
    try:
        with (gzip.open(path, 'rt', encoding='utf8') if path.suffix == '.gz'
              else path.open(encoding='utf8')) as stream:
            for line in stream:
                if not line.endswith('\n'):
                    raise IncompleteTrainingLogs('Unterminated source record: ' + path.name)
                if not line.strip():
                    raise ValueError('Blank source record: ' + path.name)
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IncompleteTrainingLogs('Incomplete source record: ' + path.name) from exc
    except (EOFError, OSError) as exc:
        raise IncompleteTrainingLogs('Truncated source stream: ' + path.name) from exc


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Nonfinite/missing active scalar: ' + name)
    return float(value)


def scan_numbers(value, prefix='root'):
    """Scan every structured value, preserving native null counts separately."""
    count = 0
    nulls = Counter()
    def walk(v, p):
        nonlocal count
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, p + '.' + k)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x, p + '[]')
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            finite(v, p)
            count += 1
        elif v is None:
            nulls[p] += 1
    walk(value, prefix)
    return dict(numeric_values=count, null_field_counts=dict(nulls), nonfinite_values=0)


def public_values(value):
    """Remove absolute private path values from optional augmentation metadata."""
    if isinstance(value, dict):
        return {k: public_values(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_values(v) for v in value]
    if isinstance(value, str) and (value.startswith('/') or value.startswith('\\\\') or re.match(r'^[A-Za-z]:[/\\]',value)):
        return '<private-path>'
    return value


def _near(a, b, message, tolerance=3e-5):
    error = abs(finite(a, message) - finite(b, message))
    if error > tolerance:
        raise ValueError(message)
    return error


def relation(r, active):
    details = r['SupCon_details']
    if r['crossview_mode'] != 'ORD128' or r['new_losses'] or r['extra_U_NN'] or len(details) != (4 if active else 0):
        raise ValueError('Changed original ordinary SupCon scope')
    if active and {(q['phase'], q['level']) for q in details} != {(p, l) for p in ('first', 'second') for l in ('time_down', 't2.pool')}:
        raise ValueError('Missing original SupCon depth/pass')
    for q in details:
        if (q['mode'], q['total_anchors'], q['ordinary_anchors']) != ('ORD128', 128, 128) or any(q[k] for k in ('additional_anchors', 'same_RX_positive_pairs', 'true_satellite_anchors')):
            raise ValueError('Changed SupCon identity membership')
    return sum(q['valid_anchors'] for q in details)


def satellite_record(q, arm, i, cumulative, inventory):
    if q['mode'] != arm or q['step'] != i or q['cumulative_step'] != cumulative or q['As_supervised'] != 64 or q['extra_NN'] or q['parameter_updates_extra']:
        raise ValueError('Satellite dropout mode/CE/count/step differs')
    calls = 2 * sum(v['p'] > 0 for v in inventory)
    if q['dropout_calls'] != calls or calls == 0:
        raise ValueError('Actual existing dropout call count differs')
    bypass = q['bypassed_elements']
    if type(bypass) is not int or (arm == 'AS_CE' and bypass != 0) or (arm == 'AS_DET' and bypass <= 0):
        raise ValueError('Only AS_DET bypasses satellite elements')
    for k in ('original_mixed_KL', 'ordinary_half_KL', 'satellite_half_KL', 'used_mixed_KL', 'removed_satellite_KL', 'satellite_pair_logit_maxdiff'):
        finite(q[k], k)
    errors = [
        _near(q['original_mixed_KL'], .5 * (q['ordinary_half_KL'] + q['satellite_half_KL']), 'Mixed KL decomposition differs'),
        _near(q['used_mixed_KL'], .5 * q['ordinary_half_KL'], 'Used ordinary-only KL differs'),
        _near(q['removed_satellite_KL'], .5 * q['satellite_half_KL'], 'Removed satellite KL differs'),
        _near(q['original_mixed_KL'], q['used_mixed_KL'] + q['removed_satellite_KL'], 'Satellite KL subtraction differs'),
    ]
    if q['satellite_pair_logit_maxdiff'] < 0 or (arm == 'AS_DET' and (q['satellite_pair_logit_maxdiff'] > 1e-5 or abs(q['satellite_half_KL']) > 1e-6)):
        raise ValueError('AS_DET satellite pair remains stochastic')
    return max(errors)


def native_epochs(b, count):
    full = list(records(b / 'metrics_epoch.jsonl'))
    compact = list(records(b / 'epoch_metrics.jsonl'))
    if len(full) < count or len(compact) < count:
        raise IncompleteTrainingLogs('Native epoch log is a prefix')
    if [v['epoch'] for v in full] != list(range(1, count + 1)) or [v['epoch'] for v in compact] != list(range(1, count + 1)):
        raise ValueError('Native epoch count/order differs')
    reports = {}
    for name, values in [('metrics_epoch', full), ('epoch_metrics', compact)]:
        path = b / (name + '.csv')
        if not path.is_file():
            raise IncompleteTrainingLogs('Missing native CSV: ' + path.name)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            actual = list(reader)
            header = reader.fieldnames or []
        if len(actual) < count:
            raise IncompleteTrainingLogs('Native CSV is a prefix')
        if len(actual) != count:
            raise ValueError('Native CSV/JSONL row count mismatch')
        # IncrementalTelemetry keeps the union of fields seen across epochs and
        # rewrites earlier CSV rows with DictWriter's empty missing-field value.
        # JSONL preserves each epoch's actual keys. Verify the exact union first;
        # only genuinely absent JSON fields may correspond to blank CSV cells.
        union = set().union(*(set(row) for row in values))
        if len(header) != len(set(header)) or set(header) != union:
            raise ValueError('Native CSV/JSONL schema union field mismatch')
        missing_blank_cells = 0
        for x, y in zip(values, actual):
            if set(y) != union:
                raise ValueError('Native CSV/JSONL field mismatch')
            for k in union-set(x):
                if y[k] != '':
                    raise ValueError('CSV value for absent JSON epoch field: ' + k)
                missing_blank_cells += 1
            for k, v in x.items():
                text = '' if v is None else str(v)
                if isinstance(v, (list, dict, tuple)):
                    try:
                        same = json.loads(y[k]) == v
                    except (ValueError, TypeError):
                        same = y[k] == str(v)
                else:
                    same = y[k] == text
                if not same:
                    raise ValueError('Native CSV/JSONL value mismatch: ' + k)
        reports[name] = dict(records=count, CSV_records=count, all_values_equal=True, CSV_schema_union_fields=len(union), CSV_missing_JSON_blank_cells=missing_blank_cells, **scan_numbers(values))
    if compact != [{k: v for k, v in row.items() if not isinstance(v, (list, dict, tuple))} for row in full]:
        raise ValueError('Compact observer/native epoch telemetry differs')
    return compact, reports


def _args(b):
    a = read(b / 'actual_native_args.json')
    return a.get('args', a)


def parse_segment(row, arm, stage, epochs, labels, offset, canonical_contract):
    b = row / stage
    done = read(b / 'completion.json')
    result = read(b / 'result.json')
    cfg = done['config']
    expected = epochs * 222
    expected_parent = [] if offset == 0 else [str(row / 'stage20' / 'final_ssdg.pth')]
    init = read(b / 'initialization.json')
    freeze = read(b / 'source_frozen.json')
    new = arm in ('AS_CE', 'AS_DET')
    run_id = RUN if new else REFERENCE
    sat = arm != 'B00'
    required_budget = (epochs, expected, epochs, labels, epochs-labels, expected, 222, offset, offset*222)
    actual_budget = (done['epoch'], done['optimizer_steps'], cfg['epochs'], cfg['label_epochs'], cfg['pseudo_epochs'], cfg['updates'], cfg['steps_per_epoch'], cfg['epoch_offset'], cfg['step_offset'])
    if actual_budget != required_budget or cfg['run_id'] != run_id or cfg['row_id'] != row.name or cfg['model_seed'] != 2026092701:
        raise ValueError('Fixed own20+10 epoch/update/config differs')
    if (cfg['data_contract'] != dict(RX=list(RX), days=[1,2,3], L=6300, U=56700, V=27000, split=392005, source_RX_holdout=None, internal_L_split=False, U_TX_truth_hidden=True)
            or cfg['new_Ag'] or cfg['Ag_mode'] != 'NONE' or cfg['As'] != sat or cfg['L_NN_per_step'] != 768 or cfg['total_L_loss_scale'] != 1.25 or cfg['target_access']):
        raise ValueError('Fixed original roles/allRX/6forward view contract differs')
    if cfg.get('satellite_mode') != (arm if new else None):
        raise ValueError('New satellite mode differs')
    if freeze['config'] != cfg or freeze['selection'] != 'final_only' or freeze['target_access'] or result['config'] != cfg:
        raise ValueError('Own frozen final config differs')
    if (init['checkpoint_sources'] != expected_parent or cfg['checkpoint_sources'] != expected_parent or init['scratch_only'] != (offset == 0)
            or init['source_roles'] != 'EXACT_MATCH' or init['target_contact'] or init['ancestors'] != ([] if offset == 0 else [str(row/'stage20')])
            or init['teacher_origin'] != ('own scratch student' if offset == 0 else 'strict own parent EMA')):
        raise ValueError('Own student/EMA inheritance differs')
    if read(b/'source_contract.json')['role_ids'] != canonical_contract:
        raise ValueError('Original source role handle differs')
    if any(done[k] for k in ('target_access','target_evaluated','head_fit_updates')) or done['native_disabled_probe_calls'] != 1 or not done['native_fitted_probes_disabled']:
        raise ValueError('Default head/source-only completion differs')
    if (result['status'] != 'SEGMENT_TRAINED' or result['optimizer_steps'] != expected or result['L_NN'] != expected*768 or result['L_As'] != expected*(64 if sat else 0)
            or result['Ag_packets'] or result['Ag_control_NN'] != expected*256 or result['total_L_loss_scale'] != 1.25
            or any(result[k] for k in ('extra_U_NN','U_As','U_truth_read','target_access','head_fit_updates'))
            or result['source_RX'] != list(RX) or result['source_RX_holdout'] is not None):
        raise ValueError('Recorded source costs/permissions differ')
    a = _args(b)
    if (a['epochs'], a['label_epochs'], a['pseudo_epochs']) != (epochs, labels, epochs-labels) or a['baseline_ckpt'] != (expected_parent[0] if offset else '') or a['teacher_ckpt'] or a['freeze_backbone'] or a.get('source_fold_held_rx') is not None:
        raise ValueError('Actual native initialization/budget differs')
    opt = read(b/'optimizer_initialization.json')
    if opt['state_entries'] or not opt['reset'] or opt['schedule'] != 'cosine'+str(epochs):
        raise ValueError('Native optimizer reset/schedule differs')
    ema = read(b/'EMA_initialization.json')
    # The native initializer describes the generic own parent; the observer
    # records the concrete stage20 EMA loaded for this fixed +10 segment.
    # Their exact producer strings differ. Parent paths are checked above.
    expected_ema_origin = 'own scratch student' if offset == 0 else 'strict own stage20 EMA'
    if (ema['origin'] != expected_ema_origin or ema['extra_teacher'] or not ema['optimizer_segment_counters_reset'] or not ema['temporal_cache_reset']):
        raise ValueError('Original EMA scope differs')
    ordinary = read(b/'ordinary_augmentation_actual.json')
    if ordinary['schedule_frozen_at'] != 13 or not ordinary['no_ordinary_before_practical'] or not ordinary['RDrop_same_IQ']:
        raise ValueError('Actual ordinary augmentation organization differs')
    nU = (epochs-labels)*56700
    expected_cost = dict(student_train_NN=expected*768+nU, student_eval_NN=epochs*27000+54002, teacher_train_NN=0, teacher_eval_NN=nU)
    if any(result[k] != v for k,v in expected_cost.items()):
        raise ValueError('Native student/teacher packet cost differs')
    teacher_offset = result['teacher_counter_offset']
    if teacher_offset != dict(train=0, eval=2):
        raise ValueError('Initial student smoke copied into EMA counter offset differs')
    inventory = result['satellite_subnetwork']['dropout_inventory'] if new else []
    if new:
        inventory_names = [v['name'] for v in inventory]
        if not inventory or len(set(inventory_names)) != len(inventory_names) or any(not v['name'].startswith('id_backbone.encoder.') or not 0 <= v['p'] < 1 for v in inventory):
            raise ValueError('Existing dropout inventory differs')
        if result['satellite_subnetwork']['mode'] != arm or result['satellite_subnetwork']['steps'] != expected or result['satellite_subnetwork']['extra_NN']:
            raise ValueError('Segment satellite scope differs')
    streams = [records(b/name) for name in ('reference_steps.jsonl.gz','shape_steps.jsonl.gz','step_metrics.jsonl.gz','unit_steps.jsonl.gz')]
    if new:
        streams.append(records(b/'satellite_subnetwork_steps.jsonl.gz'))
    sums = defaultdict(Counter)
    count = Counter()
    max_errors = Counter()
    ordinary_stream = []
    channel_stream = []
    bypass_set = set()
    numeric_count = 0
    for i, items in enumerate(zip_longest(*streams), 1):
        if any(q is None for q in items):
            raise IncompleteTrainingLogs('Unequal full step stream lengths')
        if i > expected:
            raise ValueError('Extra source update outside fixed budget')
        r,s,n,u = items[:4]
        cumulative = offset*222+i
        epoch = offset+(i-1)//222+1
        local = (i-1)//222+1
        numeric_count += scan_numbers(items)['numeric_values']
        if any(q['step'] != i for q in items) or any(q['cumulative_step'] != cumulative for q in (r,s,u)) or n['epoch'] != local or u['physical_clip_calls'] != i:
            raise ValueError('Contiguous own20+10 step/Sinc counters differ')
        if r['L_forwards'] != 6 or r['As_packets'] != (64 if sat else 0) or r['Ag_packets'] or r['Ag_mode'] != 'NONE' or r['extra_control_packets'] != 128 or r['ordinary_redraws'] or not r['same_IQ_each_pair'] or not r['all_satellite_L_supervised']:
            raise ValueError('Actual supervised view scope differs')
        if (s['mode'],s['NN'],s['packets'],s['applied'],s['weight']) != ('NONE',256,128,0,.25) or not s['same_IQ_each_pair'] or s['new_boundary_samples'] or s['relative_noise_RMS'] or s['noise_peak_fraction_mean'] or s['SNR_added_db'] is not None:
            raise ValueError('Raw matched third-pair slot changed')
        weight = .1*min(max((epoch-5)/10,0),1)
        max_errors['SupCon_weight'] = max(max_errors['SupCon_weight'], _near(r['SupCon_weight'],weight,'Original SupCon schedule differs',1e-12))
        for prefix, v in [('common',r),('extra',r),('shape',s)]:
            max_errors[prefix+'_RD'] = max(max_errors[prefix+'_RD'], _near(v[prefix+'_RD'],v[prefix+'_CE']+v[prefix+'_KL'],'RDrop CE/KL decomposition differs'))
        total = .5*r['common_RD']+.5*r['extra_RD']+weight*r['SupCon']+.25*s['shape_RD']
        max_errors['L_total'] = max(max_errors['L_total'], _near(r['total_L'],total,'Total sixforward L objective differs'))
        max_errors['L_slot'] = max(max_errors['L_slot'], _near(r['total_L'],r['base_L_before_shape']+.25*s['shape_RD'],'Additional raw slot reduction differs'))
        m = n['metrics']
        if m['concat/L_forwards'] != 4 or m['concat/As_packets'] != 64 or any(m.get(k) or 0 for k in ('train/pseudo_truth_available','train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')):
            raise ValueError('Original native branch/U truth/numerical scope differs')
        _near(m['concat/total_L'], r['base_L_before_shape'], 'Native inherited branch telemetry differs')
        _near(m['shape/RD'],s['shape_RD'],'Native extra-slot telemetry differs',1e-6)
        finite(m['train/loss'],'native loss')
        if local <= labels:
            max_errors['native_supervised_L'] = max(max_errors['native_supervised_L'],_near(m['train/loss'],total,'Actual supervised native L total differs'))
        else:
            ptotal, selected = m['train/pseudo_total'], m['train/pseudo_selected']
            if type(ptotal) not in (int,float) or int(ptotal)!=ptotal or not 0 <= selected <= ptotal <= 256 or ptotal <= 0:
                raise ValueError('Original hidden-U packet/selected count differs')
            count['native_U_packets'] += int(ptotal)
            count['selected_U_packets'] += int(selected)
        if sum(r['scene_counts'].values()) != 64 or sum(r['state_counts'].values()) != 64 or (sat and set(r['scene_counts'])-set(SCENES)):
            raise ValueError('All64 complete practical observations missing')
        if any(type(v) is not int or v<0 for d in (r['scene_counts'],r['state_counts']) for v in d.values()):
            raise ValueError('Invalid scene/state counts')
        mse = finite(r['ordinary_raw_MSE'],'actual ordinary IQ MSE')
        if mse < 0:
            raise ValueError('Negative actual ordinary IQ MSE')
        ordinary_stream.append(mse)
        channel_stream.append((r['scene_counts'],r['state_counts']))
        if new:
            q = items[4]
            if r['satellite_subnetwork'] != {k:v for k,v in q.items() if k!='cumulative_step'}:
                raise ValueError('Nested satellite telemetry differs from independent step stream')
            max_errors['removed_KL'] = max(max_errors['removed_KL'],satellite_record(q,arm,i,cumulative,inventory))
            max_errors['used_extra_KL'] = max(max_errors['used_extra_KL'],_near(r['extra_KL'],q['used_mixed_KL'],'Actual extra KL includes removed satellite KL'))
            count['dropout_calls'] += q['dropout_calls']
            count['bypassed_elements'] += q['bypassed_elements']
            bypass_set.add(q['bypassed_elements'])
            for k in ('original_mixed_KL','ordinary_half_KL','satellite_half_KL','used_mixed_KL','removed_satellite_KL','satellite_pair_logit_maxdiff'):
                sums[epoch][k] += q[k]
        count['steps'] += 1
        count['SupCon_active_steps'] += weight>0
        count['SupCon_valid_anchor_terms'] += relation(r,weight>0)
        count['As_packets'] += r['As_packets']
        sums[epoch].update({k:r[k] for k in ('common_CE','common_KL','extra_CE','extra_KL','SupCon','SupCon_weight','total_L','ordinary_raw_MSE')})
        sums[epoch].update({k:s[k] for k in ('shape_CE','shape_KL','shape_RD')})
        sums[epoch]['native_loss'] += m['train/loss']
        sums[epoch]['weighted_SupCon'] += weight*r['SupCon']
        sums[epoch]['steps'] += 1
    if count['steps'] < expected:
        raise IncompleteTrainingLogs('Step log is a prefix, not completed training')
    if count['native_U_packets'] != nU or count['As_packets'] != expected*(64 if sat else 0) or any(v['steps'] != 222 for v in sums.values()):
        raise ValueError('Complete segment native U/As budget differs')
    if new and (len(bypass_set)!=1 or result['satellite_subnetwork']['bypassed_elements'] != count['bypassed_elements']):
        raise ValueError('Satellite bypass shape/total inconsistent')
    epoch_data, telemetry = native_epochs(b, epochs)
    curves = []
    for v in epoch_data:
        ep = v['epoch']+offset
        curve = dict(epoch=ep,segment=20 if not offset else 30,**{k:x/222 for k,x in sums[ep].items() if k!='steps'})
        curve.update({k:v[k] for k in ('lr','epoch_time_s','val_tx_acc','train_grad_backbone','train_grad_total')})
        curve.update({k:x for k,x in v.items() if k.startswith('aug_')})
        curves.append(curve)
    sequence_count = Counter()
    unique = {'L':set(),'U':set()}
    batches = Counter()
    for v in records(b/'training_sequence.jsonl.gz'):
        role = v['role']
        if role not in unique or len(v['ids']) != (128 if role=='L' else (256 if batches['U']%222!=221 else 124)):
            raise ValueError('Original physical loader role/batch size differs')
        for pid in v['ids']:
            if len(pid)!=5 or any(type(x) is not int for x in pid) or pid[1] not in RX or pid[2] not in (1,2,3):
                raise ValueError('Original source physical metadata differs')
            unique[role].add(tuple(pid))
        sequence_count[role] += len(v['ids'])
        batches[role] += 1
    if batches['L'] < expected or batches['U'] < (epochs-labels)*222:
        raise IncompleteTrainingLogs('Physical loader sequence is a prefix')
    if dict(batches) != dict(L=expected, U=(epochs-labels)*222) or sequence_count['L'] != expected*128 or sequence_count['U'] != nU or len(unique['L']) != 6300 or len(unique['U']) != (56700 if nU else 0) or unique['L']&unique['U']:
        raise ValueError('Full original physical L6300/U56700 exposure differs')
    for role in ('L','U'):
        if unique[role] and ({v[1] for v in unique[role]}!=set(RX) or {v[2] for v in unique[role]}!={1,2,3}):
            raise ValueError('Source RX/day omitted from real loader stream')
    wall = finite(done['elapsed_seconds'],'actual segment wall seconds')
    return dict(segment=stage,epochs=epochs,updates=expected,cost=expected_cost,teacher_counter_offset=teacher_offset,
                wall_seconds=wall,counts=dict(count),loader_packets=dict(sequence_count),loader_batches=dict(batches),
                unique_physical={k:len(v) for k,v in unique.items()},max_errors=dict(max_errors),epoch_curves=curves,
                native_telemetry=telemetry,scanned_step_numeric_values=numeric_count,
                ordinary_draws=len(ordinary_stream),ordinary_augmentation=public_values(ordinary),
                satellite_inventory=inventory,scope_restoration_evidence='Producer finish asserts engine inactive; final flag not serialized' if new else None), dict(ordinary=ordinary_stream,channel=channel_stream,args=a)


def native_arg_differences(a, b, arm):
    expected_run = RUN if arm in ('AS_CE','AS_DET') else REFERENCE
    if arm not in ARMS or a.get('run_id') != expected_run or b.get('run_id') != REFERENCE:
        raise ValueError('Actual native run identity differs from registered row')
    keys = sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    if set(keys)-{'candidate_id','output_dir','baseline_ckpt','run_id'}:
        raise ValueError('Matched native training arguments differ')
    return keys


def compare_streams(pa, pb):
    n = mismatch = 0
    for a,b in zip_longest(records(pa),records(pb)):
        n += 1
        mismatch += a != b
    return dict(records=n,mismatch=mismatch,all_equal=mismatch==0)


def run(project):
    started = time.perf_counter()
    project = Path(project)
    roots = {a:project/'runs'/(RUN if a.startswith('AS_') else REFERENCE)/(a+'-s2026092701') for a in ARMS}
    # Only already-authorized source role handles; no data, validation rebuild, CP or truth.
    contract = read(roots['AS_CE']/'stage20'/'completion.json')['config']['source_contract']
    contract = read(contract)['role_ids']
    rows, private = {}, {}
    for arm, row in roots.items():
        segments, diagnostics = [], {}
        for stage,ep,labels,offset in SEGMENTS:
            segment, diag = parse_segment(row,arm,stage,ep,labels,offset,contract)
            segments.append(segment)
            diagnostics[stage] = diag
        textpath = row.parent/'logs'/(row.name+'.log')
        if not textpath.is_file():
            raise IncompleteTrainingLogs('Missing complete stdout')
        text = textpath.read_text(encoding='utf8',errors='replace')
        markers = {p:len(re.findall(p,text)) for p in ('Traceback','CUDA out of memory',r'\bKilled\b','RuntimeError','AssertionError',r'\[ERROR\]')}
        if any(markers.values()):
            raise ValueError('Technical marker in completed full stdout')
        cost = {k:sum(v['cost'][k] for v in segments) for k in COST_KEYS}
        if sum(v['updates'] for v in segments)!=6660 or sum(v['loader_packets']['L'] for v in segments)!=852480 or sum(v['loader_packets']['U'] for v in segments)!=963900:
            raise ValueError('Complete scratch E30 original packet budget differs')
        rows[arm] = dict(status='VERIFIED_COMPLETE_TRAINING_LOGS',source_RX=list(RX),source_days=[1,2,3],source_RX_holdout=None,
                         segments=segments,epoch_curves=[c for s in segments for c in s['epoch_curves']],
                         cost=dict(**cost,native_training_and_validation_NN=sum(cost.values()),L_NN=5114880,new_epochs=30,new_updates=6660,own_stage20_included=True,external_parent_epochs=0,teacher_offset_excluded=True,wall_seconds=sum(s['wall_seconds'] for s in segments)),
                         full_stdout=dict(lines=len(text.splitlines()),all_scanned=True,error_marker_counts=markers,warning_lines=sum(bool(re.search(r'warning|warn:',v,re.I)) for v in text.splitlines()),NaN_word_mentions=len(re.findall(r'\bnan\b',text,re.I))),
                         source_only=True,U_truth_hidden=True,head_fit_updates=0)
        private[arm] = diagnostics
    matched = {}
    for arm in ('B01','AS_CE','AS_DET'):
        matched[arm] = {}
        for stage,_,_,_ in SEGMENTS:
            seq = compare_streams(roots[arm]/stage/'training_sequence.jsonl.gz',roots['B00']/stage/'training_sequence.jsonl.gz')
            a,b = private[arm][stage],private['B00'][stage]
            ordinary_mismatch = sum(x!=y for x,y in zip_longest(a['ordinary'],b['ordinary']))
            keys = native_arg_differences(a['args'],b['args'],arm)
            if not seq['all_equal'] or ordinary_mismatch:
                raise ValueError('Matched physical/ordinary/native argument streams differ')
            # Parents differ only by the row whose own scratch model was trained.
            if stage=='source' and (a['args']['baseline_ckpt'] != str(roots[arm]/'stage20/final_ssdg.pth') or b['args']['baseline_ckpt']!=str(roots['B00']/'stage20/final_ssdg.pth')):
                raise ValueError('Own-row continuation mismatch')
            channel_mismatch = None if arm=='B01' else sum(x!=y for x,y in zip_longest(a['channel'],private['B01'][stage]['channel']))
            if channel_mismatch:
                raise ValueError('Matched complete practical scene/state sequence differs')
            matched[arm][stage] = dict(physical_sequence=seq,ordinary_IQ_MSE_records=len(a['ordinary']),ordinary_IQ_MSE_mismatch=ordinary_mismatch,
                                     channel_scene_state_mismatch_vs_B01=channel_mismatch,different_actual_arg_keys=keys,
                                     ordinary_IQ_MSE_scope='Realized scalar input MSE equality; not bytewise IQ equality')
    if any(rows[a]['cost'][k]!=rows['B00']['cost'][k] for a in ARMS for k in COST_KEYS):
        raise ValueError('Matched native NN costs differ')
    return dict(status='VERIFIED_COMPLETE_SOURCE_TRAINING_LOGS',run_id=RUN,reference_run=REFERENCE,rows=rows,matched_comparisons=matched,
                new_training=dict(models=2,seed_count=1,model_seed=2026092701,epochs=60,updates=13320,native_NN=sum(rows[a]['cost']['native_training_and_validation_NN'] for a in ('AS_CE','AS_DET'))),
                reused_training=dict(models=2,epochs=60,updates=13320,native_NN=sum(rows[a]['cost']['native_training_and_validation_NN'] for a in ('B00','B01'))),
                analysis_cost=dict(IQ_reads=0,CP_reads=0,NN=0,updates=0,target_scores_read=False,raw_truth_read=False),
                cost_scope='Native student/EMA training,validation,2 synthetic smoke packets per segment; excludes frozen source readout and formal predictions, to add from their completion artifacts',
                complete=True,performance_conclusion=False,wall_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    value = run(args.project)
    Path(args.output).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
