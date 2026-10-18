"""Nonblocking source-L coverage attribution; no IQ, checkpoint or model access.

P67 directly saves RX/day/scene/state/lock counts. Its TX joint table was not
serialized. Reuse the fixed P66 MASS_UNIFORM joint table only when the same
validated labeled-data semantics, complete ordered L stream and deterministic
channel context match. Do not derive TX from physical-ID numbers or U records.
"""
import argparse
import ast
from collections import Counter
import csv
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import re
import tempfile
import time

RUN = '20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01'
REFERENCE = '20261018-phase1-FcrossRX-Uquality-e30-s2026092701-m2-r01'
PARENT = '20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
P43_COMMIT = '19aedc3604fed9c193237dd92366164178af0983'
ROW = 'B01-s2026092701'
REFERENCE_ROW = 'MASS_UNIFORM-s2026092701'
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']
RX, DAYS = (1, 3, 4, 6, 8), (1, 2, 3)
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
STATES, LOCKS = ('LOS', 'PARTIAL', 'BLOCKED'), ('locked', 'degraded', 'unlocked')
FIELDS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
ROLE = re.compile(r'^\s*\{\s*"role"\s*:\s*"([^"]+)"')
INPUT_ARGS = ('dataset', 'dataset_dir', 'wisig_pkl', 'wisig_equalized', 'wisig_domain',
    'wisig_out_len', 'wisig_train_days', 'wisig_train_rxs', 'wisig_split_strategy',
    'wisig_cap_strategy', 'wisig_guard_gap', 'wisig_max_day123_per_combo',
    'wisig_max_train_per_combo', 'wisig_max_val_per_combo', 'wisig_max_test_per_combo',
    'split_mode', 'phase1_source_role_protocol', 'labeled_ratio', 'unlabeled_ratio',
    'source_val_ratio', 'seed', 'num_classes', 'sample_rate_hz')
WORDING = ('P67 directly observed the RX/day/scene/state/lock coverage. The TX joint '
    'counts below are reused from fixed P66 MASS_UNIFORM after matching the complete '
    'ordered labeled stream, validated source/label semantics and deterministic '
    'channel context; they were not directly serialized by P67. No physical replay '
    'or inference of TX from U or physical-ID values was performed.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def table(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labeled_records(path):
    """Read only each line's role prefix before decoding labeled records.

    Native RecordingLoader emits role first. U lines are skipped without JSON
    decoding, inspecting their IDs or reading any U label field.
    """
    with gzip.open(path, 'rt', encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            match = ROLE.match(line)
            require(match is not None, 'Training sequence is not the native role-first format')
            role = match.group(1)
            if role == 'U':
                continue
            require(role == 'L', 'Unexpected training sequence role')
            record = json.loads(line)
            require(record.get('role') == 'L', 'L sequence role differs')
            ids = record.get('ids')
            require(isinstance(ids, list) and len(ids) == 128, 'Expected every original L128 batch')
            require(all(isinstance(k, list) and len(k) == 5 and all(type(v) is int for v in k)
                and k[0] == 1 and k[1] in RX and k[2] in DAYS for k in ids), 'Invalid source L physical keys')
            yield ids


def compare_sequences(first, second, expected_steps=2220):
    count, packets, selected = 0, 0, Counter()
    for count, (a, b) in enumerate(zip_longest(labeled_records(first), labeled_records(second)), 1):
        require(a is not None and b is not None and a == b, 'Complete ordered L physical sequences differ')
        packets += len(a)
        # P43 uses local step parity; its +4440 channel offset is even.
        for position, key in enumerate(a):
            if (position + count) % 2 != 0:
                selected[(key[1], key[2])] += 1
    require(count == expected_steps and packets == 128 * expected_steps, 'Complete 2220-batch L sequence is missing')
    return dict(steps=count, physical_L_visits=packets, selected_As_visits=sum(selected.values()),
        physical_key_order=list(FIELDS), exact_ordered_ID_match=True,
        label_semantics='Reused from identical validated labeled source handle/roles/class mapping; never inferred from IDs',
        U_records_decoded=0), selected


def channel_table(channel):
    require(channel.get('calls') == 2220 and channel.get('packets') == 142080
        and channel.get('session_count') == 15 and channel.get('dropped') == 0
        and channel.get('confidence_filter') is False and channel.get('ordinary_before_satellite') is False
        and channel.get('formal_test_changed') is False and channel.get('all_original_source_RX') == list(RX),
        'Actual complete ungated original L channel coverage differs')
    require(set(channel.get('sessions', {})) == {f'rx{r}:day{day}:eq1' for r in RX for day in DAYS},
        'Actual RX/day hardware session set differs')
    counts = Counter()
    for row in channel.get('strata', []):
        key = (row['RX'], row['day'], row['scene'], row['state'], row['lock'])
        require(key[0] in RX and key[1] in DAYS and key[2] in SCENES and key[3] in STATES
            and key[4] in LOCKS and type(row['count']) is int and row['count'] > 0,
            'Invalid directly observed source channel coverage cell')
        require(key not in counts, 'Duplicate directly observed source coverage cell')
        counts[key] = row['count']
    require(sum(counts.values()) == 142080, 'Actual channel state coverage total differs')
    return counts


def reconcile_joint(joint, direct):
    counts, marginal = Counter(), Counter()
    for row in joint:
        key = (row['TX'], row['RX'], row['scene'], row['state'], row['lock'])
        require(type(key[0]) is int and key[0] in range(6) and key[1] in RX and key[2] in SCENES
            and key[3] in STATES and key[4] in LOCKS and type(row['count']) is int and row['count'] > 0,
            'Invalid reference TX joint coverage cell')
        require(key not in counts, 'Duplicate reference TX joint coverage cell')
        counts[key] = row['count']
        marginal[key[1:]] += row['count']
    observed = Counter()
    for (rx, day, scene, state, lock), count in direct.items():
        observed[(rx, scene, state, lock)] += count
    require(marginal == observed, 'Reference TX joint coverage does not reconcile to P67 actual RX/scene/state/lock marginals')
    return counts


def runtime_reference(cwd, package):
    path = Path(cwd) / 'experiments' / package / 'runner.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == 'REFERENCE_COMMIT':
            values[name] = ast.literal_eval(node.value)
        elif name == 'REFERENCE' and isinstance(node.value, ast.Call) and node.value.args:
            values[name] = ast.literal_eval(node.value.args[0])
    require(values.get('REFERENCE_COMMIT') == P43_COMMIT
        and str(values.get('REFERENCE', '')).replace('\\', '/').endswith('/releases/cvs_FcrossRX_impulse_factorial_E30_20261018_r01'),
        'The declared shared pinned P43 training runtime differs')
    return dict(reference=values['REFERENCE'], commit=values['REFERENCE_COMMIT'], runner=str(path))


def attribute(project):
    project = Path(project)
    first = project / 'runs' / RUN / ROW / 'source'
    second = project / 'runs' / REFERENCE / REFERENCE_ROW / 'source'
    report = dict(status='NOT_REUSED', nonblocking=True, candidate_run=RUN, candidate_row=ROW,
        fixed_reference_run=REFERENCE, fixed_reference_row=REFERENCE_ROW,
        missing=[], reused_evidence={}, direct_coverage_available=False,
        NN=0, IQ_reads=0, checkpoint_reads=0, physical_channel_calls=0, updates=0,
        target_access=False, formal_files_read=False, U_truth_read=False,
        candidate_promotion_changed=False, scientific_goal_claimed=False)
    direct_rows = []
    try:
        current = read(first / 'result.json')
        direct = channel_table(current['L_channel'])
        direct_rows = current['L_channel']['strata']
        report['direct_coverage_available'] = True
        report['direct_observation'] = dict(path=str(first / 'result.json'), key='L_channel.strata',
            dimensions=['RX', 'day', 'scene', 'state', 'lock'], cells=len(direct), count=sum(direct.values()))
        reference = read(second / 'result.json')
        previous = channel_table(reference['L_channel'])
        require(direct == previous and current['L_channel']['sessions'] == reference['L_channel']['sessions'],
            'Actual L channel strata or session hardware differs')
        config, old_config = current['config'], reference['config']
        parent = config.get('parent_source')
        require(parent == old_config.get('parent_source') and isinstance(parent, str)
            and parent.replace('\\', '/').endswith(f'/runs/{PARENT}/{ROW}/stage20'),
            'The corresponding own P43 B01 As20 parent source differs')
        for root, result, c, run, row in ((first, current, config, RUN, ROW), (second, reference, old_config, REFERENCE, REFERENCE_ROW)):
            done, init = read(root / 'completion.json'), read(root / 'initialization.json')
            require(c.get('run_id') == run and c.get('row_id') == row and c.get('As') is True
                and c.get('Ag_mode') == 'NONE' and c.get('new_Ag') is False
                and (c.get('epoch_offset'), c.get('step_offset')) == (20, 4440)
                and c.get('L_NN_per_step') == 768 and c.get('total_L_loss_scale') == 1.25,
                'Original labeled-view routing/clock/source configuration differs')
            require(done.get('config') == c and done.get('epoch') == 10 and done.get('optimizer_steps') == 2220
                and not done.get('target_access') and result.get('head_fit_updates') == 0
                and result.get('U_truth_read') is False and result.get('L_As') == 142080,
                'The complete source-only tail/default-head coverage is unavailable')
            require(init.get('source_roles') == 'EXACT_MATCH' and init.get('target_contact') is False
                and init.get('scratch_only') is False and init.get('checkpoint_sources') == [parent + '/final_ssdg.pth'],
                'The source-data/own-parent initialization metadata differs')
        source, old_source = read(first / 'source_contract.json'), read(second / 'source_contract.json')
        require(source.get('role_ids') == old_source.get('role_ids') and source.get('counts') == old_source.get('counts')
            and source.get('counts') == {'L_s': 6300, 'U_s': 56700, 'V': 27000}
            and source.get('native_role_comparison') == old_source.get('native_role_comparison') == 'EXACT_MATCH',
            'Validated source role handles differ')
        for key in ('schema', 'source_rxs', 'source_days', 'num_classes', 'domain_kind', 'domain_map', 'split_rule', 'split_seed'):
            require(key in source and source[key] == old_source.get(key), 'Source role/class/domain semantics differ: ' + key)
        args, old_args = read(first / 'actual_native_args.json'), read(second / 'actual_native_args.json')
        args, old_args = args.get('args', args), old_args.get('args', old_args)
        for key in INPUT_ARGS:
            require(key in args and args[key] == old_args.get(key), 'Actual labeled data handle/encoding differs: ' + key)
        require(args['num_classes'] == 6 and args['wisig_equalized'] == '1'
            and args['wisig_train_rxs'] == '1,3,4,6,8' and args['wisig_train_days'] == '1,2,3'
            and args['wisig_out_len'] == 256, 'Current full-source label/input semantics differ')
        audit, old_audit = read(first / 'actual_channel.json'), read(second / 'actual_channel.json')
        require(audit == old_audit, 'Actual L channel modules/configuration differ')
        require(config.get('channel') == old_config.get('channel') and config.get('channel', {}).get('seeds') ==
            {'train': 20261016421, 'validation': 20261016422, 'receiver': 20261016423},
            'The per-ID L stream/channel hardware seeds differ')
        resolved, old_resolved = read(first / 'resolved_config.json'), read(second / 'resolved_config.json')
        require(resolved.get('identity_input_policy') == old_resolved.get('identity_input_policy')
            and resolved.get('identity_input_policy') is not None, 'Actual identity input normalization/policy differs')
        runtime = runtime_reference(resolved['cwd'], 'cvs_temporal_bank67')
        old_runtime = runtime_reference(old_resolved['cwd'], 'cvs_u_quality66')
        require(runtime['reference'] == old_runtime['reference'], 'Shared deterministic P43 adapter differs')
        sequence, selected = compare_sequences(first / 'training_sequence.jsonl.gz', second / 'training_sequence.jsonl.gz')
        observed_rx_day = Counter()
        for (rx, day, scene, state, lock), count in direct.items():
            observed_rx_day[(rx, day)] += count
        require(selected == observed_rx_day, 'Selected L64 physical RX/day counts differ from actual channel observations')
        joint = reference.get('TX_RX_scene_state_lock')
        require(isinstance(joint, list) and joint, 'The fixed P66 true-L-label joint coverage was not saved')
        counts = reconcile_joint(joint, direct)
        require(sum(counts.values()) == 142080, 'Reference joint coverage total differs')
        report.update(status='REUSED_MATCHED_DETERMINISTIC_L_STREAM', wording=WORDING,
            reused_evidence=dict(reference_result=str(second / 'result.json'), key='TX_RX_scene_state_lock',
                copied_joint_cells=len(counts), copied_joint_count=sum(counts.values()),
                complete_ordered_L_sequence=sequence, source_roles_equal=True, actual_data_args_equal=list(INPUT_ARGS),
                fixed_class_mapping=CLASSES, label_mapping_basis='Same validated labeled dataset and common pinned native class encoding; no new label reads',
                corresponding_parent=parent, actual_channel_config_equal=True, actual_state_lock_strata_equal=True,
                actual_hardware_sessions_equal=True, TX_marginals_reconcile=True, actual_selected_RX_day_reconcile=True,
                shared_runtime=dict(current=runtime, reference=old_runtime),
                deterministic_namespace='FcrossRX/train/e{local_epoch+20}/step{local_step+4440}/L_S',
                namespace_source='Shared pinned P43 dependency experiments/cvs_clock_factorial30/channel.py:Practical.apply',
                pid_order=list(FIELDS), selection='(batch_position+local_step)%2 != 0;64 of128;4440 offset is even',
                ordinary_before_satellite=False, input_source='Same raw labeled IQ before ordinary, same native crop/pad/RMS processing'),
            joint_counts_directly_serialized_by_P67=False)
        return report, direct_rows, joint
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError, SyntaxError) as exc:
        report['missing'].append(str(exc))
        report['wording'] = ('TX joint coverage was not reused because the saved source evidence is incomplete or differs. '
            'Report only the available P67 directly observed RX/day/scene/state/lock counts. No replay, model inference or training change follows.')
        return report, direct_rows, []


def self_test():
    # Two focused metadata-only checks: exact L ordering/selection, then a
    # changed ID and a wrong TX marginal must prevent attribution.
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        ids = [[1, 1 if i % 2 else 3, 1, i, 100 + i] for i in range(128)]
        for name in ('a.gz', 'b.gz'):
            with gzip.open(folder / name, 'wt', encoding='utf8') as f:
                for _ in range(2):
                    f.write(json.dumps(dict(role='L', ids=ids)) + '\n')
                    f.write('{"role": "U", this deliberately undecodable U body is skipped}\n')
        summary, selected = compare_sequences(folder / 'a.gz', folder / 'b.gz', expected_steps=2)
        assert summary['steps'] == 2 and summary['selected_As_visits'] == 128 and summary['U_records_decoded'] == 0
        assert sum(selected.values()) == 128
        direct = Counter({(1, 1, SCENES[0], 'LOS', 'locked'): 3})
        joint = [dict(TX=0, RX=1, scene=SCENES[0], state='LOS', lock='locked', count=3)]
        assert sum(reconcile_joint(joint, direct).values()) == 3
        changed = [list(k) for k in ids]
        changed[0][-1] += 1
        with gzip.open(folder / 'b.gz', 'wt', encoding='utf8') as f:
            f.write(json.dumps(dict(role='L', ids=changed)) + '\n')
        rejected = 0
        for fn in (lambda: compare_sequences(folder / 'a.gz', folder / 'b.gz', expected_steps=2),
                   lambda: reconcile_joint([dict(joint[0], count=4)], direct)):
            try:
                fn()
            except ValueError:
                rejected += 1
        assert rejected == 2
    return dict(status='PASS', focused_checks=2, negative_cases=2, NN=0, IQ_reads=0,
        checkpoint_reads=0, physical_channel_calls=0, target_access=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    if args.project is None or args.out is None:
        parser.error('--project and --out are required')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report, direct, joint = attribute(args.project)
    report['wall_seconds'] = time.perf_counter() - started
    table(args.out / 'direct_RX_day_scene_state_lock.csv', direct)
    table(args.out / 'reused_TX_RX_scene_state_lock.csv',
        [dict(candidate_row=ROW, evidence_kind='reused_from_fixed_P66_after_deterministic_L_stream_match', **row) for row in joint])
    write(args.out / 'reused_evidence.json', report)
    write(args.out / 'complete.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'nonblocking', 'direct_coverage_available', 'missing', 'NN', 'IQ_reads', 'checkpoint_reads', 'wall_seconds')}))


if __name__ == '__main__':
    main()
