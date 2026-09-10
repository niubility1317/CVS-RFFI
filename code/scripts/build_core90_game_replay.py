"""Freeze cross-seed source action schedules with exactly matched action counts.

All three JSONL schedules are loaded with --game_control replay. The names
fixed/random describe permutation of a completed OTHER seed's action schedule,
not runtime's fraction-based fixed/random control. No recipient measurements
or future recipient outcomes are read. Matching means preregistered actions;
successful updates, solver refusals, audit/HVP cost and elapsed time must be
measured and reported separately after recipient execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random


SCHEMA = 'core90_game_matched_source_schedule_v1'
ACTIONS = ('NORMAL', 'CATCHUP', 'CORRECT', 'HOLD_CURRICULUM')


def _json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def _jsonl(path):
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'{Path(path).name}:{index} is not a record')
        rows.append(row)
    return rows


def _count(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(name + ' must be a ' + ('positive' if positive else 'nonnegative') + ' integer')
    return value


def _no_target_contact(record):
    for key in ('target_contact', 'target_evaluated', 'target_evaluation', 'uses_target', 'target_used'):
        if key in record and record[key] is not False:
            raise ValueError('donor contains target contact or unknown target state: ' + key)
    if record.get('source_only') is False:
        raise ValueError('donor is not source-only')


def load_donor(donor_dir, recipient_seed):
    """Check a completed, untruncated source donor's mutually consistent files."""
    donor_dir = Path(donor_dir)
    config = _json(donor_dir / 'resolved_config.json')
    completion = _json(donor_dir / 'completion.json')
    rows = _jsonl(donor_dir / 'game_actions.jsonl')
    donor_seed = _count(config.get('seed'), 'donor seed')
    _count(recipient_seed, 'recipient seed')
    if donor_seed == recipient_seed:
        raise ValueError('donor_seed and recipient_seed must differ; no same-seed future replay')
    for artifact in (config, completion):
        _no_target_contact(artifact)
    if completion.get('target_evaluated') is not False:
        raise ValueError('completion must explicitly establish target_evaluated=false')
    if not (completion.get('source_only') is True or completion.get('scientific_verdict') == 'SOURCE_ONLY_NO_PROMOTION'):
        raise ValueError('completion lacks explicit source-only evidence')
    if completion.get('status') not in ('TRAINING_COMPLETE', 'SOURCE_ARTIFACTS_COMPLETE'):
        raise ValueError('donor training is not complete')
    epochs = _count(config.get('epochs'), 'configured epochs', positive=True)
    if completion.get('epoch') != epochs:
        raise ValueError('donor stopped before the full configured epoch budget')
    if config.get('game_max_steps_per_epoch', 0) != 0:
        raise ValueError('step-truncated donors cannot define matched full-run schedules')
    if config.get('from_scratch') is not True or config.get('baseline_ckpt', ''):
        raise ValueError('donor initialization must have a scratch source lineage')
    if (config.get('best_metric', 'clean_val_tx') != 'clean_val_tx'
            or config.get('enable_joint_safe_guard', False) or config.get('paic_guard_enabled', False)):
        raise ValueError('donor selection is not source-only')
    if config.get('wisig_train_rxs') and config.get('wisig_test_rxs'):
        if set(config['wisig_train_rxs'].split(',')) & set(config['wisig_test_rxs'].split(',')):
            raise ValueError('donor source and target RX overlap')
    if not rows or completion.get('step') != len(rows):
        raise ValueError('completion horizon disagrees with the complete action stream')
    epoch_counts = Counter()
    previous_epoch = 1
    for expected_step, row in enumerate(rows):
        _no_target_contact(row)
        if row.get('step') != expected_step:
            raise ValueError('action steps must be contiguous, unique and ordered from zero')
        epoch = _count(row.get('epoch'), 'action epoch', positive=True)
        if epoch > epochs or epoch < previous_epoch or epoch > previous_epoch + 1:
            raise ValueError('action epochs are missing or out of order')
        previous_epoch = epoch
        epoch_counts[epoch] += 1
        if row.get('action') not in ACTIONS:
            raise ValueError('unsupported donor action: ' + str(row.get('action')))
        requested = _count(row.get('requested_head_steps'), 'requested_head_steps')
        committed = _count(row.get('committed_head_steps'), 'committed_head_steps')
        if committed > requested:
            raise ValueError('committed head steps exceed requested steps')
        if type(row.get('accepted')) is not bool:
            raise ValueError('donor must record actual main-update acceptance')
        _count(row.get('field_evaluations'), 'field_evaluations')
        if 'forward_calls' in row:
            _count(row['forward_calls'], 'forward_calls')
        if row['accepted'] and row['field_evaluations'] < 1:
            raise ValueError('accepted main update has no field evaluation')
    if set(epoch_counts) != set(range(1, epochs + 1)) or len(set(epoch_counts.values())) != 1:
        raise ValueError('donor epochs lack a complete, consistent per-epoch action horizon')
    return config, completion, rows


def _packet(row):
    return dict(action=row['action'], requested_head_steps=row['requested_head_steps'],
                donor_committed_head_steps=row['committed_head_steps'], donor_step=row['step'],
                donor_accepted=row['accepted'])


def _signature(rows):
    return Counter((row['action'], row['requested_head_steps'], row['donor_committed_head_steps']) for row in rows)


def _summary(rows):
    pairs = Counter((row['requested_head_steps'], row['donor_committed_head_steps'])
                    for row in rows if row['action'] == 'CATCHUP')
    return dict(action_counts=dict(sorted(Counter(row['action'] for row in rows).items())),
                correct_count=sum(row['action'] == 'CORRECT' for row in rows),
                requested_head_steps=sum(row['requested_head_steps'] for row in rows),
                donor_committed_head_steps=sum(row['donor_committed_head_steps'] for row in rows),
                catchup_step_multiset=[dict(requested=requested, donor_committed=committed, count=count)
                                      for (requested, committed), count in sorted(pairs.items())])


def _costs(donor_dir, rows):
    result = dict(requested_head_steps=sum(row['requested_head_steps'] for row in rows),
                  committed_head_steps=sum(row['committed_head_steps'] for row in rows),
                  field_evaluations=sum(row['field_evaluations'] for row in rows),
                  committed_main_updates=sum(row['accepted'] for row in rows),
                  rejected_main_updates=sum(not row['accepted'] for row in rows),
                  committed_correct_actions=sum(row['action'] == 'CORRECT' and row['accepted'] for row in rows),
                  committed_corrections=sum(row['field_evaluations'] > 1 and row['accepted'] for row in rows),
                  forward_calls=sum(row['forward_calls'] for row in rows) if all('forward_calls' in row for row in rows) else None,
                  elapsed_seconds=None, audit_seconds=None,
                  cost_scope='reported donor work, including failed field evaluations; missing counters are UNKNOWN')
    resource_path = Path(donor_dir) / 'resource_summary.json'
    if resource_path.exists():
        resource = _json(resource_path)
        _no_target_contact(resource)
        if resource.get('full_budget_completed') is False:
            raise ValueError('resource summary reports truncated donor budget')
        result['elapsed_seconds'] = resource.get('total_seconds')
        result['audit_seconds'] = resource.get('budget', {}).get('lifetime', {}).get('audit_seconds')
    for name, value in result.items():
        if name == 'cost_scope' or value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid actual donor cost ' + name)
    return result


def build_schedules(donor_dir, recipient_seed, *, schedule_seed=None, include_curriculum=False):
    """Create replay, uniformly spaced fixed, and independently shuffled packets.

    Keeping whole packets preserves requested AND donor-committed catch-up
    counts jointly, including mixed correction/head actions and rejected work.
    The donor committed count is evidence only, never a claimed recipient result.
    """
    config, completion, rows = load_donor(donor_dir, recipient_seed)
    donor_seed, horizon = config['seed'], len(rows)
    if schedule_seed is None:
        schedule_seed = recipient_seed * 1000003 + donor_seed
    _count(schedule_seed, 'schedule seed')
    packets = [_packet(row) for row in rows]
    replay = [dict(packet, step=step) for step, packet in enumerate(packets)]
    active = [packet for packet in packets if packet['action'] in ('CATCHUP', 'CORRECT') or packet['requested_head_steps']]
    passive = iter(packet for packet in packets if packet['action'] not in ('CATCHUP', 'CORRECT') and not packet['requested_head_steps'])
    # Midpoints in equal-width bins give distinct positions for K <= horizon.
    fixed_positions = [(2 * index + 1) * horizon // (2 * len(active)) for index in range(len(active))] if active else []
    by_position = dict(zip(fixed_positions, active))
    fixed = [dict(by_position[step] if step in by_position else next(passive), step=step) for step in range(horizon)]
    shuffled = list(packets)
    random.Random(schedule_seed).shuffle(shuffled)
    randomized = [dict(packet, step=step) for step, packet in enumerate(shuffled)]
    schedules = dict(replay=replay, fixed=fixed, random=randomized)
    expected = _signature(replay)
    if not all(_signature(schedule) == expected for schedule in schedules.values()):
        raise AssertionError('schedule construction changed the joint action multiset')
    curriculum = []
    if include_curriculum:
        curriculum = _jsonl(Path(donor_dir) / 'curriculum_events.jsonl')
        previous = -1
        for event in curriculum:
            _no_target_contact(event)
            step = _count(event.get('step'), 'curriculum step')
            if not previous < step < horizon:
                raise ValueError('curriculum steps must be unique, ordered and inside donor horizon')
            previous = step
    metadata = dict(schema=SCHEMA, status='FROZEN_SCHEDULES_NOT_EXECUTED', donor_seed=donor_seed,
                    recipient_seed=recipient_seed, schedule_seed=schedule_seed, source_only=True,
                    target_evaluated=False, horizon=horizon, epochs=config['epochs'],
                    steps_per_epoch=horizon // config['epochs'], game_split_seed=config.get('game_split_seed'),
                    donor_directory=str(Path(donor_dir).resolve()), donor_completion=completion,
                    donor_schedule_context={name: config.get(name) for name in
                                            ('game_solver', 'game_curriculum', 'game_fixed_head_steps',
                                             'game_response_tracking', 'game_control', 'game_synthetic')},
                    donor_actual_costs=_costs(donor_dir, rows), matched_actions=_summary(replay),
                    schedule_summaries={name: _summary(schedule) for name, schedule in schedules.items()},
                    matching_scope='preregistered action counts and joint requested/donor-committed head-step multiset only',
                    recipient_realized_costs='UNKNOWN until executed; rejections, actual head/field work, audit/HVP cost and wall time must be reported separately',
                    uses_recipient_measurements=False,
                    runtime=dict(game_control='replay', game_solver='simultaneous', game_fixed_head_steps=0,
                                 game_curriculum='fixed', game_response_tracking=False,
                                 required_preflight=['recipient seed differs from donor and equals metadata recipient_seed',
                                                     'same source split, epochs and steps-per-epoch horizon',
                                                     'no dynamic correction/catchup/controller overrides',
                                                     'verify action multiset; keep actual rejection/cost ledger']),
                    curriculum=dict(included=bool(include_curriculum), event_count=len(curriculum),
                                    applies_to='replay_only',
                                    execution='NOT_CONSUMED_BY_EXISTING_RUNTIME; optional archived donor event interface, no automatic curriculum change'))
    return schedules, metadata, curriculum


def write_schedules(donor_dir, output_dir, recipient_seed, *, schedule_seed=None, include_curriculum=False):
    schedules, metadata, curriculum = build_schedules(donor_dir, recipient_seed,
        schedule_seed=schedule_seed, include_curriculum=include_curriculum)
    output = Path(output_dir)
    filenames = [name + '.jsonl' for name in schedules] + ['metadata.json']
    if include_curriculum:
        filenames.append('curriculum_replay.jsonl')
    if any((output / filename).exists() for filename in filenames):
        raise FileExistsError('schedule artifacts already exist; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, schedule in schedules.items():
        with (output / (name + '.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for row in schedule:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    if include_curriculum:
        with (output / 'curriculum_replay.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for row in curriculum:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    metadata['schedule_files'] = {name: str((output / (name + '.jsonl')).resolve()) for name in schedules}
    with (output / 'metadata.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    # Independent persisted readback validates precisely the artifacts consumed
    # by runtime, rather than treating a successful write as sufficient.
    saved = [_jsonl(output / (name + '.jsonl')) for name in schedules]
    if not all(_signature(rows) == _signature(saved[0]) for rows in saved):
        raise RuntimeError('persisted schedule count mismatch')
    if _json(output / 'metadata.json')['recipient_seed'] != recipient_seed:
        raise RuntimeError('persisted recipient seed mismatch')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--recipient-seed', required=True, type=int)
    parser.add_argument('--schedule-seed', type=int)
    parser.add_argument('--include-curriculum', action='store_true')
    args = parser.parse_args(argv)
    metadata = write_schedules(args.donor_dir, args.output_dir, args.recipient_seed,
                              schedule_seed=args.schedule_seed, include_curriculum=args.include_curriculum)
    print(json.dumps(dict(status=metadata['status'], horizon=metadata['horizon'],
                         recipient_seed=metadata['recipient_seed'], metadata=str((Path(args.output_dir) / 'metadata.json').resolve()))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
