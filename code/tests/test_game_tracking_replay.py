"""Count-matched schedules are other-seed source artifacts, not cost claims."""
from collections import Counter
import json
import random

import pytest

from scripts.build_core90_game_replay import build_schedules, load_donor, write_schedules


def _write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def donor(tmp_path):
    root = tmp_path / 'donor'
    root.mkdir()
    config = dict(seed=392002, epochs=2, game_max_steps_per_epoch=0, from_scratch=True,
                  baseline_ckpt='', game_split_seed=42, game_solver='simultaneous',
                  game_control='both', game_curriculum='fixed', game_fixed_head_steps=0,
                  game_response_tracking=False, game_synthetic=False, best_metric='clean_val_tx')
    _write(root / 'resolved_config.json', config)
    completion = dict(status='SOURCE_ARTIFACTS_COMPLETE', epoch=2, step=12,
                      target_evaluated=False, scientific_verdict='SOURCE_ONLY_NO_PROMOTION')
    _write(root / 'completion.json', completion)
    rows = [dict(step=step, epoch=step // 6 + 1, action='NORMAL', requested_head_steps=0,
                 committed_head_steps=0, accepted=True, field_evaluations=1, forward_calls=2)
            for step in range(12)]
    rows[0].update(action='CATCHUP', requested_head_steps=3, committed_head_steps=2)
    rows[1].update(action='CORRECT', field_evaluations=2)
    rows[4].update(action='CATCHUP', requested_head_steps=1, committed_head_steps=1)
    rows[8].update(action='CORRECT', accepted=False, field_evaluations=2)
    # A compound correction/head action must remain one packet when moved.
    rows[11].update(action='CORRECT', requested_head_steps=2, committed_head_steps=1, field_evaluations=2)
    (root / 'game_actions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    _write(root / 'resource_summary.json', dict(source_only=True, full_budget_completed=True,
           total_seconds=54.2, budget=dict(lifetime=dict(audit_seconds=7.3))))
    return root


def _signature(rows, donor=False):
    key = 'committed_head_steps' if donor else 'donor_committed_head_steps'
    return Counter((row['action'], row['requested_head_steps'], row[key]) for row in rows)


def test_replay_fixed_random_preserve_joint_action_counts(donor):
    _, _, original = load_donor(donor, 392003)
    rng = random.getstate()
    schedules, meta, events = build_schedules(donor, 392003)
    assert random.getstate() == rng
    assert not events and meta['horizon'] == 12 and meta['source_only']
    assert meta['donor_seed'] == 392002 and meta['recipient_seed'] == 392003
    assert not meta['uses_recipient_measurements']
    for schedule in schedules.values():
        assert [row['step'] for row in schedule] == list(range(12))
        assert _signature(schedule) == _signature(original, donor=True)
        assert sum(row['action'] == 'CORRECT' for row in schedule) == 3
    assert [row['donor_step'] for row in schedules['replay']] == list(range(12))
    assert [row['step'] for row in schedules['fixed'] if row['action'] in ('CATCHUP', 'CORRECT')] == [1, 3, 6, 8, 10]
    assert schedules['random'] != schedules['replay']
    assert build_schedules(donor, 392003)[0] == schedules
    assert build_schedules(donor, 392003, schedule_seed=7)[0]['random'] != schedules['random']
    assert meta['matched_actions']['catchup_step_multiset'] == [
        dict(requested=1, donor_committed=1, count=1), dict(requested=3, donor_committed=2, count=1)]
    assert meta['donor_actual_costs']['committed_head_steps'] == 4
    assert meta['donor_actual_costs']['requested_head_steps'] == 6
    assert meta['donor_actual_costs']['field_evaluations'] == 15
    assert meta['donor_actual_costs']['committed_corrections'] == 2
    assert meta['donor_actual_costs']['elapsed_seconds'] == 54.2
    assert meta['donor_actual_costs']['audit_seconds'] == 7.3
    assert 'UNKNOWN' in meta['recipient_realized_costs']


@pytest.mark.parametrize('mutation', ['same_seed', 'target', 'missing_source_evidence', 'partial_epoch',
    'truncated_config', 'missing_step', 'duplicate_step', 'epoch_hole', 'unknown_action',
    'bad_committed', 'target_guard', 'invalid_lineage', 'missing_acceptance'])
def test_invalid_or_incomplete_donor_fails_closed(donor, mutation):
    config, completion, rows = load_donor(donor, 392003)
    recipient = 392003
    if mutation == 'same_seed': recipient = config['seed']
    elif mutation == 'target': completion['target_evaluated'] = True
    elif mutation == 'missing_source_evidence': completion.pop('scientific_verdict')
    elif mutation == 'partial_epoch': completion['epoch'] = 1
    elif mutation == 'truncated_config': config['game_max_steps_per_epoch'] = 6
    elif mutation == 'missing_step': rows.pop()
    elif mutation == 'duplicate_step': rows[4]['step'] = 3
    elif mutation == 'epoch_hole': rows[6]['epoch'] = 1
    elif mutation == 'unknown_action': rows[0]['action'] = 'TARGET_SELECT'
    elif mutation == 'bad_committed': rows[0]['committed_head_steps'] = 4
    elif mutation == 'target_guard': config['enable_joint_safe_guard'] = True
    elif mutation == 'invalid_lineage': config['from_scratch'] = False
    elif mutation == 'missing_acceptance': rows[2].pop('accepted')
    _write(donor / 'resolved_config.json', config)
    _write(donor / 'completion.json', completion)
    (donor / 'game_actions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        build_schedules(donor, recipient)


def test_persisted_runtime_schedule_and_optional_curriculum_interface(donor, tmp_path):
    events = [dict(step=2, level=.1, changed=True), dict(step=9, level=.2, changed=True)]
    (donor / 'curriculum_events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events), encoding='utf-8')
    output = tmp_path / 'schedules'
    meta = write_schedules(donor, output, 392003, include_curriculum=True)
    assert json.loads((output / 'metadata.json').read_text(encoding='utf-8')) == meta
    assert meta['runtime']['game_control'] == 'replay'
    assert meta['curriculum']['applies_to'] == 'replay_only'
    assert 'NOT_CONSUMED' in meta['curriculum']['execution']
    assert [json.loads(line) for line in (output / 'curriculum_replay.jsonl').read_text().splitlines()] == events
    for path in meta['schedule_files'].values():
        rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        assert len(rows) == 12
        assert all('step' in row and 'action' in row and 'requested_head_steps' in row for row in rows)
        assert all('committed_head_steps' not in row for row in rows)  # do not claim future recipient commits
    with pytest.raises(FileExistsError):
        write_schedules(donor, output, 392003)


def test_missing_resource_cost_is_unknown_and_zero_active_schedule_valid(donor):
    (donor / 'resource_summary.json').unlink()
    config, completion, rows = load_donor(donor, 392003)
    for row in rows:
        row.update(action='NORMAL', requested_head_steps=0, committed_head_steps=0)
    (donor / 'game_actions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    schedules, meta, _ = build_schedules(donor, 392003)
    assert meta['donor_actual_costs']['elapsed_seconds'] is None
    assert meta['donor_actual_costs']['audit_seconds'] is None
    assert meta['donor_actual_costs']['committed_correct_actions'] == 0
    assert meta['donor_actual_costs']['committed_corrections'] == 2
    assert all(row['action'] == 'NORMAL' for schedule in schedules.values() for row in schedule)


def test_curriculum_outside_complete_horizon_is_rejected(donor):
    (donor / 'curriculum_events.jsonl').write_text(json.dumps(dict(step=12, level=.3)) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='horizon'):
        build_schedules(donor, 392003, include_curriculum=True)
