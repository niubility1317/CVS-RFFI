import json

import pytest

from scripts.analyze_core90_game import analyze_run, first_capability, main, paired_differences, read_artifact, report


def write(directory, name, value, stream=False):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(r) for r in value) if stream else json.dumps(value))+'\n', encoding='utf-8')


def complete_run(directory, row='B0', loss=1., seconds=12., score=.6):
    write(directory, 'resolved_config.json', {'seed': 17, 'game_row': row})
    write(directory, 'logs.jsonl', [{'epoch': 1, 'steps': 2, 'accepted': 1, 'total_step': 2,
                                   'epoch_seconds': seconds, 'elapsed_seconds': seconds, 'peak_memory_bytes': 1024}], True)
    write(directory, 'game_actions.jsonl', [
        {'step': 0, 'accepted': True, 'algorithm': 'adamw_extragradient', 'field_evaluations': 2,
         'committed_head_steps': 2, 'requested_head_steps': 2, 'forward_calls': 3, 'sample_count': 8,
         'satellite_count': 4, 'loss': loss, 'reason': 'confirmed_catchup'},
        {'step': 1, 'accepted': False, 'algorithm': 'adamw_extragradient', 'field_evaluations': 1,
         'committed_head_steps': 1, 'requested_head_steps': 2, 'forward_calls': 2, 'sample_count': 8,
         'satellite_count': 4, 'loss': loss, 'failure_stage': 'origin:gradient', 'reason': 'confirmed_catchup'}], True)
    write(directory, 'resource_summary.json', {'total_seconds': seconds, 'full_budget_completed': True,
                                              'budget': {'lifetime': {'head_steps': 3}}})
    write(directory, 'completion.json', {'status': 'SOURCE_ARTIFACTS_COMPLETE'})
    write(directory, 'source_final_eval/source_scores.json', {'complete': True, 'source_only': True,
          'resources': {'total_wall_seconds': 2.}, 'scenes': {s: {'accuracy': score, 'worst_rx_accuracy': score-.1,
          'per_rx': {'0': {'accuracy': score}}} for s in ('clean', 'leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')}})


def test_missing_artifacts_are_pending_not_zero(tmp_path):
    run = analyze_run(tmp_path)
    assert run['status'] == 'pending_or_partial'
    assert run['mechanisms']['committed_head_steps'] is None
    assert run['mechanisms']['jacobian']['observations'] is None
    assert run['source_performance'] is None
    assert run['resources']['audit_seconds'] is None
    assert run['scientific_verdict'] == 'SOURCE_ONLY_NO_PROMOTION'


def test_full_stream_reads_late_rows_and_survives_corrupt_tail(tmp_path):
    path = tmp_path/'full.jsonl'
    path.write_text('\n'.join(json.dumps({'step': i}) for i in range(1501))+'\n{broken', encoding='utf-8')
    rows, evidence = read_artifact(path, stream=True)
    assert len(rows) == 1501 and rows[-1]['step'] == 1500
    assert evidence['status'] == 'incomplete' and evidence['errors'][0]['line'] == 1502


def test_actual_activation_failure_and_resource_totals(tmp_path):
    complete_run(tmp_path)
    run = analyze_run(tmp_path)
    assert run['status'] == 'artifacts_complete'
    assert run['mechanisms']['accepted_main_updates'] == 1
    assert run['mechanisms']['committed_head_steps'] == 3
    assert run['mechanisms']['failed_main_updates'] == 1
    assert run['mechanisms']['failure_stages'] == {'origin:gradient': 1}
    assert run['resources']['main_field_evaluations'] == 3
    assert run['resources']['training_plus_source_eval_wall_seconds'] == 14
    assert run['source_performance']['scenes']['clean']['per_rx']['0']['accuracy'] == .6


def test_incomplete_counter_does_not_silently_underreport(tmp_path):
    complete_run(tmp_path)
    path = tmp_path/'game_actions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    del rows[-1]['forward_calls']
    write(tmp_path, path.name, rows, True)
    run = analyze_run(tmp_path)
    assert run['resources']['logged_training_forward_calls'] is None
    assert run['resources']['main_field_evaluations'] == 3


def test_capability_uses_epoch_upper_bound_not_audit_duration():
    calibration = {'curriculum': {'identity_enter': .7, 'margin_enter': .1, 'lag_max': .1, 'consistency_enter': .8}}
    audits = [{'step': 1, 'valid': True, 'identity': .9, 'margin': .2, 'G_lag': .02, 'consistency': .2},
              {'step': 5, 'valid': True, 'identity': .9, 'margin': .2, 'G_lag': .02, 'consistency': .9, 'elapsed_seconds': .5}]
    epochs = [{'total_step': 5, 'elapsed_seconds': 10}, {'total_step': 10, 'elapsed_seconds': 20}]
    result = first_capability(audits, [], epochs, calibration)
    assert result['first_threshold_step'] == 5
    assert result['training_seconds_upper_bound'] == 20
    assert first_capability(audits, [], epochs, None)['status'] == 'pending'


def test_pairing_is_same_seed_not_false_compute_matching(tmp_path):
    complete_run(tmp_path/'a')
    complete_run(tmp_path/'b', row='B5', seconds=24., score=.7)
    pairs = paired_differences([analyze_run(tmp_path/'a'), analyze_run(tmp_path/'b')])
    assert pairs[0]['source_fraction_differences']['clean']['accuracy'] == pytest.approx(.1)
    assert pairs[0]['actual_costs']['training_wall_seconds']['difference'] == 12
    assert pairs[0]['matched_compute_claim'] is False


def test_duplicate_seed_row_rejected_and_duplicate_log_not_complete(tmp_path):
    complete_run(tmp_path)
    run = analyze_run(tmp_path)
    with pytest.raises(ValueError, match='ambiguous'):
        paired_differences([run, run])
    path = tmp_path/'game_actions.jsonl'
    path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
    result = analyze_run(tmp_path)
    assert not result['log_closure']['actions_match_epoch_total']
    assert result['status'] == 'pending_or_partial'


def test_cli_writes_reviewable_json_markdown_without_overwrite(tmp_path):
    complete_run(tmp_path/'run')
    args = ['--run', str(tmp_path/'run'), '--output', str(tmp_path/'analysis')]
    assert main(args) == 0
    result = json.loads((tmp_path/'analysis.json').read_text(encoding='utf-8'))
    assert result['scientific_verdict'] == 'SOURCE_ONLY_NO_PROMOTION'
    assert '四场景' in (tmp_path/'analysis.md').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        main(args)
