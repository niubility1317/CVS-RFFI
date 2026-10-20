import runpy
from pathlib import Path
import pytest

from experiments.pamf_r38_dense_steps import design as d
from tools.analyze_pamf_r38_dense_steps_results import completed_metrics, METHOD

summary = runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))['summary']


def fixture():
    rows = {}
    for arm, errors in zip(d.ARMS, (1444, 2000, 1445)):
        task = d.TASK[arm]
        records = []
        for epoch in range(1, 11):
            terms = dict(iq_nmse=.1/epoch, teacher_stage_nmse=.01/epoch)
            if task:
                terms.update(task_KL=.2/epoch, task_feature_distance=.05/epoch)
            records.append(dict(epoch=epoch, updates=985*epoch, L_exposures=6300*epoch,
                U_exposures=56700*epoch, teacher_IQ_examples=63000*epoch, source_IQ_examples=63000*epoch,
                F_inputs=126000*epoch*int(task), seconds=100.*epoch, clip_fraction=.8,
                learning_rate_last=d.learning_rate(985*epoch-1,arm), losses=dict(loss=terms['teacher_stage_nmse'], **terms)))
        trained = dict(status='SOURCE_TRAINED', arm=arm, epochs=10, updates=9850,
            L_exposures=63000, U_exposures=567000, source_IQ_examples=630000,
            teacher_IQ_examples=630000, student_context_examples=0, shared_context_examples=630000, parameters=d.PARAMETERS[arm],
            trainable_parameters=d.PRIOR_PARAMETERS[arm], context_parameters=d.CONTEXT_PARAMETERS[arm], context_trainable=False,
            context_updates=0, coarse_updates=0, action_updates=0, prior_updates=9850,
            cumulative_coarse_epochs=40, cumulative_action_epochs=40, cumulative_prior_epochs=50,
            cumulative_component_epoch_sum=130, cumulative_serial_training_passes=90,
            initialization_IQ_examples=0, TX_labels_read=0, V_reads=0, target_reads=0,
            training_F_access=task, F_weights_read=int(task), F_outputs_read=1260000*int(task),
            teacher_parameter_updates=0, F_parameter_updates=0, head_fit_updates=0, paired_clean_optimized=False)
        profile = dict(status='PROFILE_COMPLETE', includes_full_context_and_CG=True, checkpoint_payload_on_GPU=False,
            optimizer_on_GPU=False, source_V_IQ=16, TX_labels_read=0, state_updates=0,
            results={name: dict(parameters=p, samples=16, warmup=2, median_ms=10., p90_ms=12.,
                resident_before_forward_bytes=500, peak_allocated_bytes=1000, incremental_peak_bytes=500)
                for name, p in dict(original_R17=4665073, parent_R38_794529=794529, compact=d.PARAMETERS[arm]).items()})
        row = dict(training_records=records, training_complete=trained, matched_profile=profile, checkpoint_bytes=100,
                   complete=dict(status='ANALYZED'), failure=None)
        for role in ('source', 'target'):
            row[role+'_summary'] = dict(summary(role, errors if role == 'target' else 180), training_F_access=task)
            row[role+'_prediction_complete'] = dict(status='PREDICTIONS_COMPLETE', role=role,
                count=1800 if role == 'source' else 168000, views=list(d.VIEWS), readouts=['restored'],
                TX_truth_reads=0, state_updates=0, head_fit_updates=0, F_run=d.F_RUN, F_seed=d.F_SEED, seconds=100.)
        rows[arm] = row
    return dict(matched_controls={a:summary('target',1780) for a in d.ARMS},run_id=d.RUN, method_commit=METHOD, failure=None, actual_live_owned_PIDs=[],
        complete=dict(status='ANALYZED', target_rows=list(d.ARMS)), rows=rows,
        baselines={name: dict(source=summary(), target=summary('target', errors)) for name, errors in (('R17', 200), ('R38', 100))})


def test_both_references_use_exact_counts_and_keep_training_permissions_separate():
    result = completed_metrics(fixture())
    assert result['accepted_rows'] == ['factor625_b64_lr1']
    assert result['accepted_IQ_rows'] == ['factor625_b64_lr1']
    row = result['rows']['dense750_b64_lr3']
    assert row['acceptance']['R17']['accepted'] and not row['acceptance']['R38']['accepted']
    assert row['acceptance']['R38']['reference'] == 'user_R38_794529'
    assert row['training']['full_log_records'] == 10
    assert not result['long_term_goal_complete'] and result['compression_accuracy_condition_satisfied']
    assert result['rows']['factor625_b64_lr1']['performance_optimization']['decreased_views'] == list(d.VIEWS)
    assert result['rows']['factor625_b64_lr1']['accepted_both']


@pytest.mark.parametrize('bad', ['missingrow', 'missingview', 'targettruth', 'Ffree', 'epochs', 'loss', 'profile', 'component'])
def test_incomplete_or_misrepresented_results_are_not_reported_complete(bad):
    value = fixture()
    row = value['rows']['dense750_b64_lr3']
    if bad == 'missingrow':
        value['rows'].pop('factor625_b64_lr1')
    elif bad == 'missingview':
        row['target_prediction_complete']['views'].pop()
    elif bad == 'targettruth':
        row['training_complete']['target_reads'] = 1
    elif bad == 'Ffree':
        row['training_records'][-1]['F_inputs'] = 1260000
    elif bad == 'epochs':
        row['training_records'].pop()
    elif bad == 'loss':
        row['training_records'][-1]['losses']['loss'] += 1
    elif bad == 'profile':
        row['matched_profile']['results'].pop('parent_R38_794529')
    else:
        row['training_complete']['coarse_updates'] = 2470
    with pytest.raises(ValueError):
        completed_metrics(value)


def test_width_comparison_uses_matching_objective_and_budget():
    result = completed_metrics(fixture())
    assert result['rows']['dense750_b64_lr3']['matched_same_budget_control']['arm'] == 'R16/dense750_fidelity'
    assert result['rows']['factor625_b64_lr1']['matched_same_budget_control']['arm']=='R16/factor625_fidelity'


def test_current_goal_uses_designated_R38_not_stricter_historical_reference():
    value = fixture()
    value['baselines']['R17']['target'] = summary('target', 0)
    result = completed_metrics(value)
    row = result['rows']['factor625_b64_lr1']
    assert row['acceptance']['R38']['accepted']
    assert not row['acceptance']['R17']['accepted'] and not row['accepted_both']
    assert 'factor625_b64_lr1' in result['accepted_rows']


@pytest.mark.parametrize('status', ['QUEUE_VERIFIED', 'RUNNING'])
def test_delivery_renders_all_three_rows_without_remote_or_IQ_access(tmp_path, monkeypatch, status):
    import json
    from tools import complete_pamf_r38_dense_steps as delivery
    local = tmp_path/'local'
    workspace = tmp_path/'workspace'
    folder = local/'automation_reports/CV-SincNet'/d.RUN
    folder.mkdir(parents=True)
    (workspace/'automation_reports/CV-SincNet'/d.RUN).mkdir(parents=True)
    (folder/'report.md').write_text('状态：'+status+'。\n', encoding='utf-8')
    (folder/'experiment.json').write_text('{}', encoding='utf-8')
    (folder/'events.jsonl').write_text('', encoding='utf-8')
    (local/'tools').mkdir()
    (local/'tools/experiment_registry.py').write_text('def event(*args, **kwargs): pass\n', encoding='utf-8')
    output = local/'artifacts'
    (output/'models').mkdir(parents=True)
    artifacts = dict(status='VERIFIED',models={arm:dict(components=dict(coarse=1, action=2, prior=3),
        conv_linear_MACs_per_IQ=100, bytes=100) for arm in d.ARMS})
    monkeypatch.setattr(delivery, 'WORKSPACE', workspace)
    delivery.report(local, output, completed_metrics(fixture()), artifacts)
    text = (folder/'report.md').read_text(encoding='utf-8')
    assert '状态：ANALYZED。' in text and all(arm in text for arm in d.ARMS)
    comparison = next(line for line in text.splitlines() if line.startswith('|dense750_b64_lr3|clean|')).split('|')
    expected = completed_metrics(fixture())['rows']['dense750_b64_lr3']['matched_same_budget_control']['target']['views']['clean']['accuracy_delta_pp']
    assert float(comparison[6]) == pytest.approx(expected, abs=1e-6)
    assert len(list((folder/'row_metrics.csv').open(encoding='utf-8'))) == 1+3*2*7*14
    saved = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    assert saved['current_goal_complete'] and not saved['long_term_goal_complete'] and len(saved['rows']) == 3
