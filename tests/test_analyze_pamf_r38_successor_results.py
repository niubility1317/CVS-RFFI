import runpy
from pathlib import Path
import pytest

from experiments.pamf_r38_successor import design as d
from tools.analyze_pamf_r38_successor_results import completed_metrics, METHOD

summary = runpy.run_path(str(Path(__file__).with_name('test_analyze_pamf_r17_task_results.py')))['summary']


def fixture():
    rows = {}
    for arm, errors in zip(d.ARMS, (1780, 1781, 2000, 100, 300, 1780)):
        task = d.TASK[arm]
        records = []
        for epoch in range(1, 11):
            terms = dict(iq_nmse=.1/epoch, teacher_stage_nmse=.01/epoch)
            if task:
                terms.update(task_KL=.2/epoch, task_feature_distance=.05/epoch)
            records.append(dict(epoch=epoch, updates=247*epoch, L_exposures=6300*epoch,
                U_exposures=56700*epoch, teacher_IQ_examples=63000*epoch, source_IQ_examples=63000*epoch,
                F_inputs=126000*epoch*int(task), seconds=100.*epoch, clip_fraction=.8,
                learning_rate_last=d.learning_rate(247*epoch-1), losses=dict(loss=sum(terms.values()), **terms)))
        trained = dict(status='SOURCE_TRAINED', arm=arm, epochs=10, updates=2470,
            L_exposures=63000, U_exposures=567000, source_IQ_examples=630000,
            teacher_IQ_examples=630000, student_context_examples=630000, parameters=d.PARAMETERS[arm],
            trainable_parameters=d.PARAMETERS[arm], context_parameters=d.CONTEXT_PARAMETERS[arm], context_trainable=True,
            context_updates=2470, coarse_updates=2470, action_updates=2470, prior_updates=2470,
            cumulative_coarse_epochs=50, cumulative_action_epochs=50, cumulative_prior_epochs=50,
            cumulative_component_epoch_sum=150, cumulative_serial_training_passes=90,
            initialization_IQ_examples=0, TX_labels_read=0, V_reads=0, target_reads=0,
            training_F_access=task, F_weights_read=int(task), F_outputs_read=1260000*int(task),
            teacher_parameter_updates=0, F_parameter_updates=0, head_fit_updates=0, paired_clean_optimized=True)
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
    return dict(run_id=d.RUN, method_commit=METHOD, failure=None, actual_live_owned_PIDs=[],
        complete=dict(status='ANALYZED', target_rows=list(d.ARMS)), rows=rows,
        baselines={name: dict(source=summary(), target=summary('target', errors)) for name, errors in (('R17', 200), ('R38', 100))})


def test_both_references_use_exact_counts_and_keep_training_permissions_separate():
    result = completed_metrics(fixture())
    assert result['accepted_rows'] == ['keep_iq', 'keep_task', 'three_quarter_task', 'half_task']
    assert result['accepted_IQ_rows'] == ['keep_iq']
    row = result['rows']['three_quarter_iq']
    assert row['acceptance']['R17']['accepted'] and not row['acceptance']['R38']['accepted']
    assert row['acceptance']['R38']['reference'] == 'user_R38_794529'
    assert row['training']['full_log_records'] == 10
    assert not result['goal_complete']
    assert result['rows']['keep_iq']['performance_optimization']['decreased_views'] == list(d.VIEWS)
    assert result['rows']['keep_iq']['accepted_both']


@pytest.mark.parametrize('bad', ['missingrow', 'missingview', 'targettruth', 'Ffree', 'epochs', 'loss', 'profile', 'component'])
def test_incomplete_or_misrepresented_results_are_not_reported_complete(bad):
    value = fixture()
    row = value['rows']['half_iq']
    if bad == 'missingrow':
        value['rows'].pop('keep_task')
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
        row['training_complete']['coarse_updates'] = 0
    with pytest.raises(ValueError):
        completed_metrics(value)


def test_width_comparison_uses_matching_objective_and_budget():
    result = completed_metrics(fixture())
    assert result['rows']['half_iq']['matched_same_budget_control']['arm'] == 'keep_iq'
    assert result['rows']['half_task']['matched_same_budget_control']['arm'] == 'keep_task'
    assert result['rows']['keep_task']['matched_same_budget_control']['target']['accuracy_nonregression']


@pytest.mark.parametrize('status', ['QUEUE_VERIFIED', 'RUNNING'])
def test_delivery_renders_all_six_rows_without_remote_or_IQ_access(tmp_path, monkeypatch, status):
    import json
    from tools import complete_pamf_r38_successor as delivery
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
    artifacts = dict(models={arm:dict(components=dict(coarse=1, action=2, prior=3),
        conv_linear_MACs_per_IQ=100, bytes=100) for arm in d.ARMS})
    monkeypatch.setattr(delivery, 'WORKSPACE', workspace)
    delivery.report(local, output, completed_metrics(fixture()), artifacts)
    text = (folder/'report.md').read_text(encoding='utf-8')
    assert '状态：ANALYZED。' in text and all(arm in text for arm in d.ARMS)
    assert len(list((folder/'row_metrics.csv').open(encoding='utf-8'))) == 1+6*2*7*14
    saved = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    assert not saved['goal_complete'] and len(saved['rows']) == 6
