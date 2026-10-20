"""Recompute the fixed R38 successor results from completed scorer aggregates."""
import math
import numpy as np

from experiments.pamf_r38_dense_steps import design as d
from tools.analyze_pamf_r17_task_results import summary_metrics, compare
from experiments.pamf_r38_dense_steps.acceptance import target_acceptance

METHOD = '6e4abad968f0f9610bbcec3f4114da259edaf8e1'


def training_metrics(records, complete, arm):
    task = d.TASK[arm]
    names = {'loss', 'iq_nmse', 'teacher_stage_nmse'}
    if task:
        names.update(('task_KL', 'task_feature_distance'))
    if len(records) != d.EPOCHS:
        raise ValueError('All ten training epochs required')
    last = 0.
    for epoch, row in enumerate(records, 1):
        expected = dict(epoch=epoch, updates=985*epoch, L_exposures=6300*epoch,
                        U_exposures=56700*epoch, teacher_IQ_examples=63000*epoch,
                        source_IQ_examples=63000*epoch, F_inputs=126000*epoch*int(task))
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('Actual source/F exposure differs from fixed budget')
        losses = row['losses']
        if set(losses) != names or any(not math.isfinite(x) or x < 0 for x in losses.values()):
            raise ValueError('Finite registered losses required')
        if not math.isclose(losses['loss'], losses['teacher_stage_nmse'],
                            rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError('Actual objective differs from registered objective')
        if not math.isfinite(row['seconds']) or row['seconds'] <= last or not 0 <= row['clip_fraction'] <= 1:
            raise ValueError('Finite monotone training progress required')
        if not math.isclose(row['learning_rate_last'], d.learning_rate(985*epoch-1,arm), rel_tol=1e-9):
            raise ValueError('Actual learning-rate schedule differs')
        last = row['seconds']
    expected = dict(status='SOURCE_TRAINED', arm=arm, epochs=10, updates=9850,
                    L_exposures=63000, U_exposures=567000, source_IQ_examples=630000,
                    teacher_IQ_examples=630000, student_context_examples=0, shared_context_examples=630000,
                    parameters=d.PARAMETERS[arm], trainable_parameters=d.PRIOR_PARAMETERS[arm],
                    context_parameters=d.CONTEXT_PARAMETERS[arm], context_trainable=False,
                    context_updates=0, coarse_updates=0, action_updates=0, prior_updates=9850,
                    cumulative_coarse_epochs=40, cumulative_action_epochs=40, cumulative_prior_epochs=50,
                    cumulative_component_epoch_sum=130, cumulative_serial_training_passes=90,
                    initialization_IQ_examples=0, TX_labels_read=0, V_reads=0, target_reads=0,
                    training_F_access=task, F_weights_read=int(task), F_outputs_read=1260000*int(task),
                    teacher_parameter_updates=0, F_parameter_updates=0, head_fit_updates=0,
                    paired_clean_optimized=False)
    if any(complete.get(k) != v for k, v in expected.items()):
        raise ValueError('Full component budget or training permissions differ')
    losses = {}
    for name in sorted(names):
        values = np.asarray([r['losses'][name] for r in records], np.float64)
        losses[name] = dict(first=float(values[0]), last=float(values[-1]),
                            minimum=float(values.min()), maximum=float(values.max()),
                            slope_per_epoch=float(np.polyfit(np.arange(1, 11), values, 1)[0]))
    intervals = np.diff([0., *[r['seconds'] for r in records]])
    return dict(full_log_records=10, losses=losses, records=records,
                epoch_seconds_median=float(np.median(intervals)), training_complete=complete,
                checkpoint_selection='fixed final E10/2470; no metric selection')


def profile_metrics(profile, arm):
    if (profile.get('status') != 'PROFILE_COMPLETE' or not profile.get('includes_full_context_and_CG')
            or profile.get('checkpoint_payload_on_GPU') is not False
            or profile.get('optimizer_on_GPU') is not False or profile.get('source_V_IQ') != 16
            or profile.get('TX_labels_read') != 0 or profile.get('state_updates') != 0):
        raise ValueError('Matched full-chain source-only profile required')
    expected = dict(original_R17=4665073, parent_R38_794529=794529, compact=d.PARAMETERS[arm])
    if set(profile['results']) != set(expected):
        raise ValueError('Both reference profiles and successor profile required')
    for name, parameters in expected.items():
        item = profile['results'][name]
        if (item['parameters'], item['samples'], item['warmup']) != (parameters, 16, 2):
            raise ValueError('Same profile scope required')
        if any(not math.isfinite(item[k]) or item[k] <= 0 for k in
               ('median_ms', 'p90_ms', 'resident_before_forward_bytes', 'peak_allocated_bytes')):
            raise ValueError('Finite latency and memory required')
        if item['incremental_peak_bytes'] != item['peak_allocated_bytes']-item['resident_before_forward_bytes']:
            raise ValueError('Incorrect incremental peak memory')
    parent = profile['results']['parent_R38_794529']
    student = profile['results']['compact']
    return dict(**profile, median_latency_ratio_to_R38=student['median_ms']/parent['median_ms'],
                limitation='16 source V samples, batch 1; sequential order and concurrent load affect timing')


def completed_metrics(value):
    if (value.get('run_id') != d.RUN or value.get('method_commit') != METHOD
            or value.get('failure') is not None or value.get('actual_live_owned_PIDs')
            or value.get('complete', {}).get('target_rows') != list(d.ARMS)
            or value.get('complete', {}).get('status') != 'ANALYZED'
            or set(value.get('rows', {})) != set(d.ARMS)):
        raise ValueError('All three original rows must complete before final analysis')
    baselines = {name: {role: summary_metrics(raw[role], role) for role in ('source', 'target')}
                 for name, raw in value['baselines'].items()}
    if set(baselines) != {'R17', 'R38'}:
        raise ValueError('Original R17 and designated R38 references required')
    result = dict(run_id=d.RUN, method_commit=METHOD, status='ANALYZED',
                  baselines=baselines, rows={}, accepted_rows=[], accepted_IQ_rows=[],
                  target_feedback=False, long_term_goal_complete=False,
                  limitation='Single seed and exposed benchmark; all rows F-free with frozen context')
    for arm in d.ARMS:
        row = value['rows'][arm]
        if row.get('complete', {}).get('status') != 'ANALYZED' or row.get('failure') is not None:
            raise ValueError('Complete independent row scoring required')
        metrics = {}
        for role in ('source', 'target'):
            done = row[role+'_prediction_complete']
            expected = dict(status='PREDICTIONS_COMPLETE', role=role, count=1800 if role == 'source' else 168000,
                            views=list(d.VIEWS), readouts=['restored'], TX_truth_reads=0,
                            state_updates=0, head_fit_updates=0, F_run=d.F_RUN, F_seed=d.F_SEED)
            if any(done.get(k) != v for k, v in expected.items()):
                raise ValueError('Complete same-head seven-view predictions required')
            metrics[role] = summary_metrics(row[role+'_summary'], role, training_F_access=d.TASK[arm])
        acceptance = {name: target_acceptance(metrics['target'], base['target'])
                      for name, base in baselines.items()}
        # The shared helper has a historical R17 label; name each actual reference explicitly.
        acceptance['R38']['reference'] = 'user_R38_794529'
        acceptance['R17']['reference'] = 'original_R17'
        deltas = {name: compare(metrics['target'], base['target']) for name, base in baselines.items()}
        accepted = all(item['accepted'] for item in acceptance.values())
        gains = {v: deltas['R38']['views'][v]['accuracy_delta_pp'] for v in d.VIEWS}
        low = metrics['target']['views']['practical_low_urban']['accuracy']
        result['rows'][arm] = dict(parameters=d.PARAMETERS[arm], context_parameters=d.CONTEXT_PARAMETERS[arm],
            reduction_from_R38_percent=100*(1-d.PARAMETERS[arm]/794529),
            reduction_from_R17_percent=100*(1-d.PARAMETERS[arm]/4665073),
            training_F_access=d.TASK[arm], training=training_metrics(row['training_records'], row['training_complete'], arm),
            source=metrics['source'], target=metrics['target'], target_vs_R38=deltas['R38'], target_vs_R17=deltas['R17'],
            acceptance=acceptance, accepted_both=accepted,
            performance_optimization=dict(improved_views=[v for v in d.VIEWS if gains[v] > 0],
                decreased_views=[v for v in d.VIEWS if gains[v] < 0], all_seven_non_decreasing=all(x >= 0 for x in gains.values()),
                low_urban_accuracy=low, low_urban_at_least_70=low >= 70,
                practical_gap_to_raw_clean_pp={v:84.67619047619047-metrics['target']['views'][v]['accuracy'] for v in d.VIEWS if v != 'clean'},
                other_five_within_5pp_of_raw_clean=all(84.67619047619047-metrics['target']['views'][v]['accuracy'] <= 5
                    for v in d.VIEWS if v not in ('clean', 'practical_low_urban')),
                acceptance_is_not_optimization_success=True),
            profile=profile_metrics(row['matched_profile'], arm), checkpoint_bytes=row['checkpoint_bytes'],
            stdout_findings=row.get('stdout_findings', []), target_prediction_seconds=row['target_prediction_complete']['seconds'])
        if acceptance['R38']['accepted']:
            result['accepted_rows'].append(arm)
            if not d.TASK[arm]:
                result['accepted_IQ_rows'].append(arm)
    controls={arm:summary_metrics(raw,'target',training_F_access=False) for arm,raw in value['matched_controls'].items()}
    result['matched_controls']=controls
    names={'factor625_b64_lr1':'R16/factor625_fidelity','factor625_b64_lr3':'R16/factor625_fidelity','dense750_b64_lr3':'R16/dense750_fidelity'}
    for arm,row in result['rows'].items():
        row['matched_same_budget_control']=dict(arm=names[arm],target=compare(row['target'],controls[arm]),
            scope='Same structure, teacher objective and source exposure;B64/9850 vs B256/2470 updates;peak LR also changes for lr3 rows')
    result['rows']['factor625_b64_lr3']['matched_LR_control']=dict(arm='factor625_b64_lr1',target=compare(result['rows']['factor625_b64_lr3']['target'],result['rows']['factor625_b64_lr1']['target']),scope='Same B64/9850/source exposure/structure/initialization;only peak LR differs')
    result['compression_accuracy_condition_satisfied']=bool(result['accepted_rows'])
    return result
