"""Prepare the single joint pilot on unchanged legal support identities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from run_d92_joint_spectral_probe import ROOT, PROBE_CONFIG, CONFIG_NAMES, EXACT_COUNTS, MAX_COUNTS, read, require, validate_spec
from run_d92_registration_diagnostic import validate_spec as validate_parent

RUN = '20260930-phase2-d92-joint-spectral-support-m2-r01'
RELEASE = 'd92_joint_spectral_support_20260930_r01'
SPEC = 'configs/d92_joint_spectral_support_20260930.json'
PARENT = 'configs/d92_registration_diagnostic_20260930.json'


def documents(parent, *, commit):
    validate_parent(parent)
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
        'Explicit preparation commit required')
    s = deepcopy(parent)
    remote = str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release = str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    s.update(run_id=RUN, group_id='d92-joint-spectral-support',
        display_name='LocalRidge与监督谱adapter联合适应及新类注册',
        description='LocalRidge为最终分类器；支持集内折监督训练两个有界谱参数，B后C顺序继承，与原方法、固定度量和C重置路径完整配对。',
        parent_run_ids=[parent['run_id']], status='PLANNED',
        tags=['d92', 'support-only', 'joint-spectral', 'supervised-adapter', 'practical-residual'],
        authorization='Existing user joint LocalRidge/target-support SFT authorization; root sole launch owner.',
        notes=[
            'R_joint is the declared sequential mainline; controls R0/R_fixed/R_reset are reported without choosing the best path.',
            'Same 160 support parents; six old classes; new0/2/5/10/20; K1/5/10/20; practical residual/post_sync/noeq/25MHz.',
            'True K1 numerical only; proxy trainK1 exact R0 reuse; no fabricated K1 learning.',
            'Each inner fold estimates geometry/kernel/head only from its inner train; inner-held labels supervise theta, outer-held only scores.',
            'B initializes theta0; C inherits theta and final old geometry, re-estimates inner-fold geometry and refits the all-class final head.',
            'Eight fixed projected-gradient steps, lr0.1; final step only; no early stopping, grid or target-query feedback.',
            'A unavailable in this support pilot; B0 is original support classifier, not ground A. No direct promotion or independent-data claim.',
            'Cost bounds include internal solves; actual two-parameter training does not establish satellite compute savings.'])
    s['code'].update(cwd=release, commit=commit)
    s['execution'].update(remote_run_root=remote, remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_joint_spectral_probe.py --spec {SPEC}')
    s['permissions'].update(regime='legal_target_support_joint_local_ridge_sft',
        adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse',
        truth_use='Legal train support labels supervise adapter and fit fold-local heads; outer-held labels only after frozen scores; no query')
    s['checkpoint'].update(current_auxiliary_training='Fixed compliant source encoder caches; no checkpoint loaded; only two target-support spectral adapter parameters trained')
    s['metrics_plan'].update(metric_names=['A_NA', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_accuracy',
        'C_new_accuracy', 'H', 'B_minus_B0', 'registration_old_loss', 'old_new_absolute_gap',
        'training_data_loss', 'training_prox_loss', 'theta', 'analytic_gradient', 'projected_update',
        'actual_head_and_geometry_counts', 'state_bytes', 'resource_cost'],
        interpretation='Complete paired support-only joint pilot; internal loss is training, outer support holdout is performance; no query claim.')
    s['probe'].update(algorithm=deepcopy(PROBE_CONFIG), candidate='R_joint', controls=['R0', 'R_fixed', 'R_reset'],
        interpretation='support_joint_pilot_no_direct_promotion', exact_counts=deepcopy(EXACT_COUNTS),
        maximum_counts=deepcopy(MAX_COUNTS), expected_head_fits=MAX_COUNTS['head_fit_count'])
    out = {}
    for name, co in s['probe']['cohorts'].items():
        co['evaluation_config'] = release+'/'+CONFIG_NAMES[name]
        out[CONFIG_NAMES[name]] = dict(algorithm=deepcopy(PROBE_CONFIG), producer_matrix=deepcopy(co['matrix']), selection=deepcopy(co['selection']))
    for row in s['rows']:
        target = remote+'/'+row['row_id']
        row.update(method=PROBE_CONFIG['method'], purpose='Joint supervised adapter with LocalRidge final classifier; full paired B/C support pilot',
            output_root=target, log_path=target+'/probe.log', config_ref=SPEC, resolved_config_ref=target+'/probe/startup.json',
            optimizer='Projected gradient, eight updates, lr0.1, two simplex-constrained parameters; no-information stages skip training',
            lr=0.1,
            budget_ref='40 parents;792 baseline heads;max234 trained stages/1872 updates/6318 inner heads/396 extra final heads',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_joint_spectral_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
        row['expected_artifacts'] = list(dict.fromkeys(row['expected_artifacts'] + [
            'probe/training_events.jsonl', 'probe/training_events_compact.jsonl',
            'probe/training_events_compact.csv']))
    validate_spec(s)
    out[SPEC] = s
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent-spec', type=Path, default=ROOT/PARENT)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    out = documents(read(a.parent_spec), commit=a.commit)
    if any((ROOT/name).exists() for name in out):
        raise FileExistsError('Preparation exists; reconcile without overwrite')
    for name, value in out.items():
        with (ROOT/name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', files=list(out))))
