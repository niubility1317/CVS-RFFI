"""Prepare the single joint pilot on unchanged legal support identities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from run_d92_fcr8_probe import ROOT, PROBE_CONFIG, CONFIG_NAMES, EXACT_COUNTS, MAX_COUNTS, read, require, validate_spec
from run_d92_registration_diagnostic import validate_spec as validate_parent

RUN = '20260930-phase2-d92-fcr8-support-m2-r01'
RELEASE = 'd92_fcr8_support_20260930_r01'
SPEC = 'configs/d92_fcr8_support_20260930.json'
PARENT = 'configs/d92_registration_diagnostic_20260930.json'


def documents(parent, *, commit):
    validate_parent(parent)
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
        'Explicit preparation commit required')
    s = deepcopy(parent)
    remote = str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release = str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    s.update(run_id=RUN, group_id='d92-fcr8-support',
        display_name='LocalRidge与合法support函数坐标Residual8监督adapter联合适应及新类注册',
        description='LocalRidge为唯一最终分类器；固定DCT/GELU字典，仅学习函数坐标Z并物化原U，最多5888参数；完整123616维interaction保留半份；函数位移近端与旧教师间隔约束，B后C精确继承原U。',
        parent_run_ids=[parent['run_id']], status='PLANNED',
        tags=['d92', 'support-only', 'fcr8', 'supervised-adapter', 'practical-residual'],
        authorization='Existing user joint LocalRidge/target-support SFT authorization; root sole launch owner.',
        notes=[
            'R_FCR8_seq is the declared sequential mainline; controls R0/R_FCR8_reset_init are reported without choosing the best path.',
            'Same 160 support parents; six old classes; new0/2/5/10/20; K1/5/10/20; practical residual/post_sync/noeq/25MHz.',
            'True K1 numerical only; proxy trainK1 exact R0 reuse; no fabricated K1 learning.',
            'Each inner fold estimates bandwidth/kernel/head only from its inner train; inner-held labels supervise theta, outer-held only scores.',
            'B initializes U0/fixed_V_DCT; C inherits actual U_B directly and resets Z in new current-support coordinates, refits all-class final head; frozen B parameters and fold-local old heads teach keep margins, raw old support and physical IDs bind inheritance.',
            'Four bounded normalized function-coordinate iterations with keep-risk halfspace and no parameter product balls; at most three trials; both Armijo and actual fixed physical-slack keep-risk conditions required; no outer/query feedback.',
            'A unavailable in this support pilot; B0 is original support classifier, not ground A. No direct promotion or independent-data claim.',
            'Cost bounds include internal solves; at most 5888 trainable function-coordinate parameters do not establish satellite compute savings; teacher heads and two adjoint channels counted.',
            'H and W use current legal outer-train support only; fixed-U forward never depends on W; folded heads/teachers estimate bandwidth and normalization only from inner train.',
            'prox=0.5||Z||^2 equals mean pre-tangent residual displacement in exact retained coordinates; no Fisher or query bound claim; no parameter grid.',
            'SVD and dictionary physical evaluation counts measured dynamically; compact logs and complete dynamic-rank NPZ states preserved.'])
    s['code'].update(cwd=release, commit=commit)
    s['execution'].update(remote_run_root=remote, remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_fcr8_probe.py --spec {SPEC}')
    s['permissions'].update(regime='legal_target_support_joint_local_ridge_sft',
        adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse',
        truth_use='Legal train support labels supervise adapter and fit fold-local heads; outer-held labels only after frozen scores; no query')
    s['checkpoint'].update(current_auxiliary_training='Fixed compliant source encoder caches; no checkpoint loaded; only at most 5888 target-support FCR8 Z coordinates trained; fixed dictionary; U materialized for deployment')
    s['metrics_plan'].update(metric_names=['A_NA', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_accuracy',
        'C_new_accuracy', 'H', 'B_minus_B0', 'registration_old_loss', 'old_new_absolute_gap',
        'training_margin_loss', 'training_prox_loss', 'U_Z_W_fixed_V0_npz_refs', 'total_and_keep_gradient_npz_refs', 'accepted_or_rejected_trial',
        'actual_head_and_adjoint_solve_counts', 'state_bytes', 'resource_cost'],
        interpretation='Complete paired support-only joint pilot; internal loss is training, outer support holdout is performance; no query claim.')
    s['probe'].update(algorithm=deepcopy(PROBE_CONFIG), candidate='R_FCR8_seq', controls=['R0', 'R_FCR8_reset_init'],
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
            optimizer='Normalized function-coordinate gradient; max4 iterations/max3 Armijo trials, functional proximal and keep-risk margin constraint; one-shot/single-class/rank-zero preserve anchor',
            lr=None, initial_step_size=0.125,
            budget_ref='40 parents;792 baseline heads;max234 trained stages/936 accepted updates/3042 objectives/9126 inner heads/234 extra final heads/432 C teacher heads/11232 adjoint triangular solves',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_fcr8_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
        row['expected_artifacts'] = list(dict.fromkeys(row['expected_artifacts'] + [
            'probe/training_events.jsonl', 'probe/training_events_compact.jsonl',
            'probe/training_events_compact.csv', 'probe/state_manifest.json',
            'probe/state_arrays/*.npz', 'probe/artifact_manifest.json', 'probe/training.log']))
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
