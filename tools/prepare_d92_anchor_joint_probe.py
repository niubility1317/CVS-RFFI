"""Prepare one AJLR structure on the unchanged legal support cache identities."""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from run_d92_anchor_joint_probe import ROOT, PROBE_CONFIG, CONFIG_NAMES, EXACT_COUNTS, MAX_COUNTS, read, require, validate_spec
from run_d92_registration_diagnostic import validate_spec as validate_parent

RUN = '20261001-phase2-d92-anchor-joint-support-m2-r01'
RELEASE = 'd92_anchor_joint_support_20261001_r01'
SPEC = 'configs/d92_anchor_joint_support_20261001.json'
PARENT = 'configs/d92_registration_diagnostic_20260930.json'


def documents(parent, *, commit):
    validate_parent(parent)
    require(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
        'Explicit preparation commit required')
    s = deepcopy(parent)
    remote = str(PurePosixPath(parent['execution']['remote_run_root']).parent/RUN)
    release = str(PurePosixPath(parent['code']['cwd']).parent/RELEASE)
    s.update(run_id=RUN, group_id='d92-anchor-joint-support',
        display_name='以实际B函数为先验的残差LocalRidge与support监督adapter联合适应注册',
        description='固定旧类参考测度与核尺度，B分类函数作为C先验；全类残差闭式LocalRidge经平滑CE联合学习函数坐标adapter；保留完整interaction，不做参数网格。',
        parent_run_ids=[parent['run_id']], status='PLANNED',
        tags=['d92', 'support-only', 'anchor-joint', 'supervised-adapter', 'practical-residual'],
        authorization='Existing user joint LocalRidge/target-support SFT authorization; root sole launch owner.',
        notes=[
            'Only R0 and declared R_AJLR_seq are compared; no reset candidate or parameter search.',
            'Same 160 physical support parents; old6; new0/2/5/10/20; K1/5/10/20; practical residual/post_sync/noeq/25MHz.',
            'True K1 and proxy trainK1 have no adapter supervision but still fit complete B/C closed heads; their unavailable held accuracy remains N/A.',
            'Fold-local old inner-train raw support supplies fixed tau/gamma; no nuisance head or solve. Old reference physical measure is fixed, but reference embeddings follow current U.',
            'B prior is zero; C final prior is the actual B function padded with new zero columns. Each C inner prior head refits only old inner-train at frozen actual U_B, with no nested B training. C-inner holdout supervision is legal training, not independent validation.',
            'C inherits actual U_B exactly and trains Z in new current-support coordinates; m is fixed in the C gradient, including its actual B kernel parameters. new0 exactly reuses B state and scores.',
            'Row/class-centered targets, no intercept or residual re-centering; alpha solves (K+I)alpha=Y-m. Regularization applies to residual g, not the B prior.',
            'Per-class mean CE then class RMS plus 0.5||Z||^2; full centered-kernel adjoint including reference terms; no keep channel, group offset or accuracy selection.',
            'Four normalized function-coordinate iterations, at most twelve bounded halving trials per iteration, initial step0.125; first actual Armijo/nonincrease trial accepted, no outer or query feedback.',
            'Rank-zero or noninformative old kernel retains anchor but still closes the head; no new-support bandwidth fallback, floor or jitter. Nonfinite/unrepresentable arithmetic is technical failure.',
            'A unavailable in this support pilot; B0/R0 cannot substitute ground A. No direct promotion or independent-data claim.',
            'Function anchoring is empirical regularized continuation, not exact sequential Bayes or a guarantee of old accuracy preservation; centered no-intercept targets have an explicit mixed-class old-reference fit limitation.',
            'Source-only encoder cache remains fixed; no encoder/checkpoint loaded, source samples, source per-record features or ground summaries. Additional ground summary transmission0 bytes.',
            'At most736r<=5888 trainable function coordinates; dictionary width8, retained r measured. Actual training/inference times, RSS/state bytes and all head/solve/distance counts reported; satellite/GPU unmeasured=N/A.',
            'Two-path AJLR and three-path FCR differ in work; raw elapsed time alone does not establish an adapter speedup. Detailed text, full numeric NPZ, compact JSONL/CSV retained.'])
    s['code'].update(cwd=release, commit=commit)
    s['execution'].update(remote_run_root=remote, remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_anchor_joint_probe.py --spec {SPEC}')
    s['permissions'].update(regime='legal_target_support_joint_local_ridge_sft',
        adapted_state_reuse='within_path_B_to_C_only; no_cross_parent_fold_or_model_reuse',
        truth_use='Legal inner-held support labels supervise adapter; outer-held labels only score frozen predictions; no query')
    s['checkpoint'].update(current_auxiliary_training='Fixed compliant source-only encoder caches; no checkpoint loaded; only at most736r<=5888 current target-support AJLR Z coordinates trained; fixed V0; U materialized; no historical adapted state')
    s['metrics_plan'].update(metric_names=['A_NA', 'B0_old_accuracy', 'B_old_accuracy', 'C_old_accuracy',
        'C_new_accuracy', 'H', 'B_minus_A_NA', 'B_minus_B0', 'registration_old_loss', 'old_new_absolute_gap',
        'training_class_RMS_CE', 'training_prox_loss', 'actual_B_prior_npz_refs', 'full_centered_kernel_adjoint_npz_refs',
        'accepted_or_rejected_trial', 'C_zero_Z_registration_and_adapter_score_decomposition',
        'actual_head_and_primal_EDF_adjoint_solve_counts', 'state_bytes', 'resource_cost'],
        interpretation='Complete paired support-only joint pilot; inner loss is supervised training, outer support holdout is performance; no query claim.')
    s['probe'].update(algorithm=deepcopy(PROBE_CONFIG), candidate='R_AJLR_seq', controls=['R0'],
        interpretation='support_joint_pilot_no_direct_promotion', exact_counts=deepcopy(EXACT_COUNTS),
        maximum_counts=deepcopy(MAX_COUNTS), expected_head_fits=MAX_COUNTS['head_fit_count'],
        expected_sequence_paths=EXACT_COUNTS['sequence_paths'])
    out = {}
    for name, co in s['probe']['cohorts'].items():
        co['evaluation_config'] = release+'/'+CONFIG_NAMES[name]
        out[CONFIG_NAMES[name]] = dict(algorithm=deepcopy(PROBE_CONFIG), producer_matrix=deepcopy(co['matrix']), selection=deepcopy(co['selection']))
    for row in s['rows']:
        target = remote+'/'+row['row_id']
        row.update(method=PROBE_CONFIG['method'], purpose='AJLR actual B function prior, residual closed head and joint support adapter; paired B/C pilot',
            output_root=target, log_path=target+'/probe.log', config_ref=SPEC, resolved_config_ref=target+'/probe/startup.json',
            optimizer='Normalized function-coordinate gradient; max4 iterations/max12 halving Armijo trials; class-RMS CE plus functional prox; no keep channel; first valid trial; K1/rank0 still close head',
            lr=None, initial_step_size=.125,
            budget_ref='40 parents;810 baseline/preparations/stages;max162 information stages/648 accepted updates/7938 objectives/23814 inner heads/810 final heads/216 prior heads/1944 CE adjoints; total head/factor upper25650; actual branches counted',
            command=f'{s["code"]["environment"]} -u {release}/tools/run_d92_anchor_joint_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
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
