"""Preregister both fixed support diagnostics without modifying their caches."""
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260929-phase2-d92-branch-local-margin-support-m4-r01'
RELEASE = 'd92_branch_local_margin_support_20260929_r01'
SPEC = 'configs/d92_branch_local_margin_support_20260929.json'


def documents():
    parent = json.loads((ROOT/'configs/d92_branch_local_ridge_support_20260929.json').read_text(encoding='utf-8'))
    spec = copy.deepcopy(parent)
    remote = str(Path(parent['execution']['remote_run_root']).parent).replace('\\', '/')+'/'+RUN
    release = str(Path(parent['code']['cwd']).parent).replace('\\', '/')+'/'+RELEASE
    algorithm = json.loads((ROOT/'configs/d92_branch_local_margin_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    spec.update(run_id=RUN, group_id='d92-branch-local-margin-support-development',
        display_name='BranchLocalMargin固定局部核平方间隔与单样本代理support诊断',
        description='保持LocalRidge训练折内核几何，比较最强错误类平方间隔目标；四臂逐折从零拟合，完整物理OOF及全部support内1-shot anchors。',
        tags=['d92', 'branch-local-margin', 'support-only', 'cached', 'no-query-access'],
        parent_run_ids=[parent['run_id']], status='PLANNED',
        notes=['LocalMargin preserves LocalRidge train-only bandwidth and trace matching; margin=1, physical-sum weight=1, RKHS penalty=1.',
            'All four fixed arms freshly refit every exact physical fold and anchor; no control artifact reuse, parameter grid or per-row selection.',
            'True K1 has no independent holdout. Proxy train K1 is reported separately by parent K5/10/20.',
            'Existing support-only raw caches; no checkpoint reload, source access, query access or fitted-state inheritance.'])
    spec['code'].update(cwd=release, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    spec['permissions'].update(regime='current_row_support_only_local_margin_diagnostic',
        truth_use='Only held legal support labels after fixed support predictions; no query scorer')
    spec['checkpoint']['current_auxiliary_training']='Fixed support-only branch / interaction / local ridge / local margin, freshly refit for every fold or anchor'
    spec['execution'].update(remote_run_root=remote, remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_branch_local_margin_probe.py --spec {SPEC}')
    spec['expected_artifacts'] = list(dict.fromkeys(spec['expected_artifacts'] + [
        'each row/probe/fit_stages.jsonl', 'each row/probe/fit_stages.csv',
        'each row/probe/solver_sweeps.jsonl', 'each row/probe/solver_sweeps.csv',
        'each row/probe/solver_sweeps.log']))
    spec['metrics_plan'].update(
        metric_names=['support_OOF_old_accuracy','support_OOF_new_accuracy','support_OOF_H',
            'support_oneshot_proxy_old_accuracy','support_oneshot_proxy_new_accuracy','support_oneshot_proxy_H',
            'uncalibrated_NLL','resource_cost'],
        dimensions=['model_seed','cohort','receiver','scenario','parent_K','train_K','new_count','support_seed','diagnostic','arm'],
        interpretation='Standard OOF and proxy separately: at each parent K5/10/20, candidate H/new must improve versus freshly fitted LocalRidge with old decline <=1pp. BranchRidge and InteractionRidge are descriptive controls only. Report strict old/new/H improvements separately from the guard. All-old and each K/new-count stratum reported without an extra all-cell-positive gate. Aggregate all anchors within parent before task means. True K1 has numerical evidence only; no query claim or parameter revision from this diagnostic.')
    for stale in ('baseline','energy_control','expected_factorizations'):
        spec['probe'].pop(stale, None)
    spec['probe'].update(algorithm_config=release+'/configs/d92_branch_local_margin_frozen_20260929.json',
        max_factorizations=158400, expected_oof_parents=3600, expected_true_k1_parents=1200,
        expected_proxy_anchors=42000, controls=['branch_ridge','interaction_ridge','local_ridge'], candidate='local_margin',
        direct_baseline='local_ridge', descriptive_controls=['branch_ridge','interaction_ridge'],
        expected_candidate_fits=52800, expected_fresh_local_ridge_fits=52800,
        control_reuse=False, maximum_sweeps_per_candidate_fit=1000,
        diagnostics=['physical_oof','support_oneshot_proxy'],
        proxy_policy='All sorted positions 0..parent_K-1 inside each current support set; true K1 excluded')
    outputs = {}
    for key, co in spec['probe']['cohorts'].items():
        name = f'configs/d92_branch_local_margin_support_{key}_20260929.json'
        co['evaluation_config'] = release+'/'+name
        outputs[name] = dict(algorithm=algorithm, matrix=co['matrix'])
    for row in spec['rows']:
        out = remote+'/'+row['row_id']
        row['expected_artifacts'] = list(dict.fromkeys(row['expected_artifacts'] + [
            'probe/fit_stages.jsonl', 'probe/fit_stages.csv', 'probe/solver_sweeps.jsonl',
            'probe/solver_sweeps.csv', 'probe/solver_sweeps.log']))
        row.update(method='D92-BranchLocalMargin fixed four-arm support probe',
            purpose='Compare fixed-kernel margin against fresh same-fold LocalRidge; older controls descriptive; separate one-shot proxy',
            optimizer='Deterministic exact row-block dual minimization, float64, zero initialization, max1000 full sweeps, gap/KKT sqrt(eps64); controls freshly solved ridge; report measured steps',
            config_ref=SPEC, resolved_config_ref=out+'/probe/startup.json', output_root=out, log_path=out+'/probe.log',
            budget_ref='All4800 parents;3600 three-fold OOF;all42000 one-shot anchors;trueK1 numeric-only',
            command=f'{spec["code"]["environment"]} -u {release}/tools/run_d92_branch_local_margin_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
    outputs[SPEC] = spec
    return outputs


if __name__ == '__main__':
    outputs = documents()
    if any((ROOT/name).exists() for name in outputs):
        raise FileExistsError('Preparation already exists; reconcile instead of overwrite')
    for name, value in outputs.items():
        with (ROOT/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', files=list(outputs))))
