"""Preregister both fixed support diagnostics without modifying their caches."""
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260929-phase2-d92-branch-metric-support-m4-r01'
RELEASE = 'd92_branch_metric_support_20260929_r01'
SPEC = 'configs/d92_branch_metric_support_20260929.json'


def documents():
    parent = json.loads((ROOT/'configs/d92_branch_interaction_support_20260929.json').read_text(encoding='utf-8'))
    spec = copy.deepcopy(parent)
    remote = str(Path(parent['execution']['remote_run_root']).parent).replace('\\', '/')+'/'+RUN
    release = str(Path(parent['code']['cwd']).parent).replace('\\', '/')+'/'+RELEASE
    algorithm = json.loads((ROOT/'configs/d92_branch_metric_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    spec.update(run_id=RUN, group_id='d92-branch-metric-support-development',
        display_name='BranchMetric类内度量与单样本代理support诊断',
        description='固定交互表示，对比岭回归、最近类中心与类内度量；完整物理OOF及当前support内所有1-shot anchors。',
        tags=['d92', 'branch-metric', 'support-only', 'cached', 'no-query-access'],
        parent_run_ids=[parent['run_id']], status='PLANNED',
        notes=['Only the support within-class metric and nearest-centre decision change; lambda=1 and scatter is a sum.',
            'All three fixed arms and all anchors are retained; no parameter grid or per-row selection.',
            'True K1 has no independent holdout. Proxy train K1 is reported separately by parent K5/10/20.',
            'Existing support-only raw caches; no checkpoint reload, source access, query access or fitted-state inheritance.'])
    spec['code'].update(cwd=release, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    spec['permissions'].update(regime='current_row_support_only_within_class_metric_diagnostic',
        truth_use='Only held legal support labels after fixed support predictions; no query scorer')
    spec['checkpoint']['current_auxiliary_training']='Fixed support-only ridge / kernel NCM / within-class metric, separately refit for every fold or anchor'
    spec['execution'].update(remote_run_root=remote, remote_log_root=remote,
        local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        launch_command=f'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_branch_metric_probe.py --spec {SPEC}')
    spec['metrics_plan'].update(
        metric_names=['support_OOF_old_accuracy','support_OOF_new_accuracy','support_OOF_H',
            'support_oneshot_proxy_old_accuracy','support_oneshot_proxy_new_accuracy','support_oneshot_proxy_H',
            'uncalibrated_NLL','resource_cost'],
        dimensions=['model_seed','cohort','receiver','scenario','parent_K','train_K','new_count','support_seed','diagnostic','arm'],
        interpretation='OOF: at each parent K5/10/20, candidate joint H/new must improve versus both controls and old decline <=1pp. Proxy: at each parent K5/10/20, candidate H/new must improve versus ridge and old decline <=1pp; candidate=NCM is a correctness identity. All-old separate. Aggregate anchors within parent before task means. No formal K1 or query claim; no parameter revision from this diagnostic.')
    for stale in ('baseline','energy_control','expected_factorizations'):
        spec['probe'].pop(stale, None)
    spec['probe'].update(algorithm_config=release+'/configs/d92_branch_metric_frozen_20260929.json',
        max_factorizations=63600, expected_oof_parents=3600, expected_true_k1_parents=1200,
        expected_proxy_anchors=42000, controls=['interaction_ridge','kernel_ncm'], candidate='within_metric',
        diagnostics=['physical_oof','support_oneshot_proxy'],
        proxy_policy='All sorted positions 0..parent_K-1 inside each current support set; true K1 excluded')
    outputs = {}
    for key, co in spec['probe']['cohorts'].items():
        name = f'configs/d92_branch_metric_support_{key}_20260929.json'
        co['evaluation_config'] = release+'/'+name
        outputs[name] = dict(algorithm=algorithm, matrix=co['matrix'])
    for row in spec['rows']:
        out = remote+'/'+row['row_id']
        row.update(method='D92-BranchMetric fixed three-arm support probe',
            purpose='Distinguish nearest-centre decision from support within-class metric; separate one-shot proxy',
            optimizer='None; exact ridge control, kernel NCM and fixed regularized within-scatter solve; nearest-centre norm intercept; no optimizer steps',
            config_ref=SPEC, resolved_config_ref=out+'/probe/startup.json', output_root=out, log_path=out+'/probe.log',
            budget_ref='All4800 parents;3600 three-fold OOF;all42000 one-shot anchors;trueK1 numeric-only',
            command=f'{spec["code"]["environment"]} -u {release}/tools/run_d92_branch_metric_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD')
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
