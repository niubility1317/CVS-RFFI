"""Register fixed interaction evidence before any new support diagnostic."""
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260929-phase2-d92-branch-interaction-support-m4-r01'
RELEASE = 'd92_branch_interaction_support_20260929_r01'
SPEC = 'configs/d92_branch_interaction_support_20260929.json'


def documents():
    old = json.loads((ROOT/'configs/d92_branch_support_probe_20260929.json').read_text(encoding='utf-8'))
    spec = copy.deepcopy(old)
    root = str(Path(old['execution']['remote_run_root']).parent).replace('\\', '/')+'/'+RUN
    release = str(Path(old['code']['cwd']).parent).replace('\\', '/')+'/'+RELEASE
    algorithm = json.loads((ROOT/'configs/d92_branch_interaction_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    spec.update(run_id=RUN, group_id='d92-branch-interaction-support-development',
        display_name='BranchRidge分支交互三臂support诊断', description='固定背景与辅助分支乘积核，对比线性基线和1.5倍能量对照；只复用support缓存，不访问query。',
        authorization='用户明确要求在BranchRidge基础上继续优化；固定Phase1、源数据禁用、合法support研发及透明重复基准的既有授权保持。',
        tags=['d92','branch-interaction','support-only','cached','no-query-access'],
        parent_run_ids=[old['run_id']], status='PLANNED',
        notes=['Single interaction mechanism, three fixed arms, lambda=1; no hyperparameter search.',
            'Existing support-only raw features reused; no fitted state inherited and no new checkpoint forward.',
            'Query results from earlier development exist but are not inputs to this study. K1 has no physical OOF evidence.'])
    spec['code'].update(cwd=release, commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    spec['data']['representation']='Existing immutable support-only z_id/fft/t_emb/f_emb/pa_local; same single received observation'
    spec['permissions'].update(regime='current_row_support_only_interaction_diagnostic')
    spec['checkpoint'].update(runtime_checkpoint_reload=False, current_auxiliary_training='Three fixed per-fold kernel ridge heads from current support only',
        feature_cache_producer_run_id=old['run_id'], feature_cache_producer_role='support-only raw frozen features; no learned state')
    spec['execution'].update(remote_run_root=root, remote_log_root=root, local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        gpu_policy='CPU only; four lanes and two BLAS threads per lane; no checkpoint reload or GPU process',
        export_device=None, export_batch_size=None,
        launch_command=f'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_branch_interaction_probe.py --spec {SPEC}')
    spec['expected_artifacts']=['startup.json','state.json','complete.json','each row/probe.log',
        'each row/probe/startup.json','each row/probe/fit_trace.jsonl','each row/probe/compact.jsonl','each row/probe/compact.csv','each row/probe/probe_complete.json']
    spec['metrics_plan'].update(metric_names=['support_OOF_old_accuracy','support_OOF_new_accuracy','support_OOF_H','kernel_energy','loss','stationarity','resource_cost'],
        dimensions=['model_seed','cohort','receiver','scenario','K','new_count','support_seed','arm'],
        interpretation='Full paired support-only diagnostic. Candidate proceeds only if interaction improves mean H and new accuracy at K5/10/20 versus both controls, with old accuracy decline at most 1pp in each K. K1 numeric pass is not performance evidence. No query evaluation or parameter revision in this study.')
    spec['probe'].update(reuse_support_cache=True, algorithm_config=release+'/configs/d92_branch_interaction_frozen_20260929.json',
        expected_factorizations=32400, baseline='linear', energy_control='energy_control', candidate='interaction')
    outputs = {}
    for key, co in spec['probe']['cohorts'].items():
        name = f'configs/d92_branch_interaction_support_{key}_20260929.json'
        co['evaluation_config'] = release+'/'+name
        outputs[name] = dict(algorithm=algorithm, matrix=co['matrix'])
    for row, original in zip(spec['rows'], old['rows']):
        out = root+'/'+row['row_id']
        row.update(method='D92-BranchInteraction fixed three-arm support probe', purpose='Test bilinear information against linear and energy controls',
            gpu='CPU only', config_ref=SPEC, resolved_config_ref=out+'/probe/startup.json', output_root=out, log_path=out+'/probe.log',
            support_features=original['output_root']+'/feature_cache',
            budget_ref='K1 numeric-only; K5/10/20 three physical folds times three fixed arms; no full-support deployment fit',
            command=f'{spec["code"]["environment"]} -u {release}/tools/run_d92_branch_interaction_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD',
            expected_artifacts=['probe/startup.json','probe/fit_trace.jsonl','probe/compact.jsonl','probe/compact.csv','probe/probe_complete.json'])
    outputs[SPEC] = spec
    return outputs


if __name__ == '__main__':
    outputs = documents()
    if any((ROOT/name).exists() for name in outputs):
        raise FileExistsError('Preparation already exists; reconcile instead of overwrite')
    for name, value in outputs.items():
        with (ROOT/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', files=list(outputs))))
