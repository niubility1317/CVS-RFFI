"""Prepare fixed local-kernel repeat; read frozen specifications, never scores."""
import copy
import json
from pathlib import Path
import subprocess
from prepare_d92_summary_joint_benchmark import ROOT, write_documents

FROZEN = 'configs/d92_branch_local_margin_frozen_20260929.json'
PRODUCER = 'configs/d92_branch_ridge_frozen_20260929.json'
SUPPORT_RUN = '20260929-phase2-d92-branch-local-margin-support-m4-r01'


def replace_names(value):
    if isinstance(value, dict): return {k: replace_names(v) for k, v in value.items()}
    if isinstance(value, list): return [replace_names(v) for v in value]
    if isinstance(value, str):
        return (value.replace('d92-branch-ridge', 'd92-branch-local-margin')
                .replace('D92-BranchRidge', 'D92-BranchLocalMargin')
                .replace('d92_branch_ridge', 'd92_branch_local_margin')
                .replace('/branch_ridge/', '/branch_local_margin/')
                .replace('branch_ridge.log', 'branch_local_margin.log'))
    return value


def documents(root, commit):
    root = Path(root)
    algorithm = json.loads((root/FROZEN).read_text(encoding='utf-8'))['algorithm']
    if algorithm['method'] != 'D92-BranchLocalMargin-v1': raise ValueError('Frozen method mismatch')
    outputs = {}
    for cohort in ('rx3', 'rx1'):
        old = json.loads((root/f'configs/d92_branch_ridge_repeat_{cohort}_20260929.json').read_text(encoding='utf-8'))
        reference = json.loads((root/f'configs/d92_branch_interaction_repeat_{cohort}_20260929.json').read_text(encoding='utf-8'))
        local_reference = json.loads((root/f'configs/d92_branch_local_ridge_repeat_{cohort}_20260929.json').read_text(encoding='utf-8'))
        if (reference['confirmation']['reuse_validated_capsule_id'] != old['confirmation']['reuse_validated_capsule_id']
                or reference['confirmation']['branch_ridge_reference_run_id'] != old['run_id']
                or len(reference['rows']) != len(old['rows'])):
            raise ValueError('Interaction reference must share exact capsule and BranchRidge parent')
        if (local_reference['confirmation']['reuse_validated_capsule_id'] != old['confirmation']['reuse_validated_capsule_id']
                or local_reference['confirmation']['branch_ridge_reference_run_id'] != old['run_id']
                or local_reference['confirmation']['branch_interaction_reference_run_id'] != reference['run_id']
                or len(local_reference['rows']) != len(old['rows'])):
            raise ValueError('LocalRidge reference must share exact capsule and frozen producer/reference lineage')
        for previous, extra in zip(old['rows'], reference['rows']):
            if any(previous[k] != extra[k] for k in ('row_id', 'seeds', 'source_root', 'expected_checkpoint_sha256', 'reuse_row_root')):
                raise ValueError('Interaction reference model lineage mismatch')
        for previous, extra in zip(old['rows'], local_reference['rows']):
            if any(previous[k] != extra[k] for k in ('row_id', 'seeds', 'source_root', 'expected_checkpoint_sha256', 'reuse_row_root')):
                raise ValueError('LocalRidge reference model lineage mismatch')
        spec = replace_names(copy.deepcopy(old))
        name = f'configs/d92_branch_local_margin_repeat_{cohort}_20260929.json'
        data_name = f'configs/d92_branch_local_margin_repeat_{cohort}_data_20260929.json'
        data = json.loads((root/f'configs/d92_branch_ridge_repeat_{cohort}_data_20260929.json').read_text(encoding='utf-8'))
        if data['shots'] != [1, 5, 10, 20] or data['new_counts'] != [0, 2, 5, 10, 20] or len(old['rows']) != 4:
            raise ValueError('Frozen full-matrix mismatch')
        outputs[data_name] = copy.deepcopy(data)
        release = spec['code']['cwd']
        spec.update(status='PLANNED', group_id='d92-fixed-phase1-branch-local-margin-repeated-benchmark',
            display_name=f'D92-BranchLocalMargin冻结方法{cohort}完整重复基准',
            description='固定Phase1及support诊断前冻结的LocalMargin公式；复用BranchRidge原始received五块特征，保持LocalRidge局部核定义，当前row support从零优化最强错误类平方间隔目标，全部K同式。',
            parent_run_ids=[local_reference['run_id'], reference['run_id'], old['run_id'], SUPPORT_RUN, old['confirmation']['baseline_source_run_id']],
            authorization='冻结入口准备不等于启动授权；完整support-only诊断完成并按预设LocalRidge直接基线规则分析后，由唯一launch owner决定已授权重复基准的启动。公式、全部模型及矩阵保持冻结。',
            tags=['d92','branch-local-margin','source-free','frozen-phase1','cached','cpu-only','repeated-benchmark','truth-last'])
        spec['code'].update(commit=commit, commit_note='Preparation parent; publisher records exact pushed runtime HEAD.')
        spec['data']['representation']='Reused immutable original-received z_id/FFT96/t_emb/f_emb/pa_local; no encoder execution or new view'
        spec['permissions'].update(regime='fixed_phase1_cached_single_view_branch_current_support_local_margin',
            summary_permission='No ground prototype/summary inputs to candidate; no source samples or sample-level source features.')
        spec['checkpoint'].update(runtime_checkpoint_reload=False,
            initialization='Reuse original source-only scratch final200 feature provenance; no checkpoint load.',
            feature_cache_producer_run_id=old['run_id'],
            feature_cache_producer_role='Immutable original received five-block features; no adapted state',
            inheritance_policy='Exact frozen Phase1 feature lineage; no fitted BranchRidge or probe state inherited.',
            current_auxiliary_training='Current-row support only, one fixed local-kernel strongest-wrong-class squared-margin fit from zero beta; deterministic row-block dual solver, <=1000 full sweeps, numerical gap/KKT certificate; no OOF or parameter search.')
        spec['execution']['gpu_policy']='CPU only; four model lanes per cohort, two BLAS threads each; no GPU, checkpoint loading, feature export or baseline re-execution.'
        spec['execution']['launch_command'] = (
            f'F:/App/miniconda3/python.exe -X utf8 tools/publish_d92_confirmation.py --spec {name} '
            f'--release {Path(release).name}')
        conf = spec['confirmation']
        conf.update(candidate=copy.deepcopy(algorithm), candidate_method='D92-BranchLocalMargin-v1',
            candidate_folder='branch_local_margin', candidate_predictor='evaluate_d92_branch_local_margin.py',
            candidate_mode='d92_branch_local_margin_registration',
            candidate_config=release+'/'+FROZEN,
            frozen_branch_feature_source_root=old['execution']['remote_run_root'],
            frozen_branch_feature_producer_config=release+'/'+PRODUCER,
            branch_ridge_reference_run_id=old['run_id'],
            branch_ridge_reference_root=old['execution']['remote_run_root'],
            branch_interaction_reference_run_id=reference['run_id'],
            branch_interaction_reference_root=reference['execution']['remote_run_root'],
            branch_local_ridge_reference_run_id=local_reference['run_id'],
            branch_local_ridge_reference_root=local_reference['execution']['remote_run_root'])
        artifacts=['artifact_reuse.json','frozen_branch_feature_reuse.json','branch_local_margin/startup.json',
            'branch_local_margin/predictions.jsonl','branch_local_margin/predictions_complete.json',
            'branch_local_margin/fit_trace.jsonl','branch_local_margin/compact.jsonl',
            'branch_local_margin/compact.csv','branch_local_margin/fit_stages.jsonl','branch_local_margin/fit_stages.csv',
            'branch_local_margin/solver_sweeps.jsonl','branch_local_margin/solver_sweeps.csv','branch_local_margin/solver_sweeps.log']
        spec['expected_artifacts']=['startup.json','state.json','data_reuse.json','complete.json','scores.json']+['each row/'+p for p in artifacts]
        spec['metrics_plan'].update(
            prediction_ref='Original D92 reuse_row_root and new output_root/branch_local_margin; frozen LocalRidge direct baseline and Interaction reference are separate post-prediction analysis inputs only',
            direct_reference_method='D92-BranchLocalRidge-v1',
            direct_reference_run_id=local_reference['run_id'],
            direct_reference_root=local_reference['execution']['remote_run_root'],
            secondary_reference_method='D92-BranchLocalRidge-v1',
            secondary_reference_run_id=local_reference['run_id'],
            secondary_reference_root=local_reference['execution']['remote_run_root'],
            tertiary_reference_method='D92-BranchInteraction-v1',
            tertiary_reference_run_id=reference['run_id'],
            tertiary_reference_root=reference['execution']['remote_run_root'])
        for row, original in zip(spec['rows'], old['rows']):
            row.update(gpu=None, expected_artifacts=artifacts,
                reuse_branch_features_root=original['output_root']+'/branch_features',
                reuse_branch_ridge_root=original['output_root']+'/branch_ridge',
                optimizer='Deterministic exact row-block dual minimization from zero beta; fixed local kernel, physical-sum margin1 RKHS penalty1, <=1000 sweeps, gap/KKT sqrt(eps64); actual steps and sweep costs recorded',
                budget_ref=f"{conf['splits_per_model']} unchanged splits; one fullsupport fit each; no folds, search, new features or checkpoint/baseline execution")
            row['provenance_refs'].update(
                frozen_branch_features_complete=original['output_root']+'/branch_features/features_complete.json',
                frozen_branch_features_startup=original['output_root']+'/branch_features/startup.json',
                frozen_branch_checkpoint_provenance=original['output_root']+'/branch_features/checkpoint_provenance.json',
                branch_ridge_predictions_complete=original['output_root']+'/branch_ridge/predictions_complete.json')
        spec['notes']=[
            'Transparent repeat on previously scored target data; not a new independent confirmation.',
            'Frozen LocalMargin formula; launch awaits complete support-only screen analysis against LocalRidge. True K1 is fitted here but has no support holdout performance evidence.',
            'Reuse existing BranchRidge received feature cache, exact original Phase1 lineage and baseline D92 predictions; no checkpoint loading, feature export, baseline rerun or source payload.',
            'Only current-row physical support fits the head. Candidate never consumes baseline predictions, scores, truth, query roles or query statistics.',
            'LocalRidge direct baseline and BranchInteraction descriptive reference paths are registered for separate post-prediction analysis; candidate and preparation never read scores. BranchRidge remains the immutable raw-cache producer.',
            'All four seeds in each of both cohorts and full K1/5/10/20 x new0/2/5/10/20 matrix remain fixed; all receiver/scene/support draws retained.',
            'Interpret jointly only after both complete; no early outcomes change formula, matrix, healthy tasks or rerun policy.',
            'Report per-K old/new/H and declines against direct LocalRidge and descriptive D92/BranchInteraction; no universal improvement claim from support screening.',
            'Support bank is current legal target support only; report its memory plus alpha and centering state, CPU time, immutable cache bytes and model-package bytes separately.',
            'New source payload=0; original model deployment state unknown. Actual row-block steps and complete sweeps measured; learning rate/source validation=null with reasons; zero Cholesky factorizations.'
        ]
        outputs[name] = spec
    return outputs


if __name__ == '__main__':
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = documents(ROOT, commit)
    write_documents(ROOT, result)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', files=list(result))))
