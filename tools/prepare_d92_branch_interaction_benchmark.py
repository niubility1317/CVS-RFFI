"""Prepare fixed interaction repeat; read frozen specifications, never scores."""
import copy
import json
from pathlib import Path
import subprocess
from prepare_d92_summary_joint_benchmark import ROOT, write_documents

FROZEN = 'configs/d92_branch_interaction_frozen_20260929.json'
PRODUCER = 'configs/d92_branch_ridge_frozen_20260929.json'
SUPPORT_RUN = '20260929-phase2-d92-branch-interaction-support-m4-r01'


def replace_names(value):
    if isinstance(value, dict): return {k: replace_names(v) for k, v in value.items()}
    if isinstance(value, list): return [replace_names(v) for v in value]
    if isinstance(value, str):
        return (value.replace('d92-branch-ridge', 'd92-branch-interaction')
                .replace('D92-BranchRidge', 'D92-BranchInteraction')
                .replace('d92_branch_ridge', 'd92_branch_interaction')
                .replace('/branch_ridge/', '/branch_interaction/')
                .replace('branch_ridge.log', 'branch_interaction.log'))
    return value


def documents(root, commit):
    root = Path(root)
    algorithm = json.loads((root/FROZEN).read_text(encoding='utf-8'))['algorithm']
    if algorithm['method'] != 'D92-BranchInteraction-v1': raise ValueError('Frozen method mismatch')
    outputs = {}
    for cohort in ('rx3', 'rx1'):
        old = json.loads((root/f'configs/d92_branch_ridge_repeat_{cohort}_20260929.json').read_text(encoding='utf-8'))
        spec = replace_names(copy.deepcopy(old))
        name = f'configs/d92_branch_interaction_repeat_{cohort}_20260929.json'
        data_name = f'configs/d92_branch_interaction_repeat_{cohort}_data_20260929.json'
        data = json.loads((root/f'configs/d92_branch_ridge_repeat_{cohort}_data_20260929.json').read_text(encoding='utf-8'))
        if data['shots'] != [1, 5, 10, 20] or data['new_counts'] != [0, 2, 5, 10, 20] or len(old['rows']) != 4:
            raise ValueError('Frozen full-matrix mismatch')
        outputs[data_name] = copy.deepcopy(data)
        release = spec['code']['cwd']
        spec.update(status='PLANNED', group_id='d92-fixed-phase1-branch-interaction-repeated-benchmark',
            display_name=f'D92-BranchInteraction冻结方法{cohort}完整重复基准',
            description='固定Phase1和三臂support诊断前预设的交互核，复用BranchRidge原始received单view五块特征；当前row support一次kernel ridge拟合，全部K同式。',
            parent_run_ids=[old['run_id'], SUPPORT_RUN, old['confirmation']['baseline_source_run_id']],
            authorization='用户已授权继续优化及透明重复基准；完整support-only诊断通过预设门槛后，原交互公式不变，覆盖全部既定模型、receiver、场景、K、新类数和support seed。',
            tags=['d92','branch-interaction','source-free','frozen-phase1','cached','cpu-only','repeated-benchmark','truth-last'])
        spec['code'].update(commit=commit, commit_note='Preparation parent; publisher records exact pushed runtime HEAD.')
        spec['data']['representation']='Reused immutable original-received z_id/FFT96/t_emb/f_emb/pa_local; no encoder execution or new view'
        spec['permissions'].update(regime='fixed_phase1_cached_single_view_branch_current_support_interaction',
            summary_permission='No ground prototype/summary inputs to candidate; no source samples or sample-level source features.')
        spec['checkpoint'].update(runtime_checkpoint_reload=False,
            initialization='Reuse original source-only scratch final200 feature provenance; no checkpoint load.',
            feature_cache_producer_run_id=old['run_id'],
            feature_cache_producer_role='Immutable original received five-block features; no adapted state',
            inheritance_policy='Exact frozen Phase1 feature lineage; no fitted BranchRidge or probe state inherited.',
            current_auxiliary_training='Current-row support only, one fixed interaction kernel ridge1 solve; no OOF or parameter search.')
        spec['execution']['gpu_policy']='CPU only; four model lanes per cohort, two BLAS threads each; no GPU, checkpoint loading, feature export or baseline re-execution.'
        conf = spec['confirmation']
        conf.update(candidate=copy.deepcopy(algorithm), candidate_method='D92-BranchInteraction-v1',
            candidate_folder='branch_interaction', candidate_predictor='evaluate_d92_branch_interaction.py',
            candidate_mode='d92_branch_interaction_registration',
            candidate_config=release+'/'+FROZEN,
            frozen_branch_feature_source_root=old['execution']['remote_run_root'],
            frozen_branch_feature_producer_config=release+'/'+PRODUCER,
            branch_ridge_reference_run_id=old['run_id'],
            branch_ridge_reference_root=old['execution']['remote_run_root'])
        artifacts=['artifact_reuse.json','frozen_branch_feature_reuse.json','branch_interaction/startup.json',
            'branch_interaction/predictions.jsonl','branch_interaction/predictions_complete.json',
            'branch_interaction/fit_trace.jsonl','branch_interaction/compact.jsonl',
            'branch_interaction/compact.csv','branch_interaction/fit_stages.jsonl','branch_interaction/fit_stages.csv']
        spec['expected_artifacts']=['startup.json','state.json','data_reuse.json','complete.json','scores.json']+['each row/'+p for p in artifacts]
        spec['metrics_plan'].update(
            prediction_ref='Original D92 reuse_row_root and new output_root/branch_interaction; frozen BranchRidge reference artifacts are independent analysis inputs only',
            secondary_reference_method='D92-BranchRidge-v1',
            secondary_reference_run_id=old['run_id'],
            secondary_reference_root=old['execution']['remote_run_root'])
        for row, original in zip(spec['rows'], old['rows']):
            row.update(gpu=None, expected_artifacts=artifacts,
                reuse_branch_features_root=original['output_root']+'/branch_features',
                reuse_branch_ridge_root=original['output_root']+'/branch_ridge',
                optimizer='None; fixed KB+KA+KB*KA support-centered physical-sum ridge1 Cholesky; zero optimizer steps',
                budget_ref=f"{conf['splits_per_model']} unchanged splits; one fullsupport fit each; no folds, search, new features or checkpoint/baseline execution")
            row['provenance_refs'].update(
                frozen_branch_features_complete=original['output_root']+'/branch_features/features_complete.json',
                frozen_branch_features_startup=original['output_root']+'/branch_features/startup.json',
                frozen_branch_checkpoint_provenance=original['output_root']+'/branch_features/checkpoint_provenance.json',
                branch_ridge_predictions_complete=original['output_root']+'/branch_ridge/predictions_complete.json')
        spec['notes']=[
            'Transparent repeat on previously scored target data; not a new independent confirmation.',
            'Frozen interaction formula unchanged after complete support-only diagnostic passed predetermined screen; K1 still has no support holdout performance evidence.',
            'Reuse existing BranchRidge received feature cache, exact original Phase1 lineage and baseline D92 predictions; no checkpoint loading, feature export, baseline rerun or source payload.',
            'Only current-row physical support fits the head. Candidate never consumes baseline predictions, scores, truth, query roles or query statistics.',
            'BranchRidge reference paths are registered for separate post-prediction analysis; candidate and preparation never read scores.',
            'All four seeds in each of both cohorts and full K1/5/10/20 x new0/2/5/10/20 matrix remain fixed; all receiver/scene/support draws retained.',
            'Interpret jointly only after both complete; no early outcomes change formula, matrix, healthy tasks or rerun policy.',
            'Report per-K old/new/H and declines against D92 and BranchRidge; no universal improvement claim from support screening.',
            'Support bank is current legal target support only; report its memory plus alpha and centering state, CPU time, immutable cache bytes and model-package bytes separately.',
            'New source payload=0; original model deployment state unknown. Steps=0, LR/epoch/source validation=null with reasons.'
        ]
        outputs[name] = spec
    return outputs


if __name__ == '__main__':
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = documents(ROOT, commit)
    write_documents(ROOT, result)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED', files=list(result))))
