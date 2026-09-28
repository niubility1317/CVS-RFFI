"""Prepare one support-justified BranchRidge candidate; never read query results."""
import copy
import json
from pathlib import Path
import subprocess

from prepare_d92_summary_joint_benchmark import ROOT, write_documents

FROZEN = 'configs/d92_branch_ridge_frozen_20260929.json'


def replace_names(value):
    if isinstance(value, dict):
        return {k: replace_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_names(v) for v in value]
    if isinstance(value, str):
        return (value.replace('d92_multiview_ridge_frozen_', 'd92_branch_ridge_frozen_')
                .replace('predict_d92_multiview_ridge.py', 'evaluate_d92_branch_ridge.py')
                .replace('D92-MVRidge', 'D92-BranchRidge').replace('d92-mvridge', 'd92-branch-ridge')
                .replace('d92_mvridge', 'd92_branch_ridge').replace('mvridge/', 'branch_ridge/')
                .replace('mvridge.log', 'branch_ridge.log'))
    return value


def documents(root, commit):
    root = Path(root)
    algorithm = json.loads((root/FROZEN).read_text(encoding='utf-8'))['algorithm']
    if algorithm['method'] != 'D92-BranchRidge-v1':
        raise ValueError('Frozen method mismatch')
    result = {}
    for cohort in ('rx3', 'rx1'):
        spec = replace_names(json.loads((root/f'configs/d92_mvridge_repeat_{cohort}_20260929.json').read_text(encoding='utf-8')))
        config_name = f'configs/d92_branch_ridge_repeat_{cohort}_20260929.json'
        data_name = f'configs/d92_branch_ridge_repeat_{cohort}_data_20260929.json'
        result[data_name] = copy.deepcopy(json.loads((root/f'configs/d92_mvridge_repeat_{cohort}_data_20260929.json').read_text(encoding='utf-8')))
        spec['code'].update(commit=commit, commit_note='Preparation parent; publisher records exact pushed runtime HEAD.')
        spec.update(group_id='d92-fixed-phase1-branch-ridge-repeated-benchmark',status='PLANNED',
            description='固定Phase1，使用原始received单view的身份、FFT及融合前时域/频域/PA分支，当前任务全部真实support拟合固定ridge1解析分类头；所有K同式。',
            tags=['d92','branch-ridge','source-free','repeated-benchmark','frozen-phase1','truth-last'])
        spec['data']['representation']='One unchanged received IQ; frozen z_id/t_emb/f_emb/pa_local160 each plus original FFT96; 736-dimensional per-sample normalized concatenation'
        spec['permissions'].update(regime='fixed_phase1_single_view_branch_current_support_ridge',
            summary_permission='No ground summary/prototype inputs; no source samples or per-record source features.')
        checkpoint=spec['checkpoint']
        checkpoint.update(initialization='Exact original source-only scratch final200 checkpoint; eval and frozen; complete lineage and SHA verified before load.',
            runtime_checkpoint_reload=True, metadata_evidence_ref='docs/D92_FIXED_PHASE1_BRANCH_METADATA_20260929.json',
            inheritance_policy='Only original source-only checkpoint; no historical adaptation state is loaded.',
            current_auxiliary_training='Current-row labeled physical support only, one closed-form ridge1 fit, zero optimizer steps.')
        for key in ('feature_cache_producer_run_id','feature_cache_producer_role'):
            checkpoint.pop(key,None)
        spec['execution']['gpu_policy']='GPU0 serial frozen singleton feature export per cohort; at most two export processes across cohorts; four CPU lanes per cohort, two BLAS threads each; no other tasks changed.'
        conf=spec['confirmation']
        conf.update(candidate=copy.deepcopy(algorithm),candidate_method='D92-BranchRidge-v1',candidate_folder='branch_ridge',
            candidate_predictor='evaluate_d92_branch_ridge.py',candidate_mode='d92_branch_ridge_registration')
        for key in ('frozen_feature_source_root','frozen_feature_producer_config'):
            conf.pop(key,None)
        artifacts=['artifact_reuse.json','branch_features/features_complete.json','branch_features/startup.json',
            'branch_features/checkpoint_provenance.json','branch_ridge/startup.json','branch_ridge/predictions.jsonl',
            'branch_ridge/predictions_complete.json','branch_ridge/fit_trace.jsonl','branch_ridge/compact.jsonl',
            'branch_ridge/compact.csv','branch_ridge/fit_stages.jsonl','branch_ridge/fit_stages.csv']
        spec['expected_artifacts']=['startup.json','state.json','data_reuse.json','complete.json','scores.json']+['each row/'+p for p in artifacts]
        for row in spec['rows']:
            row.pop('reuse_multiview_features_root',None)
            row['provenance_refs']={k:v for k,v in row['provenance_refs'].items() if not k.startswith('frozen_orbit_')}
            row.update(gpu='0 for frozen export; CPU fit/predict',
                optimizer='None; exact centered physical-sum ridge1 Cholesky solve with unregularized intercept; zero optimizer steps',
                budget_ref=f"{conf['splits_per_model']} fixed splits; one fullsupport fit each; no folds, grid, checkpoint selection or query feedback",
                seed_notes='Deterministic original view and analytic classifier introduce no new RNG role; all six original seed roles retained.',
                expected_artifacts=artifacts)
        spec['notes']=[
            'User authorized existing full benchmark reuse; previously scored targets are not new independent confirmation.',
            'Single candidate justified by completed support-only branch diagnostic, not query results; docs/D92_BRANCH_AUGMENT_DESIGN_20260929.md.',
            'One original received view, frozen Phase1, 736D unit-block features and ridge1 fixed for every K and class; no candidate/parameter search.',
            'One analytic fit on current-row physical support. Existing support OOF diagnostic is not rerun; K1 has no independent within-class holdout evidence.',
            'No source IQ/loader/sample features/replay or ground summary/prototype fitting; additional source payload0B.',
            'Baseline prediction artifacts reused unchanged for independent scoring only; candidate never consumes baseline predictions, query truth or scores.',
            'Both complete cohorts finish before either scores file is read; no selective reruns or changes from target outcomes.',
            'Report complete four-RX/all-K old/new/H, seeds, scenes, new counts, local declines and resource costs; same preregistered acceptance retained.',
            'Model deployment unknown; report checkpoint file sizes separately from incremental transmission, which is null.',
            'Detailed measured analytic loss/regularization/gradient residual and time/state bytes; epoch/LR/source validation null with reasons.'
        ]
        result[config_name]=spec
    return result


if __name__=='__main__':
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=documents(ROOT,commit)
    write_documents(ROOT,result)
    print(json.dumps(dict(status='PREPARED_NOT_LAUNCHED',files=list(result))))
