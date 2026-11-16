"""Create a minimal managed record for the fixed source AdamW diagnostic."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

RUN = '20261116-diagnostic-pamf-tri-adamw-e20-s2026092701-r01'


def make(template):
    result = deepcopy(template)
    root = 'PROJECT/runs/' + RUN
    result.update(run_id=RUN, group_id='phase1-pamf-repl-weight-clean', kind='diagnostic',
                  display_name='Tri-SIG-TIME 源域 AdamW 更新方向诊断',
                  description='十类和六源各三批次，比较同一 E20 的 L-only 与原保护 L+U 实际首步。',
                  aliases=['tri-adamw-source'], tags=['PAMF', 'REPL', 'gradient', 'source-only'],
                  parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03'], replaces_run_id=None,
                  authorization='用户要求优化 Tri 梯度冲突并详细记录；不改变当前 U 标签权限。')
    result['code'].update(commit='release HEAD commit', checkout='task worktree',
                          environment='N607 ordinary account, existing CVS-RFFI Python, CPU 2, FP32',
                          cwd='PROJECT/releases/'+RUN+'-release')
    result['data'].update(dataset='original ten-class ManyRx and six-source receiver contracts',
                          roles={'L_s':'visible TX', 'U_s':'source IQ, TX hidden', 'V':'not read'},
                          contract_ref='original paper source config source_contract',
                          physical_ids_ref='original paper source contract',
                          train_ratio='original fixed labeled/unlabeled/validation roles',
                          support_query_ref='not applicable; source-only, no target query',
                          leo_config_ref='not applicable')
    result['permissions'].update(regime='source_only', query_use='none',
                                 claim_scope='six correlated source snapshots, fresh AdamW moments only')
    result['checkpoint'].update(initialization='matching original paper same-setting scratch E20',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='checked_e20_parent plus paper.validate_config',
        provenance_verdict='checked at execution: scratch, source-only, same seed and source contract',
        selection_rule='both original user-requested settings, fixed seed and first three source batches')
    result['execution'].update(host='N607 ordinary user', launch_owner='codex-'+RUN,
        gpu_policy='CPU only, two CPU threads; no GPU workers', remote_run_root=root,
        remote_log_root=root+'/diagnostic.log', local_artifact_root='local_artifacts/pamf_tri_adamw_source_20261116',
        launch_command='python -m experiments.pamf_repl_weight.source_adamw_diagnostic EXISTING_RUNTIME_JSON',
        stop_rule='Stop diagnostic on identity, role, finite-value, derivative or guard-backward mismatch; do not affect other jobs.')
    result['expected_artifacts']=['two original E20 checkpoint no-query source smokes',
        'six complete source snapshot observations with 12 disposable AdamW steps',
        'per-group L/protected-U gradient and actual incremental/total descent relations']
    result['metrics_plan'].update(metric_names=['L/U gradient norms and dot/cosine before and after guard',
        'L versus AdamW incremental U descent dot/cosine', 'optimizer clipping and unit metadata', 'U acceptance'],
        dimensions=['ten-class and six-source', 'three fixed source batches', 't2, cls_head, other'],
        prediction_ref=None, scorer_ref=None)
    result['rows'] = []
    for setting, protocol in [('both_settings',['ManyRx','m6-natural'])]:
        row = deepcopy(template['rows'][0])
        row.update(row_id=setting+'-source-adamw-e20-s2026092701', method='original Tri-SIG-TIME',
                   purpose='compare L-only and guarded L+U actual first-step AdamW',
                   data_overrides={'protocol_id':protocol},
                   seeds={'model':2026092701, 'split':392005, 'data':None, 'augmentation':None,
                          'support':None, 'evaluation':None},
                   seed_notes='Fixed offsets 0/1/2 use CPU random seeds 2026092701+offset.',
                   scenario='source-only L and U; no target', config_ref='original paper source resolved_config',
                   resolved_config_ref='original paper source complete.json config',
                   budget_ref='6 disposable source batches, 12 disposable parameter-only AdamW steps',
                   output_root=root, log_path=root+'/diagnostic.log',
                   command='source_adamw_diagnostic with both fixed settings',
                   expected_artifacts=['diagnostic.log contains complete JSON result'])
        result['rows'].append(row)
    result['notes']=['U TX truth, V and target are never read; no formal training updates.',
        'Optimizer moments are fresh, so do not infer the rest of the trajectory or target performance.',
        'Source inputs and E20 parents follow original paper roles; selected seed was historically target-exposed.',
        'The experiment is fixed before any new source observation; no candidate training or target prediction.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
