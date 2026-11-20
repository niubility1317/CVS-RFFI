"""Generate the immutable 11-setting, three-seed clean preregistration."""
import argparse
from copy import deepcopy

from experiments.cric_dura_comparison.io import read, write
from . import full_suite as suite
from . import receiver_balance as balance, transfer


def make(template):
    result = deepcopy(template)
    root = f'PROJECT/runs/{suite.RUN}'
    cmd = f'python -m {suite.MODULE} publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    result.update(
        run_id=suite.RUN, group_id='phase1-pamf-source-adaptation',
        display_name='Tri 融合正对自适应完整 11 设置 clean 对比',
        description='11 设置各 3 种子，统一融合正对 RX 均衡，与原 Tri 和普通 PL 按物理样本配对。',
        parent_run_ids=[suite.paper.RUN, balance.RUN, balance.CONFIRM_RUN],
        replaces_run_id=suite.REPLACES, authorization='用户要求选四张 GPU 发布 11 设置各三种子与普通 PL 的完整对比。')
    result['code'].update(commit='actual release git OID', checkout='current task worktree',
                          environment='N607 CVS-RFFI', cwd=f'PROJECT/releases/{suite.RUN}-release')
    result['data'].update(
        dataset='11 frozen registered Phase1 data settings', contract_ref='original matched per-setting handles',
        physical_ids_ref='exact ordered physical IDs as both matched baselines',
        roles={'L_s': 'original labeled source', 'U_s': 'original TX-blind source', 'V': 'original source validation'},
        support_query_ref='Phase1 clean: cross_date, cross_rx and cross_both; only the latter two pooled')
    result['permissions'].update(
        query_use='Same-row frozen E50 clean prediction followed by independent truth-last scoring',
        claim_scope='Fixed sample-adaptive candidate on 11 existing clean settings; no target-selected weights')
    result['checkpoint'].update(
        initialization='same-setting, same-seed, same-contract original Tri E20 for new continuations',
        sources=[suite.paper.RUN, balance.RUN, balance.CONFIRM_RUN],
        contract_check_ref='full_suite.prepare and transfer.parent',
        provenance_verdict='explicit known source-only ancestors, matching data roles and original recipe',
        selection_rule='Fixed E50. Primary common rule uses U weight 6 and positive target mixture .5 for all rows.')
    result['execution'].update(
        host='N607 ordinary account', gpu_policy='GPU 0,4,5,6 only; max two total per GPU; min 12 GiB admission',
        owner=f'codex-{suite.RUN}', launch_owner=f'codex-{suite.RUN}',
        remote_run_root=root, remote_log_root=root+'/logs', launch_command=cmd,
        local_artifact_root='local_artifacts/pamf_tri_adaptive_full_20261120',
        stop_rule='Stop new dispatch on blocking technical failure; retain healthy workers and all outputs.')
    result['expected_artifacts'] = ['15 new real E20 no-query smokes and same-contract continuations',
        '15 new frozen E50 raw clean predictions and independent scores',
        '12 verified exact fusion-balance rows and 6 exact original-Tri identity rows',
        '33 paired comparisons against original Tri and ordinary PL and an 11-setting summary']
    result['metrics_plan'].update(
        metric_names=['overall clean accuracy', 'paired gain, damage and net',
                      'cross_both accuracy and net', 'three-seed mean per setting',
                      'source acceptance and affected fusion anchor fraction'],
        dimensions=['setting', 'seed', 'clean test subset'],
        prediction_ref=root+'/sources.json', scorer_ref=root+'/analysis/results.json')
    result['rows'] = []
    for row in suite.rows():
        rid, entry = row['row_id'], deepcopy(template['rows'][0])
        if row['reuses_original']:
            prior = f"{suite.paper.RUN}/F/{transfer.original_row(row)}/source"
        elif row['reuse_run']:
            prior = f"{row['reuse_run']}/F/{row['setting']}-tri-balance-joint-s{row['model_seed']}/source"
        else:
            prior = None
        out = f'PROJECT/runs/{prior}' if prior else f'{root}/F/{rid}/source'
        entry.update(row_id=rid, method='Tri, fusion RX positive balancing, U=6',
                     purpose='Exact frozen row reuse' if prior else 'Previously untested data design',
                     config_ref=f'{out}/resolved_config.json' if prior else f'{root}/configs/train__{rid}.json',
                     resolved_config_ref=f'{out}/resolved_config.json',
                     data_overrides={'protocol_id': row['protocol_id'], 'U_weight': 6., 'mixture': .5,
                                     'reuse_completed': bool(prior), 'native_identity': row['reuses_original']},
                     epochs=50, budget_ref='reuse exact fixed E50' if prior else 'original 4440 updates + own 6660',
                     scenario='raw clean only', output_root=out,
                     log_path=f'{out}/segment50/metrics_epoch.jsonl' if prior else f'{root}/logs/train__{rid}.log',
                     command=cmd, expected_artifacts=['existing source, prediction and score'] if prior else
                     ['source/complete.json', 'predictions/complete.json', 'scores/complete.json'])
        entry['seeds'].update(model=row['model_seed'], split=None, data=None)
        entry['seed_notes'] = 'Only model seed varies. Source role split and other fixed seeds are bound by the matched protocol.'
        result['rows'].append(entry)
    result['notes'] = [
        'Both methods use same setting, seed, physical clean records and fixed E50.',
        'Preserve original Tri except the fusion positive target. The per-batch mix redistributes probability across supported RX.',
        'Single- and two-source settings are exact identities: at most one alternative RX is available per anchor.',
        'F2 U=3 is a previously replicated setting-specific reference, outside the common primary policy.',
        'No checkpoint or coefficient selection uses target outputs. Existing clean targets are historically exposed.',
    ]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write(args.output, make(read(args.template)))
