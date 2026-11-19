"""Register each E50 transfer row from the frozen source-only decisions."""
import argparse
from copy import deepcopy

from experiments.cric_dura_comparison.io import read, write
from .pilot import paper
from .transfer import DEVELOPMENT_RUN, RUN, original_row, selected_rows


def make(template, decisions=None, *, receiver_balance=False):
    data = deepcopy(template)
    run_id = RUN
    if receiver_balance:
        from .receiver_balance import RUN as run_id, rows
        candidates = rows()
    else:
        candidates = selected_rows(decisions)
    root = f'PROJECT/runs/{run_id}'
    cmd = 'python -m experiments.pamf_repl_source_adapt.transfer_publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    if receiver_balance:
        cmd += ' --receiver-balance'
    data.update(run_id=run_id, group_id='phase1-pamf-source-adaptation', display_name='Tri 自适应源域收益 E50 校准',
                description='四种设置×三种子。源域决策固定后按相同物理样本对照原 Tri 和普通 PL。',
                parent_run_ids=[DEVELOPMENT_RUN, paper.RUN], authorization='用户要求在其他数据设置保留原 Tri 对普通 PL 的领先幅度。')
    data['code'].update(commit='actual runtime release OID', checkout='current task worktree',
                        environment='N607 CVS-RFFI', cwd=f'PROJECT/releases/{run_id}-release')
    data['data'].update(dataset='Four fixed source and raw clean Phase1 contracts', contract_ref='original per-protocol roles',
                        physical_ids_ref='same ordered physical IDs as both ordinary PL and original Tri',
                        roles={'L_s': 'original full labeled source', 'U_s': 'original full TX-blind source',
                               'V': 'original source validation, no checkpoint selection'},
                        support_query_ref='Three existing clean subsets, frozen prediction before independent truth-last scoring')
    data['permissions'].update(query_use='Frozen E50 clean only; independent truth-last scoring',
                               claim_scope='Retrospective source-rule calibration on development settings; same decision across three matched seeds')
    data['checkpoint'].update(initialization='original full same-contract same-seed E20',
                              contract_check_ref='checked_e20_parent plus original paper row matching',
                              provenance_verdict='No held-out pilot checkpoint or target ancestry',
                              selection_rule='Frozen E50; use exact source-specified weight with no target feedback')
    data['execution'].update(host='N607 ordinary user', launch_owner='codex-'+run_id, gpu_policy='GPU 0–6, max two total per GPU, 12 GiB admission reserve',
                             remote_run_root=root, remote_log_root=root+'/logs', local_artifact_root='local_artifacts/pamf_repl_source_adapt_transfer_20261119',
                             launch_command=cmd, stop_rule='Stop new dispatch on first blocking technical failure; preserve healthy workers')
    data['expected_artifacts'] = ['real full-contract E20 smoke for each activated row', 'fixed E50 changed weights only',
                                  'clean predictions, independent scores, and same-physical-ID paired net corrections']
    data['metrics_plan'].update(metric_names=['cross_rx + cross_both accuracy', 'adaptive minus native lead', 'paired net corrections'],
                                dimensions=['setting', 'seed', 'clean test subset'],
                                prediction_ref=root+'/readouts/ROW/predictions', scorer_ref=root+'/readouts/ROW/scores')
    data['rows'] = []
    for row in candidates:
        rid = row['row_id']
        entry = deepcopy(template['rows'][0])
        reused = row['reuses_original']
        previous = f'PROJECT/runs/{paper.RUN}/F/{original_row(row)}/source'
        cfg = f'{previous}/resolved_config.json' if reused else f'{root}/configs/train__{rid}.json'
        entry.update(row_id=rid, method=f"Tri with frozen source-selected U weight {row['weight']}",
                     purpose='Exact original fallback' if reused else 'Source-triggered full transfer',
                     config_ref=cfg, resolved_config_ref=cfg,
                     data_overrides={'protocol_id': row['protocol_id'], 'source_selected_weight': row['weight'], 'reuse_original': reused},
                     epochs=50, budget_ref='original matched E20 plus own E21–E50' if not reused else 'reuse original fixed E50',
                     output_root=previous if reused else f'{root}/F/{rid}/source',
                     log_path=f'{previous}/segment50/metrics_epoch.jsonl' if reused else f'{root}/logs/train__{rid}.log', command=cmd,
                     expected_artifacts=['same historical prediction and score'] if reused else ['E50 final', 'clean prediction and independent score'])
        entry['seeds']['model'] = row['model_seed']
        data['rows'].append(entry)
    data['notes'] = ['Seeds 2026092702 and 2026092703 did not choose the source weights. All seeds use identical decision per data setting.',
                     'Fallback rows reuse the original Tri and represent zero gain by construction. Compare activated rows separately.']
    if receiver_balance:
        data.update(display_name='Tri 接收机正对分布自适应',
                    description='四设置两条损失通道，按当前已接受的跨 RX 正对重分配概率，总权重保持原 Tri。', parent_run_ids=[paper.RUN, RUN])
        data['permissions']['claim_scope'] = 'Matched fixed exploratory clean comparisons of receiver-balance routes'
        data['checkpoint']['selection_rule'] = 'Original weight 6 and fixed E50 for every arm; no target feedback'
        data['execution']['local_artifact_root'] = 'local_artifacts/pamf_repl_source_adapt_rxbalance_20261119'
        data['expected_artifacts'] = ['eight original full-contract E20 smokes', 'eight E50 raw clean independent scores',
                                      'same-ID paired outcomes and supervised, U, and RX reweighting telemetry']
        data['metrics_plan']['metric_names'].append('eligible anchor fraction and positive distribution total variation')
        data['notes'] = ['Only the selected existing shallow or joint U contrast uses positive distribution balancing.',
                         'Keep all original U thresholds, CE, KL and gradient protection. Same RX-pair target rule across settings.']
        for entry, row in zip(data['rows'], candidates, strict=True):
            entry['method'] = f"Native Tri weight 6, RX balance of {row['route']} U contrast"
            entry['purpose'] = 'Matched loss attribution for source-only positive distribution reweighting'
            entry['data_overrides'] = dict(protocol_id=row['protocol_id'], route=row['route'], mixture=.5)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('template', 'output'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--source-decisions')
    parser.add_argument('--receiver-balance', action='store_true')
    args = parser.parse_args()
    if not args.receiver_balance and args.source_decisions is None:
        parser.error('Calibration requires --source-decisions')
    write(args.output, make(read(args.template), read(args.source_decisions) if args.source_decisions else None,
                            receiver_balance=args.receiver_balance))
