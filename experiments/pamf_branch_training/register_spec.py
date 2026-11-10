"""Fill the project registry template for this fixed pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .plan import RUN, candidate_rows


def make(template):
    s = deepcopy(template)
    run = 'PROJECT/runs/' + RUN
    release = 'PROJECT/releases/' + RUN + '-release'
    s.update(run_id=RUN, group_id='phase1-pamf-branch-relations',
             display_name='完整 PAMF 多分支标签与伪标签关系优化单种子验证',
             description='旧版含参考响应三视图 PAMF，单种子四组检验 L、U 分支关系损失；普通 PL 与监督对照复用。',
             aliases=['PAMF-branch-relations-pilot'], tags=['PAMF', 'SSL', 'branch', 'truth-last'],
             parent_run_ids=['20261022-phase1-T1-DURA-F-ablation-binding-m63-r01',
                             '20261024-phase1-PAMF-REPL-completion-m201-r01'],
             authorization='2026-11-10 用户授权使用 N607 GPU 0、1，每卡本实验最多两个进程，继续多分支半监督优化。')
    s['code'].update(commit='Recorded immutable release commit in runtime.json at launch',
                     checkout='current PAMF branch-training Git worktree',
                     environment='N607 existing project Python, FP32/TF32 off, no installs', cwd=release)
    s['data'].update(dataset='ManySig F0, six classes', version='existing VALIDATED_ONCE',
                     representation='received IQ float32 [2,256]',
                     contract_ref='F0 same source contract and validated target handles in private runtime context',
                     physical_ids_ref='F0 target physical IDs; weak view manifest physical_ids_ref',
                     label_map_ref='F0 source contract classes', source_receivers=[1, 3, 4, 6, 8],
                     target_receivers=[0, 2, 5, 7, 9, 10, 11], source_days=[1, 2, 3], target_days=[0, 1, 2, 3],
                     tx_sets_ref='F0 source contract classes',
                     roles=dict(L_s='6300 labeled; branch positives share true label across source RX',
                                U_s='56700 hidden labels; threshold .95 student pseudo labels only for PL rows',
                                V='27000 source validation, no training or checkpoint selection'),
                     train_ratio='F0 fixed source roles', capsule_id=None, split_id=None,
                     validation_ref='previous VALIDATED_ONCE; Phase2 capsule/split not applicable',
                     support_query_ref='Phase1 fixed prediction only; Phase2 support/query not applicable',
                     leo_config_ref='existing weak-view manifest, fixed seeds 2026100912/13/14')
    s['permissions'].update(regime='source_only', query_use='test_only_after_fixed_final_checkpoint',
                            claim_scope='single-seed exploratory, previously exposed benchmark; no general or statistical claim')
    s['checkpoint'].update(initialization='F20/F21 same-contract scratch E20, then own E10+E20',
                           sources=['original F20 and F21 E20; ordinary PL and supervised E50 comparison readouts'],
                           contract_check_ref='source.checked_parent; frozen_inference.load_F',
                           provenance_verdict='Actual parent contract and scratch checkpoint inspected at load; fail on mismatch',
                           selection_rule='fixed final E50; no target selection or tuning')
    s['execution'].update(host='N607', launch_owner='codex-pamf-branch-training-20261110',
                          gpu_policy='GPU0/1, max two this-run workers each; free-memory admission; preserve other owners',
                          remote_run_root=run, remote_log_root=run + '/logs',
                          local_artifact_root='local_artifacts/pamf_branch_training_20261110',
                          launch_command='PYTHON -u -m experiments.pamf_branch_training.runner dispatch --config ' + release + '/runtime.json',
                          stop_rule='Any technical failure stops new dispatch; preserve healthy workers and all artifacts; no target-driven retry')
    s['expected_artifacts'] = ['real E20 no-query four-arm forward/backward smoke',
        'four own E30 and E50 frozen source trajectories', 'fixed practical and weak predictions',
        'independent truth-last per-row scores, accuracy and macro-F1',
        'same-seed F20/F21 comparisons and overall.csv', 'PID/CWD/command/GPU/log readback']
    s['metrics_plan'].update(metric_names=['accuracy', 'macro_f1', 'paired percentage-point differences'],
        dimensions=['one seed', 'cross_date/cross_rx/cross_both', 'clean and six practical',
                    'weak combined_target/cross_rx/cross_both', 'receiver/day/transmitter'],
        prediction_ref=run + '/readouts/ROW', scorer_ref='separate CPU scorer, after all same-row predictions')
    rows = []
    for branch in candidate_rows():
        rid = branch['row_id']
        r = deepcopy(template['rows'][0])
        r.update(row_id=rid, branch_row=branch, method='source branch relation; L=%s U=%s' %
                 (branch['l_weight'], branch['u_weight']), purpose='fixed factorial pilot',
                 config_ref=run + '/configs/train__' + rid + '.json',
                 resolved_config_ref=run + '/F/' + rid + '/source/resolved_config.json',
                 data_overrides={'protocol_id': 'F0'},
                 seeds=dict(model=2026092701, split=392005, data=None, augmentation=20261016421,
                            support=None, evaluation=None),
                 seed_notes='Inherited F0 seed and source simulation; weak seeds fixed in manifest; no support seed.',
                 scenario='clean + six practical for three F0 populations; three weak on combined target, raw',
                 optimizer='original matched native AdamW/CENTER_UNITS', lr='original segment schedule',
                 epochs=50, budget_ref='E20/4440 reused + own 10/2220 + own 20/4440',
                 output_root=run + '/F/' + rid + '/source', log_path=run + '/logs/train__' + rid + '.log',
                 command='worker --kind train --config ' + run + '/configs/train__' + rid + '.json',
                 expected_artifacts=['source/complete.json', 'readouts/ROW/predictions/complete.json',
                                     'readouts/ROW/scores/complete.json', 'readouts/ROW/weak_scores/complete.json'])
        rows.append(r)
    s['rows'] = rows
    s['notes'] = ['Pilot is a falsification experiment, not evidence of universal improvement.',
        'Primary comparison pl_lu against the matched F21 ordinary PL; pl_l and pl_u isolate contributions.',
        'Only progress to a separate wider matrix when fixed-source diagnostics and same-row results justify it; no target tuning.']
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
