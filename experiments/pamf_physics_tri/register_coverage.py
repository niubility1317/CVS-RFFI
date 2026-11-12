"""Register the fixed source-reference and tiered-gate factorial study."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .coverage_plan import RUN, candidate_rows


def make(template):
    result = deepcopy(template)
    run = 'PROJECT/runs/' + RUN
    result.update(run_id=RUN, display_name='10 类 PAMF 分支覆盖率与可靠性实验',
                  description='原始完整训练数据上比较参考库、3/5 低权重分层及两者结合，3 个匹配种子。',
                  aliases=['PAMF-tenclass-coverage-factorial'],
                  tags=['PAMF', 'ten-class', 'reference-memory', 'tiered-coverage'],
                  authorization='2026-11-12 用户要求发布覆盖率实验、保持原训练数据并自动七景测试。')
    result['code']['cwd'] = 'PROJECT/releases/' + RUN + '-release'
    result['data'].update(dataset='ManyRx ten-class, original complete L/U/V roles',
        contract_ref='context.handles.ManyRx.source_contract', physical_ids_ref='original ManyRx source/test contract',
        label_map_ref='original ManyRx class labels', tx_sets_ref='original ManyRx class labels',
        roles=dict(L_s='all 6533 original labeled source samples, no held-out subset',
                   U_s='all 58828 original source U; TX hidden', V='original source V unchanged'),
        train_ratio='original role split, unchanged', source_days=[1, 2, 3],
        support_query_ref='per-row frozen predictions followed by independent truth-last scoring')
    result['permissions'].update(regime='source_only', query_use='frozen predictions and independent truth-last scoring',
        claim_scope='previously exposed test set; exploratory comparisons without target feedback into training')
    result['checkpoint'].update(initialization='same-setting same-seed original full-L Tri E20, then own E10+E20',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='coverage_plan.parent -> full_matrix_plan.parent and checked_e20_parent',
        provenance_verdict='matching original full-L same-seed scratch E20; no held-out-L or target ancestry',
        selection_rule='all fixed rows, final E50, no target-driven selection')
    result['execution'].update(launch_owner='codex-' + RUN, remote_run_root=run, remote_log_root=run + '/logs',
        gpu_policy='N607 GPUs 0-7; max one this-run worker and two total GPU processes per card, other owners counted',
        local_artifact_root='local_artifacts/pamf_coverage_tenclass_20261112',
        launch_command='python -m experiments.pamf_physics_tri.coverage_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Systemic technical failure stops new dispatch; preserve outputs, healthy workers and other owners')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', 'nine full-source matched E30/E50 continuations',
        'nine direct seven-view predictions and independent truth-last scores',
        'coverage, rejection, pseudo-class, source labeled reliability, auxiliary loss and gradient-conflict diagnostics',
        'matched three-seed comparisons with prior ordinary PL and original 4/5 prototype']
    result['metrics_plan'].update(metric_names=['accuracy', 'six-scene accuracy', 'coverage', 'source labeled tier precision',
                                                'class counts', 'auxiliary losses', 'sampled gradient conflict'],
        dimensions=['three seeds', 'strict and weak tier', '10 classes', 'clean and six practical', 'TX/RX/day'],
        prediction_ref=run + '/readouts/ROW/predictions', scorer_ref=run + '/readouts/ROW/scores')
    result['rows'] = []
    for row in candidate_rows():
        item = deepcopy(template['rows'][0])
        rid = row['row_id']
        item.update(row_id=rid, method=f"reference_memory={row['reference_memory']}; tiered={row['tiered']}; weak_weight=0.25",
            purpose='fixed 10-class coverage factorial', data_overrides={'protocol_id': 'ManyRx'},
            seeds=dict(model=row['model_seed'], split=392005, data=None, augmentation=20261016421,
                       support=None, evaluation=None),
            seed_notes='Original fixed data roles; no L holdout.', scenario='direct raw clean and six practical',
            config_ref=run + '/configs/train__' + rid + '.json',
            resolved_config_ref=run + '/F/' + rid + '/source/resolved_config.json',
            budget_ref='matching original scratch E20 reused; own E10 + E20, 6660 new updates',
            output_root=run + '/F/' + rid + '/source', log_path=run + '/logs/train__' + rid + '.log',
            command='worker --kind train --config ' + run + '/configs/train__' + rid + '.json',
            expected_artifacts=['source/complete.json', 'readouts/' + rid + '/predictions/complete.json',
                                'readouts/' + rid + '/scores/complete.json'])
        result['rows'].append(item)
    result['notes'] = [
        'Original 4/5 full-weight prototype and ordinary PL controls are reused; no matched control is retrained.',
        'Three-vote full-weight comparison reuses already running ten_class gate3 rows in 20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01.',
        'A rolling memory holds at most eight source-L samples per class/day and refreshes feature snapshots every 20 optimizer steps.',
        'Both views must reach 4/5 for strong or 3/5 for weak; weak auxiliary terms use 0.25 weight. Physical routing of weak examples also affects the existing U path.',
        'In-training source-L diagnostic samples every 50 steps and uses cross-day references; it is not independent held-out accuracy.',
        'No category-free consistency arm is mixed into the factorial; isolate memory and weighted tiers.',
        'Previously exposed test data: exploratory comparison, no target-driven candidate selection, stopping, or reorder.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
