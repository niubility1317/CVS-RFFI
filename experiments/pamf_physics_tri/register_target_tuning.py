"""Register the fixed full-source, three-seed, seven-view parameter comparison."""
import argparse
import json
from pathlib import Path

from . import register_source_tuning as base
from .target_tuning_plan import RUN, candidate_rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, candidate_rows
    try:
        result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    run = 'PROJECT/runs/' + RUN
    result.update(display_name='完整源数据 PAMF 原型参数三种子七景对照',
        description='固定比较三种数据配置的 11 个新参数，每个种子继承匹配原 E20，逐行自动预测并独立评分。',
        aliases=['PAMF-fullsource-prototype-parameter-comparison'],
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03',
                        '20261112-phase1-pamf-tri-two-methods-full-m60-r02'],
        authorization='2026-11-12 用户要求不留源域开发集，使用原数据和对应权重，训练后自动七景测试。')
    result['data'].update(dataset='ManyRx ten-class, ManySig 1%-labeled and ManySig six-source-receiver',
        contract_ref='context.handles indexed by each row protocol_id', physical_ids_ref='each original source and test contract',
        label_map_ref='each matching source contract classes', source_receivers=[], target_receivers=[],
        source_days=[1, 2, 3], target_days=[0, 1, 2, 3], tx_sets_ref='each matching source contract',
        roles=dict(L_s='entire original labeled role for each configuration; no internal holdout',
                   U_s='original source U; true TX hidden', V='original source V; health only'),
        train_ratio='original per-configuration roles, unchanged', capsule_id=None, split_id=None,
        support_query_ref='Phase1 frozen direct predictions; independent truth-last per row',
        leo_config_ref='original six practical scenario views, identical physical IDs per matched row')
    result['permissions'].update(regime='source_only', query_use='frozen predictions then independent truth-last per row',
        claim_scope='exploratory on previously exposed test sets; fixed matrix, no target feedback into training or selection')
    result['checkpoint'].update(initialization='matching original full-L Tri E20 by configuration and model seed, then own E10+E20',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='target_tuning_plan.parent -> full_matrix_plan.parent and checked_e20_parent',
        provenance_verdict='original same-data same-seed scratch E20; no development holdout or target ancestry',
        selection_rule='all 33 declared new parameter rows; fixed final E50; no target-driven selection')
    result['execution'].update(host='N607 ordinary user', launch_owner='codex-' + RUN,
        gpu_policy='GPUs 0,1,3,4,5,7,2,6; at most one this-run worker and two total processes per card, counting other owners; check free memory',
        local_artifact_root='local_artifacts/pamf_fullsource_tuning_20261112',
        launch_command='python -m experiments.pamf_physics_tri.target_tuning_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Systemic technical failure stops new dispatch; preserve healthy workers, outputs and other owners')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', '33 own full-source E30/E50 continuations',
        '33 direct seven-view predictions', '33 per-row independent truth-last scores',
        'same-setting same-seed matched PL and original prototype comparisons, three-seed summary']
    result['metrics_plan'].update(metric_names=['accuracy', 'macro-F1', 'paired percentage-point differences'],
        dimensions=['three model seeds', 'clean and six practical', 'cross-date, cross-RX, cross-both', 'TX/RX/day'],
        prediction_ref=run + '/readouts/ROW/predictions', scorer_ref=run + '/readouts/ROW/scores')
    for entry, row in zip(result['rows'], candidate_rows()):
        rid = row['row_id']
        entry.update(method=f"prototype temperature={row['temperature']}, votes={row['required_votes'] or 4}, weight={row['prototype_weight'] if row['prototype_weight'] is not None else 1}",
            purpose='fixed full-source parameter comparison', data_overrides={'protocol_id': row['protocol_id']},
            seeds=dict(model=row['model_seed'], split=392005, data=None, augmentation=20261016421,
                       support=None, evaluation=None),
            seed_notes='Original fixed physical split; data, support and evaluation seeds are not separately controlled.',
            scenario='clean and six practical; direct raw default head',
            config_ref=run + '/configs/train__' + rid + '.json',
            resolved_config_ref=run + '/F/' + rid + '/source/resolved_config.json',
            budget_ref='original matched scratch E20 reused; own E10 + E20 = 6660 new updates',
            output_root=run + '/F/' + rid + '/source', log_path=run + '/logs/train__' + rid + '.log',
            command='worker --kind train --config ' + run + '/configs/train__' + rid + '.json',
            expected_artifacts=['source/complete.json', 'readouts/' + rid + '/predictions/complete.json',
                                'readouts/' + rid + '/scores/complete.json'])
    result['notes'] = ['11 parameter variants across the three requested data settings, each with the original three model seeds.',
        'Previously completed original prototype and ordinary PL scores are reused, not retrained.',
        'Each row automatically predicts all seven views and independently scores after that row completes.',
        'The same benchmark was previously exposed; comparisons are exploratory and do not feed training, candidate selection or reruns.',
        'The separate completed development-only models are tested as requested but are not mixed with these full-source matched comparisons.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
