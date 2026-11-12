"""Registry entry for seven-view testing of the three completed models."""
import argparse
import json
from pathlib import Path

from . import register_source_tuning as base, source_tuning_plan
from .development_evaluation_plan import RUN


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, source_tuning_plan.candidate_rows
    try:
        result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    run = 'PROJECT/runs/' + RUN
    result.update(display_name='已完成 10 类开发模型的 clean 与六场景补测',
        description='对三枚已完成模型执行冻结默认头直接预测和逐行独立评分，不重新训练。',
        aliases=['PAMF-development-model-seven-view-evaluation'], parent_run_ids=[source_tuning_plan.RUN],
        authorization='2026-11-12 用户要求已完成训练的模型补做 clean 和六场景测试。')
    result['data']['roles'] = dict(L_s='原 6533 中训练 5228、开发 1305；历史事实，未改标为完整 L',
                                    U_s='原源 U 真值隐藏', V='原源 V 仅健康诊断')
    result['permissions'].update(regime='frozen_inference', query_use='per-row prediction then independent truth-last',
        claim_scope='探索性补测；模型使用 80% 原 L，不能当作与完整 L 的普通 PL 公平匹配')
    result['checkpoint'].update(initialization='three existing frozen source-development E50 models; no training',
        sources=[source_tuning_plan.RUN], contract_check_ref='frozen model source contract and recorded L split',
        provenance_verdict='own completed source-only models, known reduced L training budget',
        selection_rule='all three previously trained temperatures; no test-driven selection')
    result['execution'].update(launch_owner='codex-' + RUN,
        gpu_policy='GPUs 0,4,5,1,3,2,6,7; one this-run worker and at most two total processes per card',
        local_artifact_root='local_artifacts/pamf_development_evaluation_20261112',
        launch_command='python -m experiments.pamf_physics_tri.development_evaluation_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Systemic technical failure stops new dispatch; preserve outputs and other owners')
    result['expected_artifacts'] = ['one real-checkpoint no-query smoke',
        'three frozen direct predictions on clean and six practical scenes',
        'three per-row independent truth-last scores', 'per-scene exploratory analysis']
    result['metrics_plan'].update(metric_names=['accuracy', 'macro-F1'],
        dimensions=['ten classes', 'seed 2026092701', 'clean and six practical scenes',
                    'cross-date, cross-receiver and cross-both'],
        prediction_ref=run + '/readouts/ROW/predictions', scorer_ref=run + '/readouts/ROW/scores')
    for row in result['rows']:
        rid = row['row_id']
        row.update(purpose='frozen evaluation of already completed source-development model',
            config_ref=run + '/configs/predict__' + rid + '.json', resolved_config_ref=None,
            scenario='clean and six practical, direct raw readout', optimizer=None, lr=None, epochs=0,
            budget_ref='no new training; existing E50 model', output_root=run + '/readouts/' + rid,
            log_path=run + '/logs/predict__' + rid + '.log',
            command='worker --kind predict --config ' + run + '/configs/predict__' + rid + '.json',
            expected_artifacts=['predictions/complete.json', 'scores/complete.json'])
    result['notes'] = ['All three frozen models are evaluated; every row scores only after all seven of its predictions finish.',
        'The 20% source-L holdout is a historical training difference; do not present these models as full-L matched controls.',
        'Previously exposed target results are exploratory and do not feed training, candidate selection or reruns.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
