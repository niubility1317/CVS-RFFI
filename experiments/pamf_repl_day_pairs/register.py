"""Register the date-aware source U contrast matrix and exact fixed anchors."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_adaptive_kl.plan import RUN as adaptive_run
from experiments.pamf_repl_aux_view_dose import register as aux_view
from . import plan


def make(template):
    original = aux_view.plan
    aux_view.plan = plan
    try:
        result = aux_view.make(template)
    finally:
        aux_view.plan = original
    result.update(
        display_name='Tri 源接收机与日期联合对比',
        description='十类与六源各 7 行。在已有 U 对比损失中重分配正对概率，固定其余损失和视图。',
        authorization='用户要求批量持续提升 Tri clean 性能并探索数据自适应，GPU 7 除外。',
    )
    result['parent_run_ids'].extend([plan.fixed.RUN, adaptive_run])
    result['execution'].update(
        gpu_policy='GPU 0–6; balanced free capacity; reserve 12 GiB per initializing worker; maximum 2 including other owners',
        local_artifact_root='local_artifacts/pamf_repl_day_pairs_20261119',
        launch_command='python -m experiments.pamf_repl_day_pairs.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG',
    )
    result['expected_artifacts'] = [
        '14 actual E20 no-query optimizer-step smokes', '14 independent E50 clean prediction and truth-last scores',
        '14 frozen-only independent U diagnostics', 'per-step source date pair availability, mass and dynamic blend telemetry',
        'per-group gradients and every-100-step AdamW counterfactual samples',
    ]
    result['notes'] = [
        'Same matched source E20 and E50 schedule, masks, views and scalar coefficients as the strongest fixed controls.',
        'Only shallow contrast changes in ten-class, only joint160 contrast changes in six-source. Negatives and normalization population stay fixed.',
        'Use source RX and dates only inside accepted U pseudo-label pairs, preserving the uniform baseline when preferred pairs are unavailable.',
        'Exploratory exposed seed; compare the pooled clean endpoints and each subgroup with ordinary PL, original Tri and the strongest fixed control.',
        'No target or U truth during training or checkpoint selection. Clean only; frozen post-training U diagnosis is independent.',
    ]
    result['metrics_plan']['metric_names'].append('valid and joint-supported anchors, uniform and applied joint positive mass, dynamic blend')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
