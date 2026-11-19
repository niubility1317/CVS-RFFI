"""Register the matched ten-class donor-by-dose matrix."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_adaptive_kl import plan as adaptive
from experiments.pamf_repl_aux_view_dose import register as aux_view
from . import plan


def make(template):
    original = aux_view.plan
    aux_view.plan = plan
    try:
        result = aux_view.make(template)
    finally:
        aux_view.plan = original
    result.update(display_name='十类 Tri 源日期风格供体与低剂量对照',
                  description='原供体、同日供体、跨日供体与 U 权重 0.375、0.75、1.5 的全因子矩阵。',
                  authorization='用户要求继续批量探索 Tri clean、自适应和更低的总 U 权重，GPU 7 除外。')
    result['parent_run_ids'].append(adaptive.RUN)
    result['execution'].update(
        gpu_policy='GPU 0–6; balanced free capacity; 12 GiB admission reserve; two processes including other owners',
        local_artifact_root='local_artifacts/pamf_repl_style_dose_20261120',
        launch_command='python -m experiments.pamf_repl_style_dose.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG',
    )
    result['expected_artifacts'] = ['9 real E20 no-query optimizer-step smokes',
        '9 fixed E50 clean predictions and independent same-row truth-last scores',
        '9 frozen-only source-U pseudo-label diagnostics',
        'stepwise donor support and use, source validation, per-group gradients and sampled AdamW updates']
    result['notes'] = [
        'Use the ten-class hard-vote ordinary KL anchor on the matched source-only seed 01 E20, continuing to fixed E50.',
        'Only vary total U dose and detached style donors. Same pseudo class, different source receiver, and original acceptance and fallback are preserved.',
        'All nine arms include the original 0.95 mask and .5 style activation. Pseudo-label CE, joint contrast and RX extrapolated CE are disabled.',
        'Historical clean is exposed. All candidates are fixed before current inference. No target or U truth in training or checkpoint selection.',
    ]
    result['metrics_plan']['metric_names'].append('per-step original-valid, same/cross-date support, selected cross-date and donor changes')
    for row in result['rows']:
        row['expected_artifacts'].extend(f'source/segment{end}/step_metrics.jsonl.gz' for end in (30, 50))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
