"""Register matched comparisons against the frozen fixed-view controls."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_aux_view_dose import plan as fixed
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
        display_name='Tri 对称 KL 按样本自适应加权',
        description='十类与六源各 7 行。保留各自最强固定视图和辅助损失，仅改变接受样本的 KL 权重。',
        authorization='用户要求继续批量优化 Tri、以性能为主，探索能适应不同样本设置的机制，GPU 7 除外。',
    )
    result['parent_run_ids'].append(fixed.RUN)
    result['execution'].update(
        gpu_policy='GPU 0–6; balance free capacity; reserve 12 GiB to admit an initializing worker; maximum 2 with other owners included',
        local_artifact_root='local_artifacts/pamf_repl_adaptive_kl_20261119',
        launch_command='python -m experiments.pamf_repl_adaptive_kl.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG',
    )
    result['expected_artifacts'] = [
        '14 real E20 checkpoint no-query optimizer-step smokes',
        '14 frozen E50 clean predictions and independent scores',
        '14 frozen post-training U diagnostics',
        'adaptive weight totals, acceptance, effective KL, group L/U gradients and same-state AdamW samples',
    ]
    result['notes'] = [
        'Same source-only E20, seed, optimizer, acceptance, forwards, gradient rules and E50 schedule as the fixed controls, except for per-record KL weights.',
        'Ten-class retains ordinary KL, shallow contrast factor 2 and no fusion contrast. Six-source retains raw KL, no shallow contrast and fusion contrast .10.',
        'Both retain U .75, KL factor 1.5, nominal view coefficient .8, no pseudo-label CE, no U projection. Rejected rows always carry zero KL weight.',
        'Seven profiles vary detached view vote, distribution gap or inverse receiver frequency. Accepted weights average 1 and the original full-batch denominator is retained.',
        'Report paired cross_rx + cross_both clean versus the matching fixed recipe, ordinary PL and original Tri. Target-exposed chosen seed is exploratory only.',
        'Use independent E50 truth-last clean and post-freeze U scoring. No target or U truth during training or model selection. No satellite views.',
    ]
    result['metrics_plan']['metric_names'].extend(['per-step adaptive weight range, mass and absolute deviation'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
