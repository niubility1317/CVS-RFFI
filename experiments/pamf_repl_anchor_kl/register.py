"""Preregister anchor KL variants, dose controls and source-only readout."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_priority import register as priority
from . import plan


def make(template):
    old = priority.plan
    priority.plan = plan
    try:
        result = priority.make(template)
    finally:
        priority.plan = old
    result.update(display_name='十类与六源原始视图参照的定向 KL 实验',
                  description='两设置各 7 条 KL 路由与两档剂量，合计 28 条。',
                  authorization='用户要求扩大 Tri 优化实验，除 GPU 7 外每卡最多 4 个计算进程。')
    result['execution'].update(
        gpu_policy='GPU 0–6 only, total max 4; source-only smoke max allocated plus 2 GiB, minimum 6 GiB per worker',
        local_artifact_root='local_artifacts/pamf_repl_anchor_kl_20261118',
        launch_command='python -m experiments.pamf_repl_anchor_kl.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts'] = ['28 actual E20 no-query optimizer smokes',
        '28 fixed E50 predictions and truth-last clean scores', '28 frozen-only independent U diagnostics',
        'stepwise KL and L/U gradients plus same-state AdamW samples']
    result['notes'] = [
        'All candidates set base pseudo-label CE=0 while retaining fixed shallow/fusion auxiliary components.',
        'Original six training forwards and original raw U acceptance forward; detached teachers are computed from these exact outputs.',
        'Same acceptance mask and full-population denominator. T=2 scales KL by T².',
        'Use historical fixed-view symmetric KL control at the matching setting and nominal dose when available; all rows compare to same-seed PL.',
        'Source-only E20 ancestry and E50 selection, then clean-only prediction with independent scoring and frozen-only U diagnostics.',
        'Stop new dispatch only on the registered systemic technical or protocol failure; preserve healthy workers.']
    for row in result['rows']:
        row['expected_artifacts'].extend(['source/segment30/adamw_LU_samples.jsonl.gz',
                                         'source/segment50/adamw_LU_samples.jsonl.gz'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
