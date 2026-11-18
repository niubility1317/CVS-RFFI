"""Register equal-dose view contrasts against existing matched controls."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_priority import register as priority
from . import plan


def make(template):
    original = priority.plan
    priority.plan = plan
    try:
        result = priority.make(template)
    finally:
        priority.plan = original
    result.update(display_name='十类与六源基础 CE 和配对 KL 的等剂量视图对照',
                  description='每设置 3 个固定总 CE 的视图变体及 3 个 CE=0 下固定总 KL 的视图变体。',
                  authorization='2026-11-18 用户设置持续优化 Tri 的 active goal，要求匹配检验 CE、辅助和视图贡献。')
    result['execution'].update(
        gpu_policy='GPU 0–7; evenly assign spare capacity; reserve 12 GiB per initializing worker and count existing/PID pending slots',
        local_artifact_root='local_artifacts/pamf_repl_view_dose_20261118',
        launch_command='python -m experiments.pamf_repl_view_dose.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts'] = ['12 source-only real checkpoint optimizer smokes',
        '12 E50 clean blind predictions and independent scores', '12 post-freeze U diagnostics',
        'per-step losses and populations per view plus group gradients', 'same-state AdamW every 100 steps']
    result['notes'] = [
        'Exposure denominator unchanged inside each view; allocation preserves total nominal CE (1) or KL (.8 times setting-specific scale).',
        'CE arm preserves the exact baseline KL component and uses ordinary, raw or 50/50. The matched control is loss-dose aux1_ce1p0.',
        'KL arm keeps CE=0 and uses ordinary, raw or 50/50. The matched control is loss-dose aux1_ce0p0.',
        'Both arms retain all forwards, pseudo-label acceptance rule, shared style plan and non-KL auxiliaries.',
        'Fixed source-only E20 ancestry, E50 endpoint and frozen independent scores. Selected seed and exposed test; exploratory, no result feedback.']
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
