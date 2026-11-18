"""Register the controlled CE by auxiliary factorial and fixed readouts."""
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
    result.update(display_name='十类与六源伪标签 CE 和辅助损失剂量对照',
                  description='各设置固定辅助项开关与四级 CE 剂量，检验对 clean 泛化的独立效应和交互。',
                  authorization='2026-11-18 用户要求继续优化 Tri，并将性能作为目标、梯度冲突作为诊断。')
    result['execution'].update(
        gpu_policy='GPU 0–7; count existing and pending slots; maximum 5 per GPU, 12 GiB for each initializing worker',
        local_artifact_root='local_artifacts/pamf_repl_loss_dose_20261118',
        launch_command='python -m experiments.pamf_repl_loss_dose.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts'] = ['16 real-checkpoint source-only optimizer smokes',
        '16 E50 freezes and truth-last clean scores', '16 post-freeze U diagnostics',
        'stepwise effective CE/aux losses and full L/U gradients', 'same-state AdamW samples every 100 steps']
    result['notes'] = [
        'Fixed CE dose 1/.5/.25/0 by auxiliary dose 1/0, identical E20, training budget and accepted-view machinery.',
        'Zero CE and auxiliary dose is a zero U-gradient continuation with U forwards, not a pure L-only training run.',
        'No gradient projection. Gradients and AdamW statistics are diagnostic; evaluate fixed E50 clean vs matched PL and factorial controls.',
        'Exposed test seed and data; exploratory only. Selection stays source-only and U truth is scored only after freeze.',
        'Only clean predictions with independent scoring. No satellite runs. Retain verified healthy workers on systemic failure.']
    for row in result['rows']:
        row['method'] = 'Tri REPL loss dose ' + row['row_id']
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
