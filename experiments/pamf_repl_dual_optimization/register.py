"""Build registration through the prior source and evaluation contract."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_priority import register as priority
from . import plan

def make(template):
    old = priority.plan; priority.plan = plan
    try:
        result = priority.make(template)
    finally:
        priority.plan = old
    result.update(display_name='十类浅层与六源 KL 双方向软协调及损失剂量',
        description='十类和六源各八行；组合参考和监督方向软协调，匹配测试 U 权重与 KL 剂量，固定 E50 clean。',
        authorization='2026-11-18 用户要求沿上一轮方式继续设计并发布优化实验；clean only；冻结后独立 U 评分。')
    result['execution'].update(gpu_policy='GPU 0–7; account for all owners and pending workers; max five/card and reserve 12 GiB per initialization',
        local_artifact_root='local_artifacts/pamf_repl_dual_20261118',
        launch_command='python -m experiments.pamf_repl_dual_optimization.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['16 actual-checkpoint no-query optimizer-step smokes',
        '16 E30/E50 source trajectories and per-step gradient records',
        '16 blind U trajectories with independently scored post-freeze correctness',
        '16 frozen clean predictions and independent truth-last scores',
        'same-state AdamW L-only, raw joint and actual-update samples every 100 steps']
    result['notes']=[
        'Previously exposed seed and target; exploratory results, not independent confirmation.',
        'Auxiliary remainder is first soft projected against shallow (ten) or paired KL (six), then the total auxiliary is soft projected against L.',
        'Supervised and base U CE summands remain intact before global clipping; direction constraints may retain residuals or interact.',
        'Compare dual strength, U weight and KL dose with matching prior-run or within-run controls; do not treat simultaneous changes as isolated effects.',
        'Only fixed E50 and pre-registered clean predictions; no satellite or weak.',
        'U truth independently scored after the model is frozen, never used in training or selection.',
        'Stop new dispatch for systemic technical or protocol failures; retain outputs and healthy workers.']
    for row in result['rows']:
        row['expected_artifacts'].extend(['source/segment30/adamw_LU_samples.jsonl.gz',
                                          'source/segment50/adamw_LU_samples.jsonl.gz'])
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--template', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); value = make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
