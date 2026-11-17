"""Minimal fixed matrix and legal post-freeze readout registration."""
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
    result.update(display_name='十类浅层与六源 KL 的无标签辅助梯度软投影',
        description='十类和六源各 8 行，匹配参考梯度软投影、监督冲突软投影及损失消融，统一 E50 clean。',
        authorization='2026-11-17 用户要求全面分析并发布多个实验，clean only，U 真值仅冻结后独立评分。')
    result['execution'].update(gpu_policy='GPU 0–7; all owners and pending <=5/card; admission reserves >=12 GiB per initializing worker',
        local_artifact_root='local_artifacts/pamf_repl_uaux_soft_20261117',
        launch_command='python -m experiments.pamf_repl_uaux_soft.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['16 actual-checkpoint no-query optimizer-step smokes',
        '16 E30/E50 frozen source trajectories with gradient, acceptance and source-V observations',
        '16 blind U trajectories and independent post-freeze correctness diagnostics',
        '16 frozen raw clean predictions and independent truth-last scores',
        'same-state AdamW L-only, raw joint and actual-update samples every 100 steps']
    result['notes']=[
        'Single previously exposed seed and previously exposed clean outcomes: exploratory, not independent confirmation.',
        'U weight .75 for every row. Ten-class reference shallow; six-source reference exact paired KL.',
        'Reference strength alpha 0/.25/.5/1 projects only other U auxiliaries, with no norm cap.',
        'Separate beta .5/1 ablation softly projects only total U auxiliaries against L; U pseudo CE and L summands are intact.',
        'Four loss ablations isolate shallow or KL with stated setting and guard; compare only matching controls.',
        'After E50 freeze only, independently score saved U pseudo observations against U truth; no feedback.',
        'Fixed clean cross_date/cross_rx/cross_both predictions precede independent truth-last scoring; no weak or satellite.',
        'Systemic protocol or technical failure stops new dispatch but preserves healthy workers and all outputs.']
    for row in result['rows']:
        row['expected_artifacts'].extend(['source/segment30/adamw_LU_samples.jsonl.gz',
                                          'source/segment50/adamw_LU_samples.jsonl.gz'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
