"""Register a nonoverlapping low-U-weight KL experiment."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_priority import register as priority
from . import plan


def make(template):
    previous=priority.plan;priority.plan=plan
    try:result=priority.make(template)
    finally:priority.plan=previous
    result.update(display_name='六源低无标签权重 KL 方向对照',
        description='在现有六源 KL 矩阵以外补充 U 权重 0.75 的四个固定条件，不复制其他运行行。',
        authorization='2026-11-17 用户要求利用其他空闲显卡、均匀优化本轮任务分配。')
    result['execution'].update(gpu_policy='GPU 4 and 6; all owners/pending <=5 per GPU; >=12 GiB free; one own worker per GPU',
        local_artifact_root='local_artifacts/pamf_repl_lowweight_kl_20261117',
        launch_command='python -m experiments.pamf_repl_priority_extension.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['metrics_plan']['dimensions']=['six-source receivers','seed 2026092701','raw clean','TX/RX/day']
    result['expected_artifacts']=['4 real-checkpoint no-query optimizer-step smokes',
        '4 E30/E50 source trajectories with group gradients and sampled actual AdamW reference updates',
        '4 blind U trajectories with independent frozen U correctness diagnostics',
        '4 raw clean predictions and independent truth-last scores']
    result['notes']=[
        'Fixed four six-source rows U weight 0.75 × KL coefficient 1 or 1.5 × native or KL-reference update.',
        'Other U factors match the six-source priority matrix: shallow 0, joint .05, extrapolation 0 with CE reallocated.',
        'Rows are distinct from the priority matrix at U weight 3/6; not a takeover or duplicate of existing jobs.',
        'Prior weight-0.75 and clean target results were exposed, making this an exploratory data-dependent extension.',
        'U truth only independently after source E50 freeze; no target or U truth is used for training or selection.',
        'Use clean only, same E20 and seed, same-step gradients, 100-step AdamW observation and paired controls.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
