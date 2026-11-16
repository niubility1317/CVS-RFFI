"""Preregister CE-only AdamW reference versus full-supervised reference."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import candidate_rows,scoped


def make(template):
    with scoped():result=original.make(template)
    result.update(display_name='Tri 有标签 CE AdamW 参考主方向实验',
        description='十类和六源各比较 U 权重 6／3，CE-only 参考外的其他监督与 U 更新上限 0.25／0.5。')
    result['execution'].update(gpu_policy='GPU2,3,4 round-robin; all owners and pending workers count toward three processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_ce_reference_20261116',
        launch_command='python -m experiments.pamf_repl_ce_reference.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['eight actual E20 no-query smokes with CE-only AdamW reference',
        'eight own E30/E50 continuations with all-step CE-reference and parameter-step telemetry',
        'eight blind U trajectories and independent post-freeze U diagnostic scores',
        'eight raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'].append('ordinary CE-only actual AdamW reference and total step cosine; non-CE extra step ratio and fallback')
    result['notes']=['Reference is the actual ordinary labeled CE-only AdamW step at the same current joint optimizer state; prior joint moments are retained.',
        'The extra step relative to CE-only contains other supervised objectives and all U objectives; it is not a pure U contribution.',
        'After the complete L and CE gradient anchor constraints, each group extra step is capped at .25 or .5 of the CE-only reference norm.',
        'Only L_s gradients define the reference; V supplies no gradients or persistent state; U truth only enters independent post-E50 diagnosis.',
        'Seed and historical target outcomes were exposed; all rows fixed before their own clean test; exploratory only.']
    for entry,row in zip(result['rows'],candidate_rows()):
        entry['method']='Tri ordinary labeled CE AdamW reference dominance '+row['recipe']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(make(json.loads(Path(a.template).read_text(encoding='utf-8-sig'))),
                              ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
