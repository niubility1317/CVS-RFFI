"""Preregister the numerical-repair comparison against the full-L reference."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri 监督参考方向数值修复对照',
                  description='十类和六源各比较 U 权重 6／3，监督参考更新上限均为 0.25。')
    result['execution'].update(gpu_policy='GPU0,1,5,6,7 round-robin; launch only on an empty card including all owners and pending workers',
        local_artifact_root='local_artifacts/pamf_repl_reference_repair_20261116',
        launch_command='python -m experiments.pamf_repl_reference_repair.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['four actual E20 no-query smokes with full-L AdamW reference and rounding repair',
        'four own E30/E50 continuations with all-step reference, repair and parameter-step telemetry',
        'four blind U trajectories and independent post-freeze U diagnostic scores',
        'four raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'].append('actual AdamW step versus full-L reference, cap repair attempts and factor, fallback and supervised-anchor conflict')
    result['notes']=['Matched comparison against the full-L reference cap .25, same setting and U weight; fixed four pairs before their tests.',
        'Only numerical handling differs: when a rounded FP32 candidate violates constraints, progressively shrink its extra step and recheck the actual rounded candidate; all failures fall back to full-L reference.',
        'Inward factors are 1, .9999, .999, .99, .95, .9, .75, .5; every accepted candidate must satisfy the same cap, nonexpansion and full-L and CE anchor conditions.',
        'The comparison was motivated by early source-training numerical fallback counts, not target results or U truth.',
        'Only L_s gradients define reference and anchors; V supplies no gradient or persistent state; U truth only enters independent post-E50 diagnosis.',
        'Seed and historical target outcomes were exposed; all rows fixed before their own clean test; exploratory only.']
    for entry, row in zip(result['rows'], candidate_rows()):
        entry['method'] = 'Tri full-L AdamW reference numerical repair ' + row['recipe']
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--template', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    Path(a.output).write_text(json.dumps(make(json.loads(Path(a.template).read_text(encoding='utf-8-sig'))),
                                        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
