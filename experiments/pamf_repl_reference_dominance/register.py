"""Preregister supervised-reference-dominant continuation runs."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri 监督 AdamW 参考主方向实验',
        description='十类和六源各比较 U 权重 6／3 与 U 额外实际更新上限 0.25／0.5。')
    result['execution'].update(gpu_policy='GPU2,3,4 round-robin; all owners and pending workers count toward three processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_reference_dominance_20261116',
        launch_command='python -m experiments.pamf_repl_reference_dominance.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['eight actual E20 no-query optimizer-step smokes with CE and reference dominance',
        'eight own E30/E50 continuations with all-step L/U gradients and actual parameter-step telemetry',
        'eight blind U observation trajectories and independent post-freeze U diagnostic scores',
        'eight raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'].append('reference and actual total step norms/cosines; per-group U extra step ratio and zero/quantization fallback')
    result['notes']=['Reference is the actual full-L-only AdamW step at the current parameter and optimizer state; it retains historical joint moments and is not an independently trained supervised trajectory.',
        'After full L and ordinary L CE actual extra-step constraints, cap each group extra U parameter-step norm at 0.25 or 0.5 times the L-only reference step norm.',
        'The norm bound enforces a total-step/reference cosine of at least sqrt(1-cap^2) when the reference is nonzero; FP32 violations fall back to the L-only reference.',
        'Only L_s gradients determine the reference; V does not supply gradients or persistent state; U truth is read only by independent post-E50 diagnosis.',
        'Seed and historical target outcomes were exposed; all rows fixed before their own clean test; exploratory only.']
    for entry, row in zip(result['rows'], candidate_rows()):
        entry['method'] = 'Tri full-L AdamW reference dominance, CE anchor, ' + row['recipe']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
