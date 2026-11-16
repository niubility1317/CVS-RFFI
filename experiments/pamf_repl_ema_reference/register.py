"""Preregister segment-local labeled-gradient EMA reference comparisons."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import candidate_rows,scoped


def make(template):
    with scoped():result=original.make(template)
    result.update(display_name='Tri 有标签梯度平滑参考主方向实验',
        description='十类／六源各比较 U 权重 6／3、EMA 监督参考外额外更新上限 0.25／0.5。')
    result['execution'].update(gpu_policy='GPU2,3,4 round-robin; launch only on an empty card including all owners and pending workers',
        local_artifact_root='local_artifacts/pamf_repl_ema_reference_20261116',
        launch_command='python -m experiments.pamf_repl_ema_reference.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['eight actual E20 no-query smokes with labeled-gradient EMA reference',
        'eight own E30/E50 continuations with all-step EMA reference and parameter-step telemetry',
        'eight blind U trajectories and independent post-freeze U diagnostic scores',
        'eight raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'].append('current full L vs EMA gradient, EMA-reference vs total step cosine, extra-step ratio and fallback')
    result['notes']=['Reference gradient is beta=.9 EMA of full L gradients from the current source segment only; the first step initializes to the current full L and E31 starts a new EMA.',
        'The actual AdamW reference step uses that EMA gradient at the current joint optimizer state; earlier optimizer moments remain.',
        'The extra step relative to the EMA reference includes current L minus EMA and all U objectives; it is not a pure U contribution.',
        'Both complete current L and ordinary CE anchor the extra step, which is capped at .25 or .5 of the EMA-reference actual step per group.',
        'Only L_s gradients enter EMA; V supplies no gradients or persistent state; U truth only enters independent post-E50 diagnosis.',
        'Source-only L-L batch variability motivates smoothing; seed and historical target outcomes were exposed, so results are exploratory.']
    for entry,row in zip(result['rows'],candidate_rows()):
        entry['method']='Tri segment-local full-L gradient EMA AdamW reference '+row['recipe']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(make(json.loads(Path(a.template).read_text(encoding='utf-8-sig'))),
                              ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
