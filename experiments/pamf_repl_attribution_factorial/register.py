"""Register every factorial cell and its frozen clean/U readouts."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_component_guard import register as original
from .plan import candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri 无标签损失机制与总权重完整因子归因',
        description='十类、六源各按 U 权重、浅中层对比和外推增量组成 2×2×2 完整矩阵。')
    result['data']['roles'] = {'L_s':'labeled training', 'U_s':'TX blind during training; frozen offline diagnostic only', 'V':'source validation'}
    result['execution'].update(gpu_policy='GPU2,3 distributed round-robin; all owners and pending workers count toward three processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_attribution_factorial_20261116',
        launch_command='python -m experiments.pamf_repl_attribution_factorial.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts'] = ['16 real E20 checkpoint no-query smokes',
        '16 own E30/E50 source continuations with all-step group gradients and 100-step component gradients',
        '16 blind U observation trajectories and independent post-freeze U diagnostic scores',
        '16 raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'] = ['clean accuracy and macro-F1', 'L CE and source validation',
        'U pseudo acceptance, unique exposure coverage and independently scored correctness',
        'group L/U gradients before and after native guard',
        'sampled shallow, joint and extrapolation-increment gradients versus complete L']
    result['notes'] = ['Full 2×2×2 matrix per setting: U weight 6/3, shallow contrast on/off, extrapolation increment on/off.',
        'When extrapolation is off, the replacement coefficient returns to ordinary CE so total CE allocation remains .5; it is not removal of all extrapolated CE without replacement.',
        'The original native U gradient guard remains; no actual AdamW update projection is introduced in this matrix.',
        'Source-only sampled component-gradient evidence motivated the factors; test and U truth are not used for training or selection.',
        'Prior test results and the chosen seed were exposed, so all factorial effects are exploratory, not independent confirmation.',
        'Every cell uses the same source-only E20 for its setting, own E30/E50, clean only, and post-freeze independent U scoring.']
    for entry, row in zip(result['rows'], candidate_rows()):
        rid = row['row_id']
        entry['method'] = 'Tri original U guard factorial ' + row['recipe']
        entry['expected_artifacts'] = ['source/complete.json',
            'source/segment30/U_pseudo_observations.jsonl.gz', 'source/segment50/U_pseudo_observations.jsonl.gz',
            'readouts/'+rid+'/U_diagnostic/scores.json', 'readouts/'+rid+'/predictions/complete.json',
            'readouts/'+rid+'/scores/complete.json']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
