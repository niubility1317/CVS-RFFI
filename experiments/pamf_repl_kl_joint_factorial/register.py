"""Preregister every U-KL and fusion-contrast factorial cell."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_component_guard import register as original
from .plan import candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri U 配对 KL、融合对比与总权重因子归因',
        description='十类、六源各按 U 权重、融合对比和 U 配对 KL 组成 2×2×2 完整矩阵。')
    result['data']['roles'] = {'L_s':'labeled training', 'U_s':'TX blind during training; frozen offline diagnostic only', 'V':'source validation'}
    result['execution'].update(gpu_policy='GPU2,3,4 round-robin; all owners and pending workers count toward three processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_kl_joint_factorial_20261116',
        launch_command='python -m experiments.pamf_repl_kl_joint_factorial.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['16 real E20 checkpoint no-query smokes',
        '16 own E30/E50 source continuations with all-step group gradients and sampled component gradients',
        '16 blind U trajectories and independent post-freeze U diagnostic scores',
        '16 clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names']=['clean accuracy and macro-F1', 'L CE and source validation',
        'U pseudo acceptance, unique exposure coverage and independently scored correctness',
        'group L/U gradients before and after native guard',
        'sampled shallow, joint and extrapolation-increment gradients versus complete L']
    result['notes']=['Full 2×2×2 matrix per setting: U weight 6/3, U joint contrast .05/0, U ordinary/mixed/raw paired KL 1/0.',
        'The U KL factor affects paired KL only; pseudo-label CE and all labeled losses are unchanged.',
        'Source-only E20 cross-component gradient diagnosis motivates the factors; historical target scores and seed were already exposed.',
        'All 16 cells fixed before their own tests; exploratory, not independent confirmation.',
        'Same source-only E20 per setting and own E30/E50; original native U gradient guard; clean only.',
        'U truth is scored only by an independent program after E50 freeze and is not used for training or selection.']
    for entry, row in zip(result['rows'],candidate_rows()):
        rid=row['row_id']
        entry['method']='Tri original U guard KL/joint factorial '+row['recipe']
        entry['expected_artifacts']=['source/complete.json','source/segment30/U_pseudo_observations.jsonl.gz',
            'source/segment50/U_pseudo_observations.jsonl.gz','readouts/'+rid+'/U_diagnostic/scores.json',
            'readouts/'+rid+'/predictions/complete.json','readouts/'+rid+'/scores/complete.json']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--template',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    template=json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
