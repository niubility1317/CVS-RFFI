"""Build the four-row preregistration without changing existing owners."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import RUN, candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri 实际更新保护与 U 强度因子对照',
        description='十类和六源各测试权重 3 与梯度限幅 0.5，两者均保留完整 L 实际更新保护。')
    result['execution'].update(gpu_policy='GPU 2,3; all owners and pending workers count toward five processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_step_factor_20261116',
        launch_command='python -m experiments.pamf_repl_step_factor.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['four actual E20 no-query optimizer-step smokes',
        'four own E30/E50 continuations with all-step gradient and AdamW step-guard telemetry',
        'four blind U observation trajectories and independent post-freeze U diagnostic scores',
        'four raw clean predictions and independent truth-last scores']
    result['notes']=['Both settings and seed fixed in advance; each factor is paired with its existing non-step-guard counterpart.',
        'All rows protect the actual AdamW increment against full-L using the current segment optimizer state.',
        'Source E21-E30 data showed frequent t2 norm capping; factor values halve the U weight or group norm cap.',
        'Historical target scores and the seed were exposed; exploratory only; no target feedback into the candidates.',
        'U true labels only in separate independent post-E50 scoring; no new practical or weak tests.']
    for entry, row in zip(result['rows'], candidate_rows()):
        entry['method'] = 'Tri AdamW full-L step guard plus ' + row['recipe']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
