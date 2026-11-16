"""Preregister the source-diagnostic-motivated dual-anchor comparison."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_step_guard import register as original
from .plan import candidate_rows, scoped


def make(template):
    with scoped():
        result = original.make(template)
    result.update(display_name='Tri 完整监督目标与有标签 CE 双锚点保护',
                  description='十类和六源各测试 U 权重 6 与 3，保护完整监督目标及有标签 CE。')
    result['execution'].update(gpu_policy='GPU4; all owners and pending workers count toward five processes per GPU',
        local_artifact_root='local_artifacts/pamf_repl_dual_ce_20261116',
        launch_command='python -m experiments.pamf_repl_dual_ce.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['four actual E20 no-query optimizer-step smokes with both anchors',
        'four own E30/E50 continuations with all-step gradient and dual-anchor telemetry',
        'four blind U observation trajectories and independent post-freeze U diagnostic scores',
        'four raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names'].append('ordinary labeled CE versus actual extra AdamW update by parameter group')
    result['notes']=['Source-only E20 diagnostic motivates protecting both full L and the existing ordinary labeled CE; no U truth or target data were read for that diagnostic.',
        'Original U weights 6 and 3 fixed for both settings; full L and CE anchors protect the actual extra step relative to L-only at the current optimizer state.',
        'The CE anchor is logged as L_clean and requires one extra autograd call; numerical fallback and zero update are recorded.',
        'Historical target scores and seed were exposed; results exploratory; no target or U truth feedback into candidates or selection.',
        'U truth is read only by an independent diagnostic after E50 freeze; no practical or weak tests.']
    for entry, row in zip(result['rows'], candidate_rows()):
        entry['method'] = 'Tri AdamW full-L and ordinary labeled CE dual-anchor guard plus ' + row['recipe']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
