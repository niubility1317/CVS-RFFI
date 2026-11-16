"""Preregister the supervised-by-unlabeled source loss comparison."""
import argparse
import json
from pathlib import Path
from . import register_source_components as base
from .publish_source_cross_loss import RUN


def make(template):
    old = base.RUN
    base.RUN = RUN
    try:
        result = base.make(template)
    finally:
        base.RUN = old
    result.update(display_name='Tri 监督与无标签损失分项交叉梯度诊断',
        description='十类和六源 E20 各三批，比较 L 三项与 U 五项的源域梯度。')
    result['execution'].update(local_artifact_root='local_artifacts/pamf_tri_source_cross_loss_20261116',
        launch_command='python -m experiments.pamf_repl_attribution_factorial.source_cross_loss_diagnostic EXISTING_RUNTIME_JSON')
    result['expected_artifacts']=['six source E20 batches with no target, V or U truth access',
        'supervised CE, supervised KL and labeled residual versus five original U components, per parameter group',
        'parameter and group gradient-sum closure for the complete L and U losses']
    result['metrics_plan']['metric_names']=['labeled and U component losses and gradient closure',
        'component-to-component per-group norms, dots and cosines']
    for row in result['rows']:
        row.update(row_id='both_settings-source-cross-loss-e20-s2026092701',
                   purpose='separate supervised CE, KL and residual from the five original U components',
                   command='source_cross_loss_diagnostic with both fixed settings')
    result['notes']=['No target, V or U true labels are read; no training state is saved.',
        'F21 exactly four labeled pair calls; As pair KL contributes zero; L residual is not assumed to be exclusively contrast.',
        'The original five U components include extrapolation new-minus-replaced CE, not the entire extrapolated CE.',
        'Three source batches per setting at E20 are local mechanism observations, not independent model seeds or target causal evidence.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
