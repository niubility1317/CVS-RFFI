"""Preregister the fixed real-graph U loss decomposition."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_weight import register_source_adamw as base
from .publish_source_components import RUN


def make(template):
    old = base.RUN
    base.RUN = RUN
    try:
        result = base.make(template)
    finally:
        base.RUN = old
    result.update(display_name='Tri 无标签损失分项源域梯度诊断',
        description='十类与六源原始 E20 各三批，逐组比较 U 五个损失分项与 L CE、完整 L 及完整 U。', replaces_run_id=None)
    result['execution'].update(local_artifact_root='local_artifacts/pamf_tri_source_components_20261116',
        launch_command='python -m experiments.pamf_repl_attribution_factorial.source_component_diagnostic EXISTING_RUNTIME_JSON')
    result['expected_artifacts']=['six fixed E20 source batches including original no-query checks',
        'actual pseudo CE, pair KL, shallow and joint contrast, extrapolation-increment gradients',
        'per-group gradient closure, norms, dots and cosines versus labeled CE, full L and full U']
    result['metrics_plan']['metric_names']=['component losses and per-parameter-group L2 gradient closure',
        'component gradient norms and dots/cosines versus labeled CE, full L and full U']
    for row in result['rows']:
        row.update(row_id='both_settings-source-component-diagnostic-e20-s2026092701',
                   purpose='attribute the full U gradient to the actual five original U components',
                   budget_ref='6 source batches and disposable original source optimizer shadows',
                   command='source_component_diagnostic with both fixed settings')
    result['notes']=['No target, source V or U true labels are read; no training state is saved.',
        'Five components reuse the exact current graph; check the component gradient sum against the native U gradient.',
        'Extrapolation is new-minus-replaced ordinary CE, not the whole extrapolated CE.',
        'Three source batches per setting at E20; no inference about later trajectory or target performance.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
