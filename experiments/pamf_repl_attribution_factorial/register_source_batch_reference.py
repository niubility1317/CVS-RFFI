"""Minimal preregistration for source minibatch gradient references."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_weight import register_source_adamw as base
from .publish_source_batch_reference import RUN


def make(template):
    old = base.RUN
    base.RUN = RUN
    try:
        result = base.make(template)
    finally:
        base.RUN = old
    result.update(display_name='Tri 源域批次梯度波动参照',
        description='十类与六源原始 E20 各 12 批，比较 L–L、U–U 及 L–U 梯度方向。', replaces_run_id=None)
    result['execution'].update(local_artifact_root='local_artifacts/pamf_tri_batch_reference_20261116',
        launch_command='python -m experiments.pamf_repl_attribution_factorial.source_batch_reference EXISTING_RUNTIME_JSON')
    result['expected_artifacts']=['24 fixed E20 source batches including original no-query checks',
        'within-setting L-L, U-U, same-batch L-U and cross-batch L-U per-group relations',
        '24 disposable source snapshots with no model or optimizer state used for training']
    result['metrics_plan']['metric_names']=['complete L and original raw U group gradient norm, dot and cosine',
        'L-L, U-U, same-batch L-U and cross-batch L-U distributions']
    for row in result['rows']:
        row.update(row_id='both_settings-source-batch-reference-e20-s2026092701',
                   purpose='distinguish L/U disagreement from source minibatch gradient variation',
                   budget_ref='24 source batches and 48 disposable fresh-moment AdamW shadow steps',
                   command='source_batch_reference with both fixed settings')
    result['notes']=['No target, source V or U true labels are read; no training state is saved.',
        '12 batches per setting share the same E20 parameter checkpoint but differ in batch, augmentation and possible forward buffer state.',
        'Pairwise batch comparisons are dependent and are not 66 independent experiments or multiple model seeds.',
        'The historically target-exposed seed is used for source diagnosis only.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
