"""Minimal record for the fixed source-only CE anchor diagnostic."""
import argparse
import json
from pathlib import Path
from . import register_source_adamw as base
from .publish_source_ce_anchor import RUN


def make(template):
    old = base.RUN
    base.RUN = RUN
    try:
        result = base.make(template)
    finally:
        base.RUN = old
    result.update(display_name='Tri 有标签 CE 与完整监督锚点源域诊断',
        description='十类和六源原始 E20 各三批，测量 CE 与完整 L、U 梯度及 fresh-moment AdamW 实际增量关系。',
        replaces_run_id=None)
    result['execution'].update(local_artifact_root='local_artifacts/pamf_tri_ce_anchor_20261116',
        launch_command='python -m experiments.pamf_repl_weight.source_ce_anchor EXISTING_RUNTIME_JSON')
    result['expected_artifacts']=['two original E20 checkpoint no-query source smokes',
        'six source batches, six extra CE derivatives and 24 disposable fresh-moment AdamW shadow steps',
        'CE vs complete L, raw/protected U and actual extra descent per parameter group']
    result['metrics_plan']['metric_names']=['source labeled CE and complete L gradient norms and cosine',
        'CE versus raw and protected U gradients', 'CE versus actual extra first-step AdamW descent']
    for row in result['rows']:
        row.update(row_id='both_settings-source-ce-anchor-e20-s2026092701',
                   purpose='compare existing labeled CE and complete L to protected U and actual AdamW first-step increment',
                   budget_ref='6 disposable source batches, 6 extra CE gradients, 24 disposable AdamW first steps',
                   command='source_ce_anchor with both fixed settings')
    result['notes']=['No target, source V or U TX truth is read; no training state is persisted.',
        'Six source snapshots share two fixed E20 models; not independent seeds or a full optimizer trajectory.',
        'All optimizer shadows use fresh moments; local directional observations cannot prove test performance.',
        'The historically target-exposed seed is used only for the fixed source diagnostic.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    template = json.loads(Path(args.template).read_text(encoding='utf-8-sig'))
    Path(args.output).write_text(json.dumps(make(template), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
