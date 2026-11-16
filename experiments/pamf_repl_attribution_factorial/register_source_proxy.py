"""Preregister a diagnostic of whether head CE predicts full-gradient agreement."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_weight import register_source_adamw as base
from .publish_source_proxy import RUN


def make(template):
    old=base.RUN;base.RUN=RUN
    try:result=base.make(template)
    finally:base.RUN=old
    result.update(display_name='Tri 有标签与伪标签分类头梯度代理诊断',
        description='十类与六源原始 E20 各 8 批，检验分类头 CE 代理对完整 L/U 方向的预测能力。',replaces_run_id=None)
    result['execution'].update(local_artifact_root='local_artifacts/pamf_tri_source_ce_proxy_20261116',
        launch_command='python -m experiments.pamf_repl_attribution_factorial.source_proxy_diagnostic EXISTING_RUNTIME_JSON')
    result['permissions']['claim_scope']='16 dependent source snapshots, original E20 parameters and fresh AdamW moments only'
    result['checkpoint']['selection_rule']='both user-requested settings, fixed seed and source offsets 0 through 7'
    result['expected_artifacts']=['16 fixed source E20 batches with real-checkpoint no-query smoke',
        'within-setting 56 ordered cross-batch CE head proxies and full L/U gradient relations per setting',
        'for each L batch the highest and lowest proxy candidates and their full-gradient relations']
    result['metrics_plan']['metric_names']=['ordinary L CE versus native U pseudo-CE classification-head norm, dot and cosine',
        'raw full L versus raw full U per-group norm, dot, cosine and top/bottom proxy comparisons']
    result['metrics_plan']['dimensions']=['ten-class and six-source','eight fixed source batches per setting','t2, cls_head, other']
    for row in result['rows']:
        row.update(row_id='both_settings-source-ce-proxy-e20-s2026092701',
            purpose='screen whether an accessible CE-head proxy predicts full L/U gradient alignment',
            seed_notes='Fixed offsets 0 through 7 use CPU random seeds 2026092701+offset.',
            budget_ref='16 original source batches, disposable CPU gradients and shadow steps',
            command='source_proxy_diagnostic with both fixed settings')
    result['notes']=['No target, source V or U true labels are read; no training state is saved.',
        'Same source-only E20 parameters but independent L128/U256 batches, augmentation and possible forward buffers; cross-batch pairs are dependent.',
        'The U CE component is the native weighted pseudo-CE across ordinary, mixed/As and raw slots; L CE is ordinary identity CE.',
        'This is a screening proxy over entire independently computed U256 batches, not a U64 candidate-pool training intervention; contrast and MixStyle cannot be inferred after recombination.',
        'No later AdamW history, target performance, sample coverage or group-selection benefit is implied.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    Path(a.output).write_text(json.dumps(make(json.loads(Path(a.template).read_text(encoding='utf-8-sig'))),
                                   ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
