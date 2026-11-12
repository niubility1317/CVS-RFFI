"""Registry entry for the parallel 1%-labeled source-development pilot."""
import argparse
import json
from pathlib import Path

from . import register_source_tuning as base
from .source_tuning_label01 import RUN, configured, rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, rows
    try:
        with configured():
            result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    result.update(display_name='1% 标签 PAMF 分支原型温度源域内部开发试验',
                  description='在 1% 标签配置从零训练三组相同划分的 Hybrid＋分支原型，仅在源 L 内部开发集选温度。',
                  aliases=['PAMF-label01-branch-prototype-source-temperature-pilot'],
                  authorization='2026-11-12 用户要求使用更多显卡并行发布不同参数实验。')
    result['data'].update(dataset='ManySig 1%-labeled six-class, previously validated original roles',
        contract_ref='context.handles.label01.source_contract', label_map_ref='source contract classes',
        source_receivers=[1, 3, 4, 6, 8], target_receivers=[0, 2, 5, 7, 9, 10, 11],
        tx_sets_ref='context.handles.label01.classes',
        roles=dict(L_s='630 original; deterministic per-class split 504 train / 126 dev',
                   U_s='62370 source U; true TX hidden', V='27000 source V; health diagnostics only'))
    result['permissions']['claim_scope'] = 'Source-domain pilot; previously exposed targets; no target performance claim'
    result['execution'].update(gpu_policy='GPUs 4,5,7; one total training or prediction process per GPU; check free memory',
        local_artifact_root='local_artifacts/pamf_source_tuning_label01_20261112',
        launch_command='python -m experiments.pamf_physics_tri.source_tuning_label01_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    result['metrics_plan']['dimensions'] = ['6 classes', '1% labels', 'seed 2026092701',
                                             'three temperatures', 'clean and fixed source simulated practical']
    for row in result['rows']:
        row['data_overrides'] = {'protocol_id': 'label01'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
