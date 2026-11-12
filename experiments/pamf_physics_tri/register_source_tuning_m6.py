"""Registry entry for the six-source-receiver source-development sweep."""
import argparse
import json
from pathlib import Path

from . import register_source_tuning as base
from .source_tuning_m6 import RUN, configured, rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, rows
    try:
        with configured():
            result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    result.update(display_name='6 个源接收机 PAMF 分支原型五温度源域调优',
        description='在 6 个源接收机配置下从零训练五种原型温度，只用源域开发集选参。',
        aliases=['PAMF-m6-branch-prototype-temperature-sweep'],
        authorization='2026-11-12 用户明确要求启动 6 个源接收机调优，增加并发并确认显存充足。')
    result['data'].update(dataset='ManySig six-source-receiver six-class, previously validated original roles',
        contract_ref='context.handles.m6-natural.source_contract', label_map_ref='source contract classes',
        source_receivers=[1, 3, 4, 6, 8, 11], target_receivers=[0, 2, 5, 7, 9, 10],
        tx_sets_ref='context.handles.m6-natural.classes',
        roles=dict(L_s='7560 original; deterministic per-class split 6048 train / 1512 dev',
                   U_s='68040 source U; true TX hidden', V='32400 source V; health diagnostics only'))
    result['permissions']['claim_scope'] = 'Source-domain pilot; previously exposed targets; no target performance claim'
    result['execution'].update(gpu_policy='GPUs 2,6,7,1,3; at most one this-run worker and two total GPU processes per card, counting other owners; check actual free memory',
        local_artifact_root='local_artifacts/pamf_source_tuning_m6_20261112',
        launch_command='python -m experiments.pamf_physics_tri.source_tuning_m6_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', 'five scratch E50 source models',
        'five held-out source L clean and simulated-physical results', 'fixed source-only selection and promotion decision']
    result['metrics_plan']['dimensions'] = ['6 source receivers', '6 classes', 'seed 2026092701',
                                             'five temperatures', 'clean and fixed source simulated practical']
    for row in result['rows']:
        row['data_overrides'] = {'protocol_id': 'm6-natural'}
    result['notes'].insert(0, 'Fixed temperatures: 0.05, 0.1, 0.2, 0.4 and 0.8; use the original 0.2 as source baseline.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
