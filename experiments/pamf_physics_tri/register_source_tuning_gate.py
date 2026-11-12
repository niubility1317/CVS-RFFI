"""Registry entry for the queued three-vote source-development pilot."""
import argparse
import json
from pathlib import Path

from . import register_source_tuning as base
from .source_tuning_gate import RUN, BASELINE_RUN, BASELINE_ROW, configured, rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, rows
    try:
        with configured():
            result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    result.update(display_name='10类 PAMF 分支筛选三票门槛源域对照',
        description='10类三票分支筛选对比原四票；复用同划分同种子的本轮 E20，从第 21 轮独立续训。',
        aliases=['PAMF-ten-class-branch-gate-three-votes'], parent_run_ids=[BASELINE_RUN],
        authorization='2026-11-12 用户要求并行发布不同参数实验，并减少不必要的额外流程。')
    result['checkpoint'].update(initialization='matching development-only scratch E20 plus own E30/E50',
        sources=[BASELINE_RUN + '/F/' + BASELINE_ROW + '/source/segment20'],
        contract_check_ref='source_tuning_gate.parent and source_development.verify_parent_partition',
        provenance_verdict='own current source-development scratch E20, same seed, source roles and exact L split',
        selection_rule='fixed final E50; source L_dev clean and physical accuracy equally weighted')
    result['execution'].update(gpu_policy='wait until matching source baseline finishes, then GPUs 0,1,3; one total worker per GPU',
        local_artifact_root='local_artifacts/pamf_source_tuning_gate3_20261112',
        launch_command='python -m experiments.pamf_physics_tri.source_tuning_gate_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Baseline failure stops queue; technical failure stops new dispatch; preserve outputs and other owners')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', 'one matched-E20 continuation to E50',
        'held-out source L clean and simulated-physical result', 'fixed comparison to four-vote source baseline']
    result['metrics_plan']['dimensions'] = ['10 classes', 'seed 2026092701', 'three versus four votes',
                                            'clean and fixed source simulated practical']
    row = result['rows'][0]
    row.update(method='Hybrid plus branch prototype, three-of-five certificate; prototype temperature 0.2',
        purpose='source-only matched gate falsification', budget_ref='matched scratch E20 reused; own 10 + 20 epochs, 6660 new updates',
        expected_artifacts=['source/complete.json', 'source_scores/' + row['row_id'] + '/complete.json'])
    result['notes'] = ['One source-only candidate; original four-vote .2 candidate is the matched baseline.',
        'The controller waits for the already-running source baseline to complete before admitting any GPU job.',
        'Target tests do not select or tune this candidate. Historical target data are previously exposed.',
        'At least 0.25 percentage point improvement in the equal clean/physical source mean is required for a separate multi-seed stage.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
