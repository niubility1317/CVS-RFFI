"""Create a minimal, exact four-row recovery registration."""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from experiments.pamf_repl_weight import register as base
from . import plan


def make(template):
    old = base.RUN, base.PREVIOUS_RUN, base.RECIPES, base.candidate_rows
    base.RUN, base.PREVIOUS_RUN, base.RECIPES, base.candidate_rows = (
        plan.RUN, plan.PREVIOUS_RUN, plan.RECIPES, plan.candidate_rows)
    try:
        result = base.make(template)
    finally:
        base.RUN, base.PREVIOUS_RUN, base.RECIPES, base.candidate_rows = old
    result.update(display_name='六源浅层与外推因子矩阵技术补跑',
                  description='补齐中断矩阵中六源权重3的四行；复用两条已冻结E50，另两条从相同源E20按原配方训练。',
                  replaces_run_id='20261117-phase1-pamf-repl-factorial-recovery-clean-m4-r01',
                  authorization='2026-11-17 用户要求补齐失败实验，只测clean。')
    result['checkpoint'].update(initialization='two source-only E50 reused; failed and queued rows retrain from same source-only E20',
        contract_check_ref='factorial_recovery.runner.prepare and component_guard.plan.parent',
        selection_rule='fixed four original missing rows; no target selection or parameter changes')
    result['execution'].update(gpu_policy='GPU 0,2,4,6; all owners and pending workers <=5/GPU, >=12 GiB free, one own worker/GPU',
        local_artifact_root='local_artifacts/pamf_repl_factorial_recovery_20261117',
        launch_command='python -m experiments.pamf_repl_factorial_recovery.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['data']['roles']['U_s'] = 'TX blind during training; independent post-freeze U truth diagnostic only'
    result['expected_artifacts'] = ['4 clean predictions and independent truth-last scores',
        '4 independent post-freeze U diagnostic scores', '2 matching original E50 sources reused',
        '2 own E20 to E50 source continuations with blind U observations', 'real E20 no-query smoke']
    result['notes'] = ['Original matrix and all outputs remain immutable; historical target and chosen seed are exposed.',
        'Original failed row OOM occurred during source validation on shared GPU; use an idle GPU and 12 GiB minimum reserve.',
        'Run failed and not-started rows from the original E20 without changing scientific recipe; retain the old failed E30 artifacts.',
        'Source validation does not update model; U truth only independently after E50 freeze; clean only.']
    for entry in result['rows']:
        entry['budget_ref'] = 'same source-only E20; own E21-E50 unless original exact row E50 already frozen'
        entry['expected_artifacts'] = ['source/complete.json or verified original source/complete.json',
            'readouts/'+entry['row_id']+'/U_diagnostic/scores.json',
            'readouts/'+entry['row_id']+'/predictions/complete.json',
            'readouts/'+entry['row_id']+'/scores/complete.json']
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--template', required=True); p.add_argument('--output', required=True)
    a=p.parse_args(); value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
