"""Freeze the source-only adaptive candidate before fixed-budget target results."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from experiments.pamf_physics_branch_balanced.plan import RUN as PREVIOUS
from .plan import RUN, candidate_rows


def make(template):
    raw = json.dumps(deepcopy(template), ensure_ascii=False)
    raw = raw.replace(PREVIOUS, RUN).replace('ptbr-refbudget-s2026092701', 'ptbr-adaptive-s2026092701')
    s = json.loads(raw)
    s.update(run_id=RUN, group_id='phase1-pamf-ptbr-adaptive-reference-budget',
             display_name='PAMF 物理分支自适应参考梯度预算单种子对照',
             description='当前源训练批次的参考参数梯度范数决定有界物理关系剂量。',
             aliases=['PAMF-PTBR-adaptive-pilot'], registered_at=datetime.now(timezone.utc).isoformat())
    s['parent_run_ids'].append(PREVIOUS)
    s['code']['cwd'] = 'PROJECT/releases/' + RUN + '-release'
    e = s['execution']
    e.update(launch_owner='codex-pamf-adaptive-20261110',
             gpu_policy='GPU 0/1; fixed reference-budget owner must be terminal with matched scores available; max two total observed/pending workers per card',
             local_artifact_root='local_artifacts/pamf_physics_branch_adaptive_20261110',
             stop_rule='Systemic training/prediction/scoring failure stops new dispatch; isolated source diagnostic failure is recorded and does not cancel frozen-model predictions; preserve healthy workers and artifacts')
    e['launch_command'] = e['launch_command'].replace('experiments.pamf_physics_branch_balanced.runner',
                                                       'experiments.pamf_physics_branch_adaptive.runner')
    r = s['rows'][0]
    r.update(physical_row=candidate_rows()[0], method='PTBR beta=0.5 + detached adaptive reference gradient budget',
             purpose='compare source-batch adaptive, equal and fixed reference relation budgets',
             budget_ref='reused E20/4440; own E10+E20/6660; same L64/U<=128 physical directions; two extra reference-gradient evaluations and reference pair-distance recomputation each step')
    s['expected_artifacts'] = ['one-row real E20 no-query finite-backward and adaptive-budget smoke',
        'own E30 and E50 source trajectories, reference scale statistics, L/U physical exposure counts and resource costs',
        'full source L/V physical diagnostics and stratified original As',
        'all practical and weak predictions frozen before independent truth-last scoring',
        'same-row comparison to equal beta=0.5, fixed reference 0.1 and ordinary PL', 'PID/CWD/GPU/log readback']
    s['notes'] = [
        'Base gradient uses original supervised and ordinary PL losses from the current source batch; relation gradient uses the physical reference L/U terms after the registered five-branch mean.',
        'Scale=clip(||base||/(||relation||+1e-12),.1,1); zero relation gives 1. Detached scalar, no V initialization, no target or U labels; other branches and all base losses unchanged.',
        'The .1 floor means the reference relation is not guaranteed below the base norm; no Adam-update or identity-preservation guarantee.',
        'Same source exposures and optimization steps as equal beta=.5; two extra reference-parameter autograd evaluations and reference distance recomputation each update, so training compute is not matched.',
        'Fixed prior to reference-budget target scoring. Historical F0 benchmark already exposed; report limitations and no target-driven selection or retries.',
        'Before expansion, final V no worse than original PL by .2 pp and equal beta=.5 by .1 pp; report nonfinite skips, anchors, scale and source diagnostics; do not cancel fixed pilot tests.',
        'Wait until fixed-budget owner is terminal with matched scores available; all candidate predictions precede truth-last scoring. Source diagnostics run but do not gate fixed candidate predictions.']
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--template', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
