"""Minimal preregistration, based only on source-side gradient observations."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from experiments.pamf_physics_branch.plan import RUN as PREVIOUS
from .plan import RUN, candidate_rows


def make(template):
    raw = json.dumps(deepcopy(template), ensure_ascii=False)
    raw = raw.replace(PREVIOUS, RUN).replace('ptbr-beta05-s2026092701', 'ptbr-refbudget-s2026092701')
    s = json.loads(raw)
    s['rows'] = [s['rows'][1]]
    s.update(run_id=RUN, group_id='phase1-pamf-ptbr-reference-budget',
             display_name='PAMF 物理分支参考响应梯度预算单种子对照',
             description='由源验证集固定模型的参考分支局部梯度诊断提出，仅缩小参考关系项权重。',
             aliases=['PAMF-PTBR-refbudget-pilot'], registered_at=datetime.now(timezone.utc).isoformat())
    s['parent_run_ids'].append(PREVIOUS)
    s['code']['cwd'] = 'PROJECT/releases/' + RUN + '-release'
    e = s['execution']
    e.update(launch_owner='codex-pamf-refbudget-20261110',
             gpu_policy='GPU 0/1; matched PTBR owner must be ANALYZED with no running jobs; at most two total observed/pending workers per card',
             local_artifact_root='local_artifacts/pamf_physics_branch_refbudget_20261110')
    e['launch_command'] = e['launch_command'].replace('experiments.pamf_physics_branch.runner',
                                                       'experiments.pamf_physics_branch_balanced.runner')
    r = s['rows'][0]
    r.update(physical_row=candidate_rows()[0], method='PTBR beta=0.5 + reference relation scale 0.1',
             purpose='matched beta=0.5 reference-budget effect; same exposure and data')
    s['expected_artifacts'] = ['one-row real E20 no-query finite-backward smoke',
        'own E30 and E50 source trajectories and identical L/U physical exposure counts',
        'full source L/V physical diagnostics and stratified original As',
        'all practical and weak predictions frozen before independent truth-last scoring',
        'same-row comparison to beta=0.5 and ordinary PL', 'PID/CWD/GPU/log readback']
    s['notes'] = [
        'Source-only fixed-F21 eval gradient probe: beta=.5 reference L+U relation norm .6353 vs direct CE+PL .0828; different train-mode batch geometry remains unverified.',
        'Only reference branch relation term is multiplied by .1 before the original five-branch mean; CE, PL, sampling, physical proxy, train budget and beta=.5 unchanged.',
        'Reference scale is a source-motivated fixed probe, not a calibrated physical truth or a performance guarantee; match beta=.5 and ordinary PL.',
        'Source V screen before expansion: final V no worse than original PL by .2 pp and matched beta=.5 by .1 pp; report anchors and nonfinite counts; never cancel preregistered test.',
        'Candidate fixed after earlier F0 benchmark exposure, but before matched beta=.5 target scoring; not a pristine unseen-data blind test.',
        'Wait until matched PTBR owner is ANALYZED; predict all fixed views before truth-last scoring; no target-based tuning, selection or retry.']
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
