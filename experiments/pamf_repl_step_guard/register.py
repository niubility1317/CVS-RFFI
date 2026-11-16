"""Register full-L optimizer-step protection and offline U diagnosis."""
import argparse,json
from pathlib import Path
from experiments.pamf_repl_weight import register as base
from . import plan


def make(template):
    old=base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows
    base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=plan.RUN,None,plan.RECIPES,plan.candidate_rows
    try:result=base.make(template)
    finally:base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=old
    root='PROJECT/runs/'+plan.RUN
    result.update(display_name='Tri 完整监督目标 AdamW 增量保护 clean 对照',
        description='十类与六源各一行，保护相对当前状态 L-only 的实际 AdamW 增量，并离线统计 U 真值正确率。',
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03'],replaces_run_id=None,
        authorization='用户要求解决监督与无标签冲突，并授权冻结后独立 U 真值离线诊断。')
    result['data']['roles']={'L_s':'labeled training','U_s':'TX blind during training; frozen offline diagnostic only','V':'source validation'}
    result['checkpoint'].update(initialization='original same-contract source-only E20, then own E10 and E20',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='experiments.pamf_repl_step_guard.plan.parent',
        provenance_verdict='original source-only scratch E20, same source roles, physical IDs, seed and graph',
        selection_rule='fixed final E50; all predeclared rows; no test feedback')
    result['execution'].update(gpu_policy='GPU4,5,7; total <=5 processes/card including other owners and pending; reserve memory',
        local_artifact_root='local_artifacts/pamf_repl_step_guard_20261116',
        launch_command='python -m experiments.pamf_repl_step_guard.publish --repo REPO --output LOCAL --context CONTEXT --ssh-config CONFIG',
        stop_rule='Stop dispatch on protocol, nonfinite, launch or systemic technical failure; preserve healthy workers.')
    result['expected_artifacts']=['two actual E20 checkpoint no-query optimizer-step smokes',
        'two own E30/E50 source continuations','all-step AdamW incremental-motion constraint and retention logs',
        'all-exposure blind U pseudo decisions, independent post-freeze U truth accuracy',
        'two frozen clean predictions and independent truth-last scoring']
    result['metrics_plan']['metric_names']=['clean accuracy and macro-F1','L CE and source validation',
        'full-L versus original/applied optimizer extra motion by parameter group','step retention and fallback',
        'U pseudo acceptance and independently scored correctness by epoch and class']
    for entry,row in zip(result['rows'],plan.candidate_rows()):
        rid=row['row_id'];entry['method']='Tri original loss with full-L actual AdamW extra-step guard'
        entry['budget_ref']='original same-contract E20 then own 30 epochs; all steps guarded'
        entry['expected_artifacts']=['source/complete.json','source/segment30/U_pseudo_observations.jsonl.gz',
            'source/segment50/U_pseudo_observations.jsonl.gz','source/segment30/step_guard_metrics.jsonl.gz',
            'source/segment50/step_guard_metrics.jsonl.gz','readouts/'+rid+'/U_diagnostic/scores.json',
            'readouts/'+rid+'/predictions/complete.json','readouts/'+rid+'/scores/complete.json']
    result['notes']=['Original Tri U weight6 and all original mechanisms retained.',
        'Per group project actual L+U parameter step relative to L-only using same current AdamW moments; anchor complete L only.',
        'Original joint AdamW moments persist; guard does not erase historical U moments.',
        'U IDs, pseudo labels and confidence logged without truth; independent scorer derives source U truth only after E50 freeze.',
        'U diagnostic is not used in training or selection; target clean is already exposed and single seed exploratory.',
        'No star-ground test, and no change to other owners.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();t=json.loads(Path(a.template).read_text(encoding='utf-8-sig'))
    Path(a.output).write_text(json.dumps(make(t),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
