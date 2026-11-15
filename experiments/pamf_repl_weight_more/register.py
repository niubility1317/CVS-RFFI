"""Create the managed preregistration for additional fixed candidates."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_weight import register as base
from . import plan


def make(template):
    old=base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows
    base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=plan.RUN,None,plan.RECIPES,plan.candidate_rows
    try:result=base.make(template)
    finally:base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=old
    root='PROJECT/runs/'+plan.RUN
    result.update(display_name='Tri-SIG-TIME 额外权重和机制 clean 对照',
        description='十类、六源各8个额外单因素候选，E20各自续训30轮，训练后自动clean测试。',
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03'],replaces_run_id=None,
        authorization='2026-11-15 用户要求多发布几个实验，保留原有运行候选、每GPU最多5个进程。')
    result['checkpoint'].update(initialization='matching original same-setting original Tri E20 then own E10+E20',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='experiments.pamf_repl_weight_more.plan.parent and checked_e20_parent',
        provenance_verdict='same source roles, physical IDs and original source-only E20, no target-trained checkpoint')
    result['execution'].update(gpu_policy='GPU2 and GPU6 only, disjoint from running r02 owner; total <=5 processes/card including other owners and pending workers; check free memory',
        local_artifact_root='local_artifacts/pamf_repl_more_clean_20261115',
        launch_command='python -m experiments.pamf_repl_weight_more.publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    result['expected_artifacts']=['16 real-checkpoint no-query smoke checks','16 E30/E50 source continuations',
        'source V high-confidence prediction precision proxies at E30 and E50',
        'all-step group L/U gradient and 100-step component gradient diagnostics',
        '16 raw clean predictions and independent truth-last scores with matched original PL and Tri controls']
    for entry,row in zip(result['rows'],plan.candidate_rows()):
        rid=row['row_id']
        entry['method']='Tri-SIG-TIME REPL additional '+row['recipe']
        entry['budget_ref']='original same-contract E20 reused; own E21-E50 6660 updates; total 11100'
        entry['expected_artifacts']=['source/complete.json','source/segment30/source_v_pseudo_proxy.json',
            'source/segment50/source_v_pseudo_proxy.json','readouts/'+rid+'/predictions/complete.json',
            'readouts/'+rid+'/scores/complete.json']
    result['notes']=[
        'Seed selected from previously exposed target clean; exploratory only. Candidate matrix fixed before new target tests.',
        'U TX truth is never read; source V confidence precision is a proxy, not U pseudo-label correctness.',
        'Eight one-factor candidates per setting supplement the unchanged previous owner; all use E20 then 30 rounds.',
        'KL factor affects only ordinary/raw paired KL, including the ordinary mixed subbatch; CE unchanged.',
        'All-step gradient telemetry plus 100-step loss-component gradients; raw clean only; independently truth-last scoring.',
        'No change, stop, restart or overwrite of other experiment owners.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
