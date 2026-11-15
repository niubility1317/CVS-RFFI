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
    result.update(display_name='Tri-SIG-TIME OOM 后未完成矩阵恢复',
        description='保留原矩阵已完成6行，恢复其余10行：5个冻结E50复用，1个自身E30续训，4个从E20启动。',
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03',plan.PREVIOUS_RUN],replaces_run_id=None,
        authorization='2026-11-15 用户要求多发布几个实验，保留原有运行候选、每GPU最多5个进程。')
    result['checkpoint'].update(initialization='5 own frozen E50 reused; 1 failed row own E30 to E50; 4 unstarted rows original same-setting Tri E20 to E50',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03', plan.PREVIOUS_RUN],
        contract_check_ref='experiments.pamf_repl_more_recovery.plan parent, reused_e50 and recovered_e30',
        provenance_verdict='same row, source roles, physical IDs, source-only E20 and own E30/E50; no target-trained checkpoint')
    result['execution'].update(gpu_policy='GPU0/1/3/4, at most 2 own workers/card; total <=5 including other owners and pending workers; require 10000 MiB free including pending reservations',
        local_artifact_root='local_artifacts/pamf_repl_more_recovery_20261115',
        launch_command='python -m experiments.pamf_repl_more_recovery.publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    result['expected_artifacts']=['10 real-checkpoint no-query smoke checks','5 E50 sources reused, 1 E30 recovery and 4 E20 continuations',
        'source V high-confidence prediction precision proxies at E30 and E50',
        'all-step group L/U gradient and 100-step component gradient diagnostics',
        '10 raw clean predictions and independent truth-last scores with matched original PL and Tri controls']
    for entry,row in zip(result['rows'],plan.candidate_rows()):
        rid=row['row_id']
        entry['method']='Tri-SIG-TIME REPL additional '+row['recipe']
        if rid in plan.REUSE_E50:
            source='PROJECT/runs/'+plan.PREVIOUS_RUN+'/F/'+rid+'/source'
            entry.update(config_ref=root+'/configs/predict__'+rid+'.json', resolved_config_ref=source+'/resolved_config.json',
                         output_root=source, log_path='PROJECT/runs/'+plan.PREVIOUS_RUN+'/logs/train__'+rid+'.log',
                         command='worker --kind predict --config '+root+'/configs/predict__'+rid+'.json',
                         budget_ref='own frozen E50 reused; zero additional training updates')
            source_artifacts=[source+'/complete.json',source+'/segment30/source_v_pseudo_proxy.json',source+'/segment50/source_v_pseudo_proxy.json']
        elif rid == plan.RECOVER_E30:
            entry['budget_ref']='reuse own completed E30; redo interrupted E31–E50, total trajectory 11100 updates'
            source_artifacts=['source/complete.json','source/recovered_e30_v/source_v_pseudo_proxy.json','source/segment50/source_v_pseudo_proxy.json']
        else:
            entry['budget_ref']='same-contract original E20 plus new E21–E50; total 11100 updates'
            source_artifacts=['source/complete.json','source/segment30/source_v_pseudo_proxy.json','source/segment50/source_v_pseudo_proxy.json']
        entry['expected_artifacts']=source_artifacts+['readouts/'+rid+'/predictions/complete.json',
                                                     'readouts/'+rid+'/scores/complete.json']
    result['notes']=[
        'Seed selected from previously exposed target clean; exploratory only. Candidate matrix fixed before new target tests.',
        'U TX truth is never read; source V confidence precision is a proxy, not U pseudo-label correctness.',
        'Fixed technical recovery only; no candidate added, changed, or selected by target score. Prior owner had one OOM in source validation, and 5 healthy rows reach E50.',
        'Failed row resumes own E30; partial E31-E47 trajectory is retained but cannot supply an E50 checkpoint; no result-informed retry.',
        'All-step gradient telemetry plus 100-step loss-component gradients; raw clean only; independently truth-last scoring.',
        'Prior owner finishes its healthy workers before recovery; no stop or overwrite of other owners or outputs.']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
