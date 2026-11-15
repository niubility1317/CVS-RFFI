"""Build the managed preregistration from the standard registry template."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from .plan import RUN, RECIPES, candidate_rows


def make(template):
    result = deepcopy(template)
    root = 'PROJECT/runs/' + RUN
    result.update(run_id=RUN, group_id='phase1-pamf-repl-weight-clean',
        display_name='Tri-SIG-TIME 十类与六源损失权重及梯度诊断',
        description='单种子十四个独立REPL变化；训练后自动raw clean预测与truth-last评分。',
        aliases=['tri-repl-clean-loss-weight'], tags=['PAMF','REPL','clean','gradient'],
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03'],
        authorization='2026-11-15 用户授权两设置最大clean差距种子调参，clean自动测试，每卡最多五进程。')
    result['code'].update(commit='actual committed release identity in runtime.json',
        checkout='isolated task worktree', environment='N607 CVS-RFFI existing Python, FP32',
        cwd='PROJECT/releases/'+RUN+'-release')
    result['data'].update(dataset='original ten-class ManyRx and six-source receiver contracts',
        roles={'L_s':'original labeled role', 'U_s':'source IQ, TX hidden', 'V':'original source validation'},
        contract_ref='context.handles per row protocol_id', physical_ids_ref='original fixed source/test contracts',
        train_ratio='original fixed labeled/unlabeled/validation roles',
        support_query_ref='fixed clean cross-date, cross-RX and double-cross; prediction before independent scoring',
        leo_config_ref='not applicable: clean-only, no new satellite observations')
    result['permissions'].update(regime='source_only', query_use='frozen raw clean prediction; independent truth-last scoring',
        claim_scope='seed selected by previously exposed clean target; exploratory, not an independent blind evaluation')
    result['checkpoint'].update(initialization='matching original fixed-permutation Tri E20 source checkpoint; own 10+20 continuation',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='experiments.pamf_repl_weight.plan.parent and checked_e20_parent',
        provenance_verdict='same source roles, physical IDs, seed, source-only scratch E20; no target-trained checkpoint',
        selection_rule='all registered candidates final E50; chosen seed based on exposed target; no result feedback into candidates')
    result['execution'].update(host='N607 ordinary user', launch_owner='codex-'+RUN,
        gpu_policy='GPU 0,1,3,4,5,7; total CUDA and pending processes <=5/card; check free memory and reserve pending initialization',
        remote_run_root=root, remote_log_root=root+'/logs', local_artifact_root='local_artifacts/pamf_repl_weight_clean_20261115',
        launch_command='python -m experiments.pamf_repl_weight.publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='On protocol, nonfinite or launch failure stop new dispatch; preserve all outputs and healthy workers; no low-score stops.')
    result['expected_artifacts']=['real checkpoint no-query smoke', '14 frozen E50 continuations with complete step and epoch telemetry',
        'source V high-confidence correctness proxy at E30/E50', '14 raw clean predictions and independent truth-last scores',
        'group pre/post U gradient norms and conflict/cap; sampled component L/U gradients; matched prior PL and Tri comparisons']
    result['metrics_plan'].update(metric_names=['clean accuracy and macro-F1','L CE and source validation',
        'U acceptance and pseudo-class histogram','V high-confidence precision proxy',
        'group L/U norms, dot, conflict and cap, sampled component norms/cosines'],
        dimensions=['ten-class and six-source','seed 2026092701','raw clean','TX/RX/day'],
        prediction_ref=root+'/readouts/ROW/predictions', scorer_ref=root+'/readouts/ROW/scores')
    result['rows'] = []
    for row in candidate_rows():
        rid=row['row_id']; item=deepcopy(template['rows'][0])
        item.update(row_id=rid, method='Tri-SIG-TIME REPL '+row['recipe'], purpose=str(RECIPES[row['recipe']]),
            data_overrides={'protocol_id':row['protocol_id']},
            seeds={'model':row['model_seed'],'split':392005,'data':None,'augmentation':20261016421,'support':None,'evaluation':None},
            seed_notes='Selected using previous target clean performance; exploratory only.',
            scenario='raw clean only', config_ref=root+'/configs/train__'+rid+'.json',
            resolved_config_ref=root+'/F/'+rid+'/source/resolved_config.json',
            budget_ref='same-contract Tri E20 reused; E21-E50 6660 steps; total 11100',
            output_root=root+'/F/'+rid+'/source', log_path=root+'/logs/train__'+rid+'.log',
            command='worker --kind train --config '+root+'/configs/train__'+rid+'.json',
            expected_artifacts=['source/complete.json','source/segment30/source_v_pseudo_proxy.json',
                                'source/segment50/source_v_pseudo_proxy.json','readouts/'+rid+'/predictions/complete.json',
                                'readouts/'+rid+'/scores/complete.json'])
        result['rows'].append(item)
    result['notes']=['No source U TX truth is read; direct source U pseudo-label precision is unavailable.',
        'Full source V high-confidence precision is a proxy, not U truth; no V gradient or persistent training update.',
        'Selected seed 2026092701 has largest matched raw clean deficit in both settings; target exposed.',
        'Each candidate changes one registered factor; no target-driven candidate selection or repeated tuning.',
        'Group gradient telemetry is every step; shallow/joint/extrap_delta versus L component cosines are every 100 steps.']
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--template',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); v=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
