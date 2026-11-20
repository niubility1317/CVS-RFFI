"""Preregister the six source-only E50 fold jobs."""
import argparse
from copy import deepcopy

from experiments.cric_dura_comparison.io import read, write
from . import joint_holdout as plan, pilot


def make(template, context):
    data = deepcopy(template)
    root = f'PROJECT/runs/{plan.RUN}'
    cmd = f'python -m {plan.MODULE} publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    data.update(run_id=plan.RUN, group_id='phase1-pamf-source-adaptation',
                display_name='Tri 1% 与 5% 标注的 E50 源域 U 权重选择',
                description='两个数据设置、三折、每折三档 U，源 RX 与日期联合留出。',
                parent_run_ids=[pilot.paper.RUN, pilot.RUN, pilot.SECOND_RUN], replaces_run_id=None,
                authorization='用户确认开展只用源域数据的 U 权重自适应，先检验低标注设置。')
    data['code'].update(commit='actual runtime release commit', checkout='current task worktree',
                        environment='N607 CVS-RFFI', cwd=f'PROJECT/releases/{plan.RUN}-release')
    data['data'].update(dataset='Existing 1% and 5% Phase1 source contracts only',
        contract_ref='Existing source role files restricted within predeclared receiver/date folds',
        physical_ids_ref='Paired predictions on fixed held source V physical samples',
        roles={'L_s':'labeled source excluding full holdout receiver and date',
               'U_s':'TX-blind source excluding full holdout receiver and date',
               'V':'fit-time validation excludes both axes; score original held V after all fold arms freeze'},
        support_query_ref='Phase1 source only, no target query')
    data['permissions'].update(query_use='none', claim_scope='Source-only rule test at E50, no target performance claim')
    data['checkpoint'].update(
        initialization='scratch E20 once per fold, then own E21–E30 and E31–E50 for each arm',
        sources=[pilot.paper.RUN],
        contract_check_ref='joint_holdout.registered and fold-bound training loader',
        provenance_verdict='Full-contract E20 is used only for no-query smoke, never inherited by folds',
        selection_rule='At least two positive folds, mean combined gain >=0.1pp, mean joint RX+date gain >=0')
    data['execution'].update(host='N607 ordinary account', launch_owner=f'codex-{plan.RUN}', owner=f'codex-{plan.RUN}',
        gpu_policy='GPU 0,4,5,6; total two GPU compute processes maximum; 12 GiB admission reserve',
        remote_run_root=root, remote_log_root=root+'/logs', local_artifact_root='local_artifacts/pamf_tri_source_dose_e50_20261120',
        launch_command=cmd, stop_rule='Stop new dispatch on blocking technical failure and preserve existing outputs')
    data['metrics_plan'].update(metric_names=['source paired gain and damage', 'source cross_rx', 'source cross_both', 'dose selected'],
        dimensions=['source receiver','source date','data setting','U dose'],
        prediction_ref='Frozen source V predictions paired in memory', scorer_ref=root+'/analysis/source_decisions.json')
    data['notes'] = ['Both label settings use the original Tri objective at all three doses.',
        'Low label coverage changes candidate evaluation order, not the eligibility rule.',
        'Decisions do not alter the separately frozen 11-setting plate.']
    data['expected_artifacts'] = ['six scratch E20 prefixes','18 E50 source models','six fold scores','two frozen source decisions']
    data['rows'] = []
    for setting in plan.SETTINGS:
        pid = pilot.paper.PROTOCOLS[setting][0]
        for name, split in plan.folds(context['protocols'][pid]['source_rxs'], (1,2,3)).items():
            jid = f'train__{setting}-{name}'
            entry = deepcopy(template['rows'][0])
            entry.update(row_id=jid, method='Original Tri with U dose 6/4.5/3', purpose='Held source receiver and date',
                config_ref=f'{root}/configs/{jid}.json', resolved_config_ref=f'{root}/configs/{jid}.json',
                data_overrides=dict(protocol_id=pid, excluded_cells=split['excluded'], scored_cells=split['scored'],
                                    joint_date=split['date'], weights=plan.WEIGHTS[setting]),
                epochs=50, budget_ref='4440 scratch steps then 2220+4440 for each matched arm',
                scenario='held source V, same data each arm', output_root=f'{root}/F/{setting}/{name}',
                log_path=f'{root}/logs/{jid}.log', command=cmd,
                expected_artifacts=['segment20/complete.json','three fixed E50 arm checkpoints','complete.json'])
            entry['seeds'].update(model=pilot.SEED, split=None, data=None)
            data['rows'].append(entry)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('template','context','output'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args()
    write(a.output, make(read(a.template),read(a.context)))
