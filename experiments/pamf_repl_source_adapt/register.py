"""Record the fixed source-only pilot and the prospective selection rule."""
import argparse
from copy import deepcopy

from experiments.cric_dura_comparison.io import read, write
from .folds import fold_groups
from .pilot import RUN, SECOND_RUN, SEED, WEIGHTS, paper, settings_for


def make(template, context, *, run_id=RUN):
    row_template = template['rows'][0]
    data = deepcopy(template)
    noise_aware = run_id == SECOND_RUN
    root = f'PROJECT/runs/{run_id}'
    launch = 'python -m experiments.pamf_repl_source_adapt.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    if noise_aware:
        launch += ' --round 2'
    data.update(run_id=run_id, group_id='phase1-pamf-source-adaptation',
                display_name='Tri 跨数据设置源留出自适应试点', description='四个设置、每设置两折、每折三剂量，固定 E24 的源域配对。',
                parent_run_ids=[paper.RUN], authorization='用户要求把研发重点转向在其他数据设置保持或增加原 Tri 相对普通 PL 的优势。')
    data['code'].update(commit='actual committed release identity in runtime.json', checkout='current task worktree',
                        environment='N607 existing CVS-RFFI', cwd=f'PROJECT/releases/{run_id}-release')
    data['data'].update(dataset='Original Phase1 source-only roles on four designs', contract_ref='context source per protocol',
                        physical_ids_ref='Original source roles restricted by preregistered held RX/date fold',
                        roles={'L_s': 'train minus held cells', 'U_s': 'TX-blind train minus held cells',
                               'V': 'held cells excluded from training validation and used for paired scoring only after all three arms freeze'},
                        source_days=[1, 2, 3], train_ratio='original per-design within retained cells',
                        support_query_ref='not applicable: no target prediction or scoring in source pilot')
    data['permissions'].update(query_use='none', claim_scope='Prospective source-only selection pilot, not E50 performance')
    data['checkpoint'].update(initialization='scratch E20 per fold then three own E21–E24 continuations',
                              contract_check_ref='exact source roles and location, plus fold-bound source loader',
                              provenance_verdict='Original complete source E20 used only for same-contract smoke, never inherited into a fold',
                              selection_rule='both folds strictly improve and every held cell has nonnegative net gain, then macro gain and nearest-to-native tie break')
    data['execution'].update(host='N607 ordinary user', launch_owner='codex-'+run_id,
                             gpu_policy='GPU 0–6, admitted max two total per GPU; reserve 12 GiB',
                             remote_run_root=root, remote_log_root=root+'/logs',
                             local_artifact_root='local_artifacts/pamf_repl_source_adapt_20261119',
                             launch_command=launch, stop_rule='Stop new dispatch on first blocking technical failure; preserve healthy workers and artifacts')
    data['metrics_plan'].update(metric_names=['held source paired net correction, both fold gains, worst-cell gain, activation'],
                                dimensions=['source receiver', 'source date', 'protocol', 'model seed', 'U dose'],
                                prediction_ref='Source-only in memory after all three arms freeze, identical held V order',
                                scorer_ref=root+'/analysis/source_decisions.json')
    data['expected_artifacts'] = ['actual-source checkpoint smoke', 'eight fold-specific initial checkpoints',
                                  '24 E24 source-only candidates', 'eight held-source paired readouts and four decisions']
    data['rows'] = []
    for setting in settings_for(run_id):
        pid = paper.PROTOCOLS[setting][0]
        for name, groups in fold_groups(context['protocols'][pid]['source_rxs'], (1, 2, 3)).items():
            job = f'train__{setting}-{name}'
            entry = deepcopy(row_template)
            entry.update(row_id=job, method='Original Tri with three U doses 6/4.5/3',
                         purpose='Paired source holdout pilot', config_ref=f'{root}/configs/{job}.json',
                         resolved_config_ref=f'{root}/configs/{job}.json',
                         data_overrides=dict(protocol_id=pid, held_rx_date=list(groups), dose_arms=list(WEIGHTS)),
                         epochs=24, budget_ref='scratch E20 once per fold; then three matched 4x222 optimizer continuations',
                         output_root=f'{root}/F/{setting}/{name}', log_path=f'{root}/logs/{job}.log',
                         command=launch, expected_artifacts=['segment20/complete.json', 'three weight directories', 'complete.json'])
            entry['seeds']['model'] = SEED
            entry['seed_notes'] = 'Original source split IDs; held cells deterministic; no separate augmentation or evaluation seed.'
            data['rows'].append(entry)
    data['notes'] = ['This E24 pilot never reads target handles or U truth. Final E50 clean validation is a separate fixed matrix.',
                     'With two RXs, hold RX-date cells to preserve both RXs in training. With at least three, hold complete source RXs.']
    if noise_aware:
        data['parent_run_ids'].append(RUN)
        data['execution']['local_artifact_root'] = 'local_artifacts/pamf_repl_source_adapt_r02_20261119'
        data['checkpoint']['selection_rule'] = 'Both source folds strictly improve, directional paired p <= .025, worst held cell delta >= -.25 percentage point; highest macro gain, closest-to-native tie-break.'
        data['notes'].append('This source-only screening rule is fixed before the second cohort. The .25 point tolerance applies to the source proxy, not formal E50 clean acceptance.')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('template', 'context', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--round', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    write(args.output, make(read(args.template), read(args.context), run_id=SECOND_RUN if args.round == 2 else RUN))
