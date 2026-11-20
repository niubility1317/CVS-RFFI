"""Register the approved setting profiles with preserved historical lineage."""
import argparse
from experiments.cric_dura_comparison.io import read, write
from . import full_suite_register, setting_profiles as profile, transfer


def make(template):
    result = full_suite_register.make(template)
    root = f'PROJECT/runs/{profile.RUN}'
    cmd = f'python -m {profile.MODULE} publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG'
    rows = list(profile.matrix())
    result.update(run_id=profile.RUN, display_name='Tri 逐设置配方与普通 PL 的三种子对比',
        description='11 设置各三种子，按用户确认的逐设置权重及损失配置。单源使用 Hybrid。',
        parent_run_ids=sorted({profile.paper.RUN, *(row['reuse_run'] for row in rows if row.get('reuse_run'))}),
        replaces_run_id=None, authorization='用户确认按前一轮所列的 11 种自适应配方执行，单源使用历史 Hybrid。')
    result['code']['cwd'] = f'PROJECT/releases/{profile.RUN}-release'
    result['permissions']['claim_scope'] = 'Fixed historical setting profiles, not an evaluation of a source-only selector'
    result['checkpoint']['sources'] = result['parent_run_ids'].copy()
    result['checkpoint']['selection_rule'] = 'Fixed E50 and recipe by setting, historical development results disclosed'
    result['execution'].update(owner='codex-'+profile.RUN, launch_owner='codex-'+profile.RUN,
        remote_run_root=root, remote_log_root=root+'/logs', launch_command=cmd,
        local_artifact_root='local_artifacts/pamf_tri_setting_profiles_20261120')
    result['expected_artifacts'] = ['2 new ten-class real E20 smokes and E50 continuations',
        '2 new independent clean scores and 2 frozen U diagnostic scores',
        '31 verified frozen profile rows and 33 paired comparisons to ordinary PL and original Tri']
    result['metrics_plan'].update(prediction_ref=root+'/sources.json', scorer_ref=root+'/analysis/results.json')
    result['notes'] = ['Fallback and specialized recipes are fixed before inference; no new clean results enter decisions.',
        'Ten-class seed 01 was used in recipe development. Seeds 02 and 03 are held out from recipe choice.',
        'Historical Hybrid uses fusion_branch mode. No new single-source training.',
        'This historical recipe table and the prospective source-only dose selector have separate evidence.']
    for entry, row in zip(result['rows'], rows, strict=True):
        rid, history = row['row_id'], row.get('reuse_run')
        if row['reuses_original']:
            history = profile.paper.RUN
        if history:
            archive_row = row.get('reuse_row') or transfer.original_row(row)
            archive = f'PROJECT/runs/{history}/sources.json#{archive_row}'
            cfg, out, log = archive, archive, archive
        else:
            out = f'{root}/F/{rid}/source'
            cfg, log = f'{root}/configs/train__{rid}.json', f'{root}/logs/train__{rid}.log'
        entry.update(row_id=rid, method=row['method'], purpose='Verified frozen reuse' if history else 'Held-seed verification',
            config_ref=cfg, resolved_config_ref=cfg if history else out+'/resolved_config.json',
            data_overrides={'protocol_id': row['protocol_id'], 'U_weight': row['weight'], 'reuse': bool(history)},
            output_root=out, log_path=log, command=cmd,
            budget_ref='exact frozen E50' if history else 'same-contract E20 + 6660 updates to fixed E50',
            expected_artifacts=['existing independent clean prediction and score'] if history else
                               ['source/complete.json', 'predictions/complete.json', 'scores/complete.json', 'U_diagnostic/scores.json'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    write(args.output, make(read(args.template)))
