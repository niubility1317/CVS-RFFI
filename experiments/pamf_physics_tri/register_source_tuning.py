"""Fill the existing experiment registry template for the source-only pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .source_tuning_plan import RUN, candidate_rows


def make(template):
    result = deepcopy(template)
    run = 'PROJECT/runs/' + RUN
    release = 'PROJECT/releases/' + RUN + '-release'
    result.update(run_id=RUN, group_id='phase1-pamf-branch-prototype-tuning',
                  display_name='10类 PAMF 分支原型温度源域内部开发试验',
                  description='从零训练三组相同划分的 Hybrid＋分支原型；仅在源 L 内部开发集选温度。',
                  aliases=['PAMF-branch-prototype-source-temperature-pilot'],
                  tags=['PAMF', 'source-development', 'prototype-temperature'],
                  authorization='2026-11-12 用户授权按不同数据配置调优 Hybrid＋分支原型。')
    result['code'].update(commit='actual committed release identity in runtime.json',
                          checkout='PAMF branch-training Git worktree', environment='N607 existing CVS-RFFI Python, FP32',
                          cwd=release)
    result['data'].update(dataset='ManyRx ten-class, previously validated original roles',
        version='existing source data', representation='IQ [2,256]',
        contract_ref='context.handles.ManyRx.source_contract', physical_ids_ref='source contract role IDs',
        label_map_ref='source contract classes', source_receivers=list(range(0, 32, 2)),
        target_receivers=list(range(1, 32, 2)), source_days=[1, 2, 3], target_days=[0, 1, 2, 3],
        tx_sets_ref='context.handles.ManyRx.classes',
        roles=dict(L_s='6533 original; deterministic per-class split 5228 train / 1305 dev',
                   U_s='58828 source U; true TX hidden', V='28023 source V; health diagnostics only'),
        train_ratio='source L split 80/20 by class with seed 20261112', capsule_id=None, split_id=None,
        validation_ref='existing data roles; Phase2 capsule not applicable', support_query_ref='no target/query access',
        leo_config_ref='source practical training simulator, fixed epoch 999 and batch index for development')
    result['permissions'].update(regime='source_only', query_use='no target access in this pilot',
        claim_scope='Source-domain pilot; historical target tests exposed; no performance claim on targets')
    result['checkpoint'].update(initialization='scratch for each candidate, own E20/E30/E50',
        sources=['historical same-contract E20 only for operational no-update smoke, never initialization'],
        contract_check_ref='source_development.installed and verify_parent_partition',
        provenance_verdict='fresh own-source trajectory; smoke checkpoint excluded from training and selection',
        selection_rule='fixed final E50; source L_dev clean and physical accuracy equally weighted')
    result['execution'].update(host='N607 ordinary user', launch_owner='codex-' + RUN,
        gpu_policy='GPUs 0,1,3,4,5,7; one total training or prediction process per GPU; check free memory',
        remote_run_root=run, remote_log_root=run + '/logs',
        local_artifact_root='local_artifacts/pamf_source_tuning_20261112',
        launch_command='python -m experiments.pamf_physics_tri.source_tuning_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Any technical job failure stops new dispatch, preserves active workers and outputs; no target-score-driven change')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', 'three scratch E50 source models',
        'three held-out source L clean and simulated-physical results', 'fixed source-only selection and promotion decision']
    result['metrics_plan'].update(metric_names=['accuracy', 'equal-weight mean clean/physical'],
        dimensions=['10 classes', 'seed 2026092701', 'three temperatures', 'clean and fixed source simulated practical'],
        prediction_ref='source L_dev only; no target predictions', scorer_ref=run + '/source_scores')
    result['rows'] = []
    for row in candidate_rows():
        rid = row['row_id']
        item = deepcopy(template['rows'][0])
        item.update(row_id=rid, method=f"Hybrid plus branch prototype, prototype temperature {row['temperature']}",
            purpose='source-only fixed candidate', config_ref=run + '/configs/train__' + rid + '.json',
            resolved_config_ref=run + '/F/' + rid + '/source/resolved_config.json',
            data_overrides={'protocol_id': 'ManyRx'},
            seeds=dict(model=row['model_seed'], split=392005, data=20261112, augmentation=20261016421,
                       support=None, evaluation=None),
            seed_notes='Internal L split seed in data; support/evaluation seeds not applicable; simulated channel uses fixed epoch and step.',
            scenario='source clean and source simulated low suburban, mid urban, low urban',
            optimizer='original matching native AdamW', lr='original native schedule', epochs=50,
            budget_ref='scratch 20 + own 10 + own 20 epochs; 11100 updates',
            output_root=run + '/F/' + rid + '/source', log_path=run + '/logs/train__' + rid + '.log',
            command='worker --kind train --config ' + run + '/configs/train__' + rid + '.json',
            expected_artifacts=['source/complete.json', 'source_scores/' + rid + '/complete.json'])
        result['rows'].append(item)
    result['notes'] = ['All temperatures use the same internal L split. U truth, target and query remain inaccessible.',
        'At least 0.25 percentage point source-development mean improvement versus temperature 0.2 is required for a separate multi-seed stage.',
        'No target predictions in this source-only pilot. Selected candidates require full-L retraining and independent truth-last target evaluation.',
        'Previous target scores motivated the overall research; resulting target reuse is exploratory, not an independent blind confirmation.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
