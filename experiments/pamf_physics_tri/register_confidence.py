"""Register the fixed confidence-only full-source comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .confidence_plan import RUN, candidate_rows


def make(template):
    result = deepcopy(template)
    root = 'PROJECT/runs/' + RUN
    result.update(run_id=RUN, display_name='PAMF 仅置信度筛选的分支辅助实验',
        description='三个完整源数据配置、三个种子；全部置信度至少 0.95 的 U 参与辅助训练，不使用分支投票。',
        aliases=['PAMF-confidence-only-auxiliary'], tags=['PAMF', 'confidence-only', 'no-branch-vote'],
        authorization='2026-11-13 用户要求只按 0.95 伪标签置信度参与后续辅助损失。')
    result['code']['cwd'] = 'PROJECT/releases/' + RUN + '-release'
    result['data'].update(dataset='original ManyRx ten-class, ManySig 1%-labeled and six-source-receiver',
        roles=dict(L_s='entire original labeled role, no held-out subset', U_s='original source U, TX hidden',
                   V='original source V unchanged'), train_ratio='original per-setting data roles unchanged',
        contract_ref='context.handles indexed by row protocol_id', physical_ids_ref='original source and test contracts',
        support_query_ref='per-row frozen seven-view prediction then independent truth-last scoring')
    result['permissions'].update(regime='source_only', query_use='frozen predictions then independent truth-last',
        claim_scope='previously exposed test set; exploratory comparison without target feedback into training')
    result['checkpoint'].update(initialization='matching original full-L same-setting same-seed Tri E20, then own E10+E20',
        contract_check_ref='confidence_plan.parent -> full_matrix_plan.parent and checked_e20_parent',
        provenance_verdict='original matching full-L scratch E20; no L holdout or target ancestry',
        selection_rule='all nine fixed rows, final E50; no target-driven selection')
    result['execution'].update(launch_owner='codex-' + RUN, remote_run_root=root, remote_log_root=root + '/logs',
        gpu_policy='shared single owner: GPU 7 total max 4; other cards total max 6; includes other users and reserves initializing workers memory',
        local_artifact_root='local_artifacts/pamf_confidence_only_20261113',
        launch_command='python -m experiments.pamf_physics_tri.confidence_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Systemic technical failure stops new dispatch; preserve running workers, outputs and other owners')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke', 'nine matching full-source E30/E50 continuations',
        'nine direct seven-view predictions and independent truth-last scores',
        'confidence-accepted and auxiliary-routed counts, pseudo-class distribution, source labeled reliability',
        'three-seed matched ordinary PL and original branch-prototype comparisons']
    result['metrics_plan'].update(metric_names=['accuracy', 'six-scene accuracy', 'accepted and routed exposures',
                                                'pseudo-class counts', 'source labeled confidence precision'],
        dimensions=['three configurations', 'three model seeds', 'clean and six practical', 'TX/RX/day'],
        prediction_ref=root + '/readouts/ROW/predictions', scorer_ref=root + '/readouts/ROW/scores')
    result['rows'] = []
    for row in candidate_rows():
        item = deepcopy(template['rows'][0])
        rid = row['row_id']
        item.update(row_id=rid, method='Hybrid plus branch prototype; all confidence >=0.95, no branch vote',
            purpose='fixed confidence-only coverage comparison', data_overrides={'protocol_id': row['protocol_id']},
            seeds=dict(model=row['model_seed'], split=392005, data=None, augmentation=20261016421,
                       support=None, evaluation=None), seed_notes='Original data roles, no internal L holdout.',
            scenario='direct raw clean and six practical', config_ref=root + '/configs/train__' + rid + '.json',
            resolved_config_ref=root + '/F/' + rid + '/source/resolved_config.json',
            budget_ref='matching original Tri scratch E20 reused; own E10 + E20 = 6660 updates',
            output_root=root + '/F/' + rid + '/source', log_path=root + '/logs/train__' + rid + '.log',
            command='worker --kind train --config ' + root + '/configs/train__' + rid + '.json',
            expected_artifacts=['source/complete.json', 'readouts/' + rid + '/predictions/complete.json',
                                'readouts/' + rid + '/scores/complete.json'])
        result['rows'].append(item)
    result['notes'] = [
        'Every confidence-accepted U packet is routed into extra same-packet auxiliary training, including the original non-physical half; no branch voting is evaluated.',
        'Prototype loss uses available cross-day true-L class centers. When the positive center is missing it uses the same U packet raw view as a detached positive; when no negative center exists it uses a same-packet cosine term. All accepted packets participate.',
        'If only one pseudo-class is present, an identity-free same-packet cosine term replaces the absent contrastive-negative contribution.',
        'Original channel-strength taper and all other mechanisms are retained; new physical U routing also changes the existing mixed U path.',
        'Source labeled confidence precision is an in-training diagnostic, not held-out accuracy.',
        'Previously exposed target tests are exploratory and do not change the fixed matrix or select training checkpoints.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
