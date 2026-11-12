"""Register the fixed ten-configuration Hybrid confidence-only comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .hybrid_confidence_plan import RUN, candidate_rows


def make(template):
    result = deepcopy(template)
    root = 'PROJECT/runs/' + RUN
    result.update(run_id=RUN, display_name='Hybrid 仅置信度辅助训练十配置实验',
        description='十种原始数据配置、三个种子；所有置信度至少 0.95 的 U 参与 Hybrid 辅助训练。',
        aliases=['PAMF-hybrid-confidence-only-ten-configs'], tags=['PAMF', 'Hybrid', 'confidence-only'],
        authorization='2026-11-13 用户将优化目标更改为 Hybrid，要求所有通过置信度的伪标签参与辅助训练。')
    result['code']['cwd'] = 'PROJECT/releases/' + RUN + '-release'
    result['data'].update(dataset='ten original Phase1 data configurations',
        roles=dict(L_s='complete original labeled role for each configuration',
                   U_s='original source U, TX hidden', V='original source V unchanged'),
        train_ratio='original per-configuration roles; no source holdout',
        contract_ref='context.handles indexed by row protocol_id', physical_ids_ref='original source and test contracts',
        support_query_ref='per-row frozen seven-view prediction then independent truth-last scoring')
    result['permissions'].update(regime='source_only', query_use='frozen direct prediction then independent truth-last',
        claim_scope='previously exposed test sets; exploratory comparison without target feedback into training')
    result['checkpoint'].update(initialization='same-setting same-seed original full-L Tri E20, then own E10+E20',
        contract_check_ref='hybrid_confidence_plan.parent -> full_matrix_plan.parent',
        provenance_verdict='matching original full-source same-seed E20; no held-out L or target ancestry',
        selection_rule='all 30 fixed rows, final E50; no target-driven checkpoint selection')
    result['execution'].update(launch_owner='codex-' + RUN, remote_run_root=root, remote_log_root=root + '/logs',
        gpu_policy='single shared owner; GPU 7 total max 4 and other cards max 6, includes other users and memory reservations',
        local_artifact_root='local_artifacts/pamf_hybrid_confidence_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_confidence_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG',
        stop_rule='Systemic technical failure stops new dispatch for the affected queue; preserve other healthy workers and outputs')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke per data configuration',
        '30 matching full-source E30/E50 continuations', '30 direct seven-view predictions and independent truth-last scores',
        'confidence acceptance, auxiliary exposure and pseudo-class counts',
        'three-seed matched ordinary PL and original Hybrid comparisons per setting']
    result['metrics_plan'].update(metric_names=['clean accuracy', 'six individual practical accuracies',
        'six-scene mean', 'accepted and routed exposures', 'pseudo-class counts'],
        dimensions=['ten configurations', 'three seeds', 'clean and six practical', 'TX/RX/day'],
        prediction_ref=root + '/readouts/ROW/predictions', scorer_ref=root + '/readouts/ROW/scores')
    result['rows'] = []
    for row in candidate_rows():
        rid = row['row_id']
        item = deepcopy(template['rows'][0])
        item.update(row_id=rid, method='Hybrid; all pseudo-label confidence >=0.95, no branch votes and no prototype',
            purpose='fixed confidence-only Hybrid comparison', data_overrides={'protocol_id': row['protocol_id']},
            seeds=dict(model=row['model_seed'], split=392005, data=None, augmentation=20261016421,
                       support=None, evaluation=None), seed_notes='Original fixed source and target roles; no internal L holdout.',
            scenario='direct raw clean and six practical', config_ref=root + '/configs/train__' + rid + '.json',
            resolved_config_ref=root + '/F/' + rid + '/source/resolved_config.json',
            budget_ref='matching original Tri scratch E20 reused; own E10 + E20 = 6660 updates',
            output_root=root + '/F/' + rid + '/source', log_path=root + '/logs/train__' + rid + '.log',
            command='worker --kind train --config ' + root + '/configs/train__' + rid + '.json',
            expected_artifacts=['source/complete.json', 'readouts/' + rid + '/predictions/complete.json',
                                'readouts/' + rid + '/scores/complete.json'])
        result['rows'].append(item)
    result['notes'] = [
        'Every confidence-accepted source U packet is routed into Hybrid auxiliary training, including the original non-physical half; branch votes and prototype loss are absent.',
        'If only one accepted pseudo-class occurs, a same-packet cosine term substitutes for unavailable contrastive negatives.',
        'The existing physical strength taper is retained; expanded physical routing also changes the original mixed U path.',
        'Prior ordinary PL and original Hybrid controls are reused by same setting and seed.',
        'Previously exposed target tests imply exploratory results; fixed matrix and training do not depend on target scores.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
