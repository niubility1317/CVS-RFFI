"""Prepare two transparent repeated benchmarks from configs and provenance only.

No registry mutation, data construction, target-score read, remote action or
algorithm change. Outputs use exclusive creation and preserve previous runs.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = '/home/szu2070436088/2510044040/CV-SincNet'
PYTHON = '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python'
SOURCE_RUN = '20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'
GROUP = 'd92-fixed-phase1-sgjoint-repeated-benchmark'
CLAIM = 'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS'
FROZEN = 'configs/d92_summary_joint_frozen_20260928.json'
DEFAULT_METADATA = Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/ground_payload_readback.json')
COHORTS = {
    'rx3': dict(source_spec='configs/d92_confirmation_recovery_20260928.json',
                source_data='configs/d92_confirmation_data_20260928.json',
                source_run='20260928-phase2-d92-scv-confirmation-manytx-m4-r02',
                capsule_id='residual-noeq-76121e6f34363fa612ec25fb', splits=900, receivers=3),
    'rx1': dict(source_spec='configs/d92_sourcefree_confirmation_20260928.json',
                source_data='configs/d92_sourcefree_data_20260928.json',
                source_run='20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01',
                capsule_id='residual-noeq-d0a99fede324159c5a4750fd', splits=300, receivers=1),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_id(cohort):
    return f'20260928-phase2-d92-sgjoint-repeat-{cohort}-m4-r01'


def provenance_rows(metadata, classes):
    if (metadata.get('source_samples_read') is not False
            or metadata.get('source_features_read') is not False
            or metadata.get('target_read') is not False):
        raise ValueError('Expected source-free, target-free provenance metadata')
    rows = metadata['rows']
    if len(rows) != 4 or sorted(r['seed'] for r in rows) != list(range(2026092701, 2026092705)):
        raise ValueError('Expected exact four frozen model rows')
    result = {}
    for row in rows:
        initial, done, manifest = row['initialization'], row['completion'], row['manifest']
        digest = manifest['checkpoint_sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or row['classes'] != classes
                or initial.get('seed') != row['seed'] or initial.get('scratch_only') is not True
                or initial.get('checkpoint_sources') != [] or initial.get('source_roles') != 'EXACT_MATCH'
                or initial.get('target_contact') is not False or initial.get('target_training_contact') is not False
                or done.get('status') != 'TRAINING_COMPLETE' or done.get('epochs') != 200
                or done.get('checkpoint') != 'final_ssdg.pth' or done.get('target_evaluated') is not False
                or manifest.get('provenance_status') != 'CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L'
                or manifest.get('formal_phase2_eligible') is not True):
            raise ValueError('Frozen source ancestry or summary binding mismatch')
        result[row['seed']] = row
    return result


def build_documents(root, metadata_path, commit):
    root, metadata_path = Path(root), Path(metadata_path)
    frozen = read(root / FROZEN)
    if frozen['algorithm']['method'] != 'D92-SGJoint-v1' or type(frozen['summary_already_deployed']) is not bool:
        raise ValueError('Frozen SGJoint config mismatch')
    metadata = read(metadata_path)
    outputs = {}
    joint = dict(cohort_run_ids=[run_id(key) for key in COHORTS], model_rows=4, receiver_count=4,
        scenario_count=3, k=[1, 5, 10, 20], new_class_counts=[0, 2, 5, 10, 20], support_seed_count=5,
        paired_cells=4800, scoring_records_including_dg=9648,
        aggregation='Equal weight for every paired cell across all four receivers; never equal weight for cohort means',
        per_k_acceptance=dict(harmonic_mean_delta_gt=0.0, new_accuracy_delta_gt=0.0,
                             old_accuracy_delta_ge=-0.01, old_drop_unit='absolute probability; 0.01 equals 1 percentage point'),
        new_and_h_scope='Only registered-new-count>0 cells; old-only new/H remain null, not invented',
        old_accuracy_scope='All paired cells at this K, including old-only registration',
        per_cell_pass_required=False,
        report_all_dimensions=['cohort', 'model_seed', 'receiver', 'scenario', 'K', 'new_count', 'support_seed'],
        interpretation_barrier='Do not download/read either scores file until both preregistered runs reach terminal state; interpret jointly only when both complete',
        early_result_policy='Launch both fixed cohorts; no early score may change, select, stop, or tune the other cohort',
        partial_policy='Preserve/report technical failures; no comprehensive acceptance claim from a partial matrix')
    for cohort, plan in COHORTS.items():
        old = read(root / plan['source_spec'])
        data = read(root / plan['source_data'])
        if old['run_id'] != plan['source_run'] or len(old['rows']) != 4 or len(data['target_receivers']) != plan['receivers']:
            raise ValueError('Reference cohort matrix mismatch')
        ancestors = provenance_rows(metadata, data['old_classes'])
        run = run_id(cohort)
        release_name = f'd92_sgjoint_repeat_{cohort}_20260928_r01'
        release, remote = f'{PROJECT}/releases/{release_name}', f'{PROJECT}/runs/{run}'
        spec_ref = f'configs/d92_sgjoint_repeat_{cohort}_20260928.json'
        data_ref = f'configs/d92_sgjoint_repeat_{cohort}_data_20260928.json'
        data_root = data['output_root']
        reference = {key: copy.deepcopy(data[key]) for key in (
            'output_root', 'old_classes', 'new_classes', 'source_receivers', 'target_receivers',
            'fs_hz', 'data_seed', 'augmentation_seed', 'receiver_seed', 'evaluation_seed',
            'support_seeds', 'scenarios', 'shots', 'new_counts')}
        reference.update(schema='cvs.phase2.validated_capsule_reference.v1', reference_only=True,
            builder_enabled=False, data_rebuilt=False, data_revalidated=False,
            capsule_id=plan['capsule_id'], protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
            capsule=data_root+'/capsule', validation_ref=data_root+'/capsule/manifest.json',
            split_count=plan['splits'], existing_data_config_ref=plan['source_data'],
            original_data_config=old['confirmation']['data_config'],
            claim_scope=CLAIM, note='Existing received IQ and immutable physical support/query splits; no data construction or revalidation')
        for key in ('support_pool_size', 'query_size'):
            if key in data:
                reference[key] = data[key]
        spec = copy.deepcopy(old)
        for key in ('registered_at', 'status', 'replaces_run_id'):
            spec.pop(key, None)
        spec.update(run_id=run, group_id=GROUP, stage='Phase2-repeated-benchmark',
            display_name=f'D92-SGJoint冻结方法{cohort}透明重复基准',
            description='固定4个Phase1模型与已验证received capsule；复用原D92预测和冻结目标特征，在新输出拟合SGJoint并与原D92配对。目标已经评分过，只声明重复基准。',
            parent_run_ids=[SOURCE_RUN, plan['source_run']], aliases=[],
            tags=['cvs', 'd92', 'sgjoint', 'source-free', 'repeated-benchmark', 'frozen_phase1', 'truth_last'],
            authorization='2026-09-28用户明确要求“那就先复用数据”，授权透明重复基准；算法已冻结，不读取旧评分来修改公式，不等待新增独立数据。',
            claim_scope=CLAIM, joint_benchmark=copy.deepcopy(joint))
        spec['code'].update(commit=commit, checkout=str(root), cwd=release,
            commit_note='Preparation parent commit; publisher records exact pushed release HEAD in launch/startup. Frozen algorithm config is unchanged.')
        spec['data'].update(dataset=data_root+'/capsule/received.npz',
            contract_ref=data_root+'/capsule/manifest.json', capsule_id=plan['capsule_id'],
            protocol_schema='p2_min_v1', phase2_data_status='VALIDATED_ONCE',
            physical_ids_ref=data_root+'/capsule/received.npz#ids',
            label_map_ref=release+'/'+data_ref+'#old_classes,new_classes',
            split_id=f'Existing {plan["splits"]} immutable split IDs; unchanged from '+plan['source_run'],
            validation_ref=data_root+'/capsule/manifest.json', support_query_ref=data_root+'/capsule/splits',
            tx_sets_ref=release+'/'+data_ref, scenarios=data['scenarios'],
            data_rebuilt=False, data_revalidated=False, target_previously_scored=True,
            reuse_policy='VALIDATED_ONCE; runtime artifact identity/binding checks only, no new data validation')
        spec['permissions'] = dict(regime='fixed_phase1_existing_quantized_summary_and_current_row_support_only',
            query_use='read-only independent sample against all registered classes; no fit/selection/query feedback',
            external_method_exception=None, claim_scope=CLAIM,
            source_samples=False, source_per_record_features=False, source_replay=False,
            old_target_scores_for_adaptation=False, summary_permission='existing checkpoint-bound Project 5.3.2 only',
            truth_use='Independent scorer after every baseline/candidate prediction in this run is fixed',
            cross_run_result_tuning=False)
        spec['checkpoint'].update(provenance_verdict='VERIFIED_EXISTING_SCRATCH_FINAL200_METADATA; exact SHA/class/feature binding checked at reuse',
            selection_rule='All four fixed model seeds, final epoch200; no checkpoint/model choice from any target score',
            metadata_evidence_ref=str(metadata_path), runtime_checkpoint_reload=False,
            source_training_contract_ref=old['checkpoint']['contract_check_ref'],
            inheritance_policy='Each Phase1 ancestor scratch_only with empty external checkpoint_sources; EMA originated from that run student; SGJoint creates no Phase1 inheritance',
            current_auxiliary_training='Only same-row legal support with immutable summary normalization; no source L/U/V access')
        spec['execution'].update(host='N607', launch_owner='codex/root/d92-upgrade-20260928',
            gpu_policy='CPU only; four model lanes, each 2 BLAS threads; no GPU, no checkpoint loading or encoder/baseline re-execution',
            cpu_lanes=4, blas_threads_per_lane=2, remote_run_root=remote, remote_log_root=remote,
            local_artifact_root='automation_reports/CV-SincNet/'+run,
            launch_command=f'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_confirmation.py --spec {spec_ref} --release {release_name}',
            stop_rule='Technical contract/process failure fails only affected lane; preserve artifacts, no automatic retry. Healthy lanes and the other preregistered cohort continue; never stop for low performance. Do not score a partial run.')
        spec['confirmation'] = dict(capsule=data_root+'/capsule', truth=data_root+'/score_only/truth.json',
            data_config=release+'/'+data_ref, candidate_config=release+'/'+FROZEN,
            native_code=old['confirmation']['native_code'], source_contract=old['confirmation']['source_contract'],
            old_classes=data['old_classes'], candidate=copy.deepcopy(frozen['algorithm']),
            candidate_method='D92-SGJoint-v1', candidate_folder='sgjoint',
            candidate_predictor='predict_d92_summary_joint.py', candidate_mode='d92_sgjoint_registration',
            model_rows=4, splits_per_model=plan['splits'], expected_split_count=plan['splits'],
            predictions_total=4*(2*plan['splits']+3*plan['receivers']),
            reuse_validated_capsule_id=plan['capsule_id'],
            baseline_source_run_id=plan['source_run'], claim_scope=CLAIM)
        spec['metrics_plan'].update(scorer_ref=release+'/tools/score_d92_confirmation.py',
            prediction_ref='Original reuse_row_root/predictions.jsonl paired with new output_root/sgjoint/predictions.jsonl',
            acceptance=dict(old_max_drop=0.01, require_new_improvement=True, require_harmonic_improvement=True,
                primary_scope='joint all-four-RX equal-cell per-K; cohort tables descriptive, not independent pass gates'),
            dimensions=['cohort', 'model_seed', 'receiver', 'scenario', 'K', 'new_count', 'support_seed'],
            resource_metrics=['fit_seconds', 'head_bytes', 'summary_operator_bytes', 'persistent_state_bytes',
                'numeric_array_bytes', 'registry_schema_bytes', 'total_file_bytes', 'incremental_transfer_bytes'],
            new_h_for_old_only=None, claim_scope=CLAIM)
        spec['expected_artifacts'] = ['startup.json', 'state.json', 'data_reuse.json', 'complete.json', 'scores.json',
            'each row/artifact_reuse.json', 'each row/sgjoint/predictions_complete.json', 'each row/sgjoint/predictions.jsonl',
            'each row/sgjoint/fit_trace.jsonl', 'each row/sgjoint/compact.jsonl', 'each row/sgjoint/compact.csv']
        for row in spec['rows']:
            seed = row['seeds']['model']; evidence = ancestors[seed]
            out = remote+'/'+row['row_id']
            origin = old['execution']['remote_run_root']+'/'+row['row_id']
            row.update(method='paired frozen D92 P2-256-FULL and D92-SGJoint-v1',
                purpose='transparent_repeated_benchmark_fixed_algorithm', gpu=None,
                config_ref=spec_ref, resolved_config_ref=out+'/sgjoint/startup.json',
                reuse_row_root=origin, expected_checkpoint_sha256=evidence['manifest']['checkpoint_sha256'],
                output_root=out, log_path=out+'/sgjoint.log',
                optimizer='Closed-form float64 shrinkage LDA; physical-support-only folds and bounded scalar inverse-temperature selection',
                lr=None, epochs=None, fl_rounds=None,
                budget_ref=f'{plan["splits"]} paired splits plus {3*plan["receivers"]} reused frozen DG rows/model; fixed16candidates, max3physical folds; K1 fixed, no holdout',
                command=f'{PYTHON} -u {release}/tools/run_d92_confirmation.py --spec {release}/{spec_ref} --commit RELEASE_HEAD',
                expected_artifacts=['artifact_reuse.json', 'sgjoint/startup.json', 'sgjoint/predictions.jsonl',
                    'sgjoint/predictions_complete.json', 'sgjoint/fit_trace.jsonl', 'sgjoint/compact.jsonl', 'sgjoint/compact.csv'],
                provenance_refs=dict(ground_manifest=origin+'/ground/manifest.json',
                    existing_frozen_export_startup=origin+'/received_features/startup.json',
                    received_features_provenance=origin+'/received_features/checkpoint_provenance.json',
                    existing_d92_startup=origin+'/d92_startup.json',
                    ground_provenance=origin+'/ground_provenance.json',
                    received_features_complete=origin+'/received_features/features_complete.json',
                    baseline_predictions_complete=origin+'/predictions_complete.json',
                    source_initialization=row['source_root']+'/initialization.json',
                    source_completion=row['source_root']+'/completion.json'))
            row['seed_notes'] = ('Model seed denotes existing frozen Phase1 weights; source split392005 is ancestry metadata only. '
                'Existing capsule split/data2026092705, augmentation2026092707, evaluation2026092706, receiver2027 are reused unchanged. '
                'Outer support seed null because each row enumerates support2026092711..15 from immutable splits. '
                'SGJoint is deterministic; it creates no new RNG roles or observations.')
        spec['notes'] = [
            'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: not a new independent/generalization holdout.',
            'Frozen algorithm from configs/d92_summary_joint_frozen_20260928.json is unchanged for both cohorts.',
            'Reuse existing validated received IQ, fixed source-trained checkpoint provenance, quantized summary, target features and D92 predictions; do not read old scores.',
            'No source L/U/V, source samples, source per-record features, source replay, new data building, encoder execution or GPU reservation.',
            'Runtime exact SHA/capsule/class/member checks reuse existing validation and are not new data validation.',
            'Both runs are authorized together. Do not inspect early scores, select rows, stop a healthy cohort or alter the other formula from early results.',
            'Interpret only the complete joint4800 paired cells/9648 scoring records; equal paired-cell weights across four receivers, not equal cohort weights.',
            'Each K jointly requires H and new accuracy improvement with old accuracy drop at most1percentage point. Old-only H/new are null; no per-cell victory requirement.',
            'Save actual candidate/fold train scales, held-support NLL, inverse temperature, selected objective, time and byte audit; closed-form fitting has no training epoch/gradient claim.',
            'Frozen config summary_already_deployed is retained exactly; report both existing-component file bytes and its explicit delivery assumption, not an invented zero transport bound.'
        ]
        outputs[data_ref] = reference
        outputs[spec_ref] = spec
    if sum(doc['confirmation']['predictions_total'] for doc in outputs.values() if 'confirmation' in doc) != 9648:
        raise AssertionError('Joint scoring count mismatch')
    return outputs


def write_documents(root, documents):
    paths = [Path(root)/relative for relative in documents]
    if any(path.exists() for path in paths):
        raise FileExistsError('An output already exists; never overwrite existing preregistration')
    for relative, value in documents.items():
        path = Path(root)/relative
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        if read(path) != value:
            raise AssertionError('UTF-8 JSON readback mismatch: '+relative)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    documents = build_documents(ROOT, args.metadata, commit)
    write_documents(ROOT, documents)
    print(json.dumps(dict(status='PREPARED_ONLY', files=list(documents),
        run_ids=[run_id(key) for key in COHORTS], paired_cells=4800,
        scoring_records_including_dg=9648, data_built=False, registry_mutated=False,
        experiment_started=False, claim_scope=CLAIM), ensure_ascii=False))


if __name__ == '__main__':
    main()
