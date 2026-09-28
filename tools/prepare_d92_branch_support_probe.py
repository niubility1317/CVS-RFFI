"""Prepare one source-free support-only probe; never read target scores or IQ."""
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '/home/szu2070436088/2510044040/CV-SincNet'
RUN = '20260929-phase2-d92-branch-support-probe-m4-r01'
RELEASE = 'd92_branch_support_probe_20260929_r01'
SPEC = 'configs/d92_branch_support_probe_20260929.json'


def documents():
    old = {c: json.loads((ROOT / f'configs/d92_mvridge_repeat_{c}_20260929.json').read_text(encoding='utf-8')) for c in ('rx3', 'rx1')}
    algorithm = json.loads((ROOT / 'configs/d92_branch_support_probe_frozen_20260929.json').read_text(encoding='utf-8'))
    root, release = f'{BASE}/runs/{RUN}', f'{BASE}/releases/{RELEASE}'
    spec = copy.deepcopy(old['rx3'])
    for key in ('confirmation', 'metrics_plan', 'rows', 'joint_benchmark'):
        spec.pop(key, None)
    spec.update(run_id=RUN, group_id='d92-fixed-phase1-branch-support-information',
        display_name='D92固定Phase1融合前分支support信息探查',
        description='固定模型和单原始view，完整support矩阵比较现有表示、重复背景控制、加入时间/频率/PA分支的物理OOF增量；不读取query样本或成绩，不产生部署分类器。',
        kind='diagnostic', stage='Phase2-support-only-diagnostic',
        tags=['d92','support-only','frozen-phase1','branch-information','no-query-access'],
        parent_run_ids=['20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01'],
        authorization='用户授权固定Phase1、源域无样本辅助训练、先复用现有数据；本次仅合法support信息探查。',
        status='PLANNED',claim_scope='SUPPORT_INFORMATION_DIAGNOSTIC_NOT_QUERY_PERFORMANCE',
        notes=['Fixed feature nodes and six arms; no parameter search, target score access or deployment head.',
               'Existing validated capsules unchanged. K1 only input/numeric diagnostics; no independent holdout.',
               'Frozen checkpoint lineage verified; actual architecture evidence retained. No source samples or new source payload.'])
    spec['code'].update(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        cwd=release, commit_note='Preparation parent; publisher records actual pushed release commit in launch/startup.')
    cohorts = {}
    outputs = {}
    for c, source in old.items():
        data = source['data']
        matrix = dict(receivers=data['target_receivers'],scenarios=data['scenarios'],ks=data['k'],
            new_counts=data['new_class_counts'],support_seeds=data['support_seeds'])
        conf = f'configs/d92_branch_support_probe_{c}_20260929.json'
        outputs[conf] = dict(algorithm=algorithm,matrix=matrix)
        cohorts[c] = dict(capsule=source['confirmation']['capsule'],capsule_id=data['capsule_id'],
            expected_split_count=source['confirmation']['expected_split_count'],matrix=matrix,
            evaluation_config=release+'/'+conf)
    spec['data'].update(dataset='Support slices only from the two existing VALIDATED_ONCE capsules',
        representation='One unchanged physical received IQ view; frozen z_id/t_emb/f_emb/pa_local and FFT96; no query IQ access',
        contract_ref='probe.cohorts.*.capsule/manifest.json',physical_ids_ref='probe.cohorts.*.capsule/received.npz#ids metadata only',
        label_map_ref='probe.cohorts.*.capsule/splits/*.json#registered_classes,support_labels',
        tx_sets_ref='probe.cohorts.*.capsule/splits/*.json#registered_classes',
        support_query_ref='Only registered support_indices/support_labels are consumed; query excluded',
        capsule_id={c:v['capsule_id'] for c,v in cohorts.items()},
        split_id='All unchanged 900 rx3 plus 300 rx1 splits, repeated across four fixed models',
        validation_ref='Existing cohort capsule manifests; no data rebuild or revalidation',
        target_receivers=old['rx3']['data']['target_receivers']+old['rx1']['data']['target_receivers'])
    spec['permissions'] = dict(regime='current_row_support_only_information_probe',query_use='none; query IQ/labels/truth/scores never read',
        source_samples=False,source_per_record_features=False,source_replay=False,summary_inputs=False,
        old_target_scores_for_adaptation=False,cross_run_result_tuning=False,adapted_state_reuse=False,
        truth_use='Only registered support labels for physical OOF; no query scorer',
        claim_scope='SUPPORT_INFORMATION_DIAGNOSTIC_NOT_QUERY_PERFORMANCE')
    spec['checkpoint'].update(initialization='Exact source-only scratch final200 model; eval and frozen; no changed buffers or weights',
        runtime_checkpoint_reload=True,metadata_evidence_ref='docs/D92_FIXED_PHASE1_BRANCH_METADATA_20260929.json',
        provenance_verdict='VERIFIED_ALL_FOUR_EXACT_SOURCE_CONTRACT_SCRATCH_FINAL200_SHA_AND_NATIVE_ARCHITECTURE',
        inheritance_policy='Only original source-only model; no historical adapted head/cache state',
        current_auxiliary_training='Per-row physical trainfold analytical probes only; no full-support deployment head')
    for key in ('feature_cache_producer_run_id','feature_cache_producer_role'):
        spec['checkpoint'].pop(key,None)
    spec['execution'].update(remote_run_root=root,remote_log_root=root,local_artifact_root='automation_reports/CV-SincNet/'+RUN,
        gpu_policy='GPU0 one serial frozen support exporter; at most four CPU probe lanes, two BLAS threads each',
        launch_command=f'C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 tools/publish_d92_branch_support_probe.py --spec {SPEC}',
        stop_rule='Technical failure fails only its lane; healthy lanes finish. No automatic retry, no stopping for weak support diagnostics.',
        cpu_lanes=4,blas_threads_per_lane=2,export_device='cuda:0',export_batch_size=32)
    spec['expected_artifacts']=['startup.json','state.json','complete.json','each row/export.log','each row/probe.log',
        'each row/feature_cache/features_complete.json','each row/feature_cache/support_splits.json',
        'each row/probe/startup.json','each row/probe/fit_trace.jsonl','each row/probe/compact.jsonl',
        'each row/probe/compact.csv','each row/probe/probe_complete.json']
    spec['metrics_plan']=dict(metric_names=['physical_OOF_accuracy','classwise_OOF_errors','held_aux_reconstruction','norms','rank','loss_components','actual_resource_cost'],
        dimensions=['model_seed','cohort','receiver','scenario','K','new_count','support_seed','background','probe_arm'],
        scorer_ref=None,prediction_ref='support OOF only; no query predictions',
        interpretation='No target-generalization claim. K1 has no independent same-class holdout. Duplicate control addresses changed kernel energy; all results retained; no parameter search.')
    spec['probe']=dict(cohorts=cohorts,native_code=old['rx3']['confirmation']['native_code'],
        source_contract=old['rx3']['confirmation']['source_contract'],source_receivers=spec['data']['source_receivers'],
        algorithm_config=release+'/configs/d92_branch_support_probe_frozen_20260929.json',model_rows=8,total_episodes=4800,
        view_count=1,source_payload_bytes=0,query_access=False,full_support_head=False)
    spec['rows']=[]
    for c, source in old.items():
        for original in source['rows']:
            row=copy.deepcopy(original);rid=c+'-'+row['row_id'];out=root+'/'+rid
            row['seeds']['evaluation']=None
            row.update(row_id=rid,cohort=c,method='fixed frozen branch support information probe',purpose='support-only paired information diagnostic',
                gpu='0 export only; CPU probe',config_ref=SPEC,resolved_config_ref=out+'/probe/startup.json',
                seed_notes='Model/split/data/augmentation roles retain source/capsule lineage. Support seed varies over explicit cohort matrix (row field null). Evaluation seed null: physical OOF is deterministic and no query evaluation is executed.',
                optimizer='None; exact ridge1 physical-sum least squares, unregularized intercept; no optimizer steps',lr=None,epochs=None,
                budget_ref=f"{cohorts[c]['expected_split_count']} episodes; K1 numeric-only; K>=2 three physical folds, six fixed arms plus reconstruction diagnostic; no full-fit deployment head",
                output_root=out,log_path=out+'/probe.log',
                command=f'{spec["code"]["environment"]} -u {release}/tools/run_d92_branch_support_probe.py --spec {release}/{SPEC} --commit RELEASE_HEAD',
                expected_artifacts=['feature_cache/features_complete.json','feature_cache/support_splits.json','probe/probe_complete.json','probe/fit_trace.jsonl','probe/compact.jsonl','probe/compact.csv'])
            row['data_overrides']=dict(capsule=cohorts[c]['capsule'],capsule_id=cohorts[c]['capsule_id'],expected_split_count=cohorts[c]['expected_split_count'])
            row['provenance_refs']={k:v for k,v in original['provenance_refs'].items() if k in ('source_initialization','source_completion')}
            for key in ('ground_source','reuse_row_root','reuse_multiview_features_root'):
                row.pop(key,None)
            spec['rows'].append(row)
    outputs[SPEC]=spec
    return outputs


def main():
    outputs=documents()
    if any((ROOT/name).exists() for name in outputs):
        raise FileExistsError('Probe preparation outputs exist; reconcile, do not overwrite')
    for name,value in outputs.items():
        with (ROOT/name).open('x',encoding='utf-8',newline='\n') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status='PREPARED_NOT_REGISTERED_OR_LAUNCHED',files=list(outputs))))


if __name__=='__main__':main()
