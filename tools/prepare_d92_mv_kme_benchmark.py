"""Prepare the fixed received-IQ multiview benchmark, without reading scores."""
import copy
import json
from pathlib import Path
import subprocess

from prepare_d92_summary_joint_benchmark import ROOT, DEFAULT_METADATA, build_documents, write_documents


def replace_names(value):
    if isinstance(value,dict):return {k:replace_names(v) for k,v in value.items()}
    if isinstance(value,list):return [replace_names(v) for v in value]
    if isinstance(value,str):
        for old,new in [('D92-SGJoint-v1','D92-MVKME-v1'),('D92-SGJoint','D92-MVKME'),
                        ('d92-sgjoint','d92-mvkme'),('d92_sgjoint','d92_mvkme'),
                        ('d92_summary_joint','d92_mv_kme'),('sgjoint','mvkme'),('SGJoint','MVKME')]:
            value=value.replace(old,new)
    return value


def documents(root,metadata,commit):
    root=Path(root)
    frozen=json.loads((root/'configs/d92_mv_kme_frozen_20260928.json').read_text(encoding='utf-8'))
    if frozen['algorithm']['method']!='D92-MVKME-v1':raise ValueError('Frozen MVKME config mismatch')
    prepared=build_documents(root,metadata,commit)
    result={replace_names(name):replace_names(value) for name,value in prepared.items()}
    for name,spec in result.items():
        if 'confirmation' not in spec:continue
        spec['description']='固定Phase1与已验证received IQ及support/query；同一IQ的固定16视图经冻结编码器和核均值表征，仅用当前row support训练岭回归头；复用原D92预测。'
        spec['permissions'].update(regime='fixed_phase1_received_only_multiview_current_row_support_ridge',
            summary_permission='No ground prototype or summary enters candidate fitting or prediction')
        spec['checkpoint'].update(runtime_checkpoint_reload=True,
            current_auxiliary_training='Frozen encoder; only current-row physical target support trains ridge head; no source L/U/V access',
            inheritance_policy='Existing scratch-only final200 Phase1 ancestry; verify_source metadata checked before exact native loading; no Phase1 update')
        spec['execution'].update(gpu_policy='GPU0 one frozen multiview exporter at a time; four CPU fit lanes with two BLAS threads; never stop unrelated processes')
        spec['confirmation']['candidate']=copy.deepcopy(frozen['algorithm'])
        spec['metrics_plan'].update(prediction_ref='Original reuse_row_root/predictions.jsonl paired with new output_root/mvkme/predictions.jsonl',
            resource_metrics=['extraction_seconds','fit_seconds','prediction_seconds','head_bytes','fourier_bytes',
                'cached_block_bytes','model_file_bytes','new_ground_statistics_bytes','peak_process_rss_bytes'])
        spec['expected_artifacts']=['startup.json','state.json','data_reuse.json','complete.json','scores.json',
            'each row/artifact_reuse.json','each row/mv_features/features_complete.json',
            'each row/mv_features/received_mv_features.npz','each row/mvkme/predictions_complete.json',
            'each row/mvkme/predictions.jsonl','each row/mvkme/fit_trace.jsonl',
            'each row/mvkme/compact.jsonl','each row/mvkme/compact.csv']
        for row in spec['rows']:
            row.update(gpu='0 (serial frozen exporter only; head CPU)',
                optimizer='Closed-form support-only ridge; nine fixed eta/gamma candidates; F=min(K,3) physical-ID folds',
                budget_ref=f"{spec['confirmation']['splits_per_model']} paired splits;16 frozen IQ views per sample;9 candidates,max3 folds;K1 fixed eta0.5/gamma0.1",
                expected_artifacts=['artifact_reuse.json','mv_features/startup.json','mv_features/features_complete.json',
                    'mv_features/received_mv_features.npz','mvkme/startup.json','mvkme/predictions.jsonl',
                    'mvkme/predictions_complete.json','mvkme/fit_trace.jsonl','mvkme/compact.jsonl','mvkme/compact.csv'])
            row['seed_notes'] += ' Fixed Fourier PCG64 seed0 is an algorithm constant, not a new physical data or model seed.'
        spec['notes']=[
            'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS; no new independent generalization claim.',
            'Formula independently designed without access to prior target scores; fixed in configs/d92_mv_kme_frozen_20260928.json.',
            'Reuse exact physical IQ/support/query and original D92 predictions; no source samples, source features, replay, ground summary fitting or Phase1 update.',
            'Each physical IQ creates16 deterministic views but remains one observation; no augmentation views split across folds and no increase of K.',
            'Fourier transform seed0/bandwidth1 are fixed mathematical constants. K1 eta0.5/gamma0.1;K>=2 selects only nine fixed candidates using current-row physical support.',
            'Both cohorts execute completely. Do not read scores until both terminate; no score-driven candidate changes, selective rows or performance stopping.',
            'Pool fourRX with equal paired-cell weights; perK H/new from new_count>0; old guard includes old-only cells; report complete matrix.',
            'No added source payload; frozen model deployment cost recorded separately from cache and local head. Sixteen encoder forwards per observation must be reported.',
            'Closed-form fit has no epochs, learning rate or iterative convergence claim; log measured loss components, timing and bytes; unavailable fields null with reasons.'
        ]
    return result


if __name__=='__main__':
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    outputs=documents(ROOT,DEFAULT_METADATA,commit)
    write_documents(ROOT,outputs)
    print(json.dumps(dict(status='PREPARED_ONLY',files=list(outputs),experiment_started=False)))
