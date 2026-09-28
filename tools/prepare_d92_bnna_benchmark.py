"""Prepare frozen BNNA cohorts from provenance/configs, never target scores."""
import copy
import json
from pathlib import Path
import subprocess

from prepare_d92_summary_joint_benchmark import ROOT, DEFAULT_METADATA, build_documents, write_documents

FROZEN = 'configs/d92_bnna_frozen_20260929.json'


def replace_names(value):
    if isinstance(value, dict): return {k: replace_names(v) for k, v in value.items()}
    if isinstance(value, list): return [replace_names(v) for v in value]
    if isinstance(value, str):
        for old, new in [('20260928-phase2-d92-sgjoint', '20260929-phase2-d92-bnna'),
                         ('d92_sgjoint_repeat_rx3_20260928', 'd92_bnna_repeat_rx3_20260929'),
                         ('d92_sgjoint_repeat_rx1_20260928', 'd92_bnna_repeat_rx1_20260929'),
                         ('d92_sgjoint_repeat_rx3_data_20260928', 'd92_bnna_repeat_rx3_data_20260929'),
                         ('d92_sgjoint_repeat_rx1_data_20260928', 'd92_bnna_repeat_rx1_data_20260929'),
                         ('d92_summary_joint_frozen_20260928', 'd92_bnna_frozen_20260929'),
                         ('D92-SGJoint', 'D92-BNNA'), ('d92-sgjoint', 'd92-bnna'),
                         ('d92_summary_joint', 'd92_bnna'), ('sgjoint', 'bnna'), ('SGJoint', 'BNNA')]:
            value = value.replace(old, new)
    return value


def documents(root, metadata, commit):
    root = Path(root)
    frozen = json.loads((root / FROZEN).read_text(encoding='utf-8'))
    if frozen['algorithm']['method'] != 'D92-BNNA-v1': raise ValueError('Frozen BNNA config mismatch')
    result = {replace_names(k): replace_names(v) for k, v in build_documents(root, metadata, commit).items()}
    for spec in result.values():
        if 'confirmation' not in spec: continue
        spec['description'] = '固定Phase1和既有received IQ及support/query；当前任务support的4个确定性相位视图训练有界非线性适应器；完整物理样本fold重拟合全部状态；复用原D92预测。'
        spec['authorization'] = '用户授权先复用数据，待新旧类和各K综合性能明显改善后再新增独立数据验证；允许不读取源域样本的辅助训练。'
        spec['permissions'].update(regime='fixed_phase1_received_only_current_row_support_bounded_nonlinear_adapter',
            summary_permission='No ground prototype or summary enters candidate fitting or prediction')
        spec['checkpoint'].update(runtime_checkpoint_reload=True,
            current_auxiliary_training='Frozen encoder; only current-row physical target support trains bounded coefficients; no source L/U/V access',
            inheritance_policy='Existing scratch-only final200 ancestry; verify_source metadata before exact native loading; no Phase1 update')
        spec['data']['representation'] = 'Unchanged received full-length IQ; four fixed quarter-phase views through frozen identity160 plus original FFT96; one physical observation remains one support'
        spec['execution']['gpu_policy'] = 'GPU0 one serial frozen exporter per cohort; at most two cohort exporters; four CPU fit lanes per cohort with two BLAS threads; preserve unrelated tasks'
        spec['confirmation']['candidate'] = copy.deepcopy(frozen['algorithm'])
        spec['metrics_plan'].update(prediction_ref='Original reuse_row_root/predictions.jsonl paired with output_root/bnna/predictions.jsonl',
            resource_metrics=['extraction_seconds', 'fit_seconds', 'query_score_seconds', 'prediction_write_seconds',
                              'persistent_state_bytes', 'support_feature_bytes', 'model_file_bytes',
                              'new_source_payload_bytes', 'peak_process_rss_bytes'])
        artifacts = ['artifact_reuse.json', 'bnna_features/startup.json', 'bnna_features/features_complete.json',
            'bnna_features/received_bnna_features.npz', 'bnna/startup.json', 'bnna/predictions.jsonl',
            'bnna/predictions_complete.json', 'bnna/fit_trace.jsonl', 'bnna/compact.jsonl', 'bnna/compact.csv',
            'bnna/training_steps.jsonl', 'bnna/training_steps.csv']
        spec['expected_artifacts'] = ['startup.json', 'state.json', 'data_reuse.json', 'complete.json', 'scores.json'] + ['each row/' + p for p in artifacts]
        for row in spec['rows']:
            row.update(gpu='0 (frozen exporter only; adapter CPU)', lr=0.03, epochs=None,
                optimizer='Float64 full-batch projected Adam,64 steps,lr0.03,beta0.9/0.999,epsilon1e-8; coefficients clipped to[0,0.5];rank0 exact identity skips updates',
                budget_ref=f"{spec['confirmation']['splits_per_model']} paired splits;4 full-IQ phase views;rank<=8;K1 fixed training/no CV;K>=2 max3 physical folds plus final fit;max256 Adam updates per split",
                expected_artifacts=artifacts)
        spec['notes'] = [
            'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS; no new independent generalization claim.',
            'Formula independently designed without historical target scores; frozen in ' + FROZEN + '.',
            'Reuse exact physical IQ/support/query and original D92 predictions; no source samples, source features, replay, ground summary fitting or Phase1 update.',
            'Four deterministic full-IQ phase views remain one physical observation; folds group every view and use only this row support.',
            'K1 has no validation or model choice; use fixed training recipe. K>=2 compares only fixed zero adapter versus trained adapter, using complete refit of basis, coefficients and prototypes inside each fold.',
            'Select support-CV arm by max old/new macro NLL then all-class macro NLL,round10decimals,tie zero adapter; no score-driven hyperparameter tuning.',
            'Both cohorts execute completely before either scores file is read; no selective rows or performance stopping.',
            'Pool fourRX with equal paired-cell weights; perK H/new use new_count>0; old guard includes old-only; report complete matrix and all model seeds.',
            'No additional source payload; report actual model checkpoint package and unknown deployment separately from local cache/adapter storage.',
            'Log actual64 optimizer steps per nonzero-rank fit: measured CE,view variance,coefficient penalty,weights,LR,gradient and time. No epoch loop; source validation null because source access is forbidden.',
            'Rank0 is the predeclared identity case, not an error fallback; nonfinite states fail technically without replacing the method.'
        ]
    return result


if __name__ == '__main__':
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    outputs = documents(ROOT, DEFAULT_METADATA, commit)
    write_documents(ROOT, outputs)
    print(json.dumps(dict(status='PREPARED_ONLY', files=list(outputs), experiment_started=False)))
