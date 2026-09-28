"""Preregister fixed OSC on existing frozen orbit features; never read scores."""
import copy
import json
from pathlib import Path
import subprocess

from prepare_d92_bnna_benchmark import documents as inherited_documents
from prepare_d92_summary_joint_benchmark import ROOT, PROJECT, DEFAULT_METADATA, write_documents

FROZEN='configs/d92_orbit_shared_frozen_20260929.json'


def names(value):
    if isinstance(value,dict): return {k:names(v) for k,v in value.items()}
    if isinstance(value,list): return [names(v) for v in value]
    if isinstance(value,str):
        return (value.replace('d92_bnna_frozen_','d92_orbit_shared_frozen_')
                .replace('D92-BNNA','D92-OSC').replace('d92-bnna','d92-osc')
                .replace('d92_bnna','d92_osc').replace('bnna','osc').replace('BNNA','OSC')
                .replace('predict_d92_osc.py','predict_d92_orbit_shared.py'))
    return value


def documents(root,metadata,commit):
    root=Path(root)
    frozen=json.loads((root/FROZEN).read_text(encoding='utf-8'))
    if frozen['algorithm']['method']!='D92-OSC-v1': raise ValueError('Frozen OSC configuration mismatch')
    result={names(k):names(v) for k,v in inherited_documents(root,metadata,commit).items()}
    for spec in result.values():
        if 'confirmation' not in spec: continue
        cohort='rx3' if len(spec['data']['target_receivers'])==3 else 'rx1'
        cache_run=f'20260929-phase2-d92-bnna-repeat-{cohort}-m4-r01'
        cache_root=f'{PROJECT}/runs/{cache_run}'
        spec['description']='固定Phase1及既有received四相位特征；当前任务全部注册类support进行物理medoid循环对齐与共享收缩协方差估计，query逐样本对四种相对相位边缘化；复用原D92预测。'
        spec['permissions'].update(regime='fixed_phase1_frozen_orbit_cache_current_row_support_shared_covariance',
            summary_permission='No ground prototype/summary/source inputs; only unchanged frozen received-view cache reused',
            adapted_state_reuse=False)
        spec['checkpoint'].update(runtime_checkpoint_reload=False,
            initialization='No runtime checkpoint loading; unchanged frozen received features retain exact source-only final200 checkpoint lineage',
            current_auxiliary_training='Current-row support only, analytical orbit alignment and covariance; no source data or previous fitted state',
            inheritance_policy='Exact source-only scratch final200 checkpoint provenance retained in frozen-view cache; BNNA adapter/teacher/head are not loaded',
            feature_cache_producer_run_id=cache_run,
            feature_cache_producer_role='Frozen encoder outputs only; not model or adapted-state inheritance')
        spec['execution']['gpu_policy']='CPU only; four model lanes with two BLAS threads; reuse frozen received features without encoder/checkpoint reload or GPU allocation'
        spec['confirmation'].update(candidate=copy.deepcopy(frozen['algorithm']),frozen_feature_source_root=cache_root,
            frozen_feature_producer_config='configs/d92_bnna_frozen_20260929.json')
        spec['metrics_plan']['resource_metrics']=['fit_seconds','alignment_seconds','covariance_seconds','solve_seconds',
            'query_score_seconds','prediction_write_seconds','persistent_state_bytes','working_array_bytes',
            'feature_cache_file_bytes','model_file_bytes','new_source_payload_bytes','peak_process_rss_bytes']
        artifacts=['artifact_reuse.json','frozen_feature_reuse.json','osc/startup.json','osc/predictions.jsonl',
            'osc/predictions_complete.json','osc/fit_trace.jsonl','osc/compact.jsonl','osc/compact.csv',
            'osc/fit_stages.jsonl','osc/fit_stages.csv']
        spec['expected_artifacts']=['startup.json','state.json','data_reuse.json','complete.json','scores.json']+['each row/'+a for a in artifacts]
        for row in spec['rows']:
            row.update(gpu=None,lr=None,epochs=None,
                reuse_multiview_features_root=cache_root+'/'+row['row_id']+'/bnna_features',
                optimizer='None; analytical medoid/orbit alignment,physical-df covariance shrinkage,Cholesky solve; zero gradient updates',
                budget_ref=f"{spec['confirmation']['splits_per_model']} paired splits;K1 one fixed spherical fit;K>=2 F=min(K,3) diagnostic physical folds plus one final fit;one formula,no candidates or parameter selection",
                expected_artifacts=artifacts)
            row['provenance_refs'].update(frozen_orbit_features_complete=row['reuse_multiview_features_root']+'/features_complete.json',
                frozen_orbit_checkpoint_provenance=row['reuse_multiview_features_root']+'/checkpoint_provenance.json')
            row['seed_notes']='Deterministic OSC introduces no new RNG roles; retain all six explicit existing seed roles. Views remain grouped by physical ID, without increasing K.'
        spec['notes']=[
            'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS; no new independent generalization claim.',
            'OSC design independently frozen before BNNA scores were read, design commit a2d721e943b4369aeb8abe71ff3537c5c74d1e56; formula fixed in '+FROZEN+'.',
            'Reuse exact original D92 predictions and unchanged frozen four-phase identity160/FFT96 cache; no BNNA fitted state, predictions or scores enter OSC.',
            'Feature cache lineage requires matching model SHA/seed/classes, capsule/physical IDs, four-phase full-IQ view rule, historical FFT and float32 schema; shape alone is insufficient.',
            'No source samples/per-record features/replay, ground prototypes/summaries, Phase1 update, new dataset validation or encoder execution.',
            'All classes use one shared covariance with physical df C*(n-1), never4*C*(n-1); fixed shrinkage256/(df+256);K1 spherical.',
            'Each physical fold refits medoid,shifts,templates,covariance and compiled W/b from that fold training support only;OOF diagnostic only,no selection.',
            'Both cohorts finish before either target scores file is read; no score-driven edits,selective rows or performance stopping.',
            'Equal paired-cell weights across fourRX;H/new fromnew_count>0,old guard fromall cells including old-only;allK/seed/scenario/class-count results retained.',
            'Log measured analytic phase costs and diagnostics;optimizer_steps=0,epoch/LR/gradient null with reasons;source validation absent by permission.',
            'New source payload0B;existing model deployment unknown,null incremental model transmission;training checkpoint package bytes are not a minimal inference package.',
            'Compiled state stores only W[C,4,256] and b[C] plus identifiers/audit;no training features or covariance persist in inference state.'
        ]
    return result


if __name__=='__main__':
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=documents(ROOT,DEFAULT_METADATA,commit)
    write_documents(ROOT,result)
    print(json.dumps(dict(status='PREPARED_ONLY',files=list(result),experiment_started=False)))
