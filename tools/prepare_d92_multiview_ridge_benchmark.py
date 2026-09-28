"""Preregister physical-weighted multiview ridge without reading any scores."""
import copy
import json
from pathlib import Path
import subprocess

from prepare_d92_orbit_shared_benchmark import documents as inherited_documents
from prepare_d92_summary_joint_benchmark import ROOT, DEFAULT_METADATA, write_documents

FROZEN = 'configs/d92_multiview_ridge_frozen_20260929.json'


def names(value):
    if isinstance(value, dict): return {k: names(v) for k, v in value.items()}
    if isinstance(value, list): return [names(v) for v in value]
    if isinstance(value, str):
        return (value.replace('d92_orbit_shared_frozen_', 'd92_multiview_ridge_frozen_')
                .replace('predict_d92_orbit_shared.py', 'predict_d92_multiview_ridge.py')
                .replace('D92-OSC', 'D92-MVRidge').replace('d92-osc', 'd92-mvridge')
                .replace('d92_osc', 'd92_mvridge').replace('osc/', 'mvridge/')
                .replace('osc.log','mvridge.log').replace('Deterministic OSC','Deterministic MVRidge'))
    return value


def documents(root, metadata, commit):
    root = Path(root)
    frozen = json.loads((root/FROZEN).read_text(encoding='utf-8'))
    if frozen['algorithm']['method'] != 'D92-MVRidge-v1': raise ValueError('Frozen method mismatch')
    result = {names(k): names(v) for k, v in inherited_documents(root, metadata, commit).items()}
    for spec in result.values():
        if 'confirmation' not in spec: continue
        spec['tags'] = ['mvridge' if tag == 'osc' else tag for tag in spec['tags']]
        spec['group_id'] = 'd92-fixed-phase1-mvridge-repeated-benchmark'
        spec['description'] = '固定Phase1与既有四相位received特征；全部注册类support以每物理样本总权重1拟合共享ridge判别头，单位ridge正则，无old/new组权或teacher。query逐样本四视图均值推理。'
        spec['permissions']['regime'] = 'fixed_phase1_current_row_support_physical_weighted_multiview_ridge'
        spec['checkpoint']['current_auxiliary_training'] = 'Current-row labeled physical support and its fixed views only; closed-form ridge, no gradient optimization or inherited head'
        spec['confirmation'].update(candidate=copy.deepcopy(frozen['algorithm']), candidate_folder='mvridge',
            candidate_method='D92-MVRidge-v1', candidate_predictor='predict_d92_multiview_ridge.py',
            candidate_mode='d92_mvridge_registration')
        spec['metrics_plan']['resource_metrics'] = ['fit_seconds','query_score_seconds','prediction_write_seconds',
            'persistent_state_bytes','working_array_bytes','feature_cache_file_bytes','model_file_bytes',
            'new_source_payload_bytes','peak_process_rss_bytes']
        for row in spec['rows']:
            row['optimizer'] = 'None; exact centered physical-weighted multiview ridge Cholesky solve, regularization 1; zero gradient steps'
            row['budget_ref'] = (f"{spec['confirmation']['splits_per_model']} fixed paired splits; K1 one fit without CV; "
                'K>=2 F=min(K,3) diagnostic physical folds plus full fit; no candidate or parameter selection')
        spec['notes'] = [
            'REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS; independent new data remains deferred until broad improvement.',
            'Formula independently designed without target results: docs/D92_SUPPORT_INFORMATION_DESIGN_20260929.md; frozen in '+FROZEN+'.',
            'Reuse only unchanged frozen four-phase received features; no BNNA/OSC/SFHead fitted state, scores or predictions enter MVRidge.',
            'Each physical observation has total training loss weight one across its four views; K remains physical sample count.',
            'Use unit([unit(identity160),4*unit(FFT96)]), centered one-hot responses, common squared-loss head and fixed ridge coefficient one.',
            'All classes use identical formula; no old/new group loss weighting, teacher distillation or per-class hyperparameters.',
            'No source data/embeddings/replay/ground summaries/prototypes, Phase1 updates, checkpoint reload, new data validation or feature extraction.',
            'Each physical fold re-estimates mean, feature Gram, view scatter, cross moment and head solely on trainfold; OOF diagnostic only.',
            'K1 has a supervised shared discriminant fit, but no independent validation or claim to identify physical within-class covariance.',
            'Both full cohorts complete before either scores file is read; no target-score-driven edits or selective reruns.',
            'Equal paired-cell means across four RX; H/new from joint tasks, old guard includes old-only; preserve complete matrix.',
            'Log measured loss components, regularization, linear-solve gradient residual and costs; optimizer_steps=0; epoch/LR/grad-update null with reasons.',
            'New source payload0B; model deployment unknown and incremental model transmission null; complete training checkpoint size is not minimum inference size.',
            'Inference state consists of W[C,256],b[C],class IDs and audit; no persistent support features/covariance; numeric bytes2056*C.'
        ]
    return result


if __name__ == '__main__':
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    result = documents(ROOT, DEFAULT_METADATA, commit)
    write_documents(ROOT, result)
    print(json.dumps(dict(status='PREPARED_ONLY',files=list(result),experiment_started=False)))
