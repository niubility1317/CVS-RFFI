"""Register fixed lower Hybrid auxiliary weights without changing U acceptance."""
import argparse
import json
from pathlib import Path

from . import register_hybrid_confidence as base
from .hybrid_low_plan import RUN, candidate_rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, candidate_rows
    try:
        result = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    root = 'PROJECT/runs/' + RUN
    result.update(display_name='Hybrid 全置信样本低辅助权重对照',
        description='四种配置、三种子比较全置信 Hybrid 辅助权重 0.05 与 0.10，并配对既有全权重。',
        aliases=['PAMF-hybrid-confidence-low-auxiliary-weight'], tags=['PAMF', 'Hybrid', 'confidence-only', 'auxiliary-weight'])
    result['data']['dataset'] = 'original receiver split 2, ten-class, 1%-labeled and six-source-receiver'
    result['execution'].update(local_artifact_root='local_artifacts/pamf_hybrid_low_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_low_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    result['expected_artifacts'] = ['real-checkpoint no-query smoke per weight and setting',
        '24 matching full-source E30/E50 continuations', '24 direct seven-view predictions and independent truth-last scores',
        'matched ordinary PL, original Hybrid and full-weight confidence Hybrid comparisons']
    result['metrics_plan']['dimensions'] = ['four configurations', 'two auxiliary weights', 'three seeds',
                                             'clean and six practical', 'TX/RX/day']
    result['metrics_plan'].update(prediction_ref=root + '/readouts/ROW/predictions', scorer_ref=root + '/readouts/ROW/scores')
    for entry, row in zip(result['rows'], candidate_rows()):
        entry['method'] = f"Hybrid all confidence >=0.95; extra auxiliary weight {row['auxiliary_weight']}; no vote or prototype"
    result['notes'] += [
        'The fixed 0.05 and 0.10 coefficients scale only the additional Hybrid fusion/branch auxiliary loss; all confidence-accepted U remain selected.',
        'The 1.0 confidence-only Hybrid is the completed 20261113-phase1-pamf-hybrid-confidence-m30-r01, reused by setting and seed.',
        'Full-weight scores never block reduced-weight training or testing; incomplete comparisons are recorded as pending.']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
