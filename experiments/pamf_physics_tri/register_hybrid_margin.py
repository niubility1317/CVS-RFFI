"""Register the fixed capped-margin replacements before launch."""
import argparse
import json
from pathlib import Path
from . import register_hybrid_structure as base
from .hybrid_margin_plan import RUN, candidate_rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, candidate_rows
    try:
        record = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    record.update(display_name='Hybrid 有上限分类间隔替换对照',
        description='固定 Main seed 01 和 10 类三个 seed，以分类间隔替换融合层辅助对比。',
        aliases=['PAMF-hybrid-capped-margin'], tags=['PAMF', 'Hybrid', 'margin', 'exploratory'])
    record['checkpoint']['selection_rule'] = 'fixed E50 for all four rows; no target-score checkpoint choice'
    record['execution'].update(local_artifact_root='local_artifacts/pamf_hybrid_margin_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_margin_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    record['expected_artifacts'] = ['two real-checkpoint no-query smokes', 'four source-L-derived fixed margin caps',
        'four complete-source E30/E50 continuations', 'raw and DURA-LU seven-view predictions and truth-last scores',
        'source U packet confidence and margin violation trajectory; same-state AdamW delta',
        'independent paired corrections and damage versus matching fusion-only C']
    record['metrics_plan']['dimensions'] = ['Main seed 01 plus 10-class seeds 01/02/03',
        'source-L capped margin', 'raw and DURA-LU clean and six practical', 'source U packet trajectory']
    for entry in record['rows']:
        entry.update(method='Tri-derived baseline with isolated capped-margin auxiliary',
                     purpose='replace fusion-only contrast with capped classification margin')
    record['notes'] = [
        'Main seed 01 and all three 10-class seeds match the selected structural C control; previously exposed tests make every result exploratory.',
        'The baseline route, physical-input schedule, original U confidence threshold and fixed auxiliary coefficient match C; only the extra objective is replaced.',
        'The cap is the median unscaled cosine margin among correctly classified original source-L examples, measured once from the E20 model at the first U update after the initial labeled BN update; the E50 segment reuses it.',
        'Original L/U/V roles remain intact, no source holdout or target labels are used for the cap, and original PL supervision uses every confidence-accepted U.',
        'Classifier weights and the clean reference are detached only in the margin auxiliary. The same model continues normal supervised and ordinary U training.',
        'Trace records per-packet confidence, pseudo-label, actual perturbation norm and margin violation, and a periodic same-state AdamW update comparison.',
        'Frozen raw and DURA-LU predictions precede independent seven-view row scoring and C/M paired repair/damage analysis.']
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
