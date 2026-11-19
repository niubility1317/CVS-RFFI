import argparse
import json
from pathlib import Path

from experiments.pamf_repl_aux_view_dose import register as anchor
from experiments.pamf_repl_day_pairs.plan import RUN as date_run
from . import plan


def make(template):
    original = anchor.plan
    anchor.plan = plan
    try:
        result = anchor.make(template)
    finally:
        anchor.plan = original
    result.update(display_name='Tri 十类三因素探索与六源独立种子复核',
                  description='十类固定起点八格完全因子矩阵，六源两配方在两个固定独立种子上配对。',
                  authorization='持续目标要求十类和六源超过匹配 PL，并对固定配方使用独立种子核验。')
    result['parent_run_ids'].extend([plan.adaptive.fixed.RUN, plan.adaptive.RUN, date_run])
    result['execution'].update(gpu_policy='GPU 0–6, up to two total processes per GPU including other owners, reserve 12 GiB',
        local_artifact_root='local_artifacts/pamf_repl_two_setting_20261119',
        launch_command='python -m experiments.pamf_repl_two_setting.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['permissions']['claim_scope'] = 'Ten-class development on exposed seed 01; fixed six-source recipes on seeds 02 and 03, historical target design'
    result['checkpoint']['selection_rule'] = 'Every row fixed E50; six-source recipes frozen before evaluating seeds 02 and 03'
    result['expected_artifacts'] = ['12 real source E20 optimizer-step smokes', '12 fixed E50 raw clean predictions and independent scores',
        '12 frozen-only U diagnostics', 'gradient, optimizer, per-loss and source validation telemetry']
    result['metrics_plan']['dimensions'] = ['ten-class seed 01 and six-source seeds 02/03', 'three raw clean subsets', 'paired physical IDs']
    result['notes'] = [
        'Ten-class factorial holds U .75, shallow 2 and ordinary KL dose 1.5 constant.',
        'Ten-class factors: raw CE .25, view-disagreement weighting .5, shallow positive RX balance .5; full switches and interactions.',
        'Six-source: fixed raw KL + fusion .10 and the identical recipe with same-source-date preferred positives .50.',
        'Report both six-source seeds individually before summarizing; benchmark against same-setting and same-seed ordinary PL and original Tri.',
        'Source U truth stays isolated until independent diagnostics after E50 freeze. No target or U truth for training.',
    ]
    for entry, row in zip(result['rows'], plan.candidate_rows(), strict=True):
        entry['seed_notes'] = 'Exploratory seed selected earlier' if row['setting'] == 'ten_class' else 'Model seed held out from these recipe choices; historical test design'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
