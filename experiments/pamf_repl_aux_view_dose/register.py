"""Register the two 3×3 fixed-view auxiliary matrices and historical anchors."""
import argparse
import json
from pathlib import Path

from experiments.pamf_repl_priority import register as priority
from . import plan


def make(template):
    old = priority.plan
    priority.plan = plan
    try:
        result = priority.make(template)
    finally:
        priority.plan = old
    result.update(display_name='十类与六源固定 KL 视图的辅助损失剂量对照',
                  description='每设置 3×3 剂量矩阵，复用 1 个精确对照，各训练 8 个新变体。',
                  authorization='2026-11-18 用户设置持续优化 Tri 的 active goal，要求批量匹配归因与源盲验证。')
    result['execution'].update(
        gpu_policy='GPU 0–7; balanced free capacity; reserve 12 GiB per initializing worker, maximum 5 with other owners included',
        local_artifact_root='local_artifacts/pamf_repl_aux_view_dose_20261118',
        launch_command='python -m experiments.pamf_repl_aux_view_dose.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts'] = ['16 source-only real checkpoint optimizer smokes',
        '16 E50 clean frozen predictions and independent scores', '16 post-freeze U diagnostics',
        'stepwise per-view CE/KL and parameter-group L/U gradients', 'same-state AdamW samples every 100 steps']
    result['notes'] = [
        'Shared fixed E20 ancestry, seed and E50 endpoints; base pseudo-label CE fixed to zero in every variant.',
        'Ten-class: ordinary KL with coefficients .5/1/1.5 crossed with shallow contrast .5/1/2.',
        'Six-source: raw KL with coefficients .75/1.5/2.25 crossed with fusion contrast 0/.05/.10.',
        'Both matrices reuse the exact central CE=0 row from the prior view experiment. All forwards and pseudo-label rule stay fixed.',
        'Independent E50 clean and frozen-only independent U diagnostics; selected seed and exposed test, exploratory only.',
        'Candidate allocation fixed before launch. No target or U truth during training or checkpoint selection.']
    for row in result['rows']:
        row['expected_artifacts'].extend(['source/segment30/adamw_LU_samples.jsonl.gz',
                                         'source/segment50/adamw_LU_samples.jsonl.gz'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
