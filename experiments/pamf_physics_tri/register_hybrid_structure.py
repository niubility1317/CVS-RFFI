"""Register the fixed structural A/B/C/D comparisons."""
import argparse
import json
from pathlib import Path
from . import register_hybrid_confidence as base
from .hybrid_structure_plan import RUN,candidate_rows


def make(template):
    old=base.RUN,base.candidate_rows;base.RUN,base.candidate_rows=RUN,candidate_rows
    try:record=base.make(template)
    finally:base.RUN,base.candidate_rows=old
    record.update(display_name='Hybrid 主路径与融合辅助 A/B/C/D 对照',
        description='Main 落后种子 01 与 10 类三个落后种子的原路由／全物理路由和零辅助／融合辅助结构对照。',
        aliases=['PAMF-hybrid-fusion-only-route-abcd'],tags=['PAMF','Hybrid','structural-ablation','exploratory'])
    record['data']['dataset']='original Main split and ten-class complete L/U/V roles'
    record['permissions']['claim_scope']='previously exposed tests selected seeds; exploratory, no checkpoint selection'
    record['checkpoint']['selection_rule']='fixed E50 on all 16 rows; no target-score checkpoint choice'
    record['execution'].update(local_artifact_root='local_artifacts/pamf_hybrid_structure_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_structure_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    record['expected_artifacts']=['real-checkpoint no-query smoke across two settings and four conditions',
        '16 complete-source E30/E50 continuations', 'raw and DURA-LU seven-view frozen predictions and truth-last scores',
        'per-exposure source U trajectory and full U loss for all arms; C/D auxiliary gradient and same-state AdamW delta',
        'same-setting same-seed comparisons to A, ordinary PL and original Hybrid']
    record['metrics_plan']['dimensions']=['Main seed 01 plus 10-class seeds 01/02/03','four structural conditions',
                                          'raw and DURA-LU clean and six practical','source U packet trajectory']
    for entry,row in zip(record['rows'],candidate_rows()):
        entry.update(method='Tri-derived source baseline; structural condition '+row['structure_arm'],
                     purpose='fixed input-routing and fusion-only auxiliary ablation')
    record['notes']=[
        'Main was explicitly prioritized after a 10-class/Main analysis; only its six-scene-underperforming seed 01 is run. All three 10-class seeds underperform PL.',
        'A ordinary/raw baseline without auxiliary; B all-confident physical main route without auxiliary; C baseline route with separate fusion-only auxiliary; D all-confident physical route with the same auxiliary.',
        'The declared baseline retains Tri-derived U style, joint contrast, RX extrapolation, cross-day relation, branch gradient guard and supervised objective. It is not ordinary PL.',
        'All conditions use the same 0.25-to-zero physical input schedule; nonzero auxiliary conditions use fixed weight 0.1, not a tuning grid.',
        'U trajectories and periodic same-state AdamW base-versus-full counterfactuals are source-only diagnostics; no U labels or target data enter training.',
        'A/B have no auxiliary term so their auxiliary-added same-state difference is mathematically zero; no shadow-update file is generated for A/B.',
        'Every row automatically evaluates raw and DURA-LU clean and six practical scenes; already exposed target results are exploratory.']
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
