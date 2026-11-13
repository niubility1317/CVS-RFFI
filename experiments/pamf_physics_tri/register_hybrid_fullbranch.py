"""Register matched fusion-plus-branch controls for fusion-only C."""
import argparse,json
from pathlib import Path
from . import register_hybrid_structure as base
from .hybrid_fullbranch_plan import RUN,candidate_rows


def make(template):
    old=base.RUN,base.candidate_rows;base.RUN,base.candidate_rows=RUN,candidate_rows
    try:record=base.make(template)
    finally:base.RUN,base.candidate_rows=old
    record.update(display_name='Hybrid 融合与五分支匹配对照',description='与 C 条件完全匹配、仅增加五分支辅助项的四个探索性种子。',
                  aliases=['PAMF-hybrid-fusion-branch-matched'],tags=['PAMF','Hybrid','structural-ablation','exploratory'])
    record['checkpoint']['selection_rule']='fixed E50 for all four selected rows; no target-score checkpoint selection'
    record['execution'].update(local_artifact_root='local_artifacts/pamf_hybrid_fullbranch_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_fullbranch_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    record['expected_artifacts']=['two real-checkpoint no-query smokes','four complete-source E30/E50 continuations',
        'raw and DURA-LU seven-view predictions and truth-last scores','source U packet trajectory and same-state AdamW delta',
        'independent paired corrections and damage vs matching fusion-only C']
    record['metrics_plan']['dimensions']=['Main seed 01 plus 10-class seeds 01/02/03','fusion plus five branches',
                                          'raw and DURA-LU clean and six practical','source U packet trajectory']
    for entry,row in zip(record['rows'],candidate_rows()):
        entry.update(method='Tri-derived baseline; independent fusion plus five-branch auxiliary',
                     purpose='matched E versus fusion-only C structural comparison')
    record['notes']=[
        'Same seeds, model initialization, complete source roles, optimizer, input routing, perturbation taper and fixed weight as C; only branch auxiliary is added.',
        'All confidence-accepted source U are eligible; no source U true labels are read.',
        'Complete source U trajectory and 50-step same-state AdamW update differences are retained for later diagnosis.',
        'Raw and DURA-LU decisions freeze before independent row scoring; paired C/E corrections and damage use truth only after both predictions freeze.',
        'Test-selected seeds and previously exposed test results are exploratory; no target feedback changes training or checkpoint selection.']
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
