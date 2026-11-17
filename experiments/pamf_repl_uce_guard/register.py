"""Preregister CE guard strength and matched auxiliary settings."""
import argparse,json
from pathlib import Path
from experiments.pamf_repl_priority import register as priority
from . import plan
def make(template):
    old=priority.plan;priority.plan=plan
    try: result=priority.make(template)
    finally: priority.plan=old
    result.update(display_name='十类与六源基础伪标签 CE 梯度软保护',
        description='两设置各8行，逐组软化基础U伪标签CE相对监督梯度的负分量，并与辅助项保护和U剂量匹配对照。',
        authorization='2026-11-18 用户授权三小时后分析并继续发布clean优化实验；冻结后独立U评分。')
    result['execution'].update(gpu_policy='GPU 0–7; account for other owners; reserve 12 GiB per initializing worker and <=5/card',
        local_artifact_root='local_artifacts/pamf_repl_uce_guard_20261118',
        launch_command='python -m experiments.pamf_repl_uce_guard.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['16 real-checkpoint no-query optimizer-step smokes',
        '16 E30/E50 source trajectories with every-step gradient and CE-guard telemetry',
        '16 blind U observations with independent frozen post-training correctness diagnostics',
        '16 frozen clean predictions and independent truth-last scores',
        'same-state AdamW L-only, raw joint and actual updates sampled every 100 steps']
    result['notes']=['Exposed target and selected single seed; exploratory, not independent confirmation.',
        'Raw and auxiliary-only controls are prior matching rows; gamma .25/.5/1 projects base U CE only.',
        'CE is recovered as full U minus the four actual weighted auxiliary graphs, not from source U truth.',
        'L gradient is preserved before global clipping; AdamW and clipping may change effective update direction.',
        'Fixed E50; independent clean truth-last and frozen-only U truth scoring; no satellite or weak.',
        'Systemic failure stops new dispatch and retains outputs and healthy workers.']
    for row in result['rows']:
        row['expected_artifacts'].extend(['source/segment30/adamw_LU_samples.jsonl.gz','source/segment50/adamw_LU_samples.jsonl.gz'])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();v=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
