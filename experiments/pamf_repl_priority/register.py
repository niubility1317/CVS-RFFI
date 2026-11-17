"""Preregister all fixed setting-specific candidates and clean readouts."""
import argparse
import json
from pathlib import Path
from experiments.pamf_repl_weight import register as base
from . import plan


def make(template):
    old=base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows
    base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=(
        plan.RUN,None,plan.RECIPES,plan.candidate_rows)
    try:result=base.make(template)
    finally:base.RUN,base.PREVIOUS_RUN,base.RECIPES,base.candidate_rows=old
    result.update(display_name='十类浅层参考与六源 KL 参考改进探索',
        description='十类 8 行与六源 12 行；固定 U 权重、损失因子与梯度规则，E20 各自续训 E50。',
        parent_run_ids=['20261027-phase1-pamf-paper-completion-m119-r03'],replaces_run_id=None,
        authorization='2026-11-17 用户要求按十类浅层、六源KL方向直接发布改进实验；clean only。')
    result['data']['roles']={'L_s':'labeled training','U_s':'TX blind in training; offline diagnostic only after E50',
                             'V':'source validation without gradients'}
    result['checkpoint'].update(initialization='matching original same-setting source-only Tri E20 then own E21–E50',
        sources=['20261027-phase1-pamf-paper-completion-m119-r03'],
        contract_check_ref='pamf_repl_component_guard.plan.parent with scoped priority matrix',
        provenance_verdict='same source roles, physical IDs, seed and original source-only E20, no target-trained checkpoint',
        selection_rule='all fixed rows E50; candidate design uses previously exposed clean, so exploration only')
    result['execution'].update(gpu_policy='GPU 1,3,5,7; all owners/pending <=5 per GPU; >=12 GiB free; one own worker per GPU',
        local_artifact_root='local_artifacts/pamf_repl_priority_20261117',
        launch_command='python -m experiments.pamf_repl_priority.publish --repo REPO --output OUTPUT --context CONTEXT --ssh-config CONFIG')
    result['expected_artifacts']=['20 real-checkpoint no-query optimizer-step smokes',
        '20 E30/E50 source trajectories with every-step gradients and sampled actual AdamW reference updates',
        '20 blind U observation files and frozen independent U correctness diagnostics',
        '20 raw clean predictions and independent truth-last scores']
    result['metrics_plan']['metric_names']=['clean accuracy and macro-F1','L CE and source validation',
        'U acceptance, unique coverage and post-freeze correctness','group raw and applied gradients, reference conflicts and caps',
        'every-100-step actual joint and same-state reference-only AdamW direction']
    result['notes']=['Design is based on exposed target clean and chosen seed; all results are exploratory.',
        'Ten-class reference is actual U shallow contrast, while raw control has no L-anchored native U guard.',
        'Six-source exact ordinary/mixed/raw U KL graph supplies reference; As KL remains 0; native-guard controls remain.',
        'Reference control projects and caps remainder at 0.5 reference norm; zero-reference falls back to raw joint gradient.',
        'No guarantee of AdamW update alignment; record every-100-step actual update versus same-state reference-only update.',
        'Blind U truth only scored independently after freeze; never used by training or model selection.',
        'Every candidate has fixed E50 and frozen clean prediction followed by truth-last independent scoring; no satellite.']
    for entry,row in zip(result['rows'],plan.candidate_rows()):
        rid=row['row_id'];entry['method']='Tri priority '+row['recipe']
        entry['budget_ref']='same-contract source E20 reused; own E21–E50 6660 updates; total 11100'
        entry['expected_artifacts']=['source/complete.json','source/segment30/U_pseudo_observations.jsonl.gz',
            'source/segment50/U_pseudo_observations.jsonl.gz','source/segment30/reference_adamw_samples.jsonl.gz',
            'source/segment50/reference_adamw_samples.jsonl.gz','readouts/'+rid+'/U_diagnostic/scores.json',
            'readouts/'+rid+'/predictions/complete.json','readouts/'+rid+'/scores/complete.json']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
