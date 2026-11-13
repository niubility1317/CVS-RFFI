"""Register selected failing seeds with explicitly exploratory scope."""
import argparse
import json
from pathlib import Path
from . import register_hybrid_confidence as base
from .hybrid_tracked_plan import RUN, candidate_rows


def make(template):
    old = base.RUN, base.candidate_rows
    base.RUN, base.candidate_rows = RUN, candidate_rows
    try:
        record = base.make(template)
    finally:
        base.RUN, base.candidate_rows = old
    record.update(display_name='Hybrid 失败种子隔离视图与样本轨迹',
        description='五配置、13 个已曝光测试落后种子，每种子比较伪类别对比和同信号正对两种辅助方式。',
        aliases=['PAMF-hybrid-tracked-failing-seeds'], tags=['PAMF','Hybrid','packet-trajectory','exploratory'])
    record['data']['dataset'] = 'receiver split 2, ten-class, 1%- and 5%-labeled, six-source-receiver original full-source roles'
    record['permissions']['claim_scope'] = 'previously exposed target selected these settings and seeds; exploratory only, never checkpoint selection'
    record['checkpoint']['selection_rule'] = 'fixed E50 for each preselected row; test-selected failing seeds are exploratory'
    record['execution'].update(local_artifact_root='local_artifacts/pamf_hybrid_tracked_20261113',
        launch_command='python -m experiments.pamf_physics_tri.hybrid_tracked_publish --repo REPO --output LOCAL_ROOT --context CONTEXT --ssh-config N607_CONFIG')
    record['expected_artifacts'] = ['real-checkpoint no-query smoke per setting and auxiliary mode',
        '26 full-source E30/E50 continuations', '26 direct seven-view predictions and independent truth-last scores',
        'source U per-exposure compressed packet trace and 50-step auxiliary gradient diagnostics',
        'matched ordinary PL and original Hybrid for each selected seed']
    record['metrics_plan']['dimensions'] = ['five configurations', '13 selected seeds', 'two auxiliary modes',
                                             'clean and six practical', 'source U packet trajectory']
    for entry, row in zip(record['rows'], candidate_rows()):
        entry.update(method=f"Hybrid confidence >=0.95; isolated auxiliary view; {row['tracked_aux_mode']}; weight 0.1",
                     purpose='test-selected failing seeds, exploratory')
    record['notes'] = [
        'Original Hybrid vs matched ordinary PL determines failing configurations and seeds by clean or six-scene mean; test already exposed, no population-average inference.',
        'All confident source U receive an isolated constant-strength 0.1 auxiliary view; standard U uses its original ordinary/raw partition.',
        'Compare pseudo-class-aware packet contrast with same-packet positive-only cosine; no U true labels are read.',
        'Each U exposure records base_index, source RX/day, confidence, pseudo-label, acceptance, auxiliary-view predictions and auxiliary loss; gradient dot/norm every 50 steps.',
        'Use own shared scheduler with existing low-weight queue and preserve all healthy workers.']
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();value=make(json.loads(Path(a.template).read_text(encoding='utf-8-sig')))
    Path(a.output).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
