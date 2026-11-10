"""Fill a minimal registry record for the matched PTBR pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from experiments.pamf_branch_training.register_spec import make as old_spec
from .plan import RUN, candidate_rows


def make(template):
    s = old_spec(template)
    run = 'PROJECT/runs/' + RUN
    release = 'PROJECT/releases/' + RUN + '-release'
    s.update(run_id=RUN, group_id='phase1-pamf-physics-branch-relations',
             display_name='PAMF 物理干预方向校正半监督单种子对照',
             description='同包有界两径扰动测量实际多分支有限差分；匹配比较 beta=0 与 beta=0.5。',
             aliases=['PAMF-PTBR-pilot'], tags=['PAMF', 'SSL', 'physics-aware', 'matched-control', 'truth-last'],
             parent_run_ids=['20261022-phase1-T1-DURA-F-ablation-binding-m63-r01',
                             '20261024-phase1-PAMF-REPL-completion-m201-r01',
                             '20261110-phase1-pamf-branch-relations-f0-s2026092701-m4-r01'],
             authorization='2026-11-10 用户要求训练机制兼具多分支与物理感知，并授权 N607 GPU 0/1 每卡两个实验进程。')
    s['code'].update(commit='Recorded immutable release commit in runtime.json at launch', cwd=release)
    s['data']['roles'].update(L_s='true labels; up to 64 source records per L step for echo directions',
                              U_s='hidden TX; ordinary PL threshold .95; select at most 128 U independently of acceptance, use accepted subset for relations')
    s['checkpoint']['initialization'] = 'matched F21 scratch E20, then own 10+20; no prior candidate checkpoint'
    s['checkpoint']['sources'] = ['same-contract F21 scratch E20; preexisting ordinary PL seven-view scores; previous PL weak scores if complete']
    s['checkpoint']['contract_check_ref'] = 'pamf_physics_branch.source.checked_parent using actual F21 payload and source roles'
    s['execution'].update(launch_owner='codex-pamf-ptbr-20261110',
        gpu_policy='GPU0/1; prior run owner must be terminal; max two total observed/pending processes per card; memory admission',
        remote_run_root=run, remote_log_root=run + '/logs',
        local_artifact_root='local_artifacts/pamf_physics_branch_20261110',
        launch_command='PYTHON -u -m experiments.pamf_physics_branch.runner dispatch --config ' + release + '/runtime.json')
    s['expected_artifacts'] = ['two-row real E20 source-only finite backward smoke',
        'two own E30 and E50 source trajectories with L/U direction exposure counts',
        'full L_s/V raw/echo, pseudo-label proxy, branch direction, L-derived class-subspace overlap and reference-delay diagnostics; 720 L and 720 V on each of three original As scenes',
        'all per-row practical and weak predictions frozen before truth-side scoring',
        'same-seed paired raw/DURA/weak comparison with reused ordinary PL', 'PID/CWD/GPU/log readback']
    rows = []
    for row in candidate_rows():
        rid = row['row_id']
        r = deepcopy(template['rows'][0])
        r.update(row_id=rid, physical_row=row,
                 method='ordinary PL + physical branch direction relations beta=' + str(row['beta']),
                 purpose='matched physical-direction ablation',
                 config_ref=run + '/configs/train__' + rid + '.json',
                 resolved_config_ref=run + '/F/' + rid + '/source/resolved_config.json',
                 data_overrides={'protocol_id': 'F0'},
                 seeds=dict(model=2026092701, split=392005, data=None, augmentation=20261016421,
                            support=None, evaluation=None),
                 seed_notes='Original F0 seeds; echo has fixed source RNG by global step; weak view seeds in manifest; no support.',
                 scenario='clean and six practical in three F0 populations; raw three weak scenes on combined target',
                 optimizer='unchanged native AdamW/CENTER_UNITS', lr='original native segment schedule', epochs=50,
                 budget_ref='reused E20/4440; own E10+E20/6660; L64/U<=128 each use two no-grad forwards',
                 output_root=run + '/F/' + rid + '/source', log_path=run + '/logs/train__' + rid + '.log',
                 command='worker --kind train --config ' + run + '/configs/train__' + rid + '.json',
                 expected_artifacts=['source/complete.json', 'source/segment30/physical_state.json',
                                     'source/segment50/physical_state.json', 'readouts/ROW/scores/complete.json',
                                     'readouts/ROW/weak_scores/complete.json'])
        rows.append(r)
    s['rows'] = rows
    s['notes'] = ['Exploratory beta=0.5 versus matched beta=0 primary test; no significance claim from one seed.',
        'Post-reception finite echo is a bounded propagation proxy, not identified real transmitter or full satellite channel.',
        'Complete candidate/baseline source diagnostics before any candidate target prediction; all fixed rows then evaluated without selection.',
        'Expansion screen: final source V beta=0.5 no worse than ordinary PL by 0.2 pp or beta=0 by 0.1 pp; report actual anchors and nonfinite counts.',
        'Existing branch pilot runs to its terminal state before this owner uses the same GPU reservation.',
        'Fixed final checkpoint; target results do not select, modify or restart candidates.']
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = make(json.loads(Path(args.template).read_text(encoding='utf-8-sig')))
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')
