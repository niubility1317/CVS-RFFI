"""Publish the fixed three-seed Tri-SIG-TIME physical-L continuation."""
import argparse

from experiments.pamf_branch_training import publish as base
from experiments.pamf_repl_view_shuffle.plan import RUN as TRI_RUN
from .plan import RUN, SEEDS


def publish(repo, output, context, ssh_config, *, pairing=False, recovery=False, partial_recovery=False,
            practical_gate=False, weak_practical=False, scheduled_practical=False, validate_taper=False,
            branch_guard=False, single_rx=False, nonmain_takeover=False, source_probe=False):
    from .pair_plan import RUN as PAIR_RUN
    from .partial_plan import RUN as PARTIAL_RUN
    from .practical_plan import RUN as PRACTICAL_RUN
    from .recovery_plan import RUN as RECOVERY_RUN
    from .weak_practical_plan import RUN as WEAK_RUN
    from .scheduled_plan import RUN as SCHEDULE_RUN
    from .validation_plan import RUN as VALIDATION_RUN, SETTINGS
    from .branchguard_plan import RUN as BRANCHGUARD_RUN
    from .single_rx_plan import RUN as SINGLE_RX_RUN, SETTING as SINGLE_SETTING, PROTOCOL as SINGLE_PROTOCOL
    from .nonmain_takeover import RUN as TAKEOVER_RUN
    SOURCE_PROBE_RUN = '20261111-phase1-pamf-single-rx-source-v-probe-r01'

    old_run, old_remote = base.RUN, base.REMOTE
    base.RUN, gpus = RUN, [0, 1]
    if pairing:
        base.RUN, gpus = PAIR_RUN, list(range(5))
    elif recovery:
        base.RUN, gpus = RECOVERY_RUN, list(range(5))
    elif partial_recovery:
        base.RUN, gpus = PARTIAL_RUN, [5, 6, 7]
    elif practical_gate:
        base.RUN, gpus = PRACTICAL_RUN, list(range(8))
    elif weak_practical:
        base.RUN, gpus = WEAK_RUN, list(range(8))
    elif scheduled_practical:
        base.RUN, gpus = SCHEDULE_RUN, list(range(8))
    elif validate_taper:
        base.RUN, gpus = VALIDATION_RUN, list(range(8))
    elif branch_guard:
        base.RUN, gpus = BRANCHGUARD_RUN, list(range(8))
    elif single_rx:
        base.RUN, gpus = SINGLE_RX_RUN, list(range(8))
    elif nonmain_takeover:
        base.RUN, gpus = TAKEOVER_RUN, [6, 7]
    elif source_probe:
        base.RUN, gpus = SOURCE_PROBE_RUN, list(range(8))
    owner = f'codex-{base.RUN}'
    runner = ('experiments.pamf_physics_tri.single_rx_source_probe' if source_probe else
              'experiments.pamf_physics_tri.nonmain_takeover' if nonmain_takeover else
              'experiments.pamf_physics_tri.pair_runner' if base.RUN != RUN else 'experiments.pamf_physics_tri.runner')
    preflight = (
        f"for seed in {SEEDS!r}:\n"
        f" path=project/'runs'/{TRI_RUN!r}/'F'/f'main_split-repl-shuffled_time-3view-s{{seed}}'/'source'/'segment20'\n"
        " if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)\n"
    )
    if validate_taper or branch_guard or single_rx:
        from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN
        selected_settings = {SINGLE_SETTING: SINGLE_PROTOCOL} if single_rx else SETTINGS
        preflight += (
            f"for setting,pid in {selected_settings!r}.items():\n"
            " for path in [ctx['handles'][pid]['source_cache'],ctx['handles'][pid]['source_contract'],"
            "*[ctx['handles'][pid]['tests'][t]['views_root'] for t in ('cross_date','cross_rx','cross_both')]]:\n"
            "  if not Path(path).exists():raise FileNotFoundError(path)\n"
            " for v in ctx['caches'][pid].values():\n"
            "  d=json.loads((Path(v['path'])/'complete.json').read_text())\n"
            "  if d['status']!='DURA_IQ_CACHE_COMPLETE' or d['truth_read']:raise ValueError('Missing matching DURA cache')\n"
            f" for seed in {SEEDS!r}:\n"
            f"  path=project/'runs'/{PAPER_RUN!r}/'F'/f'{{setting}}-factorial-ssl-full-permuted-s{{seed}}'/'source'/'segment20'\n"
            "  if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)\n"
        )
    base.REMOTE = (old_remote.replace("rt=p['runtime']; project=",
                                    f"rt=p['runtime']; rt['parent_seeds']=[]; rt['allowed_gpus']={gpus!r}; project=")
        .replace("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
                 "[rt['python'],*rt['frozen_sources'].values()]")
        .replace("for v in ctx['caches']['F0'].values():", preflight + "for v in ctx['caches']['F0'].values():")
        .replace("str(release/'experiments/pamf_branch_training')",
                 "str(release/'experiments/pamf_physics_tri'),str(release/'experiments/pamf_physics_branch'),"
                 "str(release/'experiments/pamf_repl_completion'),str(release/'experiments/pamf_repl_view_shuffle'),"
                 "str(release/'experiments/pamf_ssl_research')")
        .replace('experiments.pamf_branch_training.runner', runner))
    try:
        base.publish(repo, output, context, ssh_config, launch_owner=owner)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'output', 'context', 'ssh_config'):
        p.add_argument('--' + key.replace('_', '-'), required=True)
    experiment = p.add_mutually_exclusive_group()
    experiment.add_argument('--pairing', action='store_true')
    experiment.add_argument('--recovery', action='store_true')
    experiment.add_argument('--partial-recovery', action='store_true')
    experiment.add_argument('--practical-gate', action='store_true')
    experiment.add_argument('--weak-practical', action='store_true')
    experiment.add_argument('--scheduled-practical', action='store_true')
    experiment.add_argument('--validate-taper', action='store_true')
    experiment.add_argument('--branch-guard', action='store_true')
    experiment.add_argument('--single-rx', action='store_true')
    experiment.add_argument('--nonmain-takeover', action='store_true')
    experiment.add_argument('--source-probe', action='store_true')
    a = p.parse_args()
    publish(a.repo, a.output, a.context, a.ssh_config, pairing=a.pairing, recovery=a.recovery,
            partial_recovery=a.partial_recovery, practical_gate=a.practical_gate, weak_practical=a.weak_practical,
            scheduled_practical=a.scheduled_practical, validate_taper=a.validate_taper, branch_guard=a.branch_guard,
            single_rx=a.single_rx, nonmain_takeover=a.nonmain_takeover, source_probe=a.source_probe)
