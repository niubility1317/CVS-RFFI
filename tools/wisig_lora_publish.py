"""Release the registered WiSig comparison with an explicit GPU sharing policy."""

import argparse
from pathlib import Path, PurePosixPath

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main import publish as transport


RUN = '20261124-phase1-wisig-lora-recovery-comparison-m4-r01'


def publish(repo, output, context, ssh_config, resource_policy):
    ctx = read(context)
    base = PurePosixPath(ctx['project'])
    runs = base / 'runs'
    expanded = base / 'releases' / '20261124-phase1-lora-dura-capacity-source-m4-r01-release'
    guard = base / 'releases' / '20261124-phase1-lora-rank-guard-extension-source-m2-r01-release'
    rt = dict(project=ctx['project'], python=ctx['python'], frozen_sources=ctx['frozen_sources'],
              run_id=RUN, code_commit=transport.release_commit(repo),
              runner='experiments.pamf_unrolled_restore.wisig_lora_dispatch',
              release_root=str(base / 'releases' / f'{RUN}-release'), run_root=str(runs / RUN),
              expanded_release=str(expanded), guard_release=str(guard),
              external_releases=[str(expanded), str(guard)],
              source_root=str(runs / '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02' / 'cache'),
              baseline_root=str(runs / '20261123-phase1-wisig-r69-paired-main-m6-r01'),
              baseline_replay=str(runs / '20261124-phase1-wisig-preamble-dura-main-m4-r01'),
              recognizer_run=str(runs / '20261027-phase1-pamf-repl-shuffled-views-m16-r02'),
              share_existing=resource_policy == 'shared', min_free_mib=8000)
    # Compile the worker and all experiment modules in the single release check.
    transport.RELEASE_CODE = transport.RELEASE_CODE.replace(
        "'experiments/pamf_lora_main', entry", "'experiments', 'tools/wisig_lora_replay.py', entry")
    transport.release(repo, output, rt, ssh_config,
                      archive_extras=('tools/wisig_lora_replay.py', 'tools/wisig_paired_main_uncertainty.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resource-policy', choices=('exclusive', 'shared'), required=True)
    publish(**vars(parser.parse_args()))
