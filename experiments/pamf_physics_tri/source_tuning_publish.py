"""Publish the source-development pilot using the existing release mechanism."""
import argparse

from experiments.pamf_branch_training import publish as base
from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN
from .source_tuning_plan import RUN


def publish(repo, output, context, ssh_config, *, run_id=RUN, protocol_id='ManyRx', setting='ten_class',
            worker_module='experiments.pamf_physics_tri.source_tuning_runner', gpus=(0, 1, 3, 4, 5, 7),
            max_total_processes_per_gpu=1):
    old_run, old_remote = base.RUN, base.REMOTE
    script = old_remote
    substitutions = [
        ("rt=p['runtime']; project=", f"rt=p['runtime']; rt['allowed_gpus']={list(gpus)!r}; "
         f"rt['worker_module']={worker_module!r}; rt['max_total_processes_per_gpu']={max_total_processes_per_gpu!r}; "
         "rt['parent_seeds']=[]; project="),
        ("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
         "[rt['python'],*rt['frozen_sources'].values()]"),
        ("h=ctx['handles']['F0']", f"h=ctx['handles'][{protocol_id!r}]"),
        ("[h['source_cache'],h['source_contract'],*[h['tests'][t]['views_root'] for t in ('cross_date','cross_rx','cross_both')]]",
         "[h['source_cache'],h['source_contract']]"),
        ("for v in ctx['caches']['F0'].values():\n d=json.loads((Path(v['path'])/'complete.json').read_text())\n if d['status']!='DURA_IQ_CACHE_COMPLETE' or d['truth_read']:raise ValueError('Frozen cache missing')",
         f"path=project/'runs'/{PAPER_RUN!r}/'F'/{(setting + '-factorial-ssl-full-permuted-s2026092701')!r}/'source'/'segment20'\n"
         "if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)"),
        ("str(release/'experiments/pamf_branch_training')", "str(release/'experiments/pamf_physics_tri')"),
        ('experiments.pamf_branch_training.runner', worker_module),
    ]
    for original, replacement in substitutions:
        if original not in script:
            raise ValueError('Release template changed: source-only preflight cannot be generated')
        script = script.replace(original, replacement)
    base.RUN, base.REMOTE = run_id, script
    try:
        return base.publish(repo, output, context, ssh_config, launch_owner='codex-' + run_id)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config)
