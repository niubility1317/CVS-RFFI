"""One immutable N607 release; no DURA or satellite test dependencies."""
import argparse
from experiments.pamf_branch_training import publish as base
from .plan import RUN, PREVIOUS_RUN, RECIPES


def publish(repo, output, context, ssh_config):
    old_run, old_remote = base.RUN, base.REMOTE
    script = old_remote
    substitutions = [
        ("rt=p['runtime']; project=", "rt=p['runtime']; rt['parent_seeds']=[]; rt['allowed_gpus']=[0,1,3,4,5,7]; project="),
        ("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]",
         "[rt['python'],*rt['frozen_sources'].values()]"),
        ("h=ctx['handles']['F0']", "h=ctx['handles']['ManyRx']"),
        ("for v in ctx['caches']['F0'].values():\n d=json.loads((Path(v['path'])/'complete.json').read_text())\n if d['status']!='DURA_IQ_CACHE_COMPLETE' or d['truth_read']:raise ValueError('Frozen cache missing')",
         "for pid in ('ManyRx','m6-natural'):\n h=ctx['handles'][pid]\n for path in [h['source_cache'],h['source_contract'],*[h['tests'][t]['views_root'] for t in ('cross_date','cross_rx','cross_both')]]:\n  if not Path(path).exists():raise FileNotFoundError(path)\nfor setting in ('ten_class','source_rx_6'):\n path=project/'runs'/'20261027-phase1-pamf-paper-completion-m119-r03'/'F'/f'{setting}-factorial-ssl-full-permuted-s2026092701'/'source'/'segment20'\n if not (path/'final_ssdg.pth').is_file() or not (path/'complete.json').is_file():raise FileNotFoundError(path)\n"
         + f" for recipe in {tuple(RECIPES)!r}:\n  prior=project/'runs'/{PREVIOUS_RUN!r}/'F'/f'{{setting}}-repl-{{recipe}}-s2026092701'/'source'/'segment30'\n  if not (prior/'final_ssdg.pth').is_file() or not (prior/'complete.json').is_file():raise FileNotFoundError(prior)"),
        ("str(release/'experiments/pamf_branch_training')", "str(release/'experiments/pamf_repl_weight')"),
        ('experiments.pamf_branch_training.runner', 'experiments.pamf_repl_weight.runner'),
    ]
    for before, after in substitutions:
        if before not in script:
            raise ValueError('Release preflight template differs')
        script = script.replace(before, after)
    base.RUN, base.REMOTE = RUN, script
    try:
        return base.publish(repo, output, context, ssh_config, launch_owner='codex-'+RUN)
    finally:
        base.RUN, base.REMOTE = old_run, old_remote


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('repo','output','context','ssh-config'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    publish(a.repo, a.output, a.context, a.ssh_config)
