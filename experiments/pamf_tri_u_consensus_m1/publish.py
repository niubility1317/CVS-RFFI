import argparse
from experiments.pamf_branch_training import publish as base
from experiments.pamf_paper_completion.plan import RUN as PAPER_RUN
from .plan import RUN,SEEDS


def publish(repo,output,context,ssh_config):
    old_run,old_remote=base.RUN,base.REMOTE;base.RUN=RUN
    preflight=(f"paper=project/'runs'/{PAPER_RUN!r}; sources=json.loads((paper/'sources.json').read_text())\n"
      f"for seed in {SEEDS!r}:\n"
      " rid=f'source_rx_1-factorial-ssl-full-permuted-s{seed}'; e=sources[rid]; parent=Path(e['source_output'])/'segment20'\n"
      " if not (parent/'complete.json').is_file() or not (parent/'final_ssdg.pth').is_file():raise FileNotFoundError(parent)\n"
      " for m,o in [('ssl','permuted'),('pl','ordered')]:\n"
      "  e=sources[f'source_rx_1-factorial-{m}-full-{o}-s{seed}']\n"
      "  for path in (Path(e['predictions'])/'complete.json',Path(e['scores'])/'complete.json'):\n"
      "   if not path.is_file():raise FileNotFoundError(path)\n"
      )
    base.REMOTE=(old_remote.replace("rt=p['runtime']; project=","rt=p['runtime']; rt.update(allowed_gpus=[0,1,2,3,4,5,6,7],parent_seeds=[]); project=")
      .replace("[rt['python'],rt['weak_root']+'/manifest.json',rt['weak_truth'],*rt['frozen_sources'].values()]","[rt['python'],*rt['frozen_sources'].values()]")
      .replace("h=ctx['handles']['F0']", "h=ctx['handles']['m1-natural']")
      .replace("for v in ctx['caches']['F0'].values():",preflight+"for v in ctx['caches']['m1-natural'].values():")
      .replace("str(release/'experiments/pamf_branch_training')","str(release/'experiments/pamf_tri_u_consensus_m1')")
      .replace('experiments.pamf_branch_training.runner','experiments.pamf_tri_u_consensus_m1.runner'))
    try:base.publish(repo,output,context,ssh_config,launch_owner='codex-pamf-tri-u-consensus-m1-20261111')
    finally:base.RUN,base.REMOTE=old_run,old_remote


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('repo','output','context','ssh_config'):p.add_argument('--'+key.replace('_','-'),required=True)
    a=p.parse_args();publish(a.repo,a.output,a.context,a.ssh_config)
