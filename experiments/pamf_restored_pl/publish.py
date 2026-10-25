"""Use the existing immutable archive release, without handing off other jobs."""
import argparse
from pathlib import Path
import subprocess
from experiments.pamf_channel_pl.publish_source import publish
from .plan import RUN


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    dirty=subprocess.check_output(['git','-C',str(Path(a.repo)),'status','--porcelain','--','experiments/pamf_restored_pl'])
    if dirty.strip():raise ValueError('Committed restored-PL adapter required')
    publish(a.repo,a.output,a.context,a.release_id,run_id=RUN,entrypoint='experiments.pamf_restored_pl.runner',
        launch_owner='codex-pamf-restored-pl-20261025',allowed_gpus=range(8))
