"""Publish the source-only CPU diagnosis without occupying training GPUs."""
import argparse
from .publish_source import publish
from .consensus_source import RUN

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    publish(a.repo,a.output,a.context,a.release_id,run_id=RUN,entrypoint='experiments.pamf_channel_pl.consensus_source',
        launch_owner='codex-pl-consensus-source-20261025',allowed_gpus=())
