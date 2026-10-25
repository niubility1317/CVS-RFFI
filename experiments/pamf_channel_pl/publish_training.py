"""The independent training/evaluation release, using the same single-archive route."""
import argparse
from .publish_source import publish
from .training_plan import TRAIN_RUN

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    p.add_argument('--context',required=True);p.add_argument('--release-id',required=True);a=p.parse_args()
    publish(a.repo,a.output,a.context,a.release_id,run_id=TRAIN_RUN,
        entrypoint='experiments.pamf_channel_pl.training_runner',launch_owner='codex-channel-pl-training-20261025')
