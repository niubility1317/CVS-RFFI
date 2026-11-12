"""Publish only the independent 1%-labeled source-development pilot."""
import argparse

from .source_tuning_label01 import RUN
from .source_tuning_publish import publish


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN, protocol_id='label01',
            setting='labeled_1pct', worker_module='experiments.pamf_physics_tri.source_tuning_label01',
            gpus=(4, 5, 7))
