"""Publish the six-source-receiver source-development sweep."""
import argparse

from .source_tuning_m6 import RUN
from .source_tuning_publish import publish


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN,
            protocol_id='m6-natural', setting='source_rx_6',
            worker_module='experiments.pamf_physics_tri.source_tuning_m6', gpus=(2, 6, 7, 1, 3),
            max_total_processes_per_gpu=2)
