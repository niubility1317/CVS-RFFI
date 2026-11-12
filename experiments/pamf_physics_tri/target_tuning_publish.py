"""Publish the fixed full-source, per-row seven-view comparison."""
import argparse

from .source_tuning_publish import publish
from .target_tuning_plan import RUN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN,
            protocol_id='ManyRx', setting='ten_class',
            worker_module='experiments.pamf_physics_tri.target_tuning_runner',
            gpus=(0, 1, 3, 4, 5, 7, 2, 6), max_total_processes_per_gpu=2)
