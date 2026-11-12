"""Publish the frozen seven-view evaluation of the completed models."""
import argparse

from .development_evaluation_plan import RUN
from .source_tuning_publish import publish


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN,
            protocol_id='ManyRx', setting='ten_class',
            worker_module='experiments.pamf_physics_tri.development_evaluation',
            gpus=(0, 4, 5, 1, 3, 2, 6, 7), max_total_processes_per_gpu=2)
