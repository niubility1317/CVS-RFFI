"""Release the fixed source-only coverage study on N607."""
import argparse

from .coverage_plan import RUN
from .source_tuning_publish import publish


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config, run_id=RUN,
            protocol_id='ManyRx', setting='ten_class',
            worker_module='experiments.pamf_physics_tri.coverage_runner',
            gpus=(0, 1, 3, 4, 5, 7, 2, 6), max_total_processes_per_gpu=2)
