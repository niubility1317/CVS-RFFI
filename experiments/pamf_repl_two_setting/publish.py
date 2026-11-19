import argparse

from experiments.pamf_repl_dual_optimization.publish import publish
from .plan import GPUS, RUN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config,
            run_id=RUN, package='experiments.pamf_repl_two_setting', gpus=GPUS)
