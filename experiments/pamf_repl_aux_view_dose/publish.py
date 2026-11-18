"""Publish the fixed-view auxiliary dose matrix on N607."""
import argparse

from experiments.pamf_repl_dual_optimization.publish import publish
from .plan import RUN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for option in ('repo', 'output', 'context', 'ssh-config'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args()
    publish(args.repo, args.output, args.context, args.ssh_config,
            run_id=RUN, package='experiments.pamf_repl_aux_view_dose')
