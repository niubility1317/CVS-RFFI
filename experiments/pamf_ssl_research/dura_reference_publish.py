"""Publish the independent nine-row evaluation supplement without changing training."""
import argparse
from .dura_reference import RUN, prepare
from .publish import publish


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('repo', 'context', 'output', 'ssh-config'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    publish(a.repo, a.context, a.output, a.ssh_config, prepare_run=prepare, run_id=RUN)
