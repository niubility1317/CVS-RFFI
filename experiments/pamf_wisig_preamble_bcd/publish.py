"""Package the pushed B/C/D release with the matched Main-split handles."""

import argparse
from pathlib import Path

from experiments.pamf_wisig_branch_style.publish import publish

from .plan import OWNER, RUN
from .runner import MODULE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    publish(Path.cwd(), args.output, args.context, run_id=RUN, module=MODULE, owner=OWNER)
