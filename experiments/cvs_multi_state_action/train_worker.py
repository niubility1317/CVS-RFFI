"""Visible pre-CUDA reservation recognized by existing capacity controllers."""
import argparse
from experiments.cvs_multi_state_action.dispatch import worker


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['audit', 'source', 'predict'], required=True)
    parser.add_argument('--row', required=True)
    args = parser.parse_args()
    worker(args.kind, args.row)
