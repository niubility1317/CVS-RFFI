"""One archive containing the reviewed common code and fixed confirmation."""
import argparse
from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original


def publish(output):
    return original(output,design=d,package='experiments/cvs_ordinary_confirm20',
        folders=['experiments/cvs_ordinary_fold20','experiments/cvs_ordinary_confirm20'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);publish(parser.parse_args().output)
