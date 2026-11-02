"""Use the established source-only DURA trainer with the new spectral stages."""

import argparse
import sys

from experiments.phase1_dura_posterior import train_source as base
from experiments.phase1_dura_spectral_prior.integrated import SpectralPosteriorInDura


class StrictParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main():
    parser = StrictParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--root')
    parser.add_argument('--out')
    parser.add_argument('--commit')
    parser.add_argument('--device')
    parser.add_argument('--posterior-kind', choices=('mean',), default='mean')
    parser.add_argument('--smoke', action='store_true')
    parser.parse_args(sys.argv[1:])
    original = base.PosteriorInDura
    base.PosteriorInDura = SpectralPosteriorInDura
    try:
        base.main()
    finally:
        base.PosteriorInDura = original


if __name__ == '__main__':
    main()
