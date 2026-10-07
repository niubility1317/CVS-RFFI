"""Reuse the audited source loop in a separate process with the new frozen design."""
import argparse
from experiments.cvs_phase1_repair import design
from experiments.cvs_phase1_stack import source


def train(c):
    source.validate = design.validate
    source.make_args = design.make_args
    source.train(c)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    train(design.read(p.parse_args().config))
