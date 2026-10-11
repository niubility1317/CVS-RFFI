"""Use the unchanged E40 optimizer and objective with a new frozen R estimator."""
import argparse
from . import design as d
from experiments.cvs_stage2_geometry.source import train

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    train(d.read(args.config),design_module=d)
