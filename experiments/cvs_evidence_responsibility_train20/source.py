"""Scratch E20 entry using the shared native source/default-head workflow."""
import argparse
from . import design as d, model as model_api
from .training_observer import TrainingObserver
from experiments.cvs_feature_sequence50.source import train as shared_train


def train(config):
    d.validate(config)
    return shared_train(config, design=d, model_api=model_api, observer=TrainingObserver(config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    train(d.read(parser.parse_args().config))
