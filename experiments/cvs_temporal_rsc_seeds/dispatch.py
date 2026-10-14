import argparse
from pathlib import Path
from types import SimpleNamespace
from . import design as d
from .source import train
from .predict import source_proof, predict
from .score import score, finish
from experiments.cvs_signature_stats50.dispatch import worker, dispatch, replace

MODULE = 'experiments.cvs_temporal_rsc_seeds.dispatch'


def restore_project(value, project):
    """Expand the public project handle; preserve every registered config field."""
    if isinstance(value,str):
        return project+value[len('PROJECT'):] if value=='PROJECT' or value.startswith('PROJECT/') else value
    if isinstance(value,list):return [restore_project(v,project) for v in value]
    if isinstance(value,dict):return {k:restore_project(v,project) for k,v in value.items()}
    return value


def registered_design():
    def read(path):
        value=d.read(path)
        return restore_project(value,d.PROJECT) if Path(path)==d.ROOT/'registration/experiment.json' else value
    return SimpleNamespace(**dict(vars(d),read=read))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker'); parser.add_argument('--predict'); parser.add_argument('--score')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, design=d, train_fn=train, proof_fn=source_proof, module=MODULE)
    elif args.predict:
        predict(args.predict); replace('--score', args.predict, MODULE)
    elif args.score:
        score(args.score)
    else:
        d.require_source()
        dispatch(design=registered_design(), finish_fn=finish, module=MODULE)


if __name__ == '__main__':
    main()
