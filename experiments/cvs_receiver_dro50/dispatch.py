import argparse
from . import design as d
from .source import train
from .predict import source_proof, predict
from .score import score, finish
from experiments.cvs_signature_stats50.dispatch import worker, dispatch, replace

MODULE = 'experiments.cvs_receiver_dro50.dispatch'


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
        dispatch(design=d, finish_fn=finish, module=MODULE)


if __name__ == '__main__':
    main()
