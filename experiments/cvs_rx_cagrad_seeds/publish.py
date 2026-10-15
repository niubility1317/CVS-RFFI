import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_rx_cagrad50.publish import configure as previous, INSPECT as first_inspect

INSPECT = first_inspect.replace('experiments.cvs_rx_cagrad50', 'experiments.cvs_rx_cagrad_seeds')
INSPECT = (INSPECT.replace("done['models']==3", "done['models']==6")
    .replace("done['new_trained_models']==2", "done['new_trained_models']==4")
    .replace("done['prediction_count']==504000", "done['prediction_count']==1008000")
    .replace('len(workers)==3', 'len(workers)==6'))


def configure():
    previous(); transport.d = d
    transport.PATHS = list(dict.fromkeys([*transport.PATHS,
        'experiments/cvs_backbone_fishr_seeds', 'experiments/cvs_rx_cagrad_seeds']))
    transport.REMOTE = transport.REMOTE.replace('experiments.cvs_rx_cagrad50.preflight',
        'experiments.cvs_rx_cagrad_seeds.preflight').replace('experiments.cvs_rx_cagrad50.dispatch',
        'experiments.cvs_rx_cagrad_seeds.dispatch')
    transport.INSPECT = INSPECT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true'); args = parser.parse_args()
    configure()
    value = transport.inspect(args.output) if args.inspect else transport.publish(args.output)
    print(json.dumps(dict(status=value['status'], workers=len(value['workers']),
        epochs=[w['epoch_count'] for w in value['workers']], completion=value['completion'], failure=value['failure'])))
