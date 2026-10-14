import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_temporal_rsc50.publish import configure as previous, INSPECT as first_inspect

INSPECT = first_inspect.replace('experiments.cvs_temporal_rsc50', 'experiments.cvs_temporal_rsc_seeds')
INSPECT = (INSPECT.replace("done['models']==4", "done['models']==8")
    .replace("done['new_trained_models']==3", "done['new_trained_models']==6")
    .replace("done['prediction_count']==672000", "done['prediction_count']==1344000")
    .replace('len(workers)==4', 'len(workers)==8'))


def configure():
    previous(); transport.d = d
    transport.PATHS = list(dict.fromkeys([*transport.PATHS,
        'experiments/cvs_backbone_fishr_seeds', 'experiments/cvs_temporal_rsc_seeds']))
    transport.REMOTE = transport.REMOTE.replace('experiments.cvs_temporal_rsc50.preflight',
        'experiments.cvs_temporal_rsc_seeds.preflight').replace('experiments.cvs_temporal_rsc50.dispatch',
        'experiments.cvs_temporal_rsc_seeds.dispatch')
    transport.INSPECT = INSPECT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true'); args = parser.parse_args()
    configure()
    value = transport.inspect(args.output) if args.inspect else transport.publish(args.output)
    print(json.dumps(dict(status=value['status'], workers=len(value['workers']),
        epochs=[w['epoch_count'] for w in value['workers']], completion=value['completion'], failure=value['failure'])))
