import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_variance_angle.publish import configure as previous
from experiments.cvs_source_gate_audit.publish import INSPECT as original_inspect

INSPECT = original_inspect.replace('experiments.cvs_source_gate_audit',
    'experiments.cvs_source_variance_angle_seeds').replace('models=3,source_only=True',
    'models=2,source_packets=1800,forward_packets=1800,gradient_batches=10,layer_records=20,saved_vectors=60,source_only=True')


def configure():
    previous(); transport.d = d
    transport.PATHS = list(dict.fromkeys([*transport.PATHS,
        'experiments/cvs_backbone_fishr_seeds', 'experiments/cvs_source_variance_angle_seeds']))
    transport.REMOTE = transport.REMOTE.replace('experiments.cvs_source_variance_angle.preflight',
        'experiments.cvs_source_variance_angle_seeds.preflight').replace('experiments.cvs_source_variance_angle.audit',
        'experiments.cvs_source_variance_angle_seeds.audit')
    transport.INSPECT = INSPECT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true'); args = parser.parse_args()
    configure()
    value = transport.inspect(args.output) if args.inspect else transport.publish(args.output)
    print(json.dumps(dict(status=value['status'], process_live=bool(value['process']),
                         completion=value['completion'], failure=value['failure'], log_bytes=value['log_bytes'])))
