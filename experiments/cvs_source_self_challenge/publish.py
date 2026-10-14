import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_variance_angle.publish import configure as previous
from experiments.cvs_source_gate_audit.publish import INSPECT as original_inspect

INSPECT=original_inspect.replace('experiments.cvs_source_gate_audit','experiments.cvs_source_self_challenge').replace(
    'models=3,source_only=True','models=1,source_packets=900,forward_packets=3600,gradient_batches=5,layer_records=20,saved_vectors=60,source_only=True')


def configure():
    previous();transport.d=d
    transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_receiver_dro50','experiments/cvs_source_self_challenge']))
    transport.REMOTE=transport.REMOTE.replace('experiments.cvs_source_variance_angle.preflight','experiments.cvs_source_self_challenge.preflight').replace(
        'experiments.cvs_source_variance_angle.audit','experiments.cvs_source_self_challenge.audit')
    transport.INSPECT=INSPECT


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
    v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
    print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
