import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_risk_variance.publish import configure as previous,INSPECT as original
OLD='models=1,source_packets=900,forward_packets=900,gradient_batches=5,layer_records=20,saved_vectors=40,source_only=True'
NEW='models=2,source_packets=1800,forward_packets=1800,gradient_batches=10,layer_records=40,saved_vectors=80,physical_unique_packets=900,source_only=True'
if OLD not in original:raise ValueError('Original source completion inspector changed')
INSPECT=original.replace('experiments.cvs_source_risk_variance','experiments.cvs_source_risk_variance_seeds').replace(OLD,NEW)
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_backbone_fishr_seeds','experiments/cvs_receiver_dro_seeds','experiments/cvs_source_risk_variance_seeds']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_source_risk_variance.preflight','experiments.cvs_source_risk_variance_seeds.preflight').replace('experiments.cvs_source_risk_variance.audit','experiments.cvs_source_risk_variance_seeds.audit')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
