import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_feature_ce_gradient.publish import configure as previous,INSPECT as old_inspect
INSPECT=old_inspect.replace('experiments.cvs_source_feature_ce_gradient','experiments.cvs_source_teacher_confidence').replace('source_packets=37800,forward_packets=75600','source_packets=199800,forward_packets=399600').replace('saved_vectors=4950','saved_vectors=7200')
def configure():
 previous();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_teacher_confidence']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_source_feature_ce_gradient.preflight','experiments.cvs_source_teacher_confidence.preflight').replace('experiments.cvs_source_feature_ce_gradient.audit','experiments.cvs_source_teacher_confidence.audit');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
