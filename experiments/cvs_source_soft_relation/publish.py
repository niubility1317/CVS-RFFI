import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_self_challenge.publish import configure as previous,INSPECT as original
INSPECT=original.replace('experiments.cvs_source_self_challenge','experiments.cvs_source_soft_relation').replace('forward_packets=3600','forward_packets=1800')
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_soft_relation']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_source_self_challenge.preflight','experiments.cvs_source_soft_relation.preflight').replace('experiments.cvs_source_self_challenge.audit','experiments.cvs_source_soft_relation.audit')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inspect',action='store_true');args=parser.parse_args();configure()
 v=transport.inspect(args.output) if args.inspect else transport.publish(args.output)
 print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
