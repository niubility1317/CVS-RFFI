import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_source_soft_relation.publish import configure as previous,INSPECT as original
INSPECT=original.replace('experiments.cvs_source_soft_relation','experiments.cvs_source_rdrop').replace('forward_packets=1800','forward_packets=2700').replace('saved_vectors=60','saved_vectors=40')
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_rdrop']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_source_soft_relation.preflight','experiments.cvs_source_rdrop.preflight').replace('experiments.cvs_source_soft_relation.audit','experiments.cvs_source_rdrop.audit')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
