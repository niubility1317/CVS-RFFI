import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_sigtime_pair_seeds.publish import configure as previous
from experiments.cvs_source_gate_audit.publish import INSPECT as source_inspect
INSPECT=source_inspect.replace('experiments.cvs_source_gate_audit','experiments.cvs_source_pseudo_gradient')
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_source_pseudo_gradient']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_sigtime_pair_seeds.preflight','experiments.cvs_source_pseudo_gradient.preflight').replace('experiments.cvs_sigtime_pair_seeds.dispatch','experiments.cvs_source_pseudo_gradient.audit').replace('env=dict(os.environ,PYTHONPATH=','env=dict(os.environ,CUDA_VISIBLE_DEVICES=\'\',PYTHONPATH=')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],process_live=bool(v['process']),completion=v['completion'],failure=v['failure'],log_bytes=v['log_bytes'])))
