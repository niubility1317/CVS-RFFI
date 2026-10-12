"""Immutable fixed-source diagnostic release; no new identity training or target."""
import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_finite_action.publish import INSPECT
def configure():
 from experiments.cvs_e20_response.publish import configure as parent
 parent();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_response_identifiability']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_e20_response.preflight','experiments.cvs_response_identifiability.preflight').replace('experiments.cvs_e20_response.dispatch','experiments.cvs_response_identifiability.dispatch')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),complete=v['completion'],failure=v['failure'])))
