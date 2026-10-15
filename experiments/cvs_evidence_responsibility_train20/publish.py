import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_crossrx_sharpness20.publish import configure as parent
from experiments.cvs_report_ssl import publish as transport

def configure():
 parent();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_evidence_responsibility_train20','experiments/cvs_evidence_responsibility20','experiments/cvs_identity_correspondence20','experiments/cvs_identity_transport20','experiments/cvs_evidence_recomposition20','experiments/cvs_identity_difference20','experiments/cvs_restart_state20',
  'experiments/cvs_crossrx_sharpness50','experiments/cvs_sigtime_heldcredit']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_crossrx_sharpness20.preflight','experiments.cvs_evidence_responsibility_train20.preflight').replace('experiments.cvs_crossrx_sharpness20.dispatch','experiments.cvs_evidence_responsibility_train20.dispatch')
 transport.INSPECT=transport.INSPECT.replace('from experiments.cvs_crossrx_sharpness20 import design as d','from experiments.cvs_evidence_responsibility_train20 import design as d').replace("done['rows']!=15 or done['readouts']!=15 or done['prediction_count']!=2520000 or len(result['workers'])!=15","done['rows']!=12 or done['readouts']!=12 or done['prediction_count']!=2016000 or len(result['workers'])!=12")

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true')
 a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
