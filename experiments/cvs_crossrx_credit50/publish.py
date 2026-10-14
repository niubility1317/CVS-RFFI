import argparse
import json
from pathlib import Path
from . import design as d
from experiments.cvs_feature_sequence50.publish import configure as parent
from experiments.cvs_report_ssl import publish as transport

def configure():
 parent();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_crossrx_credit50',
                                  'experiments/cvs_semantic_order50','experiments/cvs_u_schedule50']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_feature_sequence50.preflight','experiments.cvs_crossrx_credit50.preflight').replace('experiments.cvs_feature_sequence50.dispatch','experiments.cvs_crossrx_credit50.dispatch')
 transport.INSPECT=transport.INSPECT.replace('from experiments.cvs_feature_sequence50 import design as d','from experiments.cvs_crossrx_credit50 import design as d').replace("done['rows']!=24 or done['readouts']!=24 or done['prediction_count']!=4032000 or len(result['workers'])!=24","done['rows']!=12 or done['readouts']!=12 or done['prediction_count']!=2016000 or len(result['workers'])!=12")
 transport.INSPECT=transport.INSPECT.replace("if result['completion'] and not result['failure']:","if (result.get('queue') or {}).get('failures'):result['status']='FAILED'\nif result['completion'] and not result['failure']:")

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true')
 a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
