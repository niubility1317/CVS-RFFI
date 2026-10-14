import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_evidence_relation.publish import configure as parent
from experiments.cvs_report_ssl import publish as transport
def configure():
 parent();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_u_schedule50']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_evidence_relation','experiments.cvs_u_schedule50')
 transport.INSPECT=transport.INSPECT.replace('from experiments.cvs_evidence_relation import design as d','from experiments.cvs_u_schedule50 import design as d').replace("done['rows']!=12 or done['readouts']!=12 or done['prediction_count']!=2016000 or len(result['workers'])!=12","done['rows']!=9 or done['readouts']!=9 or done['prediction_count']!=1512000 or len(result['workers'])!=9")
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
