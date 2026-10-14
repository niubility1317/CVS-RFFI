import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_feature_sequence50 import publish as parent
from experiments.cvs_report_ssl import publish as transport
def configure():
 parent.configure();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_time_factorial50','experiments/cvs_trusted_correction','experiments/cvs_evidence_relation']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_feature_sequence50.preflight','experiments.cvs_evidence_relation.preflight').replace('experiments.cvs_feature_sequence50.dispatch','experiments.cvs_evidence_relation.dispatch')
 transport.INSPECT=parent.INSPECT.replace('from experiments.cvs_feature_sequence50 import design as d','from experiments.cvs_evidence_relation import design as d').replace("done['rows']!=24 or done['readouts']!=24 or done['prediction_count']!=4032000 or len(result['workers'])!=24","done['rows']!=12 or done['readouts']!=12 or done['prediction_count']!=2016000 or len(result['workers'])!=12")
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],workers=len(v['workers']),completion=v['completion'],failure=v['failure'])))
