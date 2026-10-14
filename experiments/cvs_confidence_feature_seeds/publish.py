import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_confidence_feature50.publish import configure as previous,INSPECT as first_inspect
INSPECT=first_inspect.replace('experiments.cvs_confidence_feature50','experiments.cvs_confidence_feature_seeds').replace("done['models']==3","done['models']==6").replace("done['new_trained_models']==2","done['new_trained_models']==4").replace("done['prediction_count']==504000","done['prediction_count']==1008000").replace('len(workers)==3','len(workers)==6')
def configure():
 previous();transport.d=d;transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_confidence_feature_seeds']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_confidence_feature50.preflight','experiments.cvs_confidence_feature_seeds.preflight').replace('experiments.cvs_confidence_feature50.dispatch','experiments.cvs_confidence_feature_seeds.dispatch');transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
