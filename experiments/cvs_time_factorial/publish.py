import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_pa_cross_delay.publish import configure as pa_configure,INSPECT as pa_inspect
INSPECT=pa_inspect.replace('experiments.cvs_pa_cross_delay','experiments.cvs_time_factorial').replace("'LEGACY50'","'OLD_GATE_COS50'").replace("done['models']==9","done['models']==15").replace("done['new_trained_models']==6","done['new_trained_models']==12").replace("done['prediction_count']==1512000","done['prediction_count']==2520000").replace('len(workers)==9','len(workers)==15')
def configure():
 pa_configure();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_time_factorial','experiments/cvs_feature_multiscale50','experiments/cvs_transfer_features','experiments/cvs_feature_diagnostics']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_pa_cross_delay.preflight','experiments.cvs_time_factorial.preflight').replace('experiments.cvs_pa_cross_delay.dispatch','experiments.cvs_time_factorial.dispatch')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
