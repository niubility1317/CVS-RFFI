import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_conditional_rx_adversary.publish import INSPECT as PARENT_INSPECT
INSPECT=PARENT_INSPECT.replace("done['models']!=9","done['models']!=12").replace("done['readouts']!=9","done['readouts']!=12").replace("done['prediction_count']!=1512000","done['prediction_count']!=2016000").replace('len(workers)!=9','len(workers)!=12')

def configure():
 from experiments.cvs_shift_consistency50.publish import configure as parent
 parent();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_legacy_no_sat','experiments/cvs_six_direction_mechanisms','experiments/cvs_source_pseudotarget_health','experiments/cvs_repeated_rx_margin','experiments/cvs_feature_multiscale50']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_shift_consistency50.preflight','experiments.cvs_feature_multiscale50.preflight').replace('experiments.cvs_shift_consistency50.dispatch','experiments.cvs_feature_multiscale50.dispatch')
 transport.INSPECT=INSPECT

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure();v=transport.inspect(a.output) if a.inspect else transport.publish(a.output);print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epoch_counts=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
