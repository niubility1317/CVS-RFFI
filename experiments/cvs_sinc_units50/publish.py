import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_temporal_rsc50.publish import configure as previous
from experiments.cvs_feature_mean_teacher50.publish import INSPECT as base_inspect
INSPECT=base_inspect.replace('experiments.cvs_feature_mean_teacher50','experiments.cvs_sinc_units50')
def configure():
 previous();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_sinc_units50','experiments/cvs_feature_mean_teacher50','experiments/cvs_sigtime_pair50','experiments/cvs_source_neighbor_pairs','experiments/cvs_source_gate_audit']))
 assert all(p in transport.PATHS for p in ('experiments/cvs_sigtime_pair50','experiments/cvs_source_neighbor_pairs','experiments/cvs_source_gate_audit'))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_temporal_rsc50.preflight','experiments.cvs_sinc_units50.preflight').replace('experiments.cvs_temporal_rsc50.dispatch','experiments.cvs_sinc_units50.dispatch')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
